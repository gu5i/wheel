"""
Wheel Options Backend
=====================
FastAPI service serving option chain data for the wheel dashboard.

Defaults tuned for wheel strategy:
  - Only expirations within next MAX_DAYS days (default 90)
  - Only strikes within ±STRIKE_PCT% of spot (default 50)

Both overridable per-request:
  /chain/AAPL/all?max_days=120&strike_pct=30

Features:
  - In-memory cache (60s) to soften Yahoo rate limits
  - Retry-with-backoff on rate-limit responses
  - fast_info + 2-day history instead of expensive .info call
"""
import os
import time
from datetime import datetime, timezone
from typing import Optional

import yfinance as yf
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Wheel Options API", version="1.2.0")

allowed = os.getenv("ALLOWED_ORIGIN", "*")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[allowed] if allowed != "*" else ["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

DEFAULT_MAX_DAYS = 90
DEFAULT_STRIKE_PCT = 50
CACHE_TTL_SECONDS = 60  # cache /chain/X/all responses for 60s
_cache: dict = {}        # key -> (timestamp, response)


# ---- Helpers ---------------------------------------------------------------

def _safe_float(v) -> float:
    try:
        f = float(v)
        if f != f:  # NaN
            return 0.0
        return f
    except (TypeError, ValueError):
        return 0.0


def _safe_int(v) -> int:
    return int(_safe_float(v))


def _fetch_quote_light(t: yf.Ticker, symbol: str) -> dict:
    """Get underlying quote using fast_info + 2-day history.
    
    Way lighter than t.info (which hits ~6 internal endpoints).
    Falls back gracefully if pieces are missing.
    """
    price = change = change_pct = bid = ask = 0.0
    fifty_low = fifty_high = 0.0
    volume = 0
    short_name = symbol

    # fast_info — single request, gives price, bid/ask, 52w range, volume
    try:
        fi = t.fast_info
        price = _safe_float(fi.get("last_price")) or _safe_float(fi.get("regular_market_previous_close"))
        bid = _safe_float(fi.get("bid"))
        ask = _safe_float(fi.get("ask"))
        fifty_low = _safe_float(fi.get("year_low"))
        fifty_high = _safe_float(fi.get("year_high"))
        volume = _safe_int(fi.get("last_volume")) or _safe_int(fi.get("regular_market_volume"))
        # change comes from previous close
        prev_close = _safe_float(fi.get("previous_close")) or _safe_float(fi.get("regular_market_previous_close"))
        if price and prev_close:
            change = price - prev_close
            change_pct = (change / prev_close) * 100 if prev_close else 0
    except Exception as e:
        print(f"[warn] fast_info failed for {symbol}: {e}")

    # If price still missing, fall back to a tiny history call
    if not price:
        try:
            hist = t.history(period="2d", auto_adjust=False)
            if not hist.empty:
                price = _safe_float(hist["Close"].iloc[-1])
                if len(hist) >= 2:
                    prev = _safe_float(hist["Close"].iloc[-2])
                    change = price - prev
                    change_pct = (change / prev) * 100 if prev else 0
                volume = volume or _safe_int(hist["Volume"].iloc[-1])
        except Exception as e:
            print(f"[warn] history fallback failed for {symbol}: {e}")

    return {
        "symbol": symbol,
        "regularMarketPrice": price,
        "regularMarketChange": change,
        "regularMarketChangePercent": change_pct,
        "bid": bid,
        "ask": ask,
        "fiftyTwoWeekLow": fifty_low,
        "fiftyTwoWeekHigh": fifty_high,
        "regularMarketVolume": volume,
        "shortName": short_name,
    }


def _option_rows(df, expiration_ts: int, spot: float, strike_pct: float) -> list[dict]:
    """Convert a yfinance options DataFrame to dicts, filtered to strikes within ±strike_pct% of spot."""
    rows = []
    if df is None or df.empty or spot <= 0:
        return rows

    low_strike = spot * (1 - strike_pct / 100)
    high_strike = spot * (1 + strike_pct / 100)

    for _, r in df.iterrows():
        strike = _safe_float(r.get("strike", 0))
        if strike <= 0 or strike < low_strike or strike > high_strike:
            continue
        rows.append({
            "contractSymbol": str(r.get("contractSymbol", "")),
            "strike": strike,
            "bid": _safe_float(r.get("bid", 0)),
            "ask": _safe_float(r.get("ask", 0)),
            "lastPrice": _safe_float(r.get("lastPrice", 0)),
            "volume": _safe_int(r.get("volume", 0)),
            "openInterest": _safe_int(r.get("openInterest", 0)),
            "impliedVolatility": _safe_float(r.get("impliedVolatility", 0)),
            "inTheMoney": bool(r.get("inTheMoney", False)),
            "expiration": expiration_ts,
        })
    return rows


def _is_rate_limit(e: Exception) -> bool:
    msg = str(e).lower()
    return "too many requests" in msg or "rate limit" in msg or "429" in msg


def _with_retry(fn, *args, attempts=2, backoff=2.0, **kwargs):
    """Call fn(); if it hits a rate-limit, sleep `backoff` seconds and retry once."""
    last_exc = None
    for i in range(attempts):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last_exc = e
            if _is_rate_limit(e) and i < attempts - 1:
                time.sleep(backoff)
                continue
            raise
    raise last_exc  # unreachable


# ---- Endpoints -------------------------------------------------------------

@app.get("/")
def root():
    return {
        "service": "Wheel Options API",
        "version": "1.2.0",
        "defaults": {"max_days": DEFAULT_MAX_DAYS, "strike_pct": DEFAULT_STRIKE_PCT},
        "cache_ttl_seconds": CACHE_TTL_SECONDS,
        "endpoints": ["/health", "/quote/{symbol}", "/chain/{symbol}", "/chain/{symbol}/all"],
    }


@app.get("/health")
def health():
    return {"status": "ok", "time": datetime.now(timezone.utc).isoformat(), "cache_entries": len(_cache)}


@app.get("/quote/{symbol}")
def quote(symbol: str):
    symbol = symbol.upper().strip()
    try:
        t = yf.Ticker(symbol)
        q = _with_retry(_fetch_quote_light, t, symbol)
        if not q["regularMarketPrice"]:
            raise HTTPException(404, f"No quote data for {symbol}")
        return q
    except HTTPException:
        raise
    except Exception as e:
        if _is_rate_limit(e):
            raise HTTPException(429, f"Rate limited by Yahoo. Try again in a minute.")
        raise HTTPException(500, f"yfinance error: {e}")


@app.get("/chain/{symbol}/all")
def chain_all(
    symbol: str,
    max_days: int = Query(DEFAULT_MAX_DAYS, ge=1, le=730),
    strike_pct: float = Query(DEFAULT_STRIKE_PCT, ge=1, le=200),
):
    symbol = symbol.upper().strip()
    cache_key = f"{symbol}|{max_days}|{strike_pct}"
    now_ts = time.time()

    # Cache hit
    cached = _cache.get(cache_key)
    if cached and now_ts - cached[0] < CACHE_TTL_SECONDS:
        return {**cached[1], "_cached": True, "_cache_age_seconds": int(now_ts - cached[0])}

    try:
        t = yf.Ticker(symbol)
        quote = _with_retry(_fetch_quote_light, t, symbol)
        spot = quote["regularMarketPrice"]
        if not spot:
            raise HTTPException(404, f"No quote data for {symbol}")

        expirations = _with_retry(lambda: list(t.options or []))
        if not expirations:
            raise HTTPException(404, f"No options for {symbol}")

        cutoff_ts = datetime.now(timezone.utc).timestamp() + (max_days * 86400)

        options_by_exp = {}
        exp_unix = []
        warnings = []
        for exp_str in expirations:
            try:
                ts = int(datetime.strptime(exp_str, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())
                if ts > cutoff_ts:
                    continue
                chain = _with_retry(t.option_chain, exp_str)
                calls = _option_rows(chain.calls, ts, spot, strike_pct)
                puts = _option_rows(chain.puts, ts, spot, strike_pct)
                if not calls and not puts:
                    warnings.append({"exp": exp_str, "reason": "no contracts in strike range"})
                    continue
                exp_unix.append(ts)
                options_by_exp[str(ts)] = {"calls": calls, "puts": puts}
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
                warnings.append({"exp": exp_str, "reason": err})
                print(f"[warn] {symbol} exp {exp_str}: {err}")
                continue

        response = {
            "quote": quote,
            "expirationDates": sorted(exp_unix),
            "optionsByExpiration": options_by_exp,
            "filters": {"max_days": max_days, "strike_pct": strike_pct},
            "warnings": warnings,
            "expirationsAvailable": expirations,
            "expirationsReturned": len(exp_unix),
        }
        _cache[cache_key] = (now_ts, response)
        return response
    except HTTPException:
        raise
    except Exception as e:
        if _is_rate_limit(e):
            raise HTTPException(429, "Rate limited by Yahoo. Try again in 30-60 seconds.")
        raise HTTPException(500, f"yfinance error: {e}")


@app.get("/chain/{symbol}")
def chain(
    symbol: str,
    expiration: Optional[int] = None,
    strike_pct: float = Query(DEFAULT_STRIKE_PCT, ge=1, le=200),
):
    symbol = symbol.upper().strip()
    try:
        t = yf.Ticker(symbol)
        quote = _with_retry(_fetch_quote_light, t, symbol)
        spot = quote["regularMarketPrice"]

        expirations = list(t.options or [])
        if not expirations:
            raise HTTPException(404, f"No options for {symbol}")

        exp_unix = [int(datetime.strptime(e, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp()) for e in expirations]

        if expiration is None:
            exp_str = expirations[0]
            exp_ts = exp_unix[0]
        else:
            matches = [(e, u) for e, u in zip(expirations, exp_unix) if u == expiration]
            if not matches:
                raise HTTPException(400, f"Expiration {expiration} not available")
            exp_str, exp_ts = matches[0]

        chain = _with_retry(t.option_chain, exp_str)
        return {
            "quote": quote,
            "expirationDates": sorted(exp_unix),
            "selectedExpiration": exp_ts,
            "calls": _option_rows(chain.calls, exp_ts, spot, strike_pct),
            "puts":  _option_rows(chain.puts,  exp_ts, spot, strike_pct),
        }
    except HTTPException:
        raise
    except Exception as e:
        if _is_rate_limit(e):
            raise HTTPException(429, "Rate limited by Yahoo. Try again in 30-60 seconds.")
        raise HTTPException(500, f"yfinance error: {e}")


@app.get("/cache/clear")
def cache_clear():
    n = len(_cache)
    _cache.clear()
    return {"cleared": n}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
