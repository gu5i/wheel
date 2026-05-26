"""
Wheel Options Backend
=====================
FastAPI service serving option chain data for the wheel dashboard.

Defaults are tuned for the wheel strategy:
  - Only expirations within the next MAX_DAYS days (default 90)
  - Only strikes within ±STRIKE_PCT% of spot (default 50)

Both can be overridden per-request via query params:
  /chain/AAPL/all?max_days=120&strike_pct=30
"""
import os
from datetime import datetime, timezone
from typing import Optional

import yfinance as yf
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Wheel Options API", version="1.1.0")

allowed = os.getenv("ALLOWED_ORIGIN", "*")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[allowed] if allowed != "*" else ["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

# Defaults — change these here if you want different system-wide defaults
DEFAULT_MAX_DAYS = 90
DEFAULT_STRIKE_PCT = 50


def _quote_dict(info: dict, symbol: str) -> dict:
    price = info.get("regularMarketPrice") or info.get("currentPrice") or info.get("previousClose") or 0
    return {
        "symbol": symbol,
        "regularMarketPrice": price,
        "regularMarketChange": info.get("regularMarketChange", 0) or 0,
        "regularMarketChangePercent": (info.get("regularMarketChangePercent", 0) or 0) * (100 if abs(info.get("regularMarketChangePercent", 0) or 0) < 1 else 1),
        "bid": info.get("bid", 0) or 0,
        "ask": info.get("ask", 0) or 0,
        "fiftyTwoWeekLow": info.get("fiftyTwoWeekLow", 0) or 0,
        "fiftyTwoWeekHigh": info.get("fiftyTwoWeekHigh", 0) or 0,
        "regularMarketVolume": info.get("regularMarketVolume", 0) or info.get("volume", 0) or 0,
        "shortName": info.get("shortName", symbol),
    }


def _safe_float(v) -> float:
    """Convert to float, handling None/NaN/strings safely."""
    try:
        f = float(v)
        if f != f:  # NaN check (NaN != NaN)
            return 0.0
        return f
    except (TypeError, ValueError):
        return 0.0


def _safe_int(v) -> int:
    """Convert to int, handling None/NaN/strings safely."""
    return int(_safe_float(v))


def _option_rows(df, expiration_ts: int, spot: float, strike_pct: float) -> list[dict]:
    """Convert a yfinance options DataFrame to dicts, filtered to strikes within ±strike_pct% of spot.

    Includes ALL contracts in the strike range, even ones with no bid/ask/last (zombie contracts).
    The frontend can decide what to do with them.
    """
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


@app.get("/")
def root():
    return {
        "service": "Wheel Options API",
        "version": "1.1.0",
        "defaults": {"max_days": DEFAULT_MAX_DAYS, "strike_pct": DEFAULT_STRIKE_PCT},
        "endpoints": ["/health", "/quote/{symbol}", "/chain/{symbol}", "/chain/{symbol}/all"],
    }


@app.get("/health")
def health():
    return {"status": "ok", "time": datetime.now(timezone.utc).isoformat()}


@app.get("/quote/{symbol}")
def quote(symbol: str):
    symbol = symbol.upper().strip()
    try:
        t = yf.Ticker(symbol)
        info = t.info
        if not info or (not info.get("regularMarketPrice") and not info.get("previousClose")):
            raise HTTPException(404, f"No quote data for {symbol}")
        return _quote_dict(info, symbol)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"yfinance error: {e}")


@app.get("/chain/{symbol}/all")
def chain_all(
    symbol: str,
    max_days: int = Query(DEFAULT_MAX_DAYS, ge=1, le=730, description="Only include expirations within this many days"),
    strike_pct: float = Query(DEFAULT_STRIKE_PCT, ge=1, le=200, description="Only include strikes within ±this % of spot"),
):
    """Return quote + every expiration within max_days, each filtered to strikes within ±strike_pct% of spot."""
    symbol = symbol.upper().strip()
    try:
        t = yf.Ticker(symbol)
        info = t.info
        quote = _quote_dict(info, symbol)
        spot = quote["regularMarketPrice"]

        expirations = list(t.options or [])
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
                    continue  # skip expirations beyond max_days
                chain = t.option_chain(exp_str)
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

        return {
            "quote": quote,
            "expirationDates": sorted(exp_unix),
            "optionsByExpiration": options_by_exp,
            "filters": {"max_days": max_days, "strike_pct": strike_pct},
            "warnings": warnings,
            "expirationsAvailable": expirations,
            "expirationsReturned": len(exp_unix),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"yfinance error: {e}")


@app.get("/chain/{symbol}")
def chain(
    symbol: str,
    expiration: Optional[int] = None,
    strike_pct: float = Query(DEFAULT_STRIKE_PCT, ge=1, le=200),
):
    """Return chain for a single expiration."""
    symbol = symbol.upper().strip()
    try:
        t = yf.Ticker(symbol)
        info = t.info
        quote = _quote_dict(info, symbol)
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

        chain = t.option_chain(exp_str)
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
        raise HTTPException(500, f"yfinance error: {e}")


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
