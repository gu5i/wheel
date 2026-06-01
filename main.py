"""
Wheel Options Backend — Massive edition
================================================
FastAPI service serving option chain data for the wheel dashboard,
backed by Massive.com (formerly Polygon.io) Options Starter plan.

Why Massive over yfinance:
  - One API call returns the whole chain WITH greeks + IV (no Black-Scholes needed)
  - Authenticated by API key, not IP -> no shared-IP rate limiting on Render
  - Server-side strike/expiration filtering via query params

Setup:
  Set environment variable MASSIVE_API_KEY (or legacy POLYGON_API_KEY) on Render.
  Never hardcode the key.

Defaults tuned for wheel strategy:
  - Only expirations within next MAX_DAYS days (default 90)
  - Only strikes within ±STRIKE_PCT% of spot (default 50)
  Overridable per-request: /chain/AAPL/all?max_days=120&strike_pct=30
"""
import os
import time
from datetime import datetime, timezone, timedelta
from typing import Optional

import requests
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Wheel Options API (Massive)", version="2.1.0")

allowed = os.getenv("ALLOWED_ORIGIN", "*")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[allowed] if allowed != "*" else ["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

API_KEY = os.getenv("MASSIVE_API_KEY") or os.getenv("POLYGON_API_KEY", "")
BASE = "https://api.massive.com"

DEFAULT_MAX_DAYS = 90
DEFAULT_STRIKE_PCT = 50
CACHE_TTL_SECONDS = 60
MAX_PAGES = 12          # safety cap on pagination (12 * 250 = 3000 contracts)
_cache: dict = {}


# ---- Helpers ---------------------------------------------------------------

def _safe_float(v) -> float:
    try:
        f = float(v)
        return 0.0 if f != f else f  # NaN guard
    except (TypeError, ValueError):
        return 0.0


def _safe_int(v) -> int:
    return int(_safe_float(v))


def _require_key():
    if not API_KEY:
        raise HTTPException(500, "API key not set on the server. Add MASSIVE_API_KEY in Render → Settings → Environment.")


def _get(url: str, params: dict | None = None) -> dict:
    """GET with API key, basic error translation."""
    p = dict(params or {})
    p["apiKey"] = API_KEY
    r = requests.get(url, params=p, timeout=20)
    if r.status_code == 401:
        raise HTTPException(401, "Massive rejected the API key (401). Check MASSIVE_API_KEY.")
    if r.status_code == 403:
        raise HTTPException(403, "Massive access forbidden (403) — your plan may not include this endpoint.")
    if r.status_code == 429:
        raise HTTPException(429, "Massive rate limit hit (429). Unusual on paid plans — try again shortly.")
    if not r.ok:
        raise HTTPException(502, f"Massive error {r.status_code}: {r.text[:200]}")
    return r.json()


def _spot_price(symbol: str) -> float:
    """Get underlying last price. Try snapshot, fall back to previous close."""
    try:
        data = _get(f"{BASE}/v2/snapshot/locale/us/markets/stocks/tickers/{symbol}")
        t = data.get("ticker", {})
        # last trade price, else day close, else prev day close
        price = _safe_float(t.get("lastTrade", {}).get("p"))
        if not price:
            price = _safe_float(t.get("day", {}).get("c"))
        if not price:
            price = _safe_float(t.get("prevDay", {}).get("c"))
        if price:
            return price
    except HTTPException:
        pass
    # fallback: previous close endpoint
    try:
        data = _get(f"{BASE}/v2/aggs/ticker/{symbol}/prev")
        results = data.get("results", [])
        if results:
            return _safe_float(results[0].get("c"))
    except HTTPException:
        pass
    return 0.0


def _prev_agg(symbol: str) -> dict:
    """Previous session's daily bar via /prev (end-of-day, free-tier entitled).

    Returns {} on any failure. Keys of interest: c (close), v (volume),
    o/h/l (OHLC), vw (vwap).
    """
    try:
        data = _get(f"{BASE}/v2/aggs/ticker/{symbol}/prev")
        results = data.get("results", []) or []
        if results:
            return results[0]
    except HTTPException:
        pass
    return {}


def _underlying_quote(symbol: str) -> dict:
    """Underlying quote for an options-plan account (no stock-quote entitlement).

    The live stock snapshot/NBBO endpoints return NOT_AUTHORIZED on an
    options-only plan, so we don't call them. Price baseline + volume come
    from the free-tier /prev daily bar. The caller (chain_all) overrides
    `regularMarketPrice` with the live underlying price from the options
    chain once it's fetched, and recomputes the change against prevClose.
    """
    prev = _prev_agg(symbol)
    prev_close = _safe_float(prev.get("c"))
    volume = _safe_int(prev.get("v"))

    return {
        "symbol": symbol,
        "regularMarketPrice": prev_close,   # provisional; overridden with live options price
        "regularMarketChange": 0.0,
        "regularMarketChangePercent": 0.0,
        "prevClose": prev_close,            # kept so caller can compute true change
        "regularMarketVolume": volume,
        "shortName": symbol,
    }


def _map_contract(c: dict, exp_ts: int) -> dict:
    """Map one Massive options snapshot contract to the frontend's shape."""
    details = c.get("details", {})
    greeks = c.get("greeks", {}) or {}
    quote = c.get("last_quote", {}) or {}
    trade = c.get("last_trade", {}) or {}
    day = c.get("day", {}) or {}

    strike = _safe_float(details.get("strike_price"))
    bid = _safe_float(quote.get("bid"))
    ask = _safe_float(quote.get("ask"))
    last = _safe_float(trade.get("price"))
    ctype = details.get("contract_type", "")  # "call" or "put"
    underlying_price = _safe_float(c.get("underlying_asset", {}).get("price"))

    # Trade recency. The chain snapshot's last_trade carries sip_timestamp in
    # NANOSECONDS. Convert to a unix-seconds ts and an age in days so the
    # frontend can flag stale/dead prices. A contract with no real trade
    # (price 0 / no timestamp) is marked hasRealTrade=False.
    last_trade_ns = trade.get("sip_timestamp") or trade.get("t") or 0
    last_trade_ts = int(last_trade_ns / 1_000_000_000) if last_trade_ns else 0
    has_real_trade = bool(last > 0 and last_trade_ts > 0)
    if last_trade_ts > 0:
        age_seconds = max(0.0, datetime.now(timezone.utc).timestamp() - last_trade_ts)
        trade_age_days = round(age_seconds / 86400.0, 2)
    else:
        trade_age_days = None

    # day OHLC — available on Starter tier even without quotes/trades
    day_close = _safe_float(day.get("close"))
    day_open = _safe_float(day.get("open"))
    day_high = _safe_float(day.get("high"))
    day_low = _safe_float(day.get("low"))
    day_vwap = _safe_float(day.get("vwap"))

    # Price fallback chain: last trade -> day close -> day vwap
    # (Starter tier lacks live bid/ask, so day.close is our best real price)
    fallback_price = last or day_close or day_vwap

    itm = False
    if underlying_price and strike:
        itm = (strike < underlying_price) if ctype == "call" else (strike > underlying_price)

    return {
        "contractSymbol": details.get("ticker", ""),
        "strike": strike,
        "bid": bid,
        "ask": ask,
        "lastPrice": last or day_close,  # use day close if no live trade
        "dayClose": day_close,
        "dayOpen": day_open,
        "dayHigh": day_high,
        "dayLow": day_low,
        "dayVwap": day_vwap,
        "fallbackPrice": fallback_price,
        "volume": _safe_int(day.get("volume")),
        "openInterest": _safe_int(c.get("open_interest")),
        "lastTradeTs": last_trade_ts,
        "tradeAgeDays": trade_age_days,
        "hasRealTrade": has_real_trade,
        "impliedVolatility": _safe_float(c.get("implied_volatility")),
        "delta": _safe_float(greeks.get("delta")),
        "gamma": _safe_float(greeks.get("gamma")),
        "theta": _safe_float(greeks.get("theta")),
        "vega": _safe_float(greeks.get("vega")),
        "inTheMoney": itm,
        "expiration": exp_ts,
        "contractType": ctype,
    }


def _exp_to_ts(date_str: str) -> int:
    """'YYYY-MM-DD' -> unix seconds at UTC midnight."""
    return int(datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())


# ---- Endpoints -------------------------------------------------------------

@app.get("/")
def root():
    return {
        "service": "Wheel Options API (Massive)",
        "version": "2.0.0",
        "key_configured": bool(API_KEY),
        "defaults": {"max_days": DEFAULT_MAX_DAYS, "strike_pct": DEFAULT_STRIKE_PCT},
        "cache_ttl_seconds": CACHE_TTL_SECONDS,
        "endpoints": ["/health", "/quote/{symbol}", "/chain/{symbol}/all"],
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "time": datetime.now(timezone.utc).isoformat(),
        "key_configured": bool(API_KEY),
        "cache_entries": len(_cache),
    }


@app.get("/quote/{symbol}")
def quote(symbol: str):
    _require_key()
    symbol = symbol.upper().strip()
    q = _underlying_quote(symbol)
    if not q["regularMarketPrice"]:
        raise HTTPException(404, f"No quote data for {symbol}")
    return q


@app.get("/chain/{symbol}/all")
def chain_all(
    symbol: str,
    max_days: int = Query(DEFAULT_MAX_DAYS, ge=1, le=730),
    strike_pct: float = Query(DEFAULT_STRIKE_PCT, ge=1, le=200),
):
    _require_key()
    symbol = symbol.upper().strip()
    cache_key = f"{symbol}|{max_days}|{strike_pct}"
    now_ts = time.time()

    cached = _cache.get(cache_key)
    if cached and now_ts - cached[0] < CACHE_TTL_SECONDS:
        return {**cached[1], "_cached": True, "_cache_age_seconds": int(now_ts - cached[0])}

    quote = _underlying_quote(symbol)
    spot = quote["regularMarketPrice"]
    if not spot:
        raise HTTPException(404, f"No quote/price data for {symbol}")

    # Server-side filters: strike window + expiration window
    low_strike = round(spot * (1 - strike_pct / 100), 2)
    high_strike = round(spot * (1 + strike_pct / 100), 2)
    today = datetime.now(timezone.utc).date()
    exp_lo = today.isoformat()
    exp_hi = (today + timedelta(days=max_days)).isoformat()

    params = {
        "strike_price.gte": low_strike,
        "strike_price.lte": high_strike,
        "expiration_date.gte": exp_lo,
        "expiration_date.lte": exp_hi,
        "limit": 250,
        "sort": "expiration_date",
        "order": "asc",
    }

    url = f"{BASE}/v3/snapshot/options/{symbol}"
    by_exp: dict = {}
    exp_set: set = set()
    pages = 0
    live_underlying = 0.0   # captured from options chain (underlying_asset.price)

    while url and pages < MAX_PAGES:
        data = _get(url, params if pages == 0 else None)
        results = data.get("results", []) or []
        for c in results:
            details = c.get("details", {})
            exp_str = details.get("expiration_date")
            ctype = details.get("contract_type")
            if not exp_str or ctype not in ("call", "put"):
                continue
            if not live_underlying:
                live_underlying = _safe_float(c.get("underlying_asset", {}).get("price"))
            ts = _exp_to_ts(exp_str)
            exp_set.add(ts)
            slot = by_exp.setdefault(str(ts), {"calls": [], "puts": []})
            mapped = _map_contract(c, ts)
            (slot["calls"] if ctype == "call" else slot["puts"]).append(mapped)

        url = data.get("next_url")
        params = None  # next_url already carries query params (except apiKey, added by _get)
        pages += 1

    if not exp_set:
        raise HTTPException(404, f"No options returned for {symbol} in the requested window")

    # The options chain carries the live (15-min delayed) underlying price.
    # Use it as the real price, and compute Day change vs the /prev close.
    prev_close = _safe_float(quote.get("prevClose"))
    if live_underlying:
        quote["regularMarketPrice"] = live_underlying
        if prev_close:
            chg = live_underlying - prev_close
            quote["regularMarketChange"] = chg
            quote["regularMarketChangePercent"] = (chg / prev_close * 100) if prev_close else 0.0

    response = {
        "quote": quote,
        "expirationDates": sorted(exp_set),
        "optionsByExpiration": by_exp,
        "filters": {"max_days": max_days, "strike_pct": strike_pct},
        "expirationsReturned": len(exp_set),
        "pagesFetched": pages,
        "source": "massive",
    }
    _cache[cache_key] = (now_ts, response)
    return response


@app.get("/cache/clear")
def cache_clear():
    n = len(_cache)
    _cache.clear()
    return {"cleared": n}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
