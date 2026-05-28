"""
Wheel Options Backend — Polygon/Massive edition
================================================
FastAPI service serving option chain data for the wheel dashboard,
backed by Polygon.io (now Massive.com) Options Starter plan.

Why Polygon over yfinance:
  - One API call returns the whole chain WITH greeks + IV (no Black-Scholes needed)
  - Authenticated by API key, not IP -> no shared-IP rate limiting on Render
  - Server-side strike/expiration filtering via query params

Setup:
  Set environment variable POLYGON_API_KEY on Render (Settings -> Environment).
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

app = FastAPI(title="Wheel Options API (Polygon)", version="2.0.0")

allowed = os.getenv("ALLOWED_ORIGIN", "*")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[allowed] if allowed != "*" else ["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

API_KEY = os.getenv("POLYGON_API_KEY", "")
BASE = "https://api.polygon.io"

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
        raise HTTPException(500, "POLYGON_API_KEY not set on the server. Add it in Render → Settings → Environment.")


def _get(url: str, params: dict | None = None) -> dict:
    """GET with API key, basic error translation."""
    p = dict(params or {})
    p["apiKey"] = API_KEY
    r = requests.get(url, params=p, timeout=20)
    if r.status_code == 401:
        raise HTTPException(401, "Polygon rejected the API key (401). Check POLYGON_API_KEY.")
    if r.status_code == 403:
        raise HTTPException(403, "Polygon access forbidden (403) — your plan may not include this endpoint.")
    if r.status_code == 429:
        raise HTTPException(429, "Polygon rate limit hit (429). Unusual on paid plans — try again shortly.")
    if not r.ok:
        raise HTTPException(502, f"Polygon error {r.status_code}: {r.text[:200]}")
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


def _underlying_quote(symbol: str) -> dict:
    """Build the quote dict the frontend expects, from Polygon snapshot."""
    price = change = change_pct = bid = ask = 0.0
    volume = 0
    try:
        data = _get(f"{BASE}/v2/snapshot/locale/us/markets/stocks/tickers/{symbol}")
        t = data.get("ticker", {})
        day = t.get("day", {})
        prev = t.get("prevDay", {})
        last_trade = t.get("lastTrade", {})
        last_quote = t.get("lastQuote", {})

        price = _safe_float(last_trade.get("p")) or _safe_float(day.get("c")) or _safe_float(prev.get("c"))
        bid = _safe_float(last_quote.get("p"))  # bid price
        ask = _safe_float(last_quote.get("P"))  # ask price
        volume = _safe_int(day.get("v")) or _safe_int(prev.get("v"))
        change = _safe_float(t.get("todaysChange"))
        change_pct = _safe_float(t.get("todaysChangePerc"))
        if not change and price and prev.get("c"):
            pc = _safe_float(prev.get("c"))
            change = price - pc
            change_pct = (change / pc * 100) if pc else 0
    except HTTPException:
        price = _spot_price(symbol)

    if bid == 0 and ask == 0 and price > 0:
        bid = ask = price  # market closed fallback

    return {
        "symbol": symbol,
        "regularMarketPrice": price,
        "regularMarketChange": change,
        "regularMarketChangePercent": change_pct,
        "bid": bid,
        "ask": ask,
        "regularMarketVolume": volume,
        "shortName": symbol,
    }


def _map_contract(c: dict, exp_ts: int) -> dict:
    """Map one Polygon options snapshot contract to the frontend's shape."""
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
        "service": "Wheel Options API (Polygon/Massive)",
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

    while url and pages < MAX_PAGES:
        data = _get(url, params if pages == 0 else None)
        results = data.get("results", []) or []
        for c in results:
            details = c.get("details", {})
            exp_str = details.get("expiration_date")
            ctype = details.get("contract_type")
            if not exp_str or ctype not in ("call", "put"):
                continue
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

    response = {
        "quote": quote,
        "expirationDates": sorted(exp_set),
        "optionsByExpiration": by_exp,
        "filters": {"max_days": max_days, "strike_pct": strike_pct},
        "expirationsReturned": len(exp_set),
        "pagesFetched": pages,
        "source": "polygon",
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
