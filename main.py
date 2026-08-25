"""
Wheel Options Backend — Massive edition
================================================
FastAPI service serving option chain data for the wheel dashboard,
backed by Massive.com (formerly Polygon.io) Options Advanced plan.

Why Massive over yfinance:
  - One API call returns the whole chain WITH greeks + IV (no Black-Scholes needed)
  - Authenticated by API key, not IP -> no shared-IP rate limiting on Render
  - Server-side strike/expiration filtering via query params

Plan entitlement (matters — diagnose data gaps against this FIRST):
  - Options Advanced  -> live NBBO bid/ask on contracts, greeks, IV, open interest,
                         and underlying_asset.price (15-min delayed spot)
  - Stocks Free       -> /v2/aggs/ticker/{sym}/prev only (previous daily close)
  - NOT entitled      -> live stock snapshot / NBBO endpoints (return NOT_AUTHORIZED)

Setup:
  Set environment variable MASSIVE_API_KEY (or legacy POLYGON_API_KEY) on Render.
  Optionally set FINNHUB_API_KEY (free tier) to enable the next-earnings-date
  lookup. Without it the chain still serves; earnings report as "unavailable".
  Never hardcode either key.

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

# Finnhub — free tier, used ONLY for the next earnings date. Entirely optional:
# if FINNHUB_API_KEY is unset the chain endpoint still works and simply reports
# earnings as unavailable. Never let this break the critical path.
FINNHUB_KEY = os.getenv("FINNHUB_API_KEY", "")
FINNHUB_BASE = "https://finnhub.io/api/v1"
EARNINGS_LOOKAHEAD_DAYS = 31    # free tier serves ~1 month of calendar
EARNINGS_CACHE_TTL = 12 * 3600  # earnings dates don't move intraday

DEFAULT_MAX_DAYS = 90
DEFAULT_STRIKE_PCT = 50
CACHE_TTL_SECONDS = 15
MAX_PAGES = 12          # safety cap on pagination (12 * 250 = 3000 contracts)
_cache: dict = {}
_earnings_cache: dict = {}


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


def _options_underlying_price(symbol: str, exp_lo: str, exp_hi: str) -> float:
    """Spot from the OPTIONS ADVANCED plan.

    Reads `underlying_asset.price` off a 1-contract options chain snapshot. This
    price travels with the options entitlement, so it's available on the options
    plan without any stock-quote subscription. Returns 0.0 if the options plan
    doesn't serve it, so the caller can fall back to the stock plan.
    """
    try:
        data = _get(
            f"{BASE}/v3/snapshot/options/{symbol}",
            {"expiration_date.gte": exp_lo, "expiration_date.lte": exp_hi, "limit": 1},
        )
    except HTTPException:
        return 0.0
    for c in (data.get("results") or []):
        price = _safe_float(c.get("underlying_asset", {}).get("price"))
        if price:
            return price
    return 0.0


def _underlying_quote(symbol: str) -> dict:
    """Underlying quote for an options-plan account (no stock-quote entitlement).

    The live stock snapshot/NBBO endpoints return NOT_AUTHORIZED on this plan,
    so we don't call them. Price baseline + volume come from the free-tier
    /prev daily bar. The caller (chain_all) overrides `regularMarketPrice` with
    the underlying price from the options chain once it's fetched, and
    recomputes the change against prevClose.
    """
    prev = _prev_agg(symbol)
    prev_close = _safe_float(prev.get("c"))
    volume = _safe_int(prev.get("v"))

    return {
        "symbol": symbol,
        "regularMarketPrice": prev_close,   # provisional; overridden with options-chain spot
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

    # day OHLC — present even on contracts that haven't printed a trade today
    day_close = _safe_float(day.get("close"))
    day_open = _safe_float(day.get("open"))
    day_high = _safe_float(day.get("high"))
    day_low = _safe_float(day.get("low"))
    day_vwap = _safe_float(day.get("vwap"))

    # Price fallback chain: last trade -> day close -> day vwap.
    # On Advanced, live bid/ask is the primary price source and the frontend
    # prefers the true (bid+ask)/2 mid; this chain only covers contracts with
    # no two-sided quote.
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


def _to_unix_seconds(ts) -> int:
    """Normalize a Massive timestamp to unix SECONDS.

    Massive returns timestamps in nanoseconds, microseconds, milliseconds, or
    seconds depending on the field, so detect the magnitude before converting.
    Returns 0 on anything unparseable.
    """
    try:
        v = int(ts)
    except (TypeError, ValueError):
        return 0
    if v <= 0:
        return 0
    if v >= 10**17:        # nanoseconds
        return v // 1_000_000_000
    if v >= 10**14:        # microseconds
        return v // 1_000_000
    if v >= 10**11:        # milliseconds
        return v // 1_000
    return v               # already seconds


def _exp_to_ts(date_str: str) -> int:
    """'YYYY-MM-DD' -> unix seconds at UTC midnight."""
    return int(datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())


def _next_earnings(symbol: str) -> dict:
    """Next scheduled earnings date for `symbol`, via Finnhub's free calendar.

    ALWAYS returns a dict — never raises, never returns None — so a Finnhub
    outage, a missing key, or a rate limit can't take down /chain. Three
    distinct states, which the frontend must render differently:

      status="found"          -> a date inside the lookahead window
      status="none_in_window" -> Finnhub answered, nothing scheduled in the
                                 next EARNINGS_LOOKAHEAD_DAYS days. This does
                                 NOT mean "no earnings coming" — the free tier
                                 only sees ~1 month ahead, so a report 40 days
                                 out looks identical to no report at all.
      status="unavailable"    -> no key, request failed, or symbol not covered
                                 (Finnhub free tier is US-only; foreign issuers
                                 and very recent listings often come back empty)

    Collapsing "none_in_window" or "unavailable" into "clear" would produce a
    false all-clear on exactly the names where coverage is weakest. Don't.
    """
    base = {
        "date": None,
        "ts": 0,
        "hour": "",
        "lookaheadDays": EARNINGS_LOOKAHEAD_DAYS,
        "source": "finnhub",
        "status": "unavailable",
    }
    if not FINNHUB_KEY:
        return base

    now_ts = time.time()
    hit = _earnings_cache.get(symbol)
    if hit and now_ts - hit[0] < EARNINGS_CACHE_TTL:
        return hit[1]

    today = datetime.now(timezone.utc).date()
    to_date = today + timedelta(days=EARNINGS_LOOKAHEAD_DAYS)
    result = dict(base)

    try:
        r = requests.get(
            f"{FINNHUB_BASE}/calendar/earnings",
            params={
                "from": today.isoformat(),
                "to": to_date.isoformat(),
                "symbol": symbol,
                "token": FINNHUB_KEY,
            },
            timeout=10,
        )
        if r.ok:
            rows = (r.json() or {}).get("earningsCalendar") or []
            # Finnhub returns the window unsorted; take the earliest date that
            # is today or later.
            upcoming = sorted(
                (x for x in rows if x.get("date") and x["date"] >= today.isoformat()),
                key=lambda x: x["date"],
            )
            if upcoming:
                first = upcoming[0]
                result.update({
                    "date": first["date"],
                    "ts": _exp_to_ts(first["date"]),
                    "hour": first.get("hour") or "",   # bmo | amc | dmh
                    "status": "found",
                })
            else:
                result["status"] = "none_in_window"
        # non-ok (401/403/429) falls through as "unavailable"
    except (requests.RequestException, ValueError, KeyError):
        result["status"] = "unavailable"

    # Cache negatives too — a symbol Finnhub doesn't cover won't start being
    # covered within 12h, and this keeps failures from hammering the API.
    _earnings_cache[symbol] = (now_ts, result)
    return result


# ---- Endpoints -------------------------------------------------------------

@app.get("/")
def root():
    return {
        "service": "Wheel Options API (Massive)",
        "version": "2.2.0",
        "key_configured": bool(API_KEY),
        "finnhub_key_configured": bool(FINNHUB_KEY),
        "defaults": {"max_days": DEFAULT_MAX_DAYS, "strike_pct": DEFAULT_STRIKE_PCT},
        "cache_ttl_seconds": CACHE_TTL_SECONDS,
        "endpoints": [
            "/health",
            "/quote/{symbol}",
            "/chain/{symbol}/all",
            "/trades/{contract}",
            "/earnings/{symbol}",
            "/cache/clear",
        ],
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "time": datetime.now(timezone.utc).isoformat(),
        "key_configured": bool(API_KEY),
        "finnhub_key_configured": bool(FINNHUB_KEY),
        "cache_entries": len(_cache),
        "earnings_cache_entries": len(_earnings_cache),
    }


@app.get("/earnings/{symbol}")
def earnings(symbol: str):
    """Next earnings date for one symbol. Handy for spot-checking coverage
    without loading a whole chain."""
    return _next_earnings(symbol.upper().strip())


@app.get("/quote/{symbol}")
def quote(symbol: str):
    _require_key()
    symbol = symbol.upper().strip()
    q = _underlying_quote(symbol)
    if not q["regularMarketPrice"]:
        raise HTTPException(404, f"No quote data for {symbol}")
    return q


@app.get("/trades/{contract}")
def trades(contract: str, limit: int = Query(10, ge=1, le=50)):
    """Last N trades (time & sales) for one option contract.

    `contract` is the OPRA option ticker, e.g. O:SN260618P00115000.
    Pulls Massive's options Trades endpoint sorted most-recent-first. Note this
    is tick-level TRADES (prints), not an order book — US options have no
    consolidated depth-of-book feed.
    """
    _require_key()
    contract = contract.strip()
    data = _get(
        f"{BASE}/v3/trades/{contract}",
        {"limit": limit, "order": "desc", "sort": "timestamp"},
    )
    out = []
    for t in (data.get("results") or []):
        ts = _to_unix_seconds(
            t.get("sip_timestamp") or t.get("participant_timestamp") or t.get("t")
        )
        out.append({
            "price": _safe_float(t.get("price")),
            "size": _safe_int(t.get("size")),
            "exchange": _safe_int(t.get("exchange")),
            "conditions": t.get("conditions") or [],
            "ts": ts,
        })
    return {"contract": contract, "trades": out, "count": len(out)}


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

    today = datetime.now(timezone.utc).date()
    exp_lo = today.isoformat()
    exp_hi = (today + timedelta(days=max_days)).isoformat()

    # Quote shell (prev close + volume) from the free /prev daily bar — used for
    # the Day-change baseline and as the spot fallback below.
    quote = _underlying_quote(symbol)
    prev_close = _safe_float(quote.get("prevClose"))

    # Spot priority:
    #   1) OPTIONS ADVANCED plan  -> underlying_asset.price from the options snapshot
    #   2) FREE STOCK plan        -> /prev daily close (only if 1 doesn't fetch)
    spot = _options_underlying_price(symbol, exp_lo, exp_hi)
    spot_source = "options_advanced"
    if not spot:
        spot = prev_close
        spot_source = "stock_prev_close"
    if not spot:
        raise HTTPException(404, f"No price data for {symbol} (options underlying and stock /prev both empty)")

    # Server-side filters: strike window + expiration window
    low_strike = round(spot * (1 - strike_pct / 100), 2)
    high_strike = round(spot * (1 + strike_pct / 100), 2)

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
    ul_timeframe = None     # "REAL-TIME" | "DELAYED" — Massive's own freshness label
    ul_asof = 0             # unix seconds of that underlying price (0 if unknown)

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
                ua = c.get("underlying_asset", {}) or {}
                live_underlying = _safe_float(ua.get("price"))
                if live_underlying:
                    ul_timeframe = ua.get("timeframe")
                    ul_asof = _to_unix_seconds(ua.get("last_updated"))
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

    # Authoritative spot. The full chain pull also carries underlying_asset.price
    # (same OPTIONS source) — prefer it if present, else use the spot resolved
    # above. Day change is computed against the free /prev close when available.
    final_spot = live_underlying or spot
    if live_underlying:
        spot_source = "options_advanced"
    quote["regularMarketPrice"] = final_spot
    if prev_close:
        chg = final_spot - prev_close
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
        "spotSource": spot_source,            # "options_advanced" | "stock_prev_close"
        "underlyingTimeframe": ul_timeframe,  # "REAL-TIME" | "DELAYED" | None
        "underlyingAsOf": ul_asof,            # unix seconds of the spot price (0 if unknown)
        "nextEarnings": _next_earnings(symbol),  # see _next_earnings for the 3 states
    }
    _cache[cache_key] = (now_ts, response)
    return response


@app.get("/cache/clear")
def cache_clear():
    n, e = len(_cache), len(_earnings_cache)
    _cache.clear()
    _earnings_cache.clear()
    return {"cleared": n, "earnings_cleared": e}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
