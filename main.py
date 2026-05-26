import os
from datetime import datetime
from typing import Optional
import yfinance as yf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Wheel Options API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET"], allow_headers=["*"])

def _quote_dict(info, symbol):
    price = info.get("regularMarketPrice") or info.get("currentPrice") or info.get("previousClose") or 0
    return {
        "symbol": symbol,
        "regularMarketPrice": price,
        "regularMarketChange": info.get("regularMarketChange", 0) or 0,
        "regularMarketChangePercent": info.get("regularMarketChangePercent", 0) or 0,
        "bid": info.get("bid", 0) or 0,
        "ask": info.get("ask", 0) or 0,
        "fiftyTwoWeekLow": info.get("fiftyTwoWeekLow", 0) or 0,
        "fiftyTwoWeekHigh": info.get("fiftyTwoWeekHigh", 0) or 0,
        "regularMarketVolume": info.get("regularMarketVolume", 0) or info.get("volume", 0) or 0,
    }

def _option_rows(df, expiration_ts):
    rows = []
    if df is None or df.empty:
        return rows
    for _, r in df.iterrows():
        bid = float(r.get("bid", 0) or 0)
        ask = float(r.get("ask", 0) or 0)
        last = float(r.get("lastPrice", 0) or 0)
        if bid == 0 and ask == 0 and last == 0:
            continue
        rows.append({
            "contractSymbol": str(r.get("contractSymbol", "")),
            "strike": float(r.get("strike", 0) or 0),
            "bid": bid, "ask": ask, "lastPrice": last,
            "volume": int(r.get("volume", 0) or 0),
            "openInterest": int(r.get("openInterest", 0) or 0),
            "impliedVolatility": float(r.get("impliedVolatility", 0) or 0),
            "inTheMoney": bool(r.get("inTheMoney", False)),
            "expiration": expiration_ts,
        })
    return rows

@app.get("/")
def root():
    return {"service": "Wheel Options API", "status": "ok"}

@app.get("/health")
def health():
    return {"status": "ok", "time": datetime.utcnow().isoformat()}

@app.get("/chain/{symbol}/all")
def chain_all(symbol: str):
    symbol = symbol.upper().strip()
    try:
        t = yf.Ticker(symbol)
        info = t.info
        quote = _quote_dict(info, symbol)
        expirations = list(t.options or [])
        if not expirations:
            raise HTTPException(404, f"No options for {symbol}")
        options_by_exp = {}
        exp_unix = []
        for exp_str in expirations:
            try:
                ts = int(datetime.strptime(exp_str, "%Y-%m-%d").timestamp())
                exp_unix.append(ts)
                chain = t.option_chain(exp_str)
                options_by_exp[str(ts)] = {
                    "calls": _option_rows(chain.calls, ts),
                    "puts": _option_rows(chain.puts, ts),
                }
            except Exception as e:
                print(f"skip {exp_str}: {e}")
                continue
        return {"quote": quote, "expirationDates": sorted(exp_unix), "optionsByExpiration": options_by_exp}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"yfinance error: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
