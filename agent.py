import os
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()

@dataclass
class AgentConfig:
    rsi_overbought: float = 75
    stop_loss_pct: float = -8
    take_profit_1_pct: float = 20
    take_profit_2_pct: float = 35
    min_volume_ratio: float = 1.15
    aggressive_max_single_position: float = 0.15
    cache_ttl_seconds: int = 3600
    request_delay_seconds: float = 1.5
    retry_delay_seconds: int = 20

CONFIG = AgentConfig()


def load_portfolio(path: str = "portfolio.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    df["target_weight"] = pd.to_numeric(df["target_weight"], errors="coerce").fillna(0)
    df["position_ils"] = pd.to_numeric(df["position_ils"], errors="coerce").fillna(0)
    df["buy_price"] = pd.to_numeric(df["buy_price"], errors="coerce")
    return df


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _cache_file(ticker: str, period: str) -> Path:
    cache_dir = Path(".cache")
    cache_dir.mkdir(exist_ok=True)
    safe = ticker.replace("/", "_").replace(".", "_")
    return cache_dir / f"{safe}_{period}.csv"


def _read_cached_history(ticker: str, period: str, max_age_seconds: Optional[int] = None) -> pd.DataFrame:
    cache_path = _cache_file(ticker, period)
    if not cache_path.exists():
        return pd.DataFrame()

    if max_age_seconds is not None:
        age = time.time() - cache_path.stat().st_mtime
        if age > max_age_seconds:
            return pd.DataFrame()

    try:
        df = pd.read_csv(cache_path, parse_dates=["Date"])
        if "Date" in df.columns:
            df = df.set_index("Date")
        return df.dropna()
    except Exception:
        return pd.DataFrame()


def _write_cached_history(ticker: str, period: str, data: pd.DataFrame) -> None:
    if data.empty:
        return
    cache_path = _cache_file(ticker, period)
    try:
        data.to_csv(cache_path)
    except Exception:
        pass


def _period_to_days(period: str) -> int:
    mapping = {"1mo": 45, "3mo": 120, "6mo": 240, "1y": 420, "2y": 800}
    return mapping.get(period, 240)


def _fetch_stooq_history(ticker: str, period: str = "6mo") -> pd.DataFrame:
    """Fallback data provider that usually avoids Yahoo Finance rate limits."""
    end = pd.Timestamp.today().normalize()
    start = end - pd.Timedelta(days=_period_to_days(period))
    symbol = ticker.lower().replace("-", ".")
    if "." not in symbol:
        symbol = f"{symbol}.us"
    url = (
        "https://stooq.com/q/d/l/"
        f"?s={symbol}&d1={start:%Y%m%d}&d2={end:%Y%m%d}&i=d"
    )
    try:
        data = pd.read_csv(url)
        if data.empty or "Date" not in data.columns or "Close" not in data.columns:
            return pd.DataFrame()
        data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
        data = data.dropna(subset=["Date", "Close"]).set_index("Date")
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col not in data.columns:
                data[col] = np.nan
        return data[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"])
    except Exception as exc:
        print(f"Stooq fetch failed for {ticker}: {exc}")
        return pd.DataFrame()


def fetch_history(ticker: str, period: str = "6mo") -> pd.DataFrame:
    """Fetch price history with cache + Stooq first + Yahoo fallback."""
    ticker = str(ticker).upper().strip()

    fresh_cache = _read_cached_history(ticker, period, CONFIG.cache_ttl_seconds)
    if not fresh_cache.empty:
        return fresh_cache

    data = _fetch_stooq_history(ticker, period)
    if not data.empty:
        _write_cached_history(ticker, period, data)
        return data

    time.sleep(CONFIG.request_delay_seconds)
    try:
        data = yf.download(
            ticker,
            period=period,
            interval="1d",
            progress=False,
            auto_adjust=True,
            threads=False,
        )
        if not data.empty:
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            data = data.dropna()
            _write_cached_history(ticker, period, data)
            return data
    except Exception as exc:
        print(f"Yahoo fetch failed for {ticker}: {exc}")

    stale_cache = _read_cached_history(ticker, period, None)
    if not stale_cache.empty:
        print(f"Using stale cached data for {ticker}")
        return stale_cache

    return pd.DataFrame()

def analyze_ticker(ticker: str, buy_price: Optional[float] = None) -> Dict:
    if ticker.upper() == "CASH":
        return {
            "ticker": ticker,
            "price": np.nan,
            "signal": "HOLD CASH",
            "score": 0,
            "reason": "מזומן להזדמנויות",
            "rsi": np.nan,
            "ma20": np.nan,
            "ma50": np.nan,
            "volume_ratio": np.nan,
            "pnl_pct": np.nan,
        }

    hist = fetch_history(ticker)
    if hist.empty or len(hist) < 60:
        return {"ticker": ticker, "signal": "NO DATA", "score": 0, "reason": "אין מספיק נתונים", "price": np.nan}

    close = hist["Close"]
    volume = hist["Volume"]
    price = float(close.iloc[-1])
    ma20 = float(close.rolling(20).mean().iloc[-1])
    ma50 = float(close.rolling(50).mean().iloc[-1])
    rsi_value = float(rsi(close).iloc[-1])
    avg_volume = volume.rolling(20).mean().iloc[-1]
    volume_ratio = float(volume.iloc[-1] / avg_volume) if avg_volume else np.nan

    score = 0
    reasons: List[str] = []

    if price > ma20:
        score += 2
        reasons.append("מעל MA20")
    else:
        reasons.append("מתחת MA20")

    if price > ma50:
        score += 2
        reasons.append("מעל MA50")
    else:
        reasons.append("מתחת MA50")

    if ma20 > ma50:
        score += 1
        reasons.append("מגמה קצרה חזקה")

    if volume_ratio >= CONFIG.min_volume_ratio:
        score += 1
        reasons.append("נפח גבוה")

    if rsi_value < CONFIG.rsi_overbought:
        score += 1
        reasons.append("RSI לא קיצוני")
    else:
        reasons.append("RSI גבוה מדי")

    pnl_pct = np.nan
    if buy_price and not np.isnan(buy_price):
        pnl_pct = (price / float(buy_price) - 1) * 100

    if not np.isnan(pnl_pct) and pnl_pct <= CONFIG.stop_loss_pct:
        signal = "SELL / STOP LOSS"
    elif price < ma50:
        signal = "SELL / REDUCE"
    elif not np.isnan(pnl_pct) and pnl_pct >= CONFIG.take_profit_2_pct:
        signal = "TAKE PROFIT 2"
    elif not np.isnan(pnl_pct) and pnl_pct >= CONFIG.take_profit_1_pct:
        signal = "TAKE PROFIT 1"
    elif score >= 6:
        signal = "BUY"
    elif score >= 4:
        signal = "WATCH"
    else:
        signal = "HOLD / WAIT"

    return {
        "ticker": ticker,
        "price": round(price, 2),
        "signal": signal,
        "score": score,
        "reason": ", ".join(reasons),
        "rsi": round(rsi_value, 1),
        "ma20": round(ma20, 2),
        "ma50": round(ma50, 2),
        "volume_ratio": round(volume_ratio, 2),
        "pnl_pct": round(pnl_pct, 2) if not np.isnan(pnl_pct) else np.nan,
    }


def analyze_portfolio(portfolio_path: str = "portfolio.csv") -> pd.DataFrame:
    portfolio = load_portfolio(portfolio_path)
    rows = []
    for _, row in portfolio.iterrows():
        result = analyze_ticker(row["ticker"], row.get("buy_price"))
        result.update({
            "name": row.get("name", ""),
            "target_weight": row.get("target_weight", 0),
            "position_ils": row.get("position_ils", 0),
            "sector": row.get("sector", ""),
        })
        rows.append(result)
    return pd.DataFrame(rows)


def format_alerts(df: pd.DataFrame) -> List[str]:
    alerts = []
    for _, row in df.iterrows():
        sig = str(row.get("signal", ""))
        if sig in ["BUY", "SELL / STOP LOSS", "SELL / REDUCE", "TAKE PROFIT 1", "TAKE PROFIT 2", "WATCH"]:
            alerts.append(
                f"📈 {sig}: {row['ticker']}\n"
                f"Price: {row.get('price')} | Score: {row.get('score')}/7\n"
                f"RSI: {row.get('rsi')} | Vol Ratio: {row.get('volume_ratio')}\n"
                f"Reason: {row.get('reason')}"
            )
    return alerts


def send_telegram(message: str) -> bool:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print("Telegram token/chat_id missing. Create .env from .env.example")
        return False
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    response = requests.post(url, json={"chat_id": chat_id, "text": message}, timeout=15)
    return response.ok


if __name__ == "__main__":
    df = analyze_portfolio()
    print(df[["ticker", "price", "signal", "score", "reason"]].to_string(index=False))
    alerts = format_alerts(df)
    if alerts:
        send_telegram("\n\n".join(alerts))
