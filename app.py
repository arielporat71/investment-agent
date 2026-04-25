import os
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Investment Agent", page_icon="📈", layout="wide")

PORTFOLIO_FILE = "portfolio.csv"
DEFAULT_CASH = 200000.0

DEFAULT_RECOMMENDED_UNIVERSE = [
    "NVDA", "AMD", "TSLA", "META", "AMZN", "PLTR", "CRWD", "AVGO", "ARM", "MSFT", "GOOGL", "QQQ", "VOO", "SMH", "SOXX"
]

SECTOR_MAP = {
    "NVDA": "AI / Semiconductors", "AMD": "Semiconductors", "AVGO": "Semiconductors", "ARM": "Semiconductors",
    "SMCI": "AI Infrastructure", "PLTR": "AI / Software", "MSFT": "Big Tech", "GOOGL": "Big Tech",
    "META": "Big Tech", "AMZN": "Big Tech", "AAPL": "Big Tech", "NFLX": "Media / Growth",
    "TSLA": "EV / Growth", "CRWD": "Cybersecurity", "PANW": "Cybersecurity", "ANET": "Networking",
    "ORCL": "Cloud / Software", "QQQ": "ETF", "VOO": "ETF", "SPY": "ETF", "SMH": "ETF", "SOXX": "ETF"
}


def normalize_ticker(t: str) -> str:
    return str(t).strip().upper()


def load_portfolio() -> pd.DataFrame:
    if os.path.exists(PORTFOLIO_FILE):
        df = pd.read_csv(PORTFOLIO_FILE)
    else:
        df = pd.DataFrame(columns=["ticker", "shares", "avg_cost", "target_pct", "sector"])
    for col in ["ticker", "shares", "avg_cost", "target_pct", "sector"]:
        if col not in df.columns:
            df[col] = "" if col in ["ticker", "sector"] else 0.0
    df["ticker"] = df["ticker"].apply(normalize_ticker)
    df["shares"] = pd.to_numeric(df["shares"], errors="coerce").fillna(0.0)
    df["avg_cost"] = pd.to_numeric(df["avg_cost"], errors="coerce").fillna(0.0)
    df["target_pct"] = pd.to_numeric(df["target_pct"], errors="coerce").fillna(0.0)
    df["sector"] = df.apply(lambda r: r["sector"] if str(r["sector"]).strip() else SECTOR_MAP.get(r["ticker"], "Other"), axis=1)
    return df[df["ticker"] != ""].drop_duplicates(subset=["ticker"], keep="last")


def save_portfolio(df: pd.DataFrame):
    df = df.copy()
    df["ticker"] = df["ticker"].apply(normalize_ticker)
    df = df[df["ticker"] != ""]
    df.to_csv(PORTFOLIO_FILE, index=False)


@st.cache_data(ttl=900, show_spinner=False)
def fetch_history(ticker: str, period: str = "6mo") -> pd.DataFrame:
    try:
        data = yf.download(ticker, period=period, interval="1d", progress=False, auto_adjust=True, threads=False, timeout=12)
        if data is None or data.empty:
            return pd.DataFrame()
        data = data.reset_index()
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [c[0] if isinstance(c, tuple) else c for c in data.columns]
        return data
    except Exception:
        return pd.DataFrame()


def calc_rsi(close: pd.Series, period: int = 14) -> float:
    if len(close) < period + 2:
        return np.nan
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1]) if not rsi.empty and pd.notna(rsi.iloc[-1]) else np.nan


def analyze_ticker(ticker: str) -> dict:
    hist = fetch_history(ticker)
    if hist.empty or "Close" not in hist.columns:
        return {"ticker": ticker, "price": np.nan, "signal": "NO DATA", "score": 0, "reason": "No market data", "rsi": np.nan, "change_1m_pct": np.nan, "ma20": np.nan, "ma50": np.nan}

    close = pd.to_numeric(hist["Close"], errors="coerce").dropna()
    volume = pd.to_numeric(hist.get("Volume", pd.Series(dtype=float)), errors="coerce").dropna()
    if len(close) < 55:
        price = float(close.iloc[-1]) if len(close) else np.nan
        return {"ticker": ticker, "price": price, "signal": "WATCH", "score": 40, "reason": "Limited history", "rsi": np.nan, "change_1m_pct": np.nan, "ma20": np.nan, "ma50": np.nan}

    price = float(close.iloc[-1])
    ma20 = float(close.rolling(20).mean().iloc[-1])
    ma50 = float(close.rolling(50).mean().iloc[-1])
    rsi = calc_rsi(close)
    change_1m_pct = float((price / close.iloc[-22] - 1) * 100) if len(close) > 22 else np.nan
    vol_ratio = float(volume.iloc[-1] / volume.rolling(20).mean().iloc[-1]) if len(volume) > 20 and volume.rolling(20).mean().iloc[-1] else 1

    score = 0
    reasons = []
    if price > ma20:
        score += 20; reasons.append("above MA20")
    if price > ma50:
        score += 25; reasons.append("above MA50")
    if ma20 > ma50:
        score += 20; reasons.append("MA20 > MA50")
    if pd.notna(rsi) and 45 <= rsi <= 70:
        score += 15; reasons.append("healthy RSI")
    elif pd.notna(rsi) and rsi < 35:
        score += 8; reasons.append("oversold")
    elif pd.notna(rsi) and rsi > 75:
        score -= 15; reasons.append("overbought")
    if pd.notna(change_1m_pct) and change_1m_pct > 0:
        score += 10; reasons.append("positive 1M momentum")
    if vol_ratio > 1.2:
        score += 10; reasons.append("volume spike")

    score = int(max(0, min(100, score)))
    if score >= 75:
        signal = "BUY / ADD"
    elif score >= 55:
        signal = "WATCH"
    elif score >= 35:
        signal = "HOLD"
    else:
        signal = "REDUCE / AVOID"

    return {"ticker": ticker, "price": price, "signal": signal, "score": score, "reason": ", ".join(reasons) if reasons else "Weak setup", "rsi": rsi, "change_1m_pct": change_1m_pct, "ma20": ma20, "ma50": ma50}


def analyze_many(tickers, limit=None):
    clean = []
    for t in tickers:
        t = normalize_ticker(t)
        if t and t not in clean:
            clean.append(t)
    if limit:
        clean = clean[:limit]
    rows = []
    progress = st.progress(0, text="Loading market data...")
    for i, t in enumerate(clean):
        rows.append(analyze_ticker(t))
        progress.progress((i + 1) / max(1, len(clean)), text=f"Loaded {i + 1}/{len(clean)}: {t}")
    progress.empty()
    return pd.DataFrame(rows)


st.title("📈 Investment Agent — Dynamic Portfolio")
st.caption("Recommendation-only tool. Not financial advice. The app opens instantly; market data loads only after you click a button.")

portfolio = load_portfolio()

with st.sidebar:
    st.header("⚙️ Settings")
    cash = st.number_input("Cash / uninvested amount", min_value=0.0, value=DEFAULT_CASH, step=1000.0)
    st.markdown("---")
    st.subheader("Dynamic recommendation universe")
    universe_input = st.text_area("Tickers to scan", value=", ".join(DEFAULT_RECOMMENDED_UNIVERSE))
    universe = [normalize_ticker(t) for t in universe_input.replace("\n", ",").split(",") if normalize_ticker(t)]
    max_recos = st.slider("How many recommendations to show", 5, 15, 8)
    max_scan = st.slider("Max tickers to scan now", 5, 15, min(12, len(universe)))
    st.markdown("---")
    uploaded = st.file_uploader("Upload actual portfolio CSV", type=["csv"])
    if uploaded is not None:
        try:
            portfolio = pd.read_csv(uploaded)
            save_portfolio(portfolio)
            st.success("Portfolio uploaded and saved.")
            st.rerun()
        except Exception as e:
            st.error(f"Could not read CSV: {e}")

st.subheader("1) Actual portfolio editor")
st.write("Edit shares, average cost, target %, or remove a stock. Click **Save portfolio** after changes.")
edit_df = portfolio.copy()
if edit_df.empty:
    edit_df = pd.DataFrame([{"ticker": "NVDA", "shares": 0, "avg_cost": 0, "target_pct": 10, "sector": "AI / Semiconductors"}])

edited = st.data_editor(
    edit_df,
    num_rows="dynamic",
    use_container_width=True,
    column_config={
        "ticker": st.column_config.TextColumn("Ticker", required=True),
        "shares": st.column_config.NumberColumn("Shares", min_value=0.0, step=1.0),
        "avg_cost": st.column_config.NumberColumn("Avg cost", min_value=0.0, step=1.0),
        "target_pct": st.column_config.NumberColumn("Target %", min_value=0.0, max_value=100.0, step=1.0),
        "sector": st.column_config.TextColumn("Sector"),
    },
)

c_save, c_download = st.columns([1, 1])
with c_save:
    if st.button("💾 Save portfolio", type="primary"):
        save_portfolio(edited)
        st.success("Saved. Removed rows will stay removed after deploy only if you commit portfolio.csv to GitHub.")
        st.rerun()
with c_download:
    st.download_button("⬇️ Download portfolio CSV", data=edited.to_csv(index=False), file_name="portfolio.csv", mime="text/csv")

portfolio = edited.copy()
portfolio["ticker"] = portfolio["ticker"].apply(normalize_ticker)
portfolio = portfolio[portfolio["ticker"] != ""]
portfolio["sector"] = portfolio.apply(lambda r: r["sector"] if str(r["sector"]).strip() else SECTOR_MAP.get(r["ticker"], "Other"), axis=1)

st.subheader("2) Load portfolio market data")
st.info("To avoid cloud timeouts, the app does not download market data on startup. Click the button when you want current prices/signals.")
load_portfolio_data = st.button("🔄 Load / refresh portfolio prices", type="secondary")

port = None
if load_portfolio_data and not portfolio.empty:
    analysis_port = analyze_many(portfolio["ticker"].tolist(), limit=20)
    port = portfolio.merge(analysis_port, on="ticker", how="left")
    port["market_value"] = port["shares"] * port["price"].fillna(0)
    invested_value = float(port["market_value"].sum())
    total_value = invested_value + cash
    port["actual_pct"] = np.where(total_value > 0, port["market_value"] / total_value * 100, 0)
    port["cost_basis"] = port["shares"] * port["avg_cost"]
    port["gain_loss"] = port["market_value"] - port["cost_basis"]
    port["gain_loss_pct"] = np.where(port["cost_basis"] > 0, port["gain_loss"] / port["cost_basis"] * 100, np.nan)
    port["target_value"] = total_value * port["target_pct"] / 100
    port["rebalance_amount"] = port["target_value"] - port["market_value"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total value", f"${total_value:,.0f}")
    c2.metric("Invested", f"${invested_value:,.0f}")
    c3.metric("Cash", f"${cash:,.0f}")
    c4.metric("Total gain/loss", f"${port['gain_loss'].sum():,.0f}")

    show_cols = ["ticker", "shares", "price", "market_value", "actual_pct", "target_pct", "gain_loss", "gain_loss_pct", "signal", "score", "rebalance_amount", "reason"]
    st.dataframe(port[show_cols].sort_values("market_value", ascending=False), use_container_width=True)

    if not port.empty and port["market_value"].sum() > 0:
        c5, c6 = st.columns(2)
        with c5:
            st.plotly_chart(px.pie(port, names="ticker", values="market_value", title="Actual allocation by ticker"), use_container_width=True)
        with c6:
            sector = port.groupby("sector", as_index=False)["market_value"].sum()
            st.plotly_chart(px.bar(sector, x="sector", y="market_value", title="Exposure by sector"), use_container_width=True)

st.subheader("3) Dynamic recommended list")
st.write("Scan a small universe first. You can expand it later if the cloud app is stable.")
if st.button("🚀 Scan recommended list"):
    scan = analyze_many(universe, limit=max_scan)
    scan = scan.sort_values(["score", "change_1m_pct"], ascending=[False, False]).head(max_recos)
    st.dataframe(scan[["ticker", "price", "signal", "score", "rsi", "change_1m_pct", "reason"]], use_container_width=True)

    if not portfolio.empty:
        owned = set(portfolio["ticker"].tolist())
        scan["owned"] = scan["ticker"].isin(owned)
        scan["suggested_action"] = np.where(
            (scan["signal"] == "BUY / ADD") & (~scan["owned"]), "Consider adding to watch/portfolio",
            np.where((scan["signal"] == "BUY / ADD") & (scan["owned"]), "Consider adding if below target",
            np.where(scan["signal"] == "REDUCE / AVOID", "Avoid / reduce", "Watch")),
        )
        st.subheader("4) Suggested actions vs your actual portfolio")
        st.dataframe(scan[["ticker", "owned", "signal", "score", "suggested_action", "reason"]], use_container_width=True)

st.caption(f"Last app render: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Data source: yfinance")
