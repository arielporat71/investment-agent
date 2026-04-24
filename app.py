import pandas as pd
import plotly.express as px
import streamlit as st

from agent import analyze_portfolio, format_alerts, send_telegram, fetch_history

st.set_page_config(page_title="Aggressive Portfolio Agent", layout="wide")
st.title("🚀 Aggressive Portfolio Agent — Recommendation Only")
st.caption("סוכן המלצות לתיק השקעות — ללא ביצוע קניות/מכירות בפועל")

portfolio_path = st.sidebar.text_input("Portfolio CSV", "portfolio.csv")
refresh = st.sidebar.button("🔄 Refresh signals")
send_alerts = st.sidebar.button("📲 Send Telegram alerts")

@st.cache_data(ttl=900)
def get_data(path):
    return analyze_portfolio(path)

df = get_data(portfolio_path)

if refresh:
    st.cache_data.clear()
    df = get_data(portfolio_path)

portfolio_value = df["position_ils"].sum()
active_value = df[df["ticker"] != "CASH"]["position_ils"].sum()
cash_value = df[df["ticker"] == "CASH"]["position_ils"].sum()
buy_count = (df["signal"] == "BUY").sum()
sell_count = df["signal"].astype(str).str.contains("SELL", na=False).sum()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Portfolio", f"₪{portfolio_value:,.0f}")
c2.metric("Invested", f"₪{active_value:,.0f}")
c3.metric("Cash", f"₪{cash_value:,.0f}")
c4.metric("BUY / SELL", f"{buy_count} / {sell_count}")

st.subheader("📊 Signals")
st.dataframe(
    df[["ticker", "name", "signal", "score", "price", "rsi", "ma20", "ma50", "volume_ratio", "pnl_pct", "position_ils", "target_weight", "reason"]],
    use_container_width=True,
    hide_index=True,
)

left, right = st.columns(2)
with left:
    st.subheader("Exposure by sector")
    sector_df = df.groupby("sector", as_index=False)["position_ils"].sum()
    fig = px.pie(sector_df, names="sector", values="position_ils")
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Signals count")
    sig_df = df.groupby("signal", as_index=False).size()
    fig = px.bar(sig_df, x="signal", y="size")
    st.plotly_chart(fig, use_container_width=True)

st.subheader("📈 Price chart")
tickers = [t for t in df["ticker"].tolist() if t != "CASH"]
selected = st.selectbox("Ticker", tickers)
period = st.selectbox("Period", ["3mo", "6mo", "1y", "2y"], index=1)
hist = fetch_history(selected, period=period)
if not hist.empty:
    chart_df = hist.reset_index()
    fig = px.line(chart_df, x="Date", y="Close", title=f"{selected} Close Price")
    st.plotly_chart(fig, use_container_width=True)

if send_alerts:
    alerts = format_alerts(df)
    if alerts:
        ok = send_telegram("\n\n".join(alerts))
        st.success("Alerts sent to Telegram" if ok else "Telegram not configured or failed")
    else:
        st.info("No active alerts right now")

st.warning("אין לראות בכך ייעוץ השקעות. זה כלי עזר לקבלת החלטות בלבד.")
