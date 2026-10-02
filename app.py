import os
from datetime import date
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Market & Portfolio Agent", page_icon="📈", layout="wide")
PF="portfolio.csv"; SCAN="market_scan.csv"; META="market_scan_date.txt"

UNIVERSE="""AAPL MSFT NVDA AMZN GOOGL META AVGO TSLA LLY JPM WMT V ORCL MA XOM NFLX COST JNJ HD PG BAC ABBV KO CRM AMD PLTR CSCO GE IBM PM CVX ABT MCD CAT NOW ISRG GS TMO MRK DIS UBER AXP QCOM INTU RTX BKNG PEP AMGN TXN SPGI DHR ACN LOW PFE HON UNP SYK ADP GILD TJX DE PANW MU BA COP ADI APP KLAC LRCX CRWD ANET MELI SHOP ARM SNOW NET DDOG COIN HOOD RBLX TTD SOFI RKLB OKLO CEG VST NEE FSLR ENPH SMCI DELL HPE MRVL MSTR AMAT ASML TSM ON INTC NXPI MCHP CDNS SNPS FTNT ZS TEAM MDB HUBS ADBE PYPL XYZ DASH ABNB SPOT RDDT DUOL CAVA CMG SBUX NKE LULU ROST MAR TGT GM F TM JPM C GS MS WFC SCHW BLK BX KKR APO COF SLB EOG OXY MPC VLO PSX NVO UNH REGN VRTX BSX MDT LMT NOC GD ETN PH URI CARR TT EMR""".split()

@st.cache_data(ttl=3600,show_spinner=False)
def hist(t):
    try:
        d=yf.download(t,period="1y",auto_adjust=True,progress=False,threads=False)
        if isinstance(d.columns,pd.MultiIndex): d.columns=[x[0] for x in d.columns]
        return d.dropna()
    except:return pd.DataFrame()

def rsi(s,n=14):
    x=s.diff(); u=x.clip(lower=0).rolling(n).mean(); d=(-x.clip(upper=0)).rolling(n).mean()
    return 100-100/(1+u/d.replace(0,np.nan))

def analyze(t):
    d=hist(t)
    if d.empty or len(d)<130:return None
    c=d.Close; p=float(c.iloc[-1]); m20=float(c.rolling(20).mean().iloc[-1]); m50=float(c.rolling(50).mean().iloc[-1]); m200=float(c.rolling(min(200,len(c))).mean().iloc[-1])
    rv=float(rsi(c).iloc[-1]); m1=(p/float(c.iloc[-22])-1)*100; m3=(p/float(c.iloc[-63])-1)*100; m6=(p/float(c.iloc[-126])-1)*100
    vol=float(c.pct_change().tail(63).std()*np.sqrt(252)*100); score=0; why=[]
    for cond,pts,label in [(p>m20,1,"above MA20"),(p>m50,2,"above MA50"),(p>m200,2,"above long trend"),(m20>m50,1,"positive short trend"),(m1>3,1,"positive 1M momentum"),(m3>8,2,"strong 3M momentum"),(m6>15,1,"strong 6M momentum"),(45<=rv<=70,1,"constructive RSI")]:
        if cond: score+=pts; why.append(label)
    if rv>78: score-=2; why.append("extended RSI")
    if vol>75: score-=1; why.append("high volatility")
    sig="STRONG CANDIDATE" if score>=9 else "BUY CANDIDATE" if score>=7 else "WATCH" if score>=5 else "HOLD / NEUTRAL" if score>=3 else "REDUCE / REVIEW"
    return {"ticker":t,"price":round(p,2),"score":score,"signal":sig,"1M %":round(m1,1),"3M %":round(m3,1),"6M %":round(m6,1),"RSI":round(rv,1),"volatility %":round(vol,1),"reason":", ".join(why)}

def scan(n):
    rows=[]; b=st.progress(0,text="Scanning broad U.S. market universe...")
    for i,t in enumerate(UNIVERSE):
        a=analyze(t)
        if a:rows.append(a)
        b.progress((i+1)/len(UNIVERSE),text=f"Scanning {i+1}/{len(UNIVERSE)} — {t}")
    b.empty()
    d=pd.DataFrame(rows).sort_values(["score","3M %","6M %"],ascending=False)
    d.to_csv(SCAN,index=False); open(META,"w").write(str(date.today()))
    return d.head(n)

def saved_scan(n):
    try:
        if open(META).read().strip()==str(date.today()): return pd.read_csv(SCAN).head(n)
    except:pass
    return scan(n)

def load_pf():
    try:return pd.read_csv(PF)
    except:return pd.DataFrame(columns=["ticker","shares","avg_buy_price"])

st.title("📈 Market & Portfolio Agent")
st.caption("Daily broad-market discovery. Your actual portfolio remains under manual control.")
with st.sidebar:
    n=st.slider("Daily recommendations",5,25,15)
    force=st.button("Run market scan now")
    cash=st.number_input("Cash / uninvested amount",0.0,value=20000.0,step=1000.0)
    st.caption(f"Scanning {len(UNIVERSE)} liquid U.S. stocks; results cached for the day.")

a,b,c=st.tabs(["🌎 Daily market scan","💼 Portfolio actions","✏️ Update portfolio"])
with a:
    rec=scan(n) if force else saved_scan(n)
    st.subheader("Today's discovered candidates")
    st.dataframe(rec,use_container_width=True,hide_index=True)
    st.info("The ticker list is discovered by the scan and can change daily. It does not change your holdings.")

with c:
    pf=load_pf()
    edit=st.data_editor(pf,num_rows="dynamic",use_container_width=True)
    if st.button("Save actual portfolio"):
        edit["ticker"]=edit.ticker.astype(str).str.upper().str.strip()
        edit.to_csv(PF,index=False); st.success("Portfolio saved.")

with b:
    pf=load_pf()
    if pf.empty: st.warning("Enter holdings in Update portfolio.")
    else:
        rows=[]; bar=st.progress(0,text="Analyzing current portfolio...")
        for i,r in pf.iterrows():
            t=str(r.ticker).upper().strip()
            x=analyze(t) if t else None
            if x:
                sh=float(r.shares or 0); avg=float(r.avg_buy_price or 0); val=sh*x["price"]; cost=sh*avg
                x.update({"shares":sh,"avg buy":avg,"value":round(val,2),"gain/loss":round(val-cost,2),"gain/loss %":round((val-cost)/cost*100,2) if cost else 0})
                rows.append(x)
            bar.progress((i+1)/max(len(pf),1))
        bar.empty(); d=pd.DataFrame(rows)
        if not d.empty:
            total=d["value"].sum()+cash; d["portfolio %"]=(d["value"]/total*100).round(2)
            st.metric("Portfolio + cash",f"${total:,.0f}")
            st.dataframe(d,use_container_width=True,hide_index=True)
            st.caption("Signals are model-generated trend/momentum/risk indicators. No trade is made automatically.")
