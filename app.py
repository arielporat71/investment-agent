import os
from datetime import date
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from supabase import create_client

st.set_page_config(page_title="Market & Portfolio Agent", page_icon="📈", layout="wide")
SCAN="market_scan.csv"; META="market_scan_date.txt"

PROFILES = {
"MPC":("Marathon Petroleum","U.S. refining, fuel marketing and midstream energy company."),
"VLO":("Valero Energy","Independent refiner producing transportation and renewable fuels."),
"PSX":("Phillips 66","Diversified energy company spanning refining, midstream and chemicals."),
"MSFT":("Microsoft","Cloud, enterprise software and AI company behind Azure and Microsoft 365."),
"ZS":("Zscaler","Cloud cybersecurity company focused on zero-trust access."),
"SNOW":("Snowflake","Cloud data platform for analytics, data engineering and AI workloads."),
"NET":("Cloudflare","Internet infrastructure, connectivity and cybersecurity platform."),
"TMO":("Thermo Fisher Scientific","Life-sciences tools, diagnostics and laboratory-products company."),
"PANW":("Palo Alto Networks","Cybersecurity platform covering network, cloud and security operations."),
"META":("Meta Platforms","Technology company behind Facebook, Instagram and WhatsApp."),
"NVDA":("NVIDIA","Semiconductor company leading in GPUs and AI computing infrastructure."),
"AMD":("AMD","Semiconductor company producing CPUs, GPUs and data-center accelerators."),
"PLTR":("Palantir","Data analytics and AI software for government and commercial customers."),
"TSLA":("Tesla","Electric vehicles, energy storage and related technology company."),
"AMZN":("Amazon","E-commerce, cloud computing and digital-services company; owner of AWS."),
"CRWD":("CrowdStrike","Cloud-native cybersecurity company focused on endpoint and threat protection."),
"AAPL":("Apple","Consumer technology company behind iPhone, Mac, services and wearables."),
"GOOGL":("Alphabet","Technology company behind Google Search, YouTube, Cloud and AI."),
"AVGO":("Broadcom","Semiconductor and infrastructure-software company."),
"ARM":("Arm Holdings","Designer of CPU architectures licensed across mobile and data-center chips."),
"TSM":("TSMC","Leading global semiconductor foundry."),
"ASML":("ASML","Supplier of advanced lithography systems used in semiconductor manufacturing."),
"MU":("Micron Technology","Memory semiconductor company producing DRAM and NAND."),
"APP":("AppLovin","Advertising-technology platform focused on app monetization."),
"ANET":("Arista Networks","Cloud-networking company supplying high-speed data-center switches."),
"COIN":("Coinbase","Cryptocurrency trading and infrastructure platform."),
"HOOD":("Robinhood","Digital brokerage and financial-services platform."),
"OKLO":("Oklo","Advanced nuclear-energy company developing compact fission power systems."),
"CEG":("Constellation Energy","Major U.S. electricity producer with a large nuclear fleet."),
"VST":("Vistra","U.S. power-generation and retail-electricity company."),
"FSLR":("First Solar","Manufacturer of thin-film solar modules."),
}

@st.cache_data(ttl=86400, show_spinner=False)
def company_profile(t):
    t=str(t).upper().strip()
    if t in PROFILES:
        return PROFILES[t]
    try:
        info=yf.Ticker(t).get_info()
        name=info.get("longName") or info.get("shortName") or t
        summary=(info.get("longBusinessSummary") or "").strip()
        if summary:
            first=summary.split(". ")[0].strip()
            if len(first)>180:
                first=first[:177].rsplit(" ",1)[0]+"..."
            if not first.endswith("."):
                first += "."
            return name, first
        sector=info.get("sector") or info.get("industry")
        return name, (f"{sector} company included in the market scan." if sector else "Public company included in the market scan.")
    except Exception:
        return t, "Public company included in the market scan."

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
    name, brief = company_profile(t)
    return {"ticker":t,"company":name,"Company brief":brief,"price":round(p,2),"score":score,"Score %":round(score/11*100),"signal":sig,"1M %":round(m1,1),"3M %":round(m3,1),"6M %":round(m6,1),"RSI":round(rv,1),"volatility %":round(vol,1),"reason":", ".join(why)}

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

def get_supabase():
    try:
        url = st.secrets["SUPABASE_URL"]
        key = st.secrets["SUPABASE_ANON_KEY"]
        return create_client(url, key)
    except Exception:
        return None

def login_gate():
    if "sb" not in st.session_state:
        st.session_state.sb = get_supabase()
    sb = st.session_state.sb
    if sb is None:
        st.error("Supabase is not configured yet. Add SUPABASE_URL and SUPABASE_ANON_KEY to Streamlit Secrets.")
        st.stop()

    if st.session_state.get("user_id"):
        return sb

    st.title("🔐 Investment Agent Login")
    st.caption("Each account has its own private portfolio.")
    mode = st.radio("Account", ["Sign in", "Create account"], horizontal=True)
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    if mode == "Sign in":
        if st.button("Sign in", type="primary"):
            try:
                res = sb.auth.sign_in_with_password({"email": email.strip(), "password": password})
                st.session_state.user_id = res.user.id
                st.session_state.user_email = res.user.email
                st.rerun()
            except Exception as e:
                st.error("Sign in failed. Check the email/password and that the account is confirmed.")
    else:
        if st.button("Create account", type="primary"):
            try:
                res = sb.auth.sign_up({"email": email.strip(), "password": password})
                if res.user:
                    st.success("Account created. If email confirmation is enabled in Supabase, confirm the email and then sign in.")
            except Exception as e:
                st.error("Could not create the account. The email may already exist or the password may be too short.")
    st.stop()

def load_pf():
    sb = st.session_state.sb
    uid = st.session_state.user_id
    try:
        res = sb.table("portfolios").select("ticker,shares,avg_buy_price").eq("user_id", uid).order("ticker").execute()
        rows = res.data or []
        return pd.DataFrame(rows, columns=["ticker","shares","avg_buy_price"])
    except Exception as e:
        st.error("Could not load your private portfolio.")
        return pd.DataFrame(columns=["ticker","shares","avg_buy_price"])

def save_pf(df):
    sb = st.session_state.sb
    uid = st.session_state.user_id
    clean = df.copy()
    if "ticker" not in clean.columns:
        clean["ticker"] = ""
    if "shares" not in clean.columns:
        clean["shares"] = 0.0
    if "avg_buy_price" not in clean.columns:
        clean["avg_buy_price"] = 0.0
    clean["ticker"] = clean["ticker"].astype(str).str.upper().str.strip()
    clean = clean[clean["ticker"].ne("") & clean["ticker"].ne("NAN")].copy()
    clean["shares"] = pd.to_numeric(clean["shares"], errors="coerce").fillna(0.0)
    clean["avg_buy_price"] = pd.to_numeric(clean["avg_buy_price"], errors="coerce").fillna(0.0)
    records = [
        {"user_id": uid, "ticker": r["ticker"], "shares": float(r["shares"]), "avg_buy_price": float(r["avg_buy_price"])}
        for _, r in clean.iterrows()
    ]
    # RLS ensures a signed-in user can only delete/write their own rows.
    sb.table("portfolios").delete().eq("user_id", uid).execute()
    if records:
        sb.table("portfolios").insert(records).execute()
    return clean

login_gate()

st.title("📈 Market & Portfolio Agent")
st.caption("Daily broad-market discovery. Your actual portfolio remains under manual control.")
with st.sidebar:
    st.caption(f"Signed in: {st.session_state.get('user_email','')}")
    if st.button("Sign out"):
        try: st.session_state.sb.auth.sign_out()
        except Exception: pass
        for k in ["user_id","user_email","sb"]:
            st.session_state.pop(k, None)
        st.rerun()
    n=st.slider("Daily recommendations",5,25,15)
    force=st.button("Run market scan now")
    cash=st.number_input("Cash / uninvested amount",0.0,value=20000.0,step=1000.0)
    st.caption(f"Scanning {len(UNIVERSE)} liquid U.S. stocks; results cached for the day.")

a,b,c=st.tabs(["🌎 Daily market scan","💼 Portfolio actions","✏️ Update portfolio"])
with a:
    rec=scan(n) if force else saved_scan(n)
    st.subheader("🏆 Top Picks")
    st.caption("Highest-ranked names from today's market scan.")
    top = rec.head(5).reset_index(drop=True)
    for _, r in top.iterrows():
        st.markdown(
            f"### {r['ticker']} — {r.get('company', r['ticker'])}\n"
            f"**{r['signal']} · Score {int(r['score'])}/11 · Score {int(r['Score %'])}% · ${float(r['price']):,.2f}**  \n"
            f"1M {float(r['1M %']):+.1f}% · 3M {float(r['3M %']):+.1f}% · RSI {float(r['RSI']):.1f}  \n"
            f"{r.get('Company brief','')}"
        )
        st.divider()

    st.subheader("📋 Full ranked list")
    mobile_cols=["ticker","company","score","Score %","signal","price","1M %","3M %"]
    st.dataframe(rec[mobile_cols],use_container_width=True,hide_index=True)

    with st.expander("Company descriptions & technical details"):
        detail_cols=["ticker","company","score","Score %","Company brief","6M %","RSI","volatility %","reason"]
        st.dataframe(rec[detail_cols],use_container_width=True,hide_index=True)

    st.info("The recommendation list is discovered by the daily scan and can change each day. It never changes your actual holdings automatically.")

with c:
    pf=load_pf()
    edit=st.data_editor(pf,num_rows="dynamic",use_container_width=True)
    if st.button("Save actual portfolio", type="primary"):
        try:
            saved = save_pf(edit)
            st.success(f"Portfolio saved privately. {len(saved)} holding(s) stored for your account.")
            st.session_state["portfolio_saved_at"] = str(pd.Timestamp.now())
        except Exception as e:
            st.error("Portfolio could not be saved. Check the Supabase table/policies and try again.")

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
