# Modified 1-17 at 421
import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from openai import OpenAI

# =========================
# Config & clients
# =========================
st.set_page_config(page_title="Quarterly Report Analyzer", layout="wide")

ALPHA_KEY = st.secrets["ALPHA_KEY"]
OPENAI_KEY = st.secrets["OPENAI_KEY"]

client = OpenAI(api_key=OPENAI_KEY)

ALPHA_BASE = "https://www.alphavantage.co/query"


# =========================
# Data fetchers
# =========================
@st.cache_data(show_spinner=False)
def fetch_quarterly_earnings(symbol: str) -> pd.DataFrame:
    params = {
        "function": "EARNINGS",
        "symbol": symbol,
        "apikey": ALPHA_KEY,
    }
    r = requests.get(ALPHA_BASE, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()

    if "quarterlyEarnings" not in data:
        return pd.DataFrame()

    df = pd.DataFrame(data["quarterlyEarnings"])
    # Normalize types
    df["reportedDate"] = pd.to_datetime(df["reportedDate"], errors="coerce")
    for col in ["reportedEPS", "estimatedEPS", "surprise", "surprisePercentage"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_values("reportedDate")


@st.cache_data(show_spinner=False)
def fetch_overview(symbol: str) -> pd.DataFrame:
    params = {
        "function": "OVERVIEW",
        "symbol": symbol,
        "apikey": ALPHA_KEY,
    }
    r = requests.get(ALPHA_BASE, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
    if not data or "Symbol" not in data:
        return pd.DataFrame()

    # Pick a subset of useful fundamentals
    fields = [
        "Symbol",
        "Name",
        "Sector",
        "MarketCapitalization",
        "PERatio",
        "PEGRatio",
        "EPS",
        "ReturnOnEquityTTM",
        "ProfitMargin",
        "OperatingMarginTTM",
        "CurrentRatio",
        "QuickRatio",
        "DebtToEquity",
    ]
    row = {k: data.get(k, None) for k in fields}
    df = pd.DataFrame([row])
    # Convert numeric fields
    numeric_cols = [
        "MarketCapitalization",
        "PERatio",
        "PEGRatio",
        "EPS",
        "ReturnOnEquityTTM",
        "ProfitMargin",
        "OperatingMarginTTM",
        "CurrentRatio",
        "QuickRatio",
        "DebtToEquity",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# =========================
# AI sentiment
# =========================
def classify_sentiment(metric_name: str, value) -> str:
    prompt = (
        f"You are an equity analyst.\n"
        f"Metric: {metric_name}\n"
        f"Value: {value}\n\n"
        "Classify this as positive, neutral, or negative for long-term investors. "
        "Respond with exactly one word: Positive, Neutral, or Negative."
    )

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {e}"


def build_sentiment_table(fund_df: pd.DataFrame) -> pd.DataFrame:
    if fund_df.empty:
        return pd.DataFrame()

    metrics = {
        "PE Ratio": fund_df["PERatio"].iloc[0],
        "PEG Ratio": fund_df["PEGRatio"].iloc[0],
        "EPS": fund_df["EPS"].iloc[0],
        "ROE (TTM)": fund_df["ReturnOnEquityTTM"].iloc[0],
        "Profit Margin": fund_df["ProfitMargin"].iloc[0],
        "Operating Margin (TTM)": fund_df["OperatingMarginTTM"].iloc[0],
        "Current Ratio": fund_df["CurrentRatio"].iloc[0],
        "Quick Ratio": fund_df["QuickRatio"].iloc[0],
        "Debt to Equity": fund_df["DebtToEquity"].iloc[0],
    }

    rows = []
    for name, val in metrics.items():
        if pd.isna(val):
            continue
        sentiment = classify_sentiment(name, val)
        rows.append({"Metric": name, "Value": val, "AI Sentiment": sentiment})

    return pd.DataFrame(rows)


# =========================
# UI
# =========================
st.title("Quarterly Report Analyzer with AI Sentiment")

col1, col2, col3 = st.columns(3)
with col1:
    ticker = st.text_input("Ticker", "AAPL").upper().strip()
with col2:
    start_date = st.date_input("Start date", pd.to_datetime("2020-01-01"))
with col3:
    end_date = st.date_input("End date", pd.to_datetime("today"))

run = st.button("Run Analysis")

if run:
    if not ticker:
        st.error("Please enter a ticker.")
        st.stop()

    st.subheader(f"Fetching data for {ticker}…")

    # ---- Fetch data ----
    with st.spinner("Fetching quarterly earnings…"):
        earnings_df = fetch_quarterly_earnings(ticker)

    with st.spinner("Fetching fundamentals…"):
        overview_df = fetch_overview(ticker)

    # ---- Fundamentals section ----
    st.markdown("## Fundamentals snapshot")

    if overview_df.empty:
        st.warning("No fundamentals found for this ticker.")
    else:
        info_cols = st.columns([2, 1, 1])
        with info_cols[0]:
            st.write(
                f"**{overview_df['Name'].iloc[0]}**  "
                f"({overview_df['Symbol'].iloc[0]})"
            )
            st.write(f"Sector: {overview_df['Sector'].iloc[0]}")
        with info_cols[1]:
            mc = overview_df["MarketCapitalization"].iloc[0]
            if pd.notna(mc):
                st.metric("Market Cap", f"${mc:,.0f}")
        with info_cols[2]:
            pe = overview_df["PERatio"].iloc[0]
            if pd.notna(pe):
                st.metric("P/E Ratio", f"{pe:.2f}")

        st.dataframe(
            overview_df.set_index("Symbol").T,
            use_container_width=True,
        )

        st.markdown("### AI sentiment on key metrics")
        sent_df = build_sentiment_table(overview_df)
        if not sent_df.empty:
            st.dataframe(sent_df, use_container_width=True)
        else:
            st.info("Not enough numeric fundamentals to run sentiment.")

    # ---- Earnings section ----
    st.markdown("## Quarterly earnings")

    if earnings_df.empty:
        st.warning("No quarterly earnings found.")
    else:
        # Filter by date range
        mask = (earnings_df["reportedDate"] >= pd.to_datetime(start_date)) & (
            earnings_df["reportedDate"] <= pd.to_datetime(end_date)
        )
        filtered = earnings_df[mask]

        if filtered.empty:
            st.warning("No quarterly earnings in the selected date range.")
        else:
            st.dataframe(filtered, use_container_width=True)

            # EPS over time
            st.markdown("### Reported vs Estimated EPS")
            eps_fig = px.line(
                filtered,
                x="reportedDate",
                y=["reportedEPS", "estimatedEPS"],
                markers=True,
                labels={"value": "EPS", "reportedDate": "Reported Date", "variable": "Series"},
                title=f"{ticker} EPS History",
            )
            st.plotly_chart(eps_fig, use_container_width=True)

            # Surprise percentage
            if filtered["surprisePercentage"].notna().any():
                st.markdown("### Earnings surprise (%)")
                sp_fig = px.bar(
                    filtered,
                    x="reportedDate",
                    y="surprisePercentage",
                    labels={
                        "reportedDate": "Reported Date",
                        "surprisePercentage": "Surprise (%)",
                    },
                    title=f"{ticker} Earnings Surprise Percentage",
                )
                st.plotly_chart(sp_fig, use_container_width=True)

    st.success("Analysis complete.")
else:
    st.info("Enter a ticker and click **Run Analysis** to begin.")
