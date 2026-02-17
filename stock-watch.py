import streamlit as st
import requests
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import openai

# -----------------------------
# CONFIG
# -----------------------------
FINNHUB_KEY = st.secrets["FINNHUB_KEY"] 
ALPHA_KEY = st.secrets["ALPHA_KEY"] 
OPENAI_KEY = st.secrets["OPENAI_KEY"]

openai.api_key = OPENAI_KEY

# -----------------------------
# DATA HELPERS
# -----------------------------
def get_alpha_fundamentals(ticker):
    """Pull quarterly fundamentals from Alpha Vantage."""
    url = (
        f"https://www.alphavantage.co/query?"
        f"function=OVERVIEW&symbol={ticker}&apikey={ALPHA_KEY}"
    )
    data = requests.get(url).json()
    return data


def get_alpha_quarterly_reports(ticker):
    """Alpha Vantage quarterly earnings."""
    url = (
        f"https://www.alphavantage.co/query?"
        f"function=EARNINGS&symbol={ticker}&apikey={ALPHA_KEY}"
    )
    data = requests.get(url).json()
    if "quarterlyEarnings" not in data:
        return pd.DataFrame()
    df = pd.DataFrame(data["quarterlyEarnings"])
    df["reportedDate"] = pd.to_datetime(df["reportedDate"])
    return df


def get_finnhub_prices(ticker, start, end):
    """Daily prices from Finnhub."""
    url = "https://finnhub.io/api/v1/stock/candle"
    params = {
        "symbol": ticker,
        "resolution": "D",
        "from": int(start.timestamp()),
        "to": int(end.timestamp()),
        "token": FINNHUB_KEY
    }
    data = requests.get(url, params=params).json()
    if data.get("s") != "ok":
        return pd.DataFrame()
    df = pd.DataFrame({
        "t": pd.to_datetime(data["t"], unit="s"),
        "c": data["c"]
    })
    df.rename(columns={"t": "date", "c": "close"}, inplace=True)
    return df


# -----------------------------
# AI SENTIMENT
# -----------------------------
def classify_sentiment(metric_name, value):
    """Use OpenAI to classify positive/neutral/negative."""
    prompt = (
        f"Metric: {metric_name}\n"
        f"Value: {value}\n"
        "Classify as positive, neutral, or negative for investors. "
        "Respond with only one word."
    )

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2,
        temperature=0
    )

    return response["choices"][0]["message"]["content"].strip()


# -----------------------------
# PRICE REACTION
# -----------------------------
def compute_price_reaction(price_df, report_date):
    """Compute % change after 1,3,10,30 days."""
    horizons = [1, 3, 10, 30]
    out = {}

    for h in horizons:
        target = report_date + dt.timedelta(days=h)
        future = price_df[price_df["date"] >= target].head(1)
        base = price_df[price_df["date"] >= report_date].head(1)

        if future.empty or base.empty:
            out[f"{h}d"] = None
        else:
            out[f"{h}d"] = (future["close"].iloc[0] / base["close"].iloc[0] - 1) * 100

    return out


# -----------------------------
# STREAMLIT UI
# -----------------------------
st.title("Quarterly Report Analyzer with AI Sentiment")

ticker = st.text_input("Ticker", "AAPL").upper()
col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("Start date", dt.date(2020, 1, 1))
with col2:
    end_date = st.date_input("End date", dt.date.today())

if st.button("Run Analysis"):
    st.write("### Fetching quarterly reports…")
    earnings = get_alpha_quarterly_reports(ticker)

    if earnings.empty:
        st.error("No quarterly earnings found.")
        st.stop()

    earnings = earnings[
        (earnings["reportedDate"].dt.date >= start_date) &
        (earnings["reportedDate"].dt.date <= end_date)
    ]

    if earnings.empty:
        st.warning("No reports in this date range.")
        st.stop()

    st.write("### Fetching fundamentals…")
    fundamentals = get_alpha_fundamentals(ticker)

    # Extract metrics
    metrics = {
        "ROE": fundamentals.get("ReturnOnEquityTTM"),
        "OCF": fundamentals.get("OperatingCashFlowTTM"),
        "Quick Ratio": fundamentals.get("QuickRatio"),
        "EBIT": fundamentals.get("EBITDA"),  # Alpha Vantage uses EBITDA
        "Revenue Growth": fundamentals.get("QuarterlyRevenueGrowthYOY"),
        "P/B": fundamentals.get("PriceToBookRatio"),
        "PEG": fundamentals.get("PEGRatio")
    }

    # AI sentiment
    st.write("### AI Sentiment Classification")
    sentiment_results = {}
    for k, v in metrics.items():
        sentiment_results[k] = classify_sentiment(k, v)

    sentiment_df = pd.DataFrame({
        "Metric": list(metrics.keys()),
        "Value": list(metrics.values()),
        "Sentiment": list(sentiment_results.values())
    })

    st.dataframe(sentiment_df)

    # Price reaction
    st.write("### Price Reaction After Reports")
    all_reactions = []

    # Pull price data once
    price_df = get_finnhub_prices(
        ticker,
        dt.datetime.combine(start_date, dt.time()),
        dt.datetime.combine(end_date + dt.timedelta(days=40), dt.time())
    )

    for _, row in earnings.iterrows():
        report_date = row["reportedDate"].date()
        reaction = compute_price_reaction(price_df, row["reportedDate"])
        reaction["report_date"] = report_date
        all_reactions.append(reaction)

    reaction_df = pd.DataFrame(all_reactions)
    st.dataframe(reaction_df)

    # Plot
    st.write("### Price Reaction Chart")
    fig, ax = plt.subplots(figsize=(8, 4))
    for h in ["1d", "3d", "10d", "30d"]:
        if h in reaction_df.columns:
            ax.plot(reaction_df["report_date"], reaction_df[h], marker="o", label=h)

    ax.axhline(0, color="gray")
    ax.set_ylabel("% change")
    ax.set_title(f"{ticker} Price Reaction After Earnings")
    ax.legend()
    plt.xticks(rotation=45)
    st.pyplot(fig)
