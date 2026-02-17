# Last updated: Feb 17, 2026 — Finnhub-expanded version

import streamlit as st
import pandas as pd
import finnhub
import openai
import plotly.express as px
from datetime import datetime

# -----------------------------
# Secrets and client setup
# -----------------------------

FINNHUB_KEY = st.secrets["FINNHUB_KEY"]
OPENAI_KEY = st.secrets["OPENAI_KEY"]

finnhub_client = finnhub.Client(api_key=FINNHUB_KEY)
openai.api_key = OPENAI_KEY

# -----------------------------
# Finnhub data fetch functions
# -----------------------------

def get_company_profile(ticker):
    try:
        return finnhub_client.company_profile2(symbol=ticker)
    except Exception:
        return None


def get_fundamentals_finnhub(ticker):
    """Basic financial metrics (ratios, margins, etc.)."""
    try:
        data = finnhub_client.company_basic_financials(ticker, 'all')
        if not data or "metric" not in data:
            return None
        return data["metric"]
    except Exception:
        return None


def get_quarterly_earnings_finnhub(ticker):
    """Quarterly EPS actual vs estimate."""
    try:
        data = finnhub_client.earnings(ticker)
        if not data:
            return None
        return data
    except Exception:
        return None


def get_financials_income_statement(ticker):
    """Quarterly income statement."""
    try:
        data = finnhub_client.financials_reported(symbol=ticker, freq="quarterly")
        if not data or "data" not in data:
            return None
        return data["data"]
    except Exception:
        return None


def get_recommendation_trends(ticker):
    """Analyst recommendation trends."""
    try:
        data = finnhub_client.recommendation_trends(ticker)
        if not data:
            return None
        return data
    except Exception:
        return None


# -----------------------------
# AI summary helpers
# -----------------------------

def summarize_text(title, payload):
    try:
        prompt = f"{title}\n\nData:\n{payload}\n\nProvide a concise, insightful summary."
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300
        )
        return response["choices"][0]["message"]["content"]
    except Exception:
        return "AI summary unavailable."


# -----------------------------
# Transformation helpers
# -----------------------------

def earnings_to_dataframe(earnings, year):
    df = pd.DataFrame(earnings)
    if df.empty:
        return df
    # period like '2025-03-31'
    df["year"] = df["period"].str[:4]
    df = df[df["year"] == str(year)]
    if df.empty:
        return df
    df["period"] = pd.to_datetime(df["period"])
    df = df.sort_values("period")
    return df


def income_to_dataframe(income_data, year):
    # income_data is a list of reports; each has 'report' with 'ic'
    rows = []
    for item in income_data:
        period = item.get("report", {}).get("period")
        if not period:
            continue
        if str(period)[:4] != str(year):
            continue
        ic = item.get("report", {}).get("ic", [])
        row = {"period": period}
        for entry in ic:
            label = entry.get("label")
            value = entry.get("value")
            if label:
                row[label] = value
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["period"] = pd.to_datetime(df["period"])
    df = df.sort_values("period")
    return df


def recommendation_to_dataframe(recs):
    df = pd.DataFrame(recs)
    if df.empty:
        return df
    df["period"] = pd.to_datetime(df["period"])
    df = df.sort_values("period")
    return df


# -----------------------------
# Streamlit UI
# -----------------------------

st.title("Quarterly Report Analyzer — Finnhub Expanded")

col1, col2 = st.columns(2)
with col1:
    ticker = st.text_input("Ticker (e.g., AAPL, MSFT, VZ):").upper().strip()
with col2:
    year = st.text_input("Year (e.g., 2025):").strip()

run = st.button("Run Analysis")

if run:
    if not ticker or not year:
        st.error("Please enter both ticker and year.")
    else:
        st.subheader(f"Results for {ticker} in {year}")

        # -----------------------------
        # Company profile
        # -----------------------------
        profile = get_company_profile(ticker)
        if profile:
            st.markdown("### Company Profile")
            prof_cols = ["name", "ticker", "exchange", "finnhubIndustry", "country", "ipo"]
            prof_df = pd.DataFrame(
                [(k, profile.get(k, "")) for k in prof_cols],
                columns=["Field", "Value"]
            )
            st.table(prof_df)
        else:
            st.info("No company profile available.")

        # -----------------------------
        # Fundamentals (metrics)
        # -----------------------------
        fundamentals = get_fundamentals_finnhub(ticker)
        if fundamentals:
            st.markdown("### Fundamentals (Key Metrics)")
            df_fund = pd.DataFrame(fundamentals.items(), columns=["Metric", "Value"])
            st.dataframe(df_fund)

            fund_summary = summarize_text(
                f"Summarize key valuation and profitability metrics for {ticker}.",
                fundamentals
            )
            st.markdown("#### AI Summary of Fundamentals")
            st.write(fund_summary)
        else:
            st.warning("No fundamentals found for this ticker.")

        # -----------------------------
        # Quarterly earnings (EPS)
        # -----------------------------
        earnings = get_quarterly_earnings_finnhub(ticker)
        df_earn = earnings_to_dataframe(earnings, year) if earnings else pd.DataFrame()

        if not df_earn.empty:
            st.markdown("### Quarterly EPS vs Estimate")
            st.dataframe(df_earn[["period", "epsActual", "epsEstimate", "surprise", "surprisePercent"]])

            # Chart: EPS actual vs estimate
            fig_eps = px.bar(
                df_earn,
                x="period",
                y=["epsActual", "epsEstimate"],
                barmode="group",
                title=f"{ticker} EPS Actual vs Estimate ({year})"
            )
            st.plotly_chart(fig_eps, use_container_width=True)

            earn_summary = summarize_text(
                f"Summarize EPS performance and surprises for {ticker} in {year}.",
                df_earn.to_dict(orient="records")
            )
            st.markdown("#### AI Summary of Quarterly Earnings")
            st.write(earn_summary)
        else:
            st.warning(f"No quarterly EPS data found for {year}.")

        # -----------------------------
        # Income statement (revenue, net income)
        # -----------------------------
        income_data = get_financials_income_statement(ticker)
        df_income = income_to_dataframe(income_data, year) if income_data else pd.DataFrame()

        if not df_income.empty:
            st.markdown("### Income Statement (Selected Lines)")
            # Try to pick common labels if present
            possible_revenue_cols = [c for c in df_income.columns if "Revenue" in c]
            possible_net_cols = [c for c in df_income.columns if "Net Income" in c]

            st.dataframe(df_income)

            # Chart: revenue and net income if we can find them
            value_cols = []
            if possible_revenue_cols:
                value_cols.append(possible_revenue_cols[0])
            if possible_net_cols:
                value_cols.append(possible_net_cols[0])

            if value_cols:
                fig_inc = px.line(
                    df_income,
                    x="period",
                    y=value_cols,
                    markers=True,
                    title=f"{ticker} Income Statement Trends ({year})"
                )
                st.plotly_chart(fig_inc, use_container_width=True)

            income_summary = summarize_text(
                f"Summarize revenue and profitability trends for {ticker} in {year}.",
                df_income.to_dict(orient="records")
            )
            st.markdown("#### AI Summary of Income Statement")
            st.write(income_summary)
        else:
            st.info(f"No quarterly income statement data found for {year}.")

        # -----------------------------
        # Analyst recommendation trends
        # -----------------------------
        recs = get_recommendation_trends(ticker)
        df_recs = recommendation_to_dataframe(recs) if recs else pd.DataFrame()

        if not df_recs.empty:
            st.markdown("### Analyst Recommendation Trends")
            st.dataframe(df_recs)

            # Chart: Strong Buy / Buy / Hold / Sell / Strong Sell over time
            melt_cols = ["strongBuy", "buy", "hold", "sell", "strongSell"]
            available_cols = [c for c in melt_cols if c in df_recs.columns]

            if available_cols:
                df_melt = df_recs.melt(
                    id_vars="period",
                    value_vars=available_cols,
                    var_name="Rating",
                    value_name="Count"
                )
                fig_rec = px.line(
                    df_melt,
                    x="period",
                    y="Count",
                    color="Rating",
                    markers=True,
                    title=f"{ticker} Analyst Recommendation Trends"
                )
                st.plotly_chart(fig_rec, use_container_width=True)

            rec_summary = summarize_text(
                f"Summarize analyst sentiment and recommendation trends for {ticker}.",
                df_recs.to_dict(orient="records")
            )
            st.markdown("#### AI Summary of Analyst Recommendations")
            st.write(rec_summary)
        else:
            st.info("No analyst recommendation trend data available.")
