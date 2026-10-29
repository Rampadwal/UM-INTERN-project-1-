import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path

st.set_page_config(
    page_title="UAC Care Load Analytics",
    page_icon="📊",
    layout="wide"
)

st.title("System Capacity & Care Load Analytics")
st.caption(
    "Daily care load, intake, transfers, discharges and "
    "capacity-pressure indicators"
)

DATA_FILE = Path("uac_data.csv")

if not DATA_FILE.exists():
    st.error(
        "Dataset not found. Place uac_data.csv in the project folder."
    )
    st.stop()

df = pd.read_csv(DATA_FILE)

# Match CSV headers while ignoring capitalization and extra spaces.
df.columns = [str(c).strip() for c in df.columns]

# Map dataset columns to short internal names.
column_map = {
    "Date": "date",
    "Children apprehended and placed in CBP custody": "daily_intake",
    "Children in CBP custody": "cbp_load",
    "Children transferred out of CBP custody": "transfers",
    "Children in HHS Care": "hhs_load",
    "Children discharged from HHS Care": "discharges",
}

missing = [c for c in column_map if c not in df.columns]

if missing:
    st.error("Missing required columns: " + ", ".join(missing))
    st.write("Columns found in the CSV:", list(df.columns))
    st.stop()

df = df.rename(columns=column_map)

df["date"] = pd.to_datetime(df["date"], errors="coerce")

metrics = [
    "daily_intake", "cbp_load", "transfers",
    "hhs_load", "discharges"
]

for col in metrics:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Record data-quality issues before removing invalid dates.
invalid_dates = int(df["date"].isna().sum())
duplicate_dates = int(df["date"].dropna().duplicated().sum())

df = df.dropna(subset=["date"]).sort_values("date")

# Aggregate duplicate dates so the dashboard has one row per day.
# Review the original duplicate records before relying on these totals.
df = df.groupby("date", as_index=False)[metrics].sum(min_count=1)

# Check gaps, but do not invent actual observations.
if df.empty:
    st.error("No valid reporting dates found.")
    st.stop()

full_dates = pd.date_range(
    df["date"].min(), df["date"].max(), freq="D"
)
df = df.set_index("date").reindex(full_dates)
df.index.name = "date"
df = df.reset_index()

missing_days = int(df[metrics].isna().any(axis=1).sum())

# Derived metrics
df["total_load"] = df["cbp_load"] + df["hhs_load"]
df["net_intake"] = df["transfers"] - df["discharges"]

df["load_change"] = df["total_load"].diff()
df["growth_rate_pct"] = (
    df["total_load"].pct_change(fill_method=None) * 100
).replace([np.inf, -np.inf], np.nan)

df["avg_7day_load"] = df["total_load"].rolling(
    7, min_periods=7
).mean()

df["avg_14day_net_intake"] = df["net_intake"].rolling(
    14, min_periods=14
).mean()

df["backlog_pressure"] = df["avg_14day_net_intake"] > 0

# Sidebar filters
st.sidebar.header("Dashboard Filters")

start_date = st.sidebar.date_input(
    "Start date", value=df["date"].min().date()
)
end_date = st.sidebar.date_input(
    "End date", value=df["date"].max().date()
)

if start_date > end_date:
    st.error("Start date must be before or equal to end date.")
    st.stop()

view = df[
    (df["date"].dt.date >= start_date)
    & (df["date"].dt.date <= end_date)
].copy()

if view.empty:
    st.warning("No observations are available for this date range.")
    st.stop()

granularity = st.sidebar.selectbox(
    "Time granularity", ["Daily", "Weekly", "Monthly"]
)

metric_choice = st.sidebar.multiselect(
    "Load metrics",
    ["CBP Load", "HHS Load", "Total Load", "7-day Average"],
    default=["CBP Load", "HHS Load", "Total Load"]
)

# KPI cards
latest = view.iloc[-1]
previous = view.iloc[-2] if len(view) > 1 else None

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Total Children Under Care",
    f"{latest['total_load']:,.0f}"
    if pd.notna(latest["total_load"]) else "N/A",
    delta=(
        f"{latest['load_change']:+,.0f}"
        if pd.notna(latest["load_change"]) else None
    )
)

c2.metric(
    "CBP Care Load",
    f"{latest['cbp_load']:,.0f}"
    if pd.notna(latest["cbp_load"]) else "N/A"
)

c3.metric(
    "HHS Care Load",
    f"{latest['hhs_load']:,.0f}"
    if pd.notna(latest["hhs_load"]) else "N/A"
)

c4.metric(
    "Net Intake",
    f"{latest['net_intake']:+,.0f}"
    if pd.notna(latest["net_intake"]) else "N/A"
)

st.caption(
    "KPI values refer to the last reporting date in the selected range."
)

# Aggregate for weekly or monthly views.
if granularity == "Weekly":
    plot_df = view.set_index("date").resample("W").mean(
        numeric_only=True
    ).reset_index()
elif granularity == "Monthly":
    plot_df = view.set_index("date").resample("MS").mean(
        numeric_only=True
    ).reset_index()
else:
    plot_df = view.copy()

# Load comparison
load_columns = {
    "CBP Load": "cbp_load",
    "HHS Load": "hhs_load",
    "Total Load": "total_load",
    "7-day Average": "avg_7day_load",
}

selected = [
    load_columns[name] for name in metric_choice
]

if selected:
    long_df = plot_df.melt(
        id_vars="date",
        value_vars=selected,
        var_name="Metric",
        value_name="Children"
    )

    fig = px.line(
        long_df,
        x="date",
        y="Children",
        color="Metric",
        title=f"Care Load Trends — {granularity}"
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Select at least one load metric.")

# Net intake
fig2 = px.bar(
    plot_df,
    x="date",
    y="net_intake",
    title="Net HHS Intake: Transfers Minus Discharges",
    labels={"net_intake": "Net Intake", "date": "Date"}
)
fig2.add_hline(y=0, line_dash="dash")
st.plotly_chart(fig2, use_container_width=True)

# Data quality and operational indicators
st.subheader("Data Quality & Pressure Indicators")

st.write({
    "Invalid dates removed": invalid_dates,
    "Duplicate dates found": duplicate_dates,
    "Days with missing metric values": missing_days,
    "Days in selected range": len(view),
})

pressure_days = int(view["backlog_pressure"].fillna(False).sum())
st.metric("Days with Positive 14-Day Average Net Intake", pressure_days)

st.caption(
    "This is a pressure indicator, not a verified facility-capacity "
    "or overcrowding measure. Missing values are not imputed."
)

st.subheader("Analyzed Data")
st.dataframe(view, use_container_width=True)

st.download_button(
    "Download analyzed data (CSV)",
    data=view.to_csv(index=False).encode("utf-8"),
    file_name="uac_analyzed_data.csv",
    mime="text/csv"
)