import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO

# ===========================================================
# PAGE CONFIG
# ===========================================================
st.set_page_config(
    page_title="Aged Pipeline Executive Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

TEMPLATE = "plotly_white"

# ===========================================================
# REQUIRED COLUMNS FOR THE FINAL CLEAN DATA
# ===========================================================
REQUIRED_COLUMNS = [
    "year",
    "period",
    "total_aged",
    "aged_amount",
    "active_amount",
    "percent_active",
    "hrec_exceeded",
    "cro",
    "direct"
]

# ===========================================================
# 1. AUTO-PREPROCESSOR FOR RAW BLOCK CSV FORMAT
# ===========================================================
def preprocess_raw_block_format(df):
    """
    Automatically transforms the raw 3-block vertical CSV
    into the long-format structure the dashboard requires.
    
    If the data is already clean, it returns it unchanged.
    """

    # If already correct format, do nothing
    if all(c in df.columns for c in REQUIRED_COLUMNS):
        return df

    # Map raw columns → periods
    if len(df.columns) < 3:
        st.error("Uploaded file does not have 3 block columns.")
        return df

    period_map = {
        df.columns[0]: "Start of Year",
        df.columns[1]: "Mid-Year",
        df.columns[2]: "Last Week"
    }

    # Raw metrics → final column names
    metric_map = {
        "Total Aged": "total_aged",
        "Aged $$": "aged_amount",
        "Aged $": "aged_amount",
        "% Active": "percent_active",
        "% HREC Exceeded": "percent_hrec_exceeded",
        "HREC Exceeded Count": "hrec_exceeded",
        "CRO Count": "cro",
        "Direct Count": "direct",
        "% Direct": "percent_direct",
        "% CRO": "percent_cro"
    }

    # Fill labels by forward filling the first column
    labels = df.iloc[:, 0].fillna(method="ffill")

    final_data = {}

    # Initialize structure
    for period in period_map.values():
        final_data[period] = {
            "year": 2025,   # Hardcode for now, can make dynamic later
            "period": period,
            "total_aged": 0,
            "aged_amount": 0,
            "active_amount": 0,
            "percent_active": 0,
            "hrec_exceeded": 0,
            "cro": 0,
            "direct": 0
        }

    # Parse each block
    for col in df.columns:
        period_name = period_map[col]

        for i in range(len(df)):
            label = str(labels.iloc[i]).strip()
            raw_value = df[col].iloc[i]

            # Skip empty and unmapped labels
            if label not in metric_map:
                continue

            mapped_field = metric_map[label]

            # Save mapped fields into period
            final_data[period_name][mapped_field] = raw_value

    # Convert dict to dataframe
    clean_df = pd.DataFrame(final_data.values())

    # Ensure required columns exist
    for col in REQUIRED_COLUMNS:
        if col not in clean_df.columns:
            clean_df[col] = 0

    return clean_df[REQUIRED_COLUMNS]


# ===========================================================
# 2. SIMPLE FILE LOADER
# ===========================================================
def load_data(upload):
    if upload is None:
        return None

    ext = upload.name.split(".")[-1].lower()

    if ext == "csv":
        return pd.read_csv(upload)
    elif ext in ["xlsx", "xls"]:
        return pd.read_excel(upload)
    elif ext == "json":
        return pd.read_json(upload)
    else:
        st.error("Unsupported file type. Please upload CSV, Excel, or JSON.")
        return None


# ===========================================================
# 3. VALIDATION
# ===========================================================
def validate_data(df):
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        st.error(f"Missing required columns: {missing}")
        return False
    return True


# ===========================================================
# 4. KPI + YEAR SUMMARY FUNCTIONS
# ===========================================================
def compute_year_summary(df):
    return df.groupby("year").agg(
        total_aged_sum=("total_aged", "sum"),
        aged_amount_sum=("aged_amount", "sum"),
        active_amount_sum=("active_amount", "sum"),
        percent_active_avg=("percent_active", "mean"),
        hrec_exceeded_sum=("hrec_exceeded", "sum"),
        cro_sum=("cro", "sum"),
        direct_sum=("direct", "sum")
    ).reset_index()


def compute_kpi_score(cur, prev=None):
    score = 50  # baseline

    # % active
    if cur["percent_active_avg"] >= 75:
        score += 20
    elif cur["percent_active_avg"] >= 65:
        score += 10
    else:
        score -= 5

    # HREC exceeded
    if cur["hrec_exceeded_sum"] <= 30:
        score += 15
    else:
        score -= 10

    # Aged amount YoY
    if prev is not None:
        if prev["aged_amount_sum"] > 0:
            change = (cur["aged_amount_sum"] - prev["aged_amount_sum"]) / prev["aged_amount_sum"]
            if change <= -0.1:
                score += 10
            elif change > 0.1:
                score -= 10

    # Bound score
    score = max(0, min(100, score))

    tier = "Green" if score >= 80 else "Amber" if score >= 60 else "Red"
    return score, tier


# ===========================================================
# 5. EXECUTIVE SUMMARY GENERATION
# ===========================================================
def exec_summary(year, cur, prev, score, tier):
    lines = []

    lines.append(f"Executive Summary — Aged Pipeline {year}")
    lines.append("")

    lines.append(f"- Total Aged Amount: ${cur['aged_amount_sum']:,.0f}")
    lines.append(f"- Total Active Amount: ${cur['active_amount_sum']:,.0f}")
    lines.append(f"- Avg % Active: {cur['percent_active_avg']:.1f}%")
    lines.append(f"- HREC Exceeded: {int(cur['hrec_exceeded_sum'])}")
    lines.append(f"- CRO Exceeded: {int(cur['cro_sum'])}")
    lines.append(f"- Direct Exceeded: {int(cur['direct_sum'])}")
    lines.append("")

    if prev is not None:
        lines.append("Year-over-Year Changes:")
        lines.append(f"- Aged Amount YoY Change: {((cur['aged_amount_sum']-prev['aged_amount_sum'])/prev['aged_amount_sum']*100):+.1f}%")
        lines.append(f"- Active Amount YoY Change: {((cur['active_amount_sum']-prev['active_amount_sum'])/prev['active_amount_sum']*100):+.1f}%")
        lines.append(f"- % Active YoY Change: {(cur['percent_active_avg']-prev['percent_active_avg']):+.1f} pts")
        lines.append(f"- HREC YoY Change: {(cur['hrec_exceeded_sum']-prev['hrec_exceeded_sum']):+.0f}")
        lines.append("")

    lines.append(f"KPI Score: {score}/100 ({tier})")

    return "\n".join(lines)


# ===========================================================
# 6. STREAMLIT UI START
# ===========================================================
st.title("📊 Aged Pipeline Dashboard — Executive View")

uploaded = st.sidebar.file_uploader(
    "Upload CSV / Excel / JSON",
    type=["csv", "xlsx", "xls", "json"]
)

if uploaded is None:
    st.info("Upload a pipeline file to begin.")
    st.stop()

# Load file
df_raw = load_data(uploaded)

# Auto-transform raw CSV block format
df = preprocess_raw_block_format(df_raw)

# Validate after preprocess
if not validate_data(df):
    st.stop()

# Display clean data
st.subheader("Cleaned Dataset")
st.dataframe(df)

# ===========================================================
# CALC YEAR SUMMARY
# ===========================================================
summary = compute_year_summary(df)

selected_year = st.sidebar.selectbox("Select Year:", summary["year"].unique())
cur = summary[summary["year"] == selected_year].iloc[0]

prev = None
if selected_year - 1 in summary["year"].unique():
    prev = summary[summary["year"] == selected_year - 1].iloc[0]

score, tier = compute_kpi_score(cur, prev)

# ===========================================================
# KPI METRIC CARDS
# ===========================================================
st.subheader(f"Key Metrics — {selected_year}")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Aged Amount", f"${cur['aged_amount_sum']:,.0f}")
c2.metric("Active Amount", f"${cur['active_amount_sum']:,.0f}")
c3.metric("Avg % Active", f"{cur['percent_active_avg']:.1f}%")
c4.metric("HREC Exceeded", int(cur['hrec_exceeded_sum']))

st.markdown(f"**KPI Score:** {score}/100 — **{tier}**")

# ===========================================================
# EXEC SUMMARY
# ===========================================================
st.subheader("Executive Summary")

summary_text = exec_summary(selected_year, cur, prev, score, tier)

st.text_area("Summary (editable):", value=summary_text, height=300)

buffer = StringIO()
buffer.write(summary_text)
buffer.seek(0)

st.download_button(
    label="Download Summary (.txt)",
    data=buffer.getvalue(),
    file_name=f"pipeline_summary_{selected_year}.txt",
    mime="text/plain"
)
