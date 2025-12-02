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

    # NEW fields
    "percent_hrec_exceeded",
    "percent_hrec_cro",
    "percent_hrec_direct",

    "hrec_exceeded",
    "cro",
    "direct"
]


# ===========================================================
# 1. AUTO-PREPROCESSOR FOR RAW BLOCK CSV FORMAT
# ===========================================================
def preprocess_raw_block_format(df):
    """
    Converts Sedrick's 3-block aged pipeline export into standard long-format rows
    including all percent-based HREC metrics.
    """

    # If already clean → return
    if all(col in df.columns for col in REQUIRED_COLUMNS):
        return df

    # Mapping of raw labels → final metric names
    metric_map = {
        "Total Aged": "total_aged",
        "Total Aged Amount": "aged_amount",
        "Total Active": "active_amount",
        "% Active": "percent_active",

        # HREC COUNT metrics
        "Total HREC Exceeded": "hrec_exceeded",
        "Total HREC Exceeded that are CRO": "cro",
        "Total of HREC Exceeded that are Direct": "direct",

        # NEW requested percent metrics
        "% HREC Exceeded": "percent_hrec_exceeded",
        "% of HREC Exceeded that are CRO": "percent_hrec_cro",
        "% of HREC Exceeded that are Direct": "percent_hrec_direct",
    }

    # Map block columns → periods
    period_map = {
        df.columns[0]: "Start of Year",
        df.columns[1]: "Mid-Year",
        df.columns[2]: "Last Week"
    }

    # Clean labels column
    labels = df.iloc[:, 0].fillna("").astype(str).str.strip()

    output = []

    # Loop through each period block
    for col in df.columns:
        period = period_map[col]

        # Initialize row with all metrics
        row_data = {
            "year": 2025,
            "period": period,
            "total_aged": 0,
            "aged_amount": 0,
            "active_amount": 0,
            "percent_active": 0,
            "percent_hrec_exceeded": 0,
            "percent_hrec_cro": 0,
            "percent_hrec_direct": 0,
            "hrec_exceeded": 0,
            "cro": 0,
            "direct": 0,
        }

        # Loop through rows inside this block
        for i in range(len(df)):
            raw_label = labels.iloc[i]
            raw_value = df[col].iloc[i]

            # Skip empty and invalid entries
            if raw_value in ["", None, "#REF!"]:
                continue

            # Only match mapped labels
            if raw_label in metric_map:
                mapped_field = metric_map[raw_label]
                row_data[mapped_field] = raw_value

        output.append(row_data)

    # Build DataFrame
    clean = pd.DataFrame(output)

    # Clean all numeric fields
    def clean_num(x):
        if isinstance(x, str):
            x = (
                x.replace("$", "")
                 .replace(",", "")
                 .replace("%", "")
                 .strip()
            )
        return pd.to_numeric(x, errors="coerce")

    numeric_cols = [
        "total_aged",
        "aged_amount",
        "active_amount",
        "percent_active",
        "percent_hrec_exceeded",
        "percent_hrec_cro",
        "percent_hrec_direct",
        "hrec_exceeded",
        "cro",
        "direct"
    ]

    for col in numeric_cols:
        clean[col] = clean[col].apply(clean_num).fillna(0)

    return clean



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
# ===========================================================
# FORCE CLEAN ALL NUMERIC COLUMNS
# ===========================================================
numeric_cols = [
    "total_aged",
    "aged_amount",
    "active_amount",
    "percent_active",
    "hrec_exceeded",
    "cro",
    "direct"
]

def clean_numeric(x):
    if isinstance(x, str):
        x = (
            x.replace("$", "")
             .replace(",", "")
             .replace("%", "")
             .strip()
        )
    return pd.to_numeric(x, errors="coerce")

for col in numeric_cols:
    if col in df.columns:
        df[col] = df[col].apply(clean_numeric)

# Fill missing numeric values with 0 for safety
df[numeric_cols] = df[numeric_cols].fillna(0)


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
