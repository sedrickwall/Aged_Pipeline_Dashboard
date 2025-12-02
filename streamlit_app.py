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
# UTIL FUNCTIONS
# ===========================================================
def load_data(uploaded_file):
    if uploaded_file is None:
        return None

    file_type = uploaded_file.name.split(".")[-1].lower()

    if file_type == "csv":
        df = pd.read_csv(uploaded_file)
    elif file_type in ["xlsx", "xls"]:
        df = pd.read_excel(uploaded_file)
    elif file_type == "json":
        df = pd.read_json(uploaded_file)
    else:
        st.error("Unsupported file type. Please upload CSV, Excel, or JSON.")
        return None

    return df


def validate_data(df: pd.DataFrame) -> bool:
    """Ensure all required columns are present."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        st.error(f"❌ Missing required columns: {missing}")
        return False
    return True


def compute_year_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate by year:
        - sums for amounts and counts
        - average for percent_active
    """
    agg = df.groupby("year").agg(
        total_aged_sum=("total_aged", "sum"),
        aged_amount_sum=("aged_amount", "sum"),
        active_amount_sum=("active_amount", "sum"),
        percent_active_avg=("percent_active", "mean"),
        hrec_exceeded_sum=("hrec_exceeded", "sum"),
        cro_sum=("cro", "sum"),
        direct_sum=("direct", "sum"),
        periods_count=("period", "nunique")
    ).reset_index()

    return agg


def compute_kpi_score(row, prev_row=None):
    """
    Simple KPI scoring logic:
        - percent_active_avg: higher is better
        - hrec_exceeded_sum: lower is better
        - aged_amount_sum: lower is better (if you want to de-age pipeline)

    Score range: 0–100, broken into Green / Amber / Red tiers.
    """
    score = 50  # base

    # Momentum via active %
    if row["percent_active_avg"] >= 75:
        score += 20
    elif row["percent_active_avg"] >= 65:
        score += 10
    else:
        score -= 5

    # HREC exceeded
    if row["hrec_exceeded_sum"] <= 30:
        score += 15
    elif row["hrec_exceeded_sum"] <= 60:
        score += 5
    else:
        score -= 10

    # Aged amount – relative change vs prior year if available
    if prev_row is not None:
        if prev_row["aged_amount_sum"] > 0:
            delta = (row["aged_amount_sum"] - prev_row["aged_amount_sum"]) / prev_row["aged_amount_sum"]
        else:
            delta = 0

        if delta <= -0.1:  # decreased by 10% or more
            score += 10
        elif delta <= 0.05:  # roughly flat or small increase
            score += 0
        else:  # significant increase in aged
            score -= 10

    # Clamp
    score = max(0, min(100, score))

    if score >= 80:
        tier = "Green"
    elif score >= 60:
        tier = "Amber"
    else:
        tier = "Red"

    return score, tier


def format_currency(value):
    return "${:,.0f}".format(value)


def format_pct(value):
    return "{:,.1f}%".format(value)


def generate_executive_summary(selected_year, cur_row, prev_row=None, kpi_score=None, kpi_tier=None):
    """
    Generate a short executive narrative based on YoY movements.
    """
    lines = []
    lines.append(f"Executive Summary — Aged Pipeline {selected_year}")
    lines.append("")

    # Basic metrics
    lines.append(
        f"- Total aged amount: {format_currency(cur_row['aged_amount_sum'])}"
    )
    lines.append(
        f"- Total active amount: {format_currency(cur_row['active_amount_sum'])}"
    )
    lines.append(
        f"- Average % active: {format_pct(cur_row['percent_active_avg'])}"
    )
    lines.append(
        f"- HREC exceeded count: {int(cur_row['hrec_exceeded_sum'])}"
    )
    lines.append(
        f"- CRO vs Direct exceeded: {int(cur_row['cro_sum'])} CRO / {int(cur_row['direct_sum'])} Direct"
    )
    lines.append("")

    if prev_row is not None:
        # YoY changes
        def delta_str(cur, prev, is_pct=False):
            if prev == 0:
                return "n/a"
            diff = cur - prev
            pct = diff / prev * 100
            if is_pct:
                return f"{pct:+.1f} pts"
            else:
                return f"{pct:+.1f}%"

        aged_delta = delta_str(cur_row["aged_amount_sum"], prev_row["aged_amount_sum"])
        active_delta = delta_str(cur_row["active_amount_sum"], prev_row["active_amount_sum"])
        pct_active_delta = delta_str(cur_row["percent_active_avg"], prev_row["percent_active_avg"], is_pct=True)
        hrec_delta = delta_str(cur_row["hrec_exceeded_sum"], prev_row["hrec_exceeded_sum"])

        lines.append(f"Year-over-Year vs {int(prev_row['year'])}:")
        lines.append(f"- Aged amount: {aged_delta} vs prior year.")
        lines.append(f"- Active amount: {active_delta} vs prior year.")
        lines.append(f"- Average % active: {pct_active_delta} vs prior year.")
        lines.append(f"- HREC exceeded: {hrec_delta} vs prior year.")
        lines.append("")

    if kpi_score is not None and kpi_tier is not None:
        lines.append(f"KPI Score: **{kpi_score}/100 ({kpi_tier})**")
        if kpi_tier == "Green":
            lines.append("- Overall pipeline health is strong with positive operational momentum.")
        elif kpi_tier == "Amber":
            lines.append("- Pipeline health is stable but there are areas that warrant attention, especially around aging or HREC thresholds.")
        else:
            lines.append("- Pipeline health is under pressure. Focused action on de-aging and HREC management is recommended.")

    lines.append("")
    lines.append("Strategic Focus Suggestions:")
    lines.append("- Reduce aged opportunities through targeted follow-up and prioritization.")
    lines.append("- Maintain or improve % active through disciplined pipeline hygiene.")
    lines.append("- Monitor HREC exceedances and address root causes with operations and BD teams.")
    lines.append("- Review CRO vs Direct distributions to ensure accountability and coverage.")

    return "\n".join(lines)


# ===========================================================
# SIDEBAR – DATA UPLOAD & YEAR SELECTION
# ===========================================================
st.sidebar.title("Data & Filters")

uploaded_file = st.sidebar.file_uploader(
    "Upload aged pipeline data (CSV, Excel, JSON)",
    type=["csv", "xlsx", "xls", "json"]
)

st.sidebar.markdown("**Required Columns:**")
st.sidebar.code(", ".join(REQUIRED_COLUMNS))

# ===========================================================
# MAIN CONTENT
# ===========================================================
st.title("📊 Aged Pipeline Dashboard — Executive View")

if uploaded_file is None:
    st.info("Upload your dataset in the sidebar to begin.")
    st.stop()

df_raw = load_data(uploaded_file)

# Automatically detect & convert your raw block format
df_raw = preprocess_raw_block_format(df_raw)
if df_raw is None:
    st.stop()

if not validate_data(df_raw):
    st.stop()

# Cast year to int (in case it's loaded as float/string)
df_raw["year"] = df_raw["year"].astype(int)

# Sort by year and optionally by a custom period order if you have one
df = df_raw.sort_values(["year", "period"]).reset_index(drop=True)

year_summary = compute_year_summary(df)

available_years = sorted(year_summary["year"].unique())
selected_year = st.sidebar.selectbox(
    "Select year to review",
    options=available_years,
    index=len(available_years) - 1  # default to latest
)

prev_year = None
prev_row = None
cur_row = year_summary[year_summary["year"] == selected_year].iloc[0]

if selected_year - 1 in available_years:
    prev_year = selected_year - 1
    prev_row = year_summary[year_summary["year"] == prev_year].iloc[0]

kpi_score, kpi_tier = compute_kpi_score(cur_row, prev_row)

# ===========================================================
# TOP METRIC CARDS
# ===========================================================
st.subheader(f"Key Metrics — {selected_year}")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Aged Amount",
        format_currency(cur_row["aged_amount_sum"]),
        delta=(
            format_currency(cur_row["aged_amount_sum"] - prev_row["aged_amount_sum"])
            if prev_row is not None else None
        )
    )

with col2:
    st.metric(
        "Total Active Amount",
        format_currency(cur_row["active_amount_sum"]),
        delta=(
            format_currency(cur_row["active_amount_sum"] - prev_row["active_amount_sum"])
            if prev_row is not None else None
        )
    )

with col3:
    st.metric(
        "Avg % Active",
        format_pct(cur_row["percent_active_avg"]),
        delta=(
            f"{cur_row['percent_active_avg'] - prev_row['percent_active_avg']:+.1f} pts"
            if prev_row is not None else None
        )
    )

with col4:
    st.metric(
        "HREC Exceeded (Total)",
        int(cur_row["hrec_exceeded_sum"]),
        delta=(
            int(cur_row["hrec_exceeded_sum"] - prev_row["hrec_exceeded_sum"])
            if prev_row is not None else None
        )
    )

st.markdown(f"**KPI Score for {selected_year}:** `{kpi_score}/100` — **{kpi_tier}**")

def preprocess_raw_block_format(df):
    """
    Detects and transforms the 'vertical block' format Sedrick uses
    into the long-format dataset required by the dashboard.
    """

    # If the dataframe already has required columns → skip
    if all(c in df.columns for c in REQUIRED_COLUMNS):
        return df  # no preprocessing needed

    # Map columns → periods
    period_map = {
        df.columns[0]: "Start of Year",
        df.columns[1]: "Mid-Year",
        df.columns[2]: "Last Week"
    }

    # Expected vertical structure
    metric_map = {
        "Total Aged": "total_aged",
        "Aged $": "aged_amount",
        "% Active": "percent_active",
        "% HREC Exceeded": "percent_hrec_exceeded",  # not needed but captured
        "HREC Exceeded Count": "hrec_exceeded",
        "CRO Count": "cro",
        "Direct Count": "direct",
        "% Direct": "percent_direct",
        "% CRO": "percent_cro"
    }

    # STEP 1 — Clean labels column
    labels = df.iloc[:, 0:1].rename(columns={df.columns[0]: "label"})
    labels = labels.fillna(method="ffill")

    # STEP 2 — Build clean long-format structure
    final_rows = []

    for col in df.columns:
        period_name = period_map[col]

        # Extract column as a series paired with labels
        for i in range(len(df)):
            label = str(labels.iloc[i, 0]).strip()
            value = df[col].iloc[i]

            # Only keep metrics we mapped
            if label in metric_map:
                metric_key = metric_map[label]

                final_rows.append({
                    "year": 2025,  # hardcode for now, or extract later
                    "period": period_name,
                    metric_key: value
                })

    # STEP 3 — Convert to DataFrame
    clean_df = pd.DataFrame(final_rows)

    # STEP 4 — Pivot data to wide format per period
    clean_df = clean_df.pivot_table(
        index=["year", "period"],
        columns=clean_df.groupby(["year", "period"]).cumcount(),
        aggfunc="first"
    )

    clean_df = clean_df.reset_index()

    # STEP 5 — Flatten duplicate columns (cleanup)
    clean_df = clean_df.loc[:, ~clean_df.columns.duplicated()]

    # STEP 6 — Final required columns (fill missing with 0)
    required_cols = [
        "year", "period",
        "total_aged",
        "aged_amount",
        "active_amount",        # not in raw file → set to 0
        "percent_active",
        "hrec_exceeded",
        "cro",
        "direct"
    ]

    for col in required_cols:
        if col not in clean_df.columns:
            clean_df[col] = 0

    return clean_df[required_cols]


# ===========================================================
# DETAILED CHARTS FOR SELECTED YEAR
# ===========================================================
st.markdown("---")
st.subheader(f"1️⃣ In-Year Trends — {selected_year}")

year_df = df[df["year"] == selected_year]

# Aged vs Active Amount by Period
fig_amount = go.Figure()
fig_amount.add_trace(go.Scatter(
    x=year_df["period"],
    y=year_df["aged_amount"],
    mode="lines+markers",
    name="Aged Amount",
    line=dict(color="red")
))
fig_amount.add_trace(go.Scatter(
    x=year_df["period"],
    y=year_df["active_amount"],
    mode="lines+markers",
    name="Active Amount",
    line=dict(color="green")
))
fig_amount.update_layout(
    title=f"Aged vs Active Amount — {selected_year}",
    xaxis_title="Period",
    yaxis_title="Amount (USD)",
    template=TEMPLATE
)

# % Active by Period
fig_pct = px.line(
    year_df,
    x="period",
    y="percent_active",
    markers=True,
    title=f"% Active by Period — {selected_year}",
    template=TEMPLATE
)

# HREC Exceeded by Period
fig_hrec = px.bar(
    year_df,
    x="period",
    y="hrec_exceeded",
    text="hrec_exceeded",
    title=f"HREC Exceeded by Period — {selected_year}",
    template=TEMPLATE
)
fig_hrec.update_traces(textposition="outside")

c1, c2 = st.columns(2)
with c1:
    st.plotly_chart(fig_amount, use_container_width=True)
with c2:
    st.plotly_chart(fig_pct, use_container_width=True)

st.plotly_chart(fig_hrec, use_container_width=True)

# CRO vs Direct — stacked bar
st.subheader(f"2️⃣ CRO vs Direct (Exceeded) — {selected_year}")

fig_cro_direct = go.Figure()
fig_cro_direct.add_trace(go.Bar(
    x=year_df["period"],
    y=year_df["cro"],
    name="CRO"
))
fig_cro_direct.add_trace(go.Bar(
    x=year_df["period"],
    y=year_df["direct"],
    name="Direct"
))
fig_cro_direct.update_layout(
    barmode="stack",
    title=f"CRO vs Direct Exceeded — {selected_year}",
    xaxis_title="Period",
    yaxis_title="Count",
    template=TEMPLATE
)

st.plotly_chart(fig_cro_direct, use_container_width=True)

# ===========================================================
# MULTI-YEAR COMPARISON
# ===========================================================
st.markdown("---")
st.subheader("3️⃣ Multi-Year Comparison")

# Aged vs Active Amount (Year Aggregates)
fig_year_amounts = go.Figure()
fig_year_amounts.add_trace(go.Bar(
    x=year_summary["year"],
    y=year_summary["aged_amount_sum"],
    name="Aged Amount"
))
fig_year_amounts.add_trace(go.Bar(
    x=year_summary["year"],
    y=year_summary["active_amount_sum"],
    name="Active Amount"
))
fig_year_amounts.update_layout(
    barmode="group",
    title="Aged vs Active Amount by Year",
    xaxis_title="Year",
    yaxis_title="Amount (USD)",
    template=TEMPLATE
)

# Avg % Active by Year
fig_year_pct = px.line(
    year_summary,
    x="year",
    y="percent_active_avg",
    markers=True,
    title="Average % Active by Year",
    template=TEMPLATE
)

c3, c4 = st.columns(2)
with c3:
    st.plotly_chart(fig_year_amounts, use_container_width=True)
with c4:
    st.plotly_chart(fig_year_pct, use_container_width=True)

# HREC Exceeded by Year
fig_year_hrec = px.bar(
    year_summary,
    x="year",
    y="hrec_exceeded_sum",
    text="hrec_exceeded_sum",
    title="HREC Exceeded by Year",
    template=TEMPLATE
)
fig_year_hrec.update_traces(textposition="outside")
st.plotly_chart(fig_year_hrec, use_container_width=True)

# ===========================================================
# EXECUTIVE SUMMARY + DOWNLOAD
# ===========================================================
st.markdown("---")
st.subheader("4️⃣ Executive Summary & Download")

exec_summary_text = generate_executive_summary(
    selected_year=selected_year,
    cur_row=cur_row,
    prev_row=prev_row,
    kpi_score=kpi_score,
    kpi_tier=kpi_tier
)

st.text_area(
    "Executive Summary (editable before sharing):",
    value=exec_summary_text,
    height=300
)

# Download as text file (you can save as PDF via browser print dialog)
buffer = StringIO()
buffer.write(exec_summary_text)
buffer.seek(0)

st.download_button(
    label="Download Executive Summary (.txt)",
    data=buffer.getvalue(),
    file_name=f"aged_pipeline_executive_summary_{selected_year}.txt",
    mime="text/plain"
)

st.caption("Tip: Open the summary, format in Word/Google Docs/PowerPoint, and export as PDF for your CEO & board review.")
