import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO

# ================================================
# PAGE CONFIG
# ================================================
st.set_page_config(
    page_title="Aged Pipeline Executive Dashboard",
    layout="wide"
)

st.markdown("""
    <style>
        .metric-card {
            background-color: #F5F7FA;
            color: #1558B0;
            padding: 20px;
            border-radius: 12px;
            font-size: 20px;
            text-align: center;
            border: 1px solid #DCE3EC;
        }

        .metric-card .value {
            font-size: 32px;
            font-weight: 700;
            color: #0F9D58; /* KPI Green */
        }

        .header {
            font-size: 28px;
            font-weight: 700;
            color: #1A73E8; /* Primary Blue */
            margin-top: 30px;
            margin-bottom: -5px;
        }

        .divider {
            height: 3px;
            background: #1A73E8;
            margin: 10px 0 25px 0;
            border-radius: 2px;
        }

        .stMetric > div {
            background-color: #F5F7FA !important;
            padding: 20px;
            border-radius: 12px;
            border: 1px solid #DCE3EC;
        }

        .stMetric label {
            color: #1558B0 !important;
        }

        .stMetric div[data-testid="stMetricValue"] {
            color: #0F9D58 !important;
            font-weight: bold !important;
        }
    </style>
""", unsafe_allow_html=True)





# ================================================
# REQUIRED COLS
# ================================================
REQUIRED_COLUMNS = [
    "year", "period",
    "total_aged", "aged_amount", "active_amount",
    "percent_active",
    "percent_hrec_exceeded", "percent_hrec_cro", "percent_hrec_direct",
    "hrec_exceeded", "cro", "direct"
]


# ================================================
# PREPROCESS — your exact CSV block structure
# ================================================
def preprocess_raw_block_format(df):
    metric_map = {
        "Total Aged": "total_aged",
        "Total Aged Amount": "aged_amount",
        "Total Active": "active_amount",
        "% Active": "percent_active",
        "Total HREC Exceeded": "hrec_exceeded",
        "Total HREC Exceeded that are CRO": "cro",
        "Total of HREC Exceeded that are Direct": "direct",
        "% HREC Exceeded": "percent_hrec_exceeded",
        "% of HREC Exceeded that are CRO": "percent_hrec_cro",
        "% of HREC Exceeded that are Direct": "percent_hrec_direct",
    }

    period_map = {
        df.columns[0]: "Beginning of Year",
        df.columns[1]: "Mid-Year",
        df.columns[2]: "To Date",
    }

    rows = []

    for col in df.columns:
        period = period_map[col]
        values = df[col].fillna("").astype(str).str.strip().tolist()

        row = {
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

        for i in range(len(values)):
            label = values[i]
            if label in metric_map:
                value = values[i + 1] if i + 1 < len(values) else ""
                row[metric_map[label]] = value

        rows.append(row)

    clean = pd.DataFrame(rows)

    numeric_cols = [
        "total_aged", "aged_amount", "active_amount",
        "percent_active",
        "percent_hrec_exceeded", "percent_hrec_cro", "percent_hrec_direct",
        "hrec_exceeded", "cro", "direct"
    ]

    def clean_num(x):
        if isinstance(x, str):
            x = x.replace("$", "").replace("%", "").replace(",", "").strip()
        return pd.to_numeric(x, errors="coerce")

    for col in numeric_cols:
        clean[col] = clean[col].apply(clean_num).fillna(0)

    return clean


# ================================================
# LOAD DATA
# ================================================
def load_data(upload):
    if upload is None:
        return None
    return pd.read_csv(upload, header=None)


# ================================================
# EXEC SUMMARY (BOARD VERSION)
# ================================================
def executive_summary(cur):
    aged = cur["aged_amount"]
    active = cur["active_amount"]
    perc_active = cur["percent_active"]
    hrec = cur["hrec_exceeded"]

    lines = []
    lines.append(f"Executive Summary — {cur['period']}")
    lines.append("")
    lines.append(f"• The aged pipeline currently stands at **${aged:,.0f}**, with active opportunities totaling **${active:,.0f}**.")
    lines.append(f"• The active coverage rate is **{perc_active:.1f}%**, indicating operational capacity and follow-up consistency.")
    lines.append(f"• There are **{hrec} HREC-exceeded opportunities**, representing compliance and follow-up risks.")
    lines.append("")
    lines.append("HREC Breakdown:")
    lines.append(f"• CRO: {cur['cro']} ({cur['percent_hrec_cro']:.1f}%)")
    lines.append(f"• Direct: {cur['direct']} ({cur['percent_hrec_direct']:.1f}%)")
    lines.append("")
    lines.append("Performance Notes:")
    if perc_active < 60:
        lines.append("⚠️ **Active % is below expectations** — potential follow-up slowdowns or prioritization issues.")
    if hrec > 75:
        lines.append("⚠️ **High HREC count** — consider escalation or targeted cleanup.")
    if aged > active:
        lines.append("ℹ️ Aged is significantly higher than active — monitor conversion & follow-up cadence.")
    lines.append("")
    lines.append("Overall Recommendation:")
    lines.append("Focus on reducing HREC exceedance, improving follow-up cadence, and increasing % Active.")

    return "\n".join(lines)


# ================================================
# STREAMLIT UI
# ================================================
st.title("📊 Aged Pipeline Dashboard — Executive View")

uploaded = st.sidebar.file_uploader("Upload Aged Pipeline CSV", type=["csv"])

if uploaded is None:
    st.info("Upload the CSV to begin.")
    st.stop()

df_raw = load_data(uploaded)
df = preprocess_raw_block_format(df_raw)

st.subheader("Cleaned Dataset")
st.dataframe(df)

# ================================================
# SELECT PERIOD
# ================================================
selected_period = st.sidebar.selectbox("Select Period:", df["period"].unique())
cur = df[df["period"] == selected_period].iloc[0]

# ================================================
# KPI CARDS
# ================================================
st.markdown("<div class='header'>Key Metrics</div><div class='divider'></div>", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

col1.metric("Aged Amount", f"${cur['aged_amount']:,.0f}")
col2.metric("Active Amount", f"${cur['active_amount']:,.0f}")
col3.metric("% Active", f"{cur['percent_active']:.1f}%")
col4.metric("HREC Exceeded", int(cur["hrec_exceeded"]))


# ================================================
# CHART: Aged vs Active
# ================================================
st.markdown("<div class='header'>Aged vs Active</div><div class='divider'></div>", unsafe_allow_html=True)

fig_bar = px.bar(
    x=["Aged Amount", "Active Amount"],
    y=[cur["aged_amount"], cur["active_amount"]],
    text_auto=True,
    title=f"Aged vs Active — {selected_period}"
)
st.plotly_chart(fig_bar, use_container_width=True)


# ================================================
# CHART: CRO vs Direct Pie
# ================================================
st.markdown("<div class='header'>CRO vs Direct Breakdown</div><div class='divider'></div>", unsafe_allow_html=True)

fig_pie = px.pie(
    values=[cur["cro"], cur["direct"]],
    names=["CRO", "Direct"],
    title=f"CRO vs Direct — {selected_period}"
)
st.plotly_chart(fig_pie, use_container_width=True)


# ================================================
# TREND CHARTS (across periods)
# ================================================
st.markdown("<div class='header'>Trends Across Periods</div><div class='divider'></div>", unsafe_allow_html=True)

trend_cols = st.columns(2)

with trend_cols[0]:
    fig = px.line(df, x="period", y="aged_amount", title="Aged Amount Trend")
    st.plotly_chart(fig, use_container_width=True)

with trend_cols[1]:
    fig = px.line(df, x="period", y="active_amount", title="Active Amount Trend")
    st.plotly_chart(fig, use_container_width=True)

trend_cols2 = st.columns(2)

with trend_cols2[0]:
    fig = px.line(df, x="period", y="percent_active", title="% Active Trend")
    st.plotly_chart(fig, use_container_width=True)

with trend_cols2[1]:
    fig = px.line(df, x="period", y="hrec_exceeded", title="HREC Exceeded Trend")
    st.plotly_chart(fig, use_container_width=True)


# ================================================
# EXECUTIVE SUMMARY
# ================================================
st.markdown("<div class='header'>Executive Summary</div><div class='divider'></div>", unsafe_allow_html=True)

summary_text = executive_summary(cur)
st.text_area("", value=summary_text, height=300)

buffer = StringIO()
buffer.write(summary_text)
buffer.seek(0)

st.download_button(
    label="Download Summary",
    data=summary_text,
    file_name=f"{selected_period}_summary.txt"
)
