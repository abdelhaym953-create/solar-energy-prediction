import numpy as np
import pandas as pd
import streamlit as st

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Solar Prediction · Overview",
    layout="wide",
    page_icon="🌤️",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1e2130, #252a3d);
        border: 1px solid #2e3450;
        border-radius: 14px;
        padding: 22px 18px;
        text-align: center;
        height: 100%;
    }
    .metric-card .label {
        font-size: 12px;
        color: #8b92a5;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin-bottom: 8px;
    }
    .metric-card .value {
        font-size: 30px;
        font-weight: 700;
        color: #f0c040;
        line-height: 1.1;
    }
    .metric-card .sub {
        font-size: 12px;
        color: #5a6070;
        margin-top: 6px;
    }
    .section-title {
        font-size: 18px;
        font-weight: 700;
        color: #e0e4f0;
        margin: 36px 0 14px 0;
        padding-bottom: 8px;
        border-bottom: 2px solid #f0c04025;
        letter-spacing: 0.02em;
    }
    .summary-box {
        background: #1a1f2e;
        border: 1px solid #2e3450;
        border-radius: 12px;
        padding: 22px 26px;
        color: #b0b8cc;
        font-size: 15px;
        line-height: 1.8;
    }
    .summary-box b { color: #e0e4f0; }
    .summary-box span.highlight { color: #f0c040; font-weight: 600; }
    .col-card {
        background: #1a1f2e;
        border: 1px solid #2e3450;
        border-radius: 10px;
        padding: 14px 16px;
        margin-bottom: 10px;
    }
    .col-card .col-name {
        font-size: 13px;
        font-weight: 700;
        color: #f0c040;
        margin-bottom: 4px;
        font-family: monospace;
    }
    .col-card .col-type {
        font-size: 11px;
        color: #4fc3f7;
        margin-bottom: 6px;
    }
    .col-card .col-desc {
        font-size: 13px;
        color: #8b92a5;
        line-height: 1.5;
    }
    .tag {
        display: inline-block;
        border-radius: 20px;
        padding: 2px 10px;
        font-size: 11px;
        font-weight: 600;
        margin-left: 6px;
    }
    .tag-target  { background: #f0c04022; color: #f0c040; }
    .tag-weather { background: #4fc3f722; color: #4fc3f7; }
    .tag-time    { background: #81c78422; color: #81c784; }
</style>
""", unsafe_allow_html=True)


# ── Load Data ──────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    raw   = pd.read_csv("SolarPrediction.csv")
    clean = pd.read_csv("Cleaned_Data.csv")
    return raw, clean

raw_df, clean_df = load_data()


# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("## 🌤️ Solar Radiation Prediction")
st.markdown(
    "<span style='color:#8b92a5;font-size:15px'>"
    "Project Overview · Dataset Summary · Key Metrics"
    "</span>",
    unsafe_allow_html=True,
)
st.divider()


# ── 1. Project Summary ─────────────────────────────────────────────────────────
st.markdown('<div class="section-title">📄 What is this dataset?</div>', unsafe_allow_html=True)

st.markdown("""
<div class="summary-box">
    This dataset contains <b>32,686 solar radiation readings</b> collected in
    <span class="highlight">Hawaii, USA</span> between
    <span class="highlight">September and December 2016</span>,
    with measurements taken approximately every <b>5 minutes</b> throughout the day.<br><br>
    The goal is to <b>predict solar radiation (W/m²)</b> using weather and time-based features
    such as temperature, humidity, pressure, wind direction, and the position of the sun
    relative to sunrise and sunset.<br><br>
    The raw data had <b>11 columns</b> including timestamps and raw time strings.
    After cleaning and feature engineering, it was reduced to
    <span class="highlight">8 ready-to-model features</span> —
    with no missing values and no duplicates.
    The best performing model was <b>LightGBM</b>, chosen after comparing 7 regression algorithms.
</div>
""", unsafe_allow_html=True)


# ── 2. KPI Cards ───────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">📊 Key Numbers</div>', unsafe_allow_html=True)

kpis = [
    ("Total Records",      f"{len(clean_df):,}",                        "after cleaning"),
    ("Original Features",  "11",                                         "raw columns"),
    ("Model Features",     "7",                                          "after engineering"),
    ("Missing Values",     "0",                                          "fully clean ✅"),
    ("Avg Radiation",      f"{clean_df['radiation'].mean():.1f} W/m²",  "overall mean"),
    ("Peak Radiation",     f"{clean_df['radiation'].max():.0f} W/m²",   "max recorded"),
    ("Night Readings",     "54%",                                        "radiation < 10 W/m²"),
    ("Best Model",         "LightGBM",                                   "highest R² score"),
]

cols = st.columns(4)
for i, (label, value, sub) in enumerate(kpis):
    cols[i % 4].markdown(
        f"""<div class="metric-card">
            <div class="label">{label}</div>
            <div class="value">{value}</div>
            <div class="sub">{sub}</div>
        </div><br>""",
        unsafe_allow_html=True,
    )


# ── 3. Cleaned Data Preview ────────────────────────────────────────────────────
st.markdown('<div class="section-title">🗂️ Cleaned Data Preview</div>', unsafe_allow_html=True)

n = st.slider("Number of rows to display", min_value=10, max_value=200, value=50, step=10)

st.dataframe(
    clean_df.head(n).style
        .format(precision=3)
        .background_gradient(subset=["radiation"], cmap="YlOrRd")
        .background_gradient(subset=["temperature"], cmap="Blues"),
    use_container_width=True,
    height=380,
)

st.caption(f"Showing {n} of {len(clean_df):,} rows · {clean_df.shape[1]} columns")


# ── 4. Column Descriptions ─────────────────────────────────────────────────────
st.markdown('<div class="section-title">📖 Column Descriptions</div>', unsafe_allow_html=True)

columns_info = [
    (
        "radiation", "float64", "Target", "tag-target",
        "Solar radiation measured in Watts per square meter (W/m²). "
        "This is the target variable the model predicts. Values range from ~1 (nighttime) to 1,601 (peak sun)."
    ),
    (
        "temperature", "int64", "Weather", "tag-weather",
        "Ambient air temperature in degrees Fahrenheit (°F). "
        "The strongest predictor of radiation with a correlation of +0.735."
    ),
    (
        "pressure", "float64", "Weather", "tag-weather",
        "Atmospheric pressure in inches of mercury (inHg). "
        "Has a mild positive correlation (+0.12) with radiation."
    ),
    (
        "humidity", "int64", "Weather", "tag-weather",
        "Relative humidity as a percentage (0–100%). "
        "Higher humidity typically means more cloud cover, reducing radiation (corr: -0.23)."
    ),
    (
        "winddirection(degrees)", "float64", "Weather", "tag-weather",
        "Wind direction in degrees (0–360°), where 0/360 = North, 90 = East. "
        "Negatively correlated with radiation (-0.23), indicating wind patterns affect cloud formation."
    ),
    (
        "hour", "int64", "Time", "tag-time",
        "Hour of the day extracted from the original timestamp (0–23). "
        "Radiation is near zero at night and peaks between 11 AM and 1 PM."
    ),
    (
        "minutes_since_sunrise", "float64", "Time", "tag-time",
        "Number of minutes elapsed since sunrise at the time of each reading. "
        "Engineered feature that captures the sun's position relative to dawn. Negative = before sunrise."
    ),
    (
        "minutes_to_sunset", "float64", "Time", "tag-time",
        "Number of minutes remaining until sunset at the time of each reading. "
        "Engineered feature that captures how close the reading is to dusk. Negative = after sunset."
    ),
]

col1, col2 = st.columns(2)
for i, (name, dtype, tag_label, tag_class, desc) in enumerate(columns_info):
    col = col1 if i % 2 == 0 else col2
    col.markdown(
        f"""<div class="col-card">
            <div class="col-name">
                {name}
                <span class="tag {tag_class}">{tag_label}</span>
            </div>
            <div class="col-type">dtype: {dtype}</div>
            <div class="col-desc">{desc}</div>
        </div>""",
        unsafe_allow_html=True,
    )


st.divider()
st.caption("🌤️ Solar Radiation Prediction · Overview Page")
