import numpy as np
import pandas as pd
import plotly.graph_objects as go
import joblib
import streamlit as st
from datetime import time as dtime

st.set_page_config(page_title="Solar · Prediction", layout="wide", page_icon="⚡")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, .stApp { background-color: #0d1117 !important; font-family: 'Inter', sans-serif; }

label[data-testid="stWidgetLabel"] p { color: #8b949e !important; font-size: 13px !important; }
.stSlider > div > div > div { background: #21262d !important; }
.stNumberInput input { background: #161b22 !important; color: #e6edf3 !important; border: 1px solid #30363d !important; border-radius: 8px !important; }
div[data-testid="stTimeInput"] input { background: #161b22 !important; color: #e6edf3 !important; border: 1px solid #30363d !important; border-radius: 8px !important; }
div[data-testid="stDivider"] hr { border-color: #21262d; }

.input-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 14px;
    padding: 22px 22px 14px;
    margin-bottom: 14px;
}
.input-card-title {
    font-size: 11px; font-weight: 700; color: #58a6ff;
    letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 16px;
}

.gauge-wrap {
    background: #161b22; border: 1px solid #30363d;
    border-radius: 16px; padding: 28px 24px 20px; text-align: center;
}
.gauge-label { font-size: 12px; color: #8b949e; letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 4px; }
.gauge-value { font-size: 56px; font-weight: 800; line-height: 1; margin: 8px 0 4px; }
.gauge-unit  { font-size: 16px; color: #8b949e; }
.gauge-tag   { display: inline-block; margin-top: 12px; padding: 4px 16px; border-radius: 20px; font-size: 13px; font-weight: 600; }

.calc-chip {
    background: #21262d; border: 1px solid #30363d; border-radius: 8px;
    padding: 8px 14px; font-size: 12px; color: #8b949e;
    display: flex; justify-content: space-between; margin-bottom: 6px;
}
.calc-chip span { color: #58a6ff; font-weight: 600; }

.history-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 10px 16px; margin-bottom: 6px;
    background: #161b22; border: 1px solid #21262d; border-radius: 8px;
    font-size: 13px; color: #8b949e;
}
.history-row .val { font-weight: 700; color: #58a6ff; font-size: 14px; }

div[data-testid="stButton"] > button {
    width: 100%; background: linear-gradient(135deg, #1f6feb, #388bfd) !important;
    color: white !important; border: none !important; border-radius: 10px !important;
    padding: 14px !important; font-size: 16px !important; font-weight: 700 !important;
    cursor: pointer; margin-top: 8px;
}
div[data-testid="stButton"] > button:hover { opacity: 0.88 !important; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────────────────────
def time_to_minutes(t):
    return t.hour * 60 + t.minute + t.second / 60


# ── Load ───────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("SolarPrediction_model.pkl")

@st.cache_data
def load_data():
    return pd.read_csv("Cleaned_Data.csv")

try:
    model = load_model()
    model_loaded = True
except:
    model_loaded = False

df = load_data()

if "history" not in st.session_state:
    st.session_state.history = []


# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("## ⚡ Prediction")
st.markdown("<span style='color:#8b949e;font-size:14px'>Enter weather conditions and time — the model does the rest.</span>", unsafe_allow_html=True)
st.divider()

if not model_loaded:
    st.error("⚠️ Could not find `SolarPrediction_model.pkl` — make sure it's in the same folder.")
    st.stop()


col_in, col_out = st.columns([1.1, 1], gap="large")

# ══════════════════════════════════════════════════════════════════════════════
# LEFT — Inputs
# ══════════════════════════════════════════════════════════════════════════════
with col_in:

    # ── Weather ────────────────────────────────────────────────────────────────
    st.markdown('<div class="input-card"><div class="input-card-title">🌤 Weather Conditions</div>', unsafe_allow_html=True)
    temperature = st.slider("🌡️ Temperature (°F)", 34, 71, 55)
    humidity    = st.slider("💧 Humidity (%)", 8, 103, 70)
    pressure    = st.slider("🔵 Pressure (inHg)", 30.19, 30.56, 30.42, step=0.01, format="%.2f")
    wind_dir    = st.slider("🧭 Wind Direction (°)", 0, 360, 140)
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Time ───────────────────────────────────────────────────────────────────
    st.markdown('<div class="input-card"><div class="input-card-title">⏰ Time</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        current_time = st.time_input("🕐 Current Time", value=dtime(12, 0))
    with c2:
        sunrise_time = st.time_input("🌅 Sunrise", value=dtime(6, 13))
    with c3:
        sunset_time  = st.time_input("🌇 Sunset", value=dtime(18, 13))
    
    # auto-calculate derived features
    cur_min = time_to_minutes(current_time)
    sun_min = time_to_minutes(sunrise_time)
    set_min = time_to_minutes(sunset_time)
    mins_since_sunrise = round(cur_min - sun_min, 1)
    mins_to_sunset     = round(set_min - cur_min, 1)
    hour               = current_time.hour

    # show calculated values as chips
    st.markdown("<div style='margin-top:14px'>", unsafe_allow_html=True)
    st.markdown(
        f"""<div class="calc-chip">Minutes since Sunrise <span>{mins_since_sunrise} min</span></div>
            <div class="calc-chip">Minutes to Sunset <span>{mins_to_sunset} min</span></div>
            <div class="calc-chip">Hour (extracted) <span>{hour}:00</span></div>""",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    predict_btn = st.button("⚡  Predict Solar Radiation")


# ══════════════════════════════════════════════════════════════════════════════
# RIGHT — Result
# ══════════════════════════════════════════════════════════════════════════════
with col_out:

    if predict_btn:
        input_df = pd.DataFrame([{
            "temperature":             temperature,
            "pressure":                pressure,
            "humidity":                humidity,
            "winddirection(degrees)":  wind_dir,
            "hour":                    hour,
            "minutes_since_sunrise":   mins_since_sunrise,
            "minutes_to_sunset":       mins_to_sunset,
        }])
        pred = float(model.predict(input_df)[0])
        pred = max(0, round(pred, 1))
        st.session_state["last_pred"]  = pred
        st.session_state["last_color"] = (
            "#6e7681" if pred < 50 else
            "#58a6ff" if pred < 200 else
            "#e3b341" if pred < 500 else "#f78166"
        )
        st.session_state.history.insert(0, {
            "time": current_time.strftime("%H:%M"),
            "temp": temperature, "humidity": humidity, "pred": pred,
        })
        if len(st.session_state.history) > 6:
            st.session_state.history = st.session_state.history[:6]

    pred  = st.session_state.get("last_pred",  None)
    color = st.session_state.get("last_color", "#58a6ff")

    if pred is not None:
        if pred < 50:
            tag, tag_bg = "🌙 Night / Overcast",   "#21262d"
        elif pred < 200:
            tag, tag_bg = "⛅ Low Radiation",       "#1f3a5f"
        elif pred < 500:
            tag, tag_bg = "🌤 Moderate Radiation",  "#3d2e0a"
        else:
            tag, tag_bg = "☀️ High Radiation",      "#3d1a0a"

        # big number
        st.markdown(
            f"""<div class="gauge-wrap">
                <div class="gauge-label">Predicted Solar Radiation</div>
                <div class="gauge-value" style="color:{color}">{pred}</div>
                <div class="gauge-unit">W/m²</div>
                <div class="gauge-tag" style="background:{tag_bg};color:{color}">{tag}</div>
            </div>""",
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)

        # gauge chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pred,
            number={"suffix": " W/m²", "font": {"size": 26, "color": color}},
            gauge={
                "axis": {"range": [0, 1650], "tickcolor": "#8b949e", "tickfont": {"color": "#8b949e", "size": 11}},
                "bar": {"color": color, "thickness": 0.25},
                "bgcolor": "#161b22", "bordercolor": "#30363d",
                "steps": [
                    {"range": [0,   50],  "color": "#21262d"},
                    {"range": [50,  200], "color": "#1f3a5f"},
                    {"range": [200, 500], "color": "#3d2e0a"},
                    {"range": [500,1650], "color": "#3d1a0a"},
                ],
                "threshold": {
                    "line": {"color": "#f0c040", "width": 3},
                    "thickness": 0.75,
                    "value": df["radiation"].mean(),
                },
            },
        ))
        fig_gauge.update_layout(
            paper_bgcolor="#0d1117", font_color="#c9d1d9",
            height=250, margin=dict(t=20, b=10, l=30, r=30),
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        # context chips
        pct      = min(100, round(pred / 1601 * 100, 1))
        avg      = df["radiation"].mean()
        diff     = pred - avg
        diff_str = f"+{diff:.0f}" if diff >= 0 else f"{diff:.0f}"
        diff_col = "#3fb950" if diff >= 0 else "#f85149"

        c1, c2, c3 = st.columns(3)
        for col_w, (lbl, val, vc) in zip(
            [c1, c2, c3],
            [("% of Max",    f"{pct}%",           "#58a6ff"),
             ("vs Average",  f"{diff_str} W/m²",  diff_col),
             ("Dataset Avg", f"{avg:.0f} W/m²",   "#8b949e")],
        ):
            col_w.markdown(
                f"""<div style="background:#161b22;border:1px solid #30363d;border-radius:10px;
                                padding:12px 14px;text-align:center;">
                    <div style="font-size:11px;color:#8b949e;margin-bottom:4px">{lbl}</div>
                    <div style="font-size:20px;font-weight:700;color:{vc}">{val}</div>
                </div>""",
                unsafe_allow_html=True,
            )

    else:
        st.markdown(
            """<div style="background:#161b22;border:2px dashed #30363d;border-radius:16px;
                           padding:70px 20px;text-align:center;color:#484f58;">
                <div style="font-size:48px;margin-bottom:12px">⚡</div>
                <div style="font-size:15px;font-weight:600;color:#6e7681">Press Predict to see the result</div>
                <div style="font-size:13px;margin-top:6px">Fill in the weather and time inputs on the left</div>
            </div>""",
            unsafe_allow_html=True,
        )

    # history
    if st.session_state.history:
        st.markdown(
            "<div style='font-size:11px;font-weight:700;color:#8b949e;letter-spacing:0.1em;"
            "text-transform:uppercase;margin:20px 0 10px'>Recent Predictions</div>",
            unsafe_allow_html=True,
        )
        for h in st.session_state.history:
            st.markdown(
                f"""<div class="history-row">
                    <span>🕐 {h['time']} &nbsp;·&nbsp; 🌡️ {h['temp']}°F &nbsp;·&nbsp; 💧 {h['humidity']}%</span>
                    <span class="val">{h['pred']} W/m²</span>
                </div>""",
                unsafe_allow_html=True,
            )


# ══════════════════════════════════════════════════════════════════════════════
# BOTTOM — Daily pattern chart
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
st.markdown(
    "<div style='font-size:11px;font-weight:700;color:#58a6ff;letter-spacing:0.12em;"
    "text-transform:uppercase;margin-bottom:4px'>Daily Pattern</div>"
    "<div style='font-size:20px;font-weight:700;color:#e6edf3;margin-bottom:4px'>Average Radiation by Hour</div>"
    "<div style='font-size:13px;color:#8b949e;margin-bottom:16px'>Your prediction plotted against the dataset average.</div>",
    unsafe_allow_html=True,
)

hourly = df.groupby("hour")["radiation"].mean().reset_index()
fig_line = go.Figure()
fig_line.add_trace(go.Scatter(
    x=hourly["hour"], y=hourly["radiation"],
    mode="lines+markers",
    line=dict(color="#388bfd", width=2.5),
    marker=dict(size=5, color="#58a6ff"),
    fill="tozeroy", fillcolor="rgba(56,139,253,0.07)",
    name="Dataset Average",
))
if pred is not None:
    fig_line.add_trace(go.Scatter(
        x=[hour], y=[pred],
        mode="markers",
        marker=dict(size=14, color=color, symbol="star",
                    line=dict(color="#0d1117", width=1.5)),
        name=f"Your Prediction  ({pred} W/m²)",
    ))
fig_line.update_layout(
    paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
    font_color="#c9d1d9", font_family="Inter, sans-serif",
    height=280,
    xaxis=dict(title="Hour of Day", tickmode="linear", dtick=2, gridcolor="#21262d", zerolinecolor="#30363d"),
    yaxis=dict(title="Avg Radiation (W/m²)", gridcolor="#21262d", zerolinecolor="#30363d"),
    legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", y=1.1),
    margin=dict(t=30, b=40, l=10, r=10),
)
st.plotly_chart(fig_line, use_container_width=True)

st.divider()
st.caption("⚡ Solar Radiation Prediction · Prediction Page")
