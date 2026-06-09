import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, .stApp {
    background-color: #0d1117 !important;
    font-family: 'Inter', sans-serif;
}

.stExpander {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 12px !important;
    margin-bottom: 12px !important;
    overflow: hidden;
}
.stExpander:hover {
    border-color: #58a6ff !important;
    box-shadow: 0 0 0 1px #58a6ff30;
}
.stExpander summary {
    padding: 18px 20px !important;
    font-size: 15px !important;
    font-weight: 600 !important;
    color: #e6edf3 !important;
    cursor: pointer;
}
.stExpander summary:hover { color: #58a6ff !important; }
details[open] > summary { color: #58a6ff !important; border-bottom: 1px solid #21262d; }
.stExpander > div > div { padding: 0 !important; }

.sec-title {
    font-size: 11px;
    font-weight: 700;
    color: #58a6ff;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin: 48px 0 6px 0;
}
.sec-heading {
    font-size: 24px;
    font-weight: 700;
    color: #e6edf3;
    margin: 0 0 6px 0;
}
.sec-sub {
    font-size: 14px;
    color: #8b949e;
    margin-bottom: 24px;
}

.story-box {
    background: #0d1117;
    border-top: 1px solid #21262d;
    padding: 20px 24px;
}
.story-headline {
    font-size: 16px;
    font-weight: 700;
    color: #e6edf3;
    margin-bottom: 6px;
}
.story-text {
    font-size: 13.5px;
    color: #8b949e;
    line-height: 1.75;
}
.story-text b { color: #c9d1d9; }
.story-text .accent { color: #58a6ff; font-weight: 600; }

.pill-row { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 14px; }
.pill {
    background: #21262d;
    border: 1px solid #30363d;
    border-radius: 20px;
    padding: 5px 14px;
    font-size: 12px;
    color: #8b949e;
}
.pill b { color: #58a6ff; }

div[data-testid="stDivider"] hr { border-color: #21262d; }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load():
    return pd.read_csv("Cleaned_Data.csv")

df = load()

DARK = dict(
    plot_bgcolor="#161b22",
    paper_bgcolor="#0d1117",
    font_color="#c9d1d9",
    font_family="Inter, sans-serif",
)

GRID = dict(gridcolor="#21262d", zerolinecolor="#30363d")


# Header
st.markdown("## 🔬 Analysis")
st.markdown(
    "<span style='color:#8b949e;font-size:14px'>Click any question to see analysis and charts</span>",
    unsafe_allow_html=True
)
st.divider()


# Q1
with st.expander("🌡️ Q1 — What most affects solar radiation?"):
    num_cols = ["temperature","pressure","humidity","winddirection(degrees)","hour","minutes_since_sunrise","minutes_to_sunset"]
    corr_vals = df[num_cols + ["radiation"]].corr()["radiation"].drop("radiation").sort_values()
    colors = ["#f85149" if v < 0 else "#3fb950" for v in corr_vals.values]

    fig = go.Figure(go.Bar(
        x=corr_vals.values, y=corr_vals.index,
        orientation="h", marker_color=colors,
        text=[f"{v:+.3f}" for v in corr_vals.values],
        textposition="outside",
        textfont=dict(size=12),
    ))
    fig.update_layout(
        **DARK, height=380,
        xaxis=dict(range=[-0.4, 0.9], title="Correlation Coefficient", **GRID),
        yaxis=dict(title=""),
        margin=dict(t=20, b=20, l=10, r=60),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div class="story-box">
        <div class="story-headline">🔍 What the data says</div>
        <div class="story-text">
            <b>Temperature</b> is the strongest feature with correlation
            <span class="accent">+0.735</span> — hotter weather usually means clearer skies and more radiation.<br><br>
            <b>Humidity</b> and <b>Wind Direction</b> have negative impact (~-0.23) — high humidity often means clouds blocking radiation.<br><br>
            Surprisingly, <b>hour</b> and <b>minutes_since_sunrise</b> have very weak correlation — but that doesn't mean they are useless! The model captures non-linear relationships.
        </div>
        <div class="pill-row">
            <div class="pill">Temperature <b>+0.735</b></div>
            <div class="pill">Humidity <b>-0.226</b></div>
            <div class="pill">Wind Direction <b>-0.230</b></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# Q2
with st.expander("⏰ Q2 — When is radiation at its peak?"):
    hourly = df.groupby("hour")["radiation"].mean().reset_index()

    fig = px.bar(
        hourly, x="hour", y="radiation",
        color="radiation", color_continuous_scale=["#21262d","#f78166","#ffa657","#e3b341"],
        text=hourly["radiation"].round(0).astype(int),
        template="plotly_dark"
    )
    fig.update_layout(
        **DARK, height=360,
        xaxis=dict(title="Hour of Day", tickmode="linear", **GRID),
        yaxis=dict(title="Avg Radiation (W/m²)", **GRID),
        coloraxis_showscale=False,
        margin=dict(t=20, b=20),
    )
    fig.add_vline(
        x=12, line_dash="dash", line_color="#58a6ff", opacity=0.6,
        annotation_text="Peak ~727 W/m²",
        annotation_font_color="#58a6ff",
        annotation_position="top right"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div class="story-box">
        <div class="story-headline">🔍 Solar day behavior</div>
        <div class="story-text">
            Radiation starts increasing at <span class="accent">6 AM</span>,
            peaks around <span class="accent">~727 W/m²</span> between 11 AM – 12 PM,
            and drops to near zero by <span class="accent">7 PM</span>.<br><br>
            The 6-hour window (9 AM – 3 PM) is the <b>core solar energy production period</b>.
        </div>
        <div class="pill-row">
            <div class="pill">Peak Hour <b>11 AM – 1 PM</b></div>
            <div class="pill">Active Window <b>6 AM – 7 PM</b></div>
            <div class="pill">Max Avg <b>727 W/m²</b></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# Q3
with st.expander("📊 Q3 — Is the data balanced or skewed?"):
    fig = px.histogram(
        df, x="radiation", nbins=50,
        color_discrete_sequence=["#58a6ff"],
        template="plotly_dark",
    )
    fig.update_layout(
        **DARK, height=340,
        bargap=0.04,
        xaxis=dict(title="Radiation (W/m²)", **GRID),
        yaxis=dict(title="Count", **GRID),
        margin=dict(t=20, b=20),
    )
    fig.add_vrect(x0=0, x1=10, fillcolor="#30363d", opacity=0.6, line_width=0,
                  annotation_text="Night 54%", annotation_font=dict(color="#8b949e", size=11))
    fig.add_vrect(x0=10, x1=500, fillcolor="#1f3a5f", opacity=0.4, line_width=0,
                  annotation_text="Partial 27%", annotation_font=dict(color="#58a6ff", size=11))
    fig.add_vrect(x0=500, x1=1650, fillcolor="#3d2e0a", opacity=0.5, line_width=0,
                  annotation_text="Peak 19%", annotation_font=dict(color="#e3b341", size=11))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div class="story-box">
        <div class="story-headline">🔍 The data is skewed — which is normal</div>
        <div class="story-text">
            <span class="accent">54%</span> of readings are below 10 W/m² — night or heavy clouds.<br>
            Only <span class="accent">19%</span> are above 500 W/m² — peak sunlight hours.<br><br>
            Skewness = <b>1.37</b> means strong right skew — the model must handle all conditions, not just averages.
        </div>
        <div class="pill-row">
            <div class="pill">Night / Cloudy <b>54%</b></div>
            <div class="pill">Partial Sun <b>27%</b></div>
            <div class="pill">Peak Sun <b>19%</b></div>
            <div class="pill">Skewness <b>1.37</b></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# Q4
with st.expander("⭐ Q4 — What are the most important model features?"):
    feat_imp = pd.DataFrame({
        "Feature": ["minutes_since_sunrise","minutes_to_sunset","humidity",
                    "pressure","temperature","winddirection(degrees)","hour"],
        "Importance": [1175, 1166, 921, 767, 753, 668, 160],
    }).sort_values("Importance")

    fig = px.bar(
        feat_imp, x="Importance", y="Feature", orientation="h",
        color="Importance",
        color_continuous_scale=["#21262d","#388bfd","#58a6ff"],
        text="Importance",
        template="plotly_dark",
    )
    fig.update_layout(
        **DARK, height=360,
        coloraxis_showscale=False,
        xaxis=dict(title="Importance Score", **GRID),
        yaxis_title="",
        margin=dict(t=10, r=70),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div class="story-box">
        <div class="story-headline">🔍 Key insight — time matters more than weather</div>
        <div class="story-text">
            <span class="accent">minutes_since_sunrise</span> and
            <span class="accent">minutes_to_sunset</span> are the most important features,
            even more than temperature.<br><br>
            This means solar position (sun angle) is the dominant factor.<br><br>
            Raw <b>hour</b> is much less important — feature engineering added real value.
        </div>
        <div class="pill-row">
            <div class="pill">Since Sunrise <b>1175</b></div>
            <div class="pill">To Sunset <b>1166</b></div>
            <div class="pill">Temperature <b>753</b></div>
            <div class="pill">Raw Hour <b>160</b></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# Q5
with st.expander("🤖 Q5 — Best model and overfitting check"):
    models_df = pd.DataFrame({
        "Model": ["Linear Regression","KNN","Decision Tree",
                  "Random Forest","CatBoost","XGBoost","LightGBM ✅"],
        "Train R²": [0.6176, 0.9378, 1.0000, 0.9896, 0.9474, 0.9665, 0.9367],
        "Test R²": [0.5724, 0.7476, 0.6643, 0.8020, 0.7970, 0.7850, 0.8076],
    }).sort_values("Test R²", ascending=True)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=models_df["Model"], x=models_df["Train R²"],
        name="Train R²", orientation="h",
        marker_color="#30363d",
        text=[f"{v:.3f}" for v in models_df["Train R²"]],
        textposition="inside",
    ))
    fig.add_trace(go.Bar(
        y=models_df["Model"], x=models_df["Test R²"],
        name="Test R²", orientation="h",
        marker_color="#388bfd",
        text=[f"{v:.4f}" for v in models_df["Test R²"]],
        textposition="inside",
    ))
    fig.update_layout(
        **DARK, barmode="overlay",
        xaxis=dict(range=[0.5, 1.05], title="R² Score", **GRID),
        height=380,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div class="story-box">
        <div class="story-headline">🔍 Winner and loser models</div>
        <div class="story-text">
            <b>LightGBM</b> achieved the best Test R² = <span class="accent">0.8076</span> with strong generalization.<br><br>
            <b>Decision Tree</b> is heavily overfitting (Train = 1.0, Test = 0.664).<br><br>
            <b>Linear Regression</b> underperforms because relationships are non-linear.
        </div>
        <div class="pill-row">
            <div class="pill">🏆 LightGBM <b>0.8076</b></div>
            <div class="pill">⚠️ Decision Tree overfit</div>
            <div class="pill">❌ Linear weak</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


st.divider()
st.caption("🔬 Solar Radiation Prediction · Analysis Page")