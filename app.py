"""
app.py — SemiSight: Semiconductor Process Intelligence System
Yield Prediction · Wafer Defect Analysis · Root Cause Ranking · Process Optimization
Built by Bara Luch
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

st.set_page_config(
    page_title="SemiSight | Process Intelligence",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Semiconductor Industry Theme ───────────────────────────────────────────────
# Inspired by KLA, Synopsys, Applied Materials dashboards
# Dark charcoal base + amber warnings + cyan data + red critical alerts
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

  :root {
    --bg:       #0e0e0e;
    --bg1:      #141414;
    --bg2:      #1a1a1a;
    --surface:  #1f1f1f;
    --border:   #2a2a2a;
    --border2:  #363636;
    --amber:    #f59e0b;
    --amber2:   #d97706;
    --cyan:     #06b6d4;
    --green:    #10b981;
    --red:      #ef4444;
    --white:    #f5f5f5;
    --muted:    #737373;
    --dim:      #404040;
  }

  .stApp {
    background-color: #0e0e0e !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    color: #f5f5f5 !important;
  }

  /* Sidebar */
  [data-testid="stSidebar"] {
    background-color: #0a0a0a !important;
    border-right: 1px solid #2a2a2a !important;
  }

  /* Metrics */
  div[data-testid="stMetricValue"] {
    color: #f59e0b !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-weight: 600 !important;
  }
  div[data-testid="stMetricLabel"] {
    color: #737373 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.7rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
  }

  /* Tabs */
  .stTabs [data-baseweb="tab-list"] {
    background-color: #141414 !important;
    border-bottom: 1px solid #2a2a2a !important;
    gap: 0px !important;
  }
  .stTabs [data-baseweb="tab"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.72rem !important;
    color: #737373 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    padding: 12px 20px !important;
    border-right: 1px solid #2a2a2a !important;
  }
  .stTabs [aria-selected="true"] {
    color: #f59e0b !important;
    border-bottom: 2px solid #f59e0b !important;
    background-color: #1a1a1a !important;
  }

  /* Buttons */
  .stButton > button {
    background: linear-gradient(135deg, #f59e0b, #d97706) !important;
    color: #0e0e0e !important;
    border: none !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-weight: 600 !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    border-radius: 3px !important;
  }
  .stButton > button:hover {
    background: linear-gradient(135deg, #fbbf24, #f59e0b) !important;
    box-shadow: 0 0 20px rgba(245,158,11,0.4) !important;
  }

  /* Dataframes */
  .stDataFrame { border: 1px solid #2a2a2a !important; border-radius: 4px !important; }

  /* Selectbox / Slider */
  .stSelectbox label, .stSlider label, .stCheckbox label {
    color: #737373 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
  }

  /* Status boxes */
  .status-ok    { color: #10b981; font-family: monospace; font-size: 0.75rem; }
  .status-warn  { color: #f59e0b; font-family: monospace; font-size: 0.75rem; }
  .status-crit  { color: #ef4444; font-family: monospace; font-size: 0.75rem; }

  /* KPI card */
  .kpi-card {
    background: #141414;
    border: 1px solid #2a2a2a;
    border-left: 3px solid #f59e0b;
    border-radius: 3px;
    padding: 14px 18px;
    margin-bottom: 8px;
  }
  .kpi-label { font-family: 'IBM Plex Mono', monospace; font-size: 0.65rem; color: #737373; letter-spacing: 0.15em; text-transform: uppercase; margin-bottom: 4px; }
  .kpi-value { font-family: 'IBM Plex Mono', monospace; font-size: 1.4rem; color: #f59e0b; font-weight: 600; }
  .kpi-sub   { font-family: 'IBM Plex Mono', monospace; font-size: 0.65rem; color: #404040; margin-top: 2px; }

  /* Section headers */
  .sec-tag { font-family: 'IBM Plex Mono', monospace; font-size: 0.62rem; color: #f59e0b; letter-spacing: 0.25em; text-transform: uppercase; opacity: 0.8; margin-bottom: 6px; }

  /* Insight panel */
  .insight {
    background: rgba(245,158,11,0.05);
    border: 1px solid rgba(245,158,11,0.2);
    border-left: 3px solid #f59e0b;
    border-radius: 3px;
    padding: 14px 18px;
    margin: 12px 0;
    font-size: 0.875rem;
    color: #d4d4d4;
  }

  /* Alert banners */
  .alert-crit {
    background: rgba(239,68,68,0.08);
    border: 1px solid rgba(239,68,68,0.3);
    border-left: 3px solid #ef4444;
    border-radius: 3px;
    padding: 10px 16px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    color: #fca5a5;
    margin: 6px 0;
  }
  .alert-ok {
    background: rgba(16,185,129,0.06);
    border: 1px solid rgba(16,185,129,0.25);
    border-left: 3px solid #10b981;
    border-radius: 3px;
    padding: 10px 16px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    color: #6ee7b7;
    margin: 6px 0;
  }

  footer { display: none !important; }
  #MainMenu { visibility: hidden !important; }
</style>
""", unsafe_allow_html=True)

from data.loader import load_secom, load_wafer_maps, generate_wafer_maps, get_secom_stats, DEFECT_CLASSES
from features.engineer import prepare_secom_features, prepare_wafer_features
from models.trainer import train_all_models
from sql.queries import build_db, run_all_queries
from explainability.shap_analysis import (
    get_shap_values, plot_feature_importance, plot_waterfall,
    plot_yield_by_step, plot_chamber_comparison, plot_weekly_trend,
    plot_wafer_defect_distribution, get_yield_insight
)
from spc.control_charts import (
    run_spc_analysis, plot_xbar_chart, plot_cpk_dashboard,
    plot_violation_heatmap, get_spc_summary
)
# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="background:#141414; border-bottom:1px solid #2a2a2a; padding:16px 24px; margin:-1rem -1rem 1.5rem -1rem; display:flex; align-items:center; justify-content:space-between;">
  <div style="display:flex; align-items:center; gap:16px;">
    <div style="width:32px; height:32px; background:linear-gradient(135deg,#f59e0b,#d97706); border-radius:3px; display:flex; align-items:center; justify-content:center; font-size:16px;">🔬</div>
    <div>
      <div style="font-family:'IBM Plex Mono',monospace; font-size:1rem; font-weight:600; color:#f5f5f5; letter-spacing:0.05em;">SemiSight</div>
      <div style="font-family:'IBM Plex Mono',monospace; font-size:0.62rem; color:#737373; letter-spacing:0.2em; text-transform:uppercase;">Semiconductor Process Intelligence System</div>
    </div>
  </div>
  <div style="display:flex; gap:24px; align-items:center;">
    <div style="text-align:center;">
      <div style="font-family:'IBM Plex Mono',monospace; font-size:0.6rem; color:#737373; letter-spacing:0.1em; text-transform:uppercase;">Platform</div>
      <div style="font-family:'IBM Plex Mono',monospace; font-size:0.75rem; color:#06b6d4;">SECOM + WM-811K</div>
    </div>
    <div style="text-align:center;">
      <div style="font-family:'IBM Plex Mono',monospace; font-size:0.6rem; color:#737373; letter-spacing:0.1em; text-transform:uppercase;">Models</div>
      <div style="font-family:'IBM Plex Mono',monospace; font-size:0.75rem; color:#06b6d4;">XGB · IF · LR</div>
    </div>
    <div style="text-align:center;">
      <div style="font-family:'IBM Plex Mono',monospace; font-size:0.6rem; color:#737373; letter-spacing:0.1em; text-transform:uppercase;">Built by</div>
      <div style="font-family:'IBM Plex Mono',monospace; font-size:0.75rem; color:#f59e0b;">Bara Luch</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sec-tag">// System Config</div>', unsafe_allow_html=True)
    test_size  = st.slider("Test Partition", 0.10, 0.30, 0.20, 0.05)
    n_wafers   = st.selectbox("Wafer Sample Size", [1000, 2000, 5000], index=1)
    run_shap   = st.checkbox("Enable SHAP Analysis", value=True)
    st.divider()
    run_btn = st.button("▶  INITIALIZE PIPELINE", use_container_width=True)
    st.divider()
    st.markdown("""
    <div style="font-family:'IBM Plex Mono',monospace; font-size:0.68rem; color:#404040; line-height:2;">
    <span style="color:#f59e0b">DATASET</span><br>
    SECOM UCI · 590 params<br>
    WM-811K · 9 defect classes<br><br>
    <span style="color:#f59e0b">PIPELINE</span><br>
    Imputation → Var filter<br>
    Feature engineering<br>
    Imbalanced learning<br>
    SHAP root cause<br><br>
    <span style="color:#f59e0b">SQL ANALYTICS</span><br>
    8 process queries<br>
    Chamber × Step × Lot<br><br>
    <span style="color:#737373">Built by Bara Luch</span><br>
    <a href="https://github.com/BARA-LUCH" style="color:#06b6d4">GitHub</a> ·
    <a href="https://linkedin.com/in/bara-luch" style="color:#06b6d4">LinkedIn</a>
    </div>
    """, unsafe_allow_html=True)

# ── Plot theme helper ─────────────────────────────────────────────────────────
PLOT_LAYOUT = dict(
    plot_bgcolor="#141414",
    paper_bgcolor="#0e0e0e",
    font=dict(family="IBM Plex Mono", color="#a3a3a3", size=11),
    xaxis=dict(gridcolor="#1f1f1f", linecolor="#2a2a2a", tickfont=dict(size=10)),
    yaxis=dict(gridcolor="#1f1f1f", linecolor="#2a2a2a", tickfont=dict(size=10)),
    title_font=dict(size=13, color="#f5f5f5"),
    legend=dict(bgcolor="#141414", bordercolor="#2a2a2a", borderwidth=1),
    margin=dict(t=40, b=40, l=40, r=20),
)

def apply_theme(fig):
    fig.update_layout(**PLOT_LAYOUT)
    return fig

# ── Pipeline ──────────────────────────────────────────────────────────────────
if run_btn:
    from sklearn.model_selection import train_test_split

    with st.spinner("LOADING SECOM PROCESS DATA..."):
        try:
            df_secom = load_secom()
            stats    = get_secom_stats(df_secom)
            st.markdown(f'<div class="alert-ok">✓ SECOM LOADED — {len(df_secom):,} manufacturing runs · {stats["Failure Rate"]} failure rate</div>', unsafe_allow_html=True)
        except Exception as e:
            st.markdown(f'<div class="alert-crit">✗ SECOM LOAD FAILED: {e}</div>', unsafe_allow_html=True)
            st.stop()

    with st.spinner("EXECUTING SQL ANALYTICS..."):
        try:
            conn        = build_db(df_secom)
            sql_results = run_all_queries(conn)
            st.markdown('<div class="alert-ok">✓ SQL ANALYTICS — 8 queries executed on process database</div>', unsafe_allow_html=True)
        except Exception as e:
            sql_results = {}

    with st.spinner("ENGINEERING PROCESS PARAMETERS..."):
        try:
            X, y, feat_names, scaler, imputer, selector, keep_cols = prepare_secom_features(df_secom)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)
            st.markdown(f'<div class="alert-ok">✓ FEATURE ENGINEERING — {len(feat_names)} features retained from 590 parameters</div>', unsafe_allow_html=True)
        except Exception as e:
            st.markdown(f'<div class="alert-crit">✗ FEATURE ENGINEERING FAILED: {e}</div>', unsafe_allow_html=True)
            st.stop()

    with st.spinner("GENERATING WAFER MAP DATASET..."):
        try:
            df_wafer = load_wafer_maps() if n_wafers == 5000 else generate_wafer_maps(n_wafers)
            Xw, yw, yw_bin, wafer_feats, wafer_scaler = prepare_wafer_features(df_wafer)
            Xw_train, Xw_test, yw_train, yw_test = train_test_split(Xw, yw, test_size=0.20, stratify=yw, random_state=42)
            st.markdown(f'<div class="alert-ok">✓ WAFER MAPS — {len(df_wafer):,} wafers · 9 defect classes</div>', unsafe_allow_html=True)
        except Exception as e:
            df_wafer = None
            Xw_train = Xw_test = yw_train = yw_test = None

    with st.spinner("TRAINING YIELD PREDICTION MODELS..."):
        try:
            results = train_all_models(
                X_train, X_test, y_train, y_test, feat_names,
                Xw_train, Xw_test, yw_train, yw_test, DEFECT_CLASSES
            )
            st.markdown(f'<div class="alert-ok">✓ MODELS TRAINED — {", ".join(results["successful_models"])}</div>', unsafe_allow_html=True)
        except Exception as e:
            st.markdown(f'<div class="alert-crit">✗ TRAINING FAILED: {e}</div>', unsafe_allow_html=True)
            st.stop()

    shap_data = None
    if run_shap and results.get("xgboost", {}).get("success"):
        with st.spinner("COMPUTING SHAP ROOT CAUSE ANALYSIS..."):
            shap_data = get_shap_values(results["xgboost"], X_test, feat_names)
            if shap_data and shap_data.get("success"):
                st.markdown('<div class="alert-ok">✓ SHAP ANALYSIS COMPLETE — root cause parameters ranked</div>', unsafe_allow_html=True)

    # SPC Analysis
    with st.spinner("RUNNING STATISTICAL PROCESS CONTROL..."):
        try:
            spc_results = run_spc_analysis(df_secom, n_params=12, subgroup_size=5)
            spc_summary = get_spc_summary(spc_results)
            st.markdown(f'<div class="alert-ok">✓ SPC COMPLETE — {spc_summary["n_params"]} parameters · {spc_summary["total_violations"]} violations · {spc_summary["critical_params"]} critical</div>', unsafe_allow_html=True)
        except Exception as e:
            spc_results = {}
            spc_summary = {}
            st.markdown(f'<div class="alert-crit">✗ SPC FAILED: {e}</div>', unsafe_allow_html=True)

    st.session_state.update({
        "results": results, "df_secom": df_secom, "stats": stats,
        "sql_results": sql_results, "feat_names": feat_names,
        "X_test": X_test, "y_test": y_test,
        "shap_data": shap_data, "df_wafer": df_wafer,
        "spc_results": spc_results, "spc_summary": spc_summary,
    })

# ── Landing ───────────────────────────────────────────────────────────────────
if "results" not in st.session_state:
    st.markdown("""
    <div style="display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin-bottom:24px;">
      <div class="kpi-card"><div class="kpi-label">Dataset</div><div class="kpi-value" style="font-size:1rem;">SECOM</div><div class="kpi-sub">UCI ML Repository</div></div>
      <div class="kpi-card"><div class="kpi-label">Process Params</div><div class="kpi-value">590</div><div class="kpi-sub">Per manufacturing run</div></div>
      <div class="kpi-card"><div class="kpi-label">ML Models</div><div class="kpi-value" style="font-size:1rem;">3 + 1</div><div class="kpi-sub">Yield + Defect classifier</div></div>
      <div class="kpi-card"><div class="kpi-label">SQL Queries</div><div class="kpi-value">8</div><div class="kpi-sub">Process analytics</div></div>
    </div>

    <div style="background:#141414; border:1px solid #2a2a2a; border-radius:4px; padding:24px; margin-bottom:16px;">
      <div class="sec-tag">// System Architecture</div>
      <div style="display:grid; grid-template-columns:repeat(2,1fr); gap:16px; margin-top:16px;">
        <div style="border-left:2px solid #f59e0b; padding-left:14px;">
          <div style="font-family:'IBM Plex Mono',monospace; font-size:0.75rem; color:#f59e0b; margin-bottom:6px;">MODULE 01 — YIELD PREDICTION</div>
          <div style="font-size:0.83rem; color:#a3a3a3;">XGBoost on 590 SECOM process parameters with <code>scale_pos_weight</code> for imbalanced yield data (~6.5% failure rate). 5-fold stratified CV.</div>
        </div>
        <div style="border-left:2px solid #06b6d4; padding-left:14px;">
          <div style="font-family:'IBM Plex Mono',monospace; font-size:0.75rem; color:#06b6d4; margin-bottom:6px;">MODULE 02 — WAFER DEFECT ANALYSIS</div>
          <div style="font-size:0.83rem; color:#a3a3a3;">9-class defect pattern recognition on WM-811K wafer maps: Center, Donut, Scratch, Edge-Ring, Edge-Loc, Loc, Random, Near-full, None.</div>
        </div>
        <div style="border-left:2px solid #10b981; padding-left:14px;">
          <div style="font-family:'IBM Plex Mono',monospace; font-size:0.75rem; color:#10b981; margin-bottom:6px;">MODULE 03 — ROOT CAUSE (SHAP)</div>
          <div style="font-size:0.83rem; color:#a3a3a3;">SHAP TreeExplainer ranks which process parameters drive yield failures — giving engineers actionable root causes per manufacturing run.</div>
        </div>
        <div style="border-left:2px solid #a78bfa; padding-left:14px;">
          <div style="font-family:'IBM Plex Mono',monospace; font-size:0.75rem; color:#a78bfa; margin-bottom:6px;">MODULE 04 — PROCESS INTELLIGENCE</div>
          <div style="font-size:0.83rem; color:#a3a3a3;">8 SQL queries on SQLite — yield by chamber, process step, lot, hour, and rolling 7-day trends. Chamber × Step failure heatmap.</div>
        </div>
      </div>
    </div>
    <div style="font-family:'IBM Plex Mono',monospace; font-size:0.7rem; color:#404040; text-align:center;">
      ▶ CLICK "INITIALIZE PIPELINE" IN SIDEBAR TO START
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Retrieve state ────────────────────────────────────────────────────────────
results     = st.session_state["results"]
df_secom    = st.session_state["df_secom"]
stats       = st.session_state["stats"]
sql_results = st.session_state["sql_results"]
feat_names  = st.session_state["feat_names"]
X_test      = st.session_state["X_test"]
y_test      = st.session_state["y_test"]
shap_data   = st.session_state["shap_data"]
df_wafer    = st.session_state.get("df_wafer")
spc_results = st.session_state.get("spc_results", {})
spc_summary = st.session_state.get("spc_summary", {})

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "01 · OVERVIEW",
    "02 · SQL ANALYTICS",
    "03 · YIELD MODELS",
    "04 · ROOT CAUSE",
    "05 · WAFER DEFECTS",
    "06 · SPC",
    "07 · CNN CLASSIFIER",
    "08 · BUSINESS IMPACT",
])

# ── TAB 1: OVERVIEW ───────────────────────────────────────────────────────────
with tab1:
    st.markdown('<div class="sec-tag">// Manufacturing KPIs</div>', unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Runs",    stats["Total Runs"])
    c2.metric("Pass",          stats["Pass"])
    c3.metric("Fail",          stats["Fail"])
    c4.metric("Failure Rate",  stats["Failure Rate"])
    c5.metric("Missing Data",  stats["Missing Data"])

    xgb_r = results.get("xgboost", {})
    if xgb_r.get("success"):
        auc = xgb_r["metrics"]["ROC-AUC"]
        st.markdown(f'<div class="insight">⚡ {get_yield_insight(results, sql_results)}</div>', unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        fig = plot_yield_by_step(sql_results)
        apply_theme(fig)
        st.plotly_chart(fig, use_container_width=True)
    with col_b:
        fig = plot_chamber_comparison(sql_results)
        apply_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

    fig = plot_weekly_trend(sql_results)
    apply_theme(fig)
    st.plotly_chart(fig, use_container_width=True)

# ── TAB 2: SQL ANALYTICS ──────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="sec-tag">// Process Database Analytics — SQLite · 8 Queries</div>', unsafe_allow_html=True)

    st.markdown("**YIELD BY PROCESS STEP**")
    st.code("""SELECT process_step, COUNT(*) AS total_runs,
       ROUND(100.0 * SUM(yield_pass) / COUNT(*), 2) AS yield_pct,
       ROUND(100.0 * SUM(CASE WHEN yield_pass=0 THEN 1 ELSE 0 END) / COUNT(*), 2) AS failure_pct
FROM runs GROUP BY process_step ORDER BY failure_pct DESC""", language="sql")
    if not sql_results.get("yield_by_step", pd.DataFrame()).empty:
        st.dataframe(sql_results["yield_by_step"], use_container_width=True, hide_index=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**CHAMBER PERFORMANCE**")
        if not sql_results.get("chamber_yield", pd.DataFrame()).empty:
            st.dataframe(sql_results["chamber_yield"], use_container_width=True, hide_index=True)
    with col_b:
        st.markdown("**TOP FAILING LOTS**")
        if not sql_results.get("lot_failures", pd.DataFrame()).empty:
            st.dataframe(sql_results["lot_failures"], use_container_width=True, hide_index=True)

    st.markdown("**CHAMBER × PROCESS STEP FAILURE HEATMAP**")
    if not sql_results.get("chamber_step_cross", pd.DataFrame()).empty:
        df_cross = sql_results["chamber_step_cross"]
        pivot    = df_cross.pivot_table(values="failure_pct", index="chamber_id",
                                         columns="process_step", fill_value=0)
        fig_heat = px.imshow(
            pivot,
            color_continuous_scale=[[0,"#141414"],[0.5,"#d97706"],[1,"#ef4444"]],
            title="Failure Rate % — Chamber × Process Step",
            aspect="auto",
            text_auto=".1f",
        )
        apply_theme(fig_heat)
        fig_heat.update_coloraxes(colorbar=dict(tickfont=dict(size=10, color="#a3a3a3")))
        st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown("**HOURLY FAILURE PATTERN**")
    hourly = sql_results.get("hourly_pattern", pd.DataFrame())
    if not hourly.empty:
        fig_h = go.Figure(go.Bar(
            x=hourly["hour"], y=hourly["failure_pct"],
            marker_color=["#ef4444" if v > hourly["failure_pct"].mean() * 1.2 else "#f59e0b"
                          for v in hourly["failure_pct"]],
            hovertemplate="Hour %{x}:00 — Failure Rate: %{y:.2f}%<extra></extra>",
        ))
        fig_h.update_layout(title="Failure Rate by Hour of Day",
                            xaxis_title="Hour (UTC)", yaxis_title="Failure Rate (%)", height=300)
        apply_theme(fig_h)
        st.plotly_chart(fig_h, use_container_width=True)

# ── TAB 3: YIELD MODELS ───────────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="sec-tag">// Yield Prediction Model Comparison</div>', unsafe_allow_html=True)

    comp = results.get("comparison", pd.DataFrame())
    if not comp.empty:
        numeric_cols = list(comp.select_dtypes(include=[float, int]).columns)
        st.dataframe(
            comp.style
                .highlight_max(subset=numeric_cols, color="#1a2e1a")
                .format("{:.4f}", subset=numeric_cols),
            use_container_width=True
        )
        best = comp["ROC-AUC"].idxmax()
        st.markdown(f'<div class="alert-ok">✓ BEST MODEL: {best} — ROC-AUC {comp["ROC-AUC"].max():.4f}</div>', unsafe_allow_html=True)

    # ROC curves
    fig_roc = go.Figure()
    model_colors = {
        "XGBoost Yield Predictor": "#f59e0b",
        "Isolation Forest":        "#06b6d4",
        "Logistic Regression":     "#10b981",
    }
    for key in ["xgboost", "isolation_forest", "logistic"]:
        r = results.get(key, {})
        if r.get("success") and "ROC Curve" in r.get("metrics", {}):
            roc = r["metrics"]["ROC Curve"]
            auc = r["metrics"]["ROC-AUC"]
            fig_roc.add_trace(go.Scatter(
                x=roc["fpr"], y=roc["tpr"], mode="lines",
                name=f"{r['name']} (AUC={auc:.3f})",
                line=dict(color=model_colors.get(r["name"], "#737373"), width=2),
            ))
    fig_roc.add_trace(go.Scatter(
        x=[0,1], y=[0,1], mode="lines",
        line=dict(dash="dash", color="#2a2a2a"), name="Random Classifier",
        showlegend=True,
    ))
    fig_roc.update_layout(
        title="ROC Curves — Yield Failure Detection",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=420,
    )
    apply_theme(fig_roc)
    st.plotly_chart(fig_roc, use_container_width=True)

    xgb = results.get("xgboost", {})
    if xgb.get("success") and "cv_scores" in xgb:
        cv = xgb["cv_scores"]
        st.markdown(f"""
        <div style="font-family:'IBM Plex Mono',monospace; font-size:0.75rem; color:#a3a3a3; background:#141414; border:1px solid #2a2a2a; padding:12px 16px; border-radius:3px;">
        5-FOLD STRATIFIED CV · Mean ROC-AUC: <span style="color:#f59e0b">{np.mean(cv):.4f}</span> ± {np.std(cv):.4f} &nbsp;|&nbsp;
        Folds: {' · '.join([f'<span style="color:#f59e0b">{s:.4f}</span>' for s in cv])}
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    | Model | Type | Key Design Decision |
    |---|---|---|
    | XGBoost | Supervised | `scale_pos_weight` handles 6.5% failure rate imbalance |
    | Isolation Forest | Unsupervised | Trained on passing runs only — detects novel anomalies |
    | Logistic Regression | Supervised | Baseline with `class_weight=balanced` |
    """)
    st.info(f"Train: {results['train_size']:,} · Test: {results['test_size']:,} · Failures in test: {results['fail_in_test']:,}")

# ── TAB 4: ROOT CAUSE ────────────────────────────────────────────────────────
with tab4:
    st.markdown('<div class="sec-tag">// SHAP Root Cause Analysis — Which Parameters Drive Failures?</div>', unsafe_allow_html=True)

    if shap_data and shap_data.get("success"):
        col_a, col_b = st.columns([3, 2])
        with col_a:
            fig = plot_feature_importance(shap_data)
            apply_theme(fig)
            fig.update_traces(marker_color="#f59e0b")
            st.plotly_chart(fig, use_container_width=True)
        with col_b:
            run_idx = st.slider("Manufacturing Run #", 0, min(99, len(shap_data["shap_values"])-1), 0)
            fig = plot_waterfall(shap_data, run_idx)
            apply_theme(fig)
            st.plotly_chart(fig, use_container_width=True)

        sv        = shap_data["shap_values"]
        fns       = shap_data["feature_names"]
        mean_shap = np.abs(sv).mean(axis=0)
        top_idx   = np.argsort(mean_shap)[::-1][:15]

        top_df = pd.DataFrame({
            "Rank":        range(1, 16),
            "Parameter":   [fns[i] for i in top_idx],
            "Mean |SHAP|": [round(mean_shap[i], 5) for i in top_idx],
            "Risk Level":  ["🔴 CRITICAL" if mean_shap[i] > np.percentile(mean_shap, 95)
                            else "🟡 HIGH" if mean_shap[i] > np.percentile(mean_shap, 80)
                            else "⚪ MEDIUM" for i in top_idx],
        })
        st.markdown('<div class="sec-tag">// Top Process Parameters by Failure Impact</div>', unsafe_allow_html=True)
        st.dataframe(top_df, hide_index=True, use_container_width=True)
    else:
        st.markdown('<div class="alert-crit">✗ SHAP NOT AVAILABLE — enable in sidebar and re-run pipeline</div>', unsafe_allow_html=True)

# ── TAB 5: WAFER DEFECTS ─────────────────────────────────────────────────────
with tab5:
    st.markdown('<div class="sec-tag">// Wafer Defect Pattern Classification — 9 Classes</div>', unsafe_allow_html=True)

    if df_wafer is not None:
        wc = results.get("wafer_classifier", {})
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Wafers",     f"{len(df_wafer):,}")
        c2.metric("Defective",        f"{df_wafer['is_defective'].mean():.1%}")
        c3.metric("Defect Classes",   "9")
        if wc.get("success"):
            c4.metric("Classifier Acc", f"{wc['accuracy']:.4f}")

        col_a, col_b = st.columns(2)
        with col_a:
            fig = plot_wafer_defect_distribution(df_wafer)
            apply_theme(fig)
            st.plotly_chart(fig, use_container_width=True)
        with col_b:
            defect_stats = df_wafer.groupby("defect_class").agg(
                count=("wafer_id", "count"),
                avg_defect_ratio=("defect_ratio", "mean"),
                avg_defect_density=("defect_density", "mean"),
            ).round(4).reset_index()
            defect_stats.columns = ["Defect Class", "Count", "Avg Defect Ratio", "Avg Density"]
            st.dataframe(defect_stats, hide_index=True, use_container_width=True)

        if wc.get("success") and wc.get("report"):
            st.markdown('<div class="sec-tag">// Classification Report</div>', unsafe_allow_html=True)
            report_df     = pd.DataFrame(wc["report"]).T
            numeric_rep   = report_df.select_dtypes(include=[float])
            st.dataframe(
                numeric_rep.style
                    .highlight_max(axis=0, color="#1a2e1a")
                    .format("{:.3f}"),
                use_container_width=True
            )
    else:
        st.markdown('<div class="alert-crit">✗ WAFER DATA NOT AVAILABLE</div>', unsafe_allow_html=True)

# ── TAB 6: SPC ───────────────────────────────────────────────────────────────
with tab6:
    st.markdown('<div class="sec-tag">// Statistical Process Control — X̄-R Charts · Cpk · Western Electric Rules</div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="background:#141414; border:1px solid #2a2a2a; border-left:3px solid #f59e0b; border-radius:3px; padding:14px 20px; margin-bottom:16px; font-size:0.83rem; color:#a3a3a3;">
    SPC is the primary quality control method in semiconductor fabs (SEMI E10 standard).
    <b style="color:#f5f5f5">X̄-R charts</b> track process mean and variability.
    <b style="color:#f5f5f5">Cpk ≥ 1.33</b> is the industry standard for process capability.
    <b style="color:#f5f5f5">Western Electric rules</b> detect non-random patterns before failures occur.
    </div>
    """, unsafe_allow_html=True)

    if spc_results:
        # SPC KPIs
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Parameters Analyzed", spc_summary.get("n_params", 0))
        c2.metric("Avg Cpk",             spc_summary.get("avg_cpk", 0))
        c3.metric("WECO Violations",     spc_summary.get("total_violations", 0))
        c4.metric("Critical (Rule 1)",   spc_summary.get("critical_params", 0))
        c5.metric("Incapable (Cpk<1)",   spc_summary.get("incapable", 0))

        # Alert banner
        n_crit = spc_summary.get("critical_params", 0)
        n_inc  = spc_summary.get("incapable", 0)
        if n_crit > 0:
            st.markdown(f'<div class="alert-crit">⚠ CRITICAL: {n_crit} parameter(s) with Rule 1 violations (beyond 3σ) — immediate fab engineer review required</div>', unsafe_allow_html=True)
        if n_inc > 0:
            st.markdown(f'<div class="alert-crit">⚠ PROCESS INCAPABLE: {n_inc} parameter(s) with Cpk &lt; 1.0 — process improvement required</div>', unsafe_allow_html=True)
        if n_crit == 0 and n_inc == 0:
            st.markdown('<div class="alert-ok">✓ ALL MONITORED PARAMETERS WITHIN CONTROL LIMITS</div>', unsafe_allow_html=True)

        # Cpk Dashboard
        st.plotly_chart(plot_cpk_dashboard(spc_results), use_container_width=True)

        # WECO violation heatmap
        st.plotly_chart(plot_violation_heatmap(spc_results), use_container_width=True)

        # Individual X-bar charts
        st.markdown('<div class="sec-tag">// Individual X̄-R Control Charts</div>', unsafe_allow_html=True)
        param_options = list(spc_results.keys())
        selected_param = st.selectbox(
            "Select parameter to chart",
            param_options,
            format_func=lambda x: f"{x} | Cpk={spc_results[x]['cpk']['cpk']:.3f} | {spc_results[x]['cpk']['status']}"
        )
        if selected_param:
            st.plotly_chart(plot_xbar_chart(selected_param, spc_results[selected_param]), use_container_width=True)

            # Cpk detail
            cpk = spc_results[selected_param]["cpk"]
            weco = spc_results[selected_param]["weco"]
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown(f"""
                <div style="background:#141414; border:1px solid #2a2a2a; border-radius:3px; padding:14px 18px; font-family:'IBM Plex Mono',monospace; font-size:0.75rem;">
                <div style="color:#f59e0b; margin-bottom:8px; letter-spacing:0.1em;">PROCESS CAPABILITY</div>
                <div style="color:#a3a3a3; line-height:2.2;">
                μ (Process Mean) &nbsp;&nbsp;&nbsp;= <span style="color:#f5f5f5">{cpk['mu']:.4f}</span><br>
                σ (Estimated) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;= <span style="color:#f5f5f5">{cpk['sigma']:.4f}</span><br>
                USL &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;= <span style="color:#f5f5f5">{cpk['usl']:.4f}</span><br>
                LSL &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;= <span style="color:#f5f5f5">{cpk['lsl']:.4f}</span><br>
                Cp &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;= <span style="color:#f5f5f5">{cpk['cp']:.4f}</span><br>
                Cpk &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;= <span style="color:{'#10b981' if cpk['cpk']>=1.33 else '#f59e0b' if cpk['cpk']>=1.0 else '#ef4444'}">{cpk['cpk']:.4f}</span><br>
                Status &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;= <span style="color:{'#10b981' if cpk['status']=='CAPABLE' else '#f59e0b' if cpk['status']=='MARGINAL' else '#ef4444'}">{cpk['status']}</span>
                </div>
                </div>
                """, unsafe_allow_html=True)
            with col_b:
                st.markdown(f"""
                <div style="background:#141414; border:1px solid #2a2a2a; border-radius:3px; padding:14px 18px; font-family:'IBM Plex Mono',monospace; font-size:0.75rem;">
                <div style="color:#f59e0b; margin-bottom:8px; letter-spacing:0.1em;">WESTERN ELECTRIC VIOLATIONS</div>
                <div style="color:#a3a3a3; line-height:2.2;">
                Rule 1 (beyond 3σ) &nbsp;&nbsp;&nbsp;= <span style="color:{'#ef4444' if weco['violations']['rule1'] else '#10b981'}">{len(weco['violations']['rule1'])}</span><br>
                Rule 2 (2/3 > 2σ) &nbsp;&nbsp;&nbsp;&nbsp;= <span style="color:{'#f59e0b' if weco['violations']['rule2'] else '#10b981'}">{len(weco['violations']['rule2'])}</span><br>
                Rule 4 (8-pt run) &nbsp;&nbsp;&nbsp;&nbsp;= <span style="color:{'#f59e0b' if weco['violations']['rule4'] else '#10b981'}">{len(weco['violations']['rule4'])}</span><br>
                Rule 5 (6-pt trend) &nbsp;&nbsp;= <span style="color:{'#f59e0b' if weco['violations']['rule5'] else '#10b981'}">{len(weco['violations']['rule5'])}</span><br>
                Total Violations &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;= <span style="color:{'#ef4444' if weco['total']>5 else '#f59e0b' if weco['total']>0 else '#10b981'}">{weco['total']}</span><br>
                Overall Status &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;= <span style="color:{'#ef4444' if weco['critical'] else '#10b981'}">{'OUT OF CONTROL' if weco['critical'] else 'IN CONTROL'}</span>
                </div>
                </div>
                """, unsafe_allow_html=True)

        # SPC table summary
        st.markdown('<div class="sec-tag">// All Parameters Summary</div>', unsafe_allow_html=True)
        spc_table = pd.DataFrame([{
            "Parameter": k,
            "Cpk":       v["cpk"]["cpk"],
            "Status":    v["cpk"]["status"],
            "Violations": v["weco"]["total"],
            "Critical":  "🔴 YES" if v["weco"]["critical"] else "✅ NO",
            "Avg Mean":  round(v["cpk"]["mu"], 4),
            "Sigma":     round(v["cpk"]["sigma"], 4),
        } for k, v in spc_results.items()])
        st.dataframe(spc_table, hide_index=True, use_container_width=True)
    else:
        st.markdown('<div class="alert-crit">✗ SPC RESULTS NOT AVAILABLE — RE-RUN PIPELINE</div>', unsafe_allow_html=True)

# ── TAB 7: CNN CLASSIFIER ────────────────────────────────────────────────────
with tab7:
    st.markdown('<div class="sec-tag">// WaferCNN — Raw Image Classification (Grad-Level CV)</div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="background:#141414; border:1px solid #2a2a2a; border-left:3px solid #f59e0b; border-radius:3px; padding:16px 20px; margin-bottom:16px;">
    <div style="font-family:'IBM Plex Mono',monospace; font-size:0.72rem; color:#f59e0b; margin-bottom:8px;">// ARCHITECTURE</div>
    <div style="font-size:0.83rem; color:#a3a3a3;">
    3-channel input (26×26): <b style="color:#f5f5f5">defect map · valid die mask · pass die map</b><br>
    3 conv blocks (3→32→64→128→256) with BatchNorm + ReLU + MaxPool<br>
    AdaptiveAvgPool → FC(512) → Dropout(0.5) → FC(256) → FC(9)<br>
    Training: CrossEntropyLoss with class weights · Adam · CosineAnnealingLR · Data augmentation (rot/flip)
    </div>
    </div>
    """, unsafe_allow_html=True)

    run_cnn = st.button("▶  TRAIN CNN ON RAW WAFER IMAGES", use_container_width=False)

    if run_cnn:
        if df_wafer is None:
            st.markdown('<div class="alert-crit">✗ RUN MAIN PIPELINE FIRST TO LOAD WAFER DATA</div>', unsafe_allow_html=True)
        else:
            epochs = st.session_state.get("cnn_epochs", 30)
            with st.spinner(f"TRAINING WaferCNN ({epochs} epochs) — this takes ~60-120 seconds on CPU..."):
                try:
                    from models.cnn_wafer import train_cnn, get_cnn_summary
                    cnn_result = train_cnn(df_wafer, epochs=epochs, batch_size=64)
                    st.session_state["cnn_result"] = cnn_result
                    st.markdown(f'<div class="alert-ok">✓ CNN TRAINED — {get_cnn_summary(cnn_result)}</div>', unsafe_allow_html=True)
                except Exception as e:
                    st.markdown(f'<div class="alert-crit">✗ CNN TRAINING FAILED: {e}</div>', unsafe_allow_html=True)

    cnn_result = st.session_state.get("cnn_result")
    xgb_wafer  = results.get("wafer_classifier", {})

    if cnn_result and cnn_result.get("success"):
        # Benchmark comparison: CNN vs feature-based XGBoost
        st.markdown('<div class="sec-tag">// Benchmark: CNN (Raw Images) vs XGBoost (Hand-crafted Features)</div>', unsafe_allow_html=True)

        bench_data = {
            "Model": ["WaferCNN (Raw Images)", "XGBoost (Feature-based)"],
            "Input": ["26×26 raw pixel map", "14 hand-crafted features"],
            "Accuracy": [
                cnn_result["accuracy"],
                xgb_wafer.get("accuracy", 0) if xgb_wafer.get("success") else 0
            ],
            "F1 (weighted)": [
                cnn_result["f1_weighted"],
                xgb_wafer.get("report", {}).get("weighted avg", {}).get("f1-score", 0)
                if xgb_wafer.get("success") else 0
            ],
            "Params / Features": [
                f"{cnn_result['params']:,} params",
                "14 features"
            ],
        }
        bench_df = pd.DataFrame(bench_data).set_index("Model")
        numeric_bench = ["Accuracy", "F1 (weighted)"]
        st.dataframe(
            bench_df.style
                .highlight_max(subset=numeric_bench, color="#1a2e1a")
                .format("{:.4f}", subset=numeric_bench),
            use_container_width=True
        )

        # Published WM-811K benchmark reference
        st.markdown("""
        <div style="background:#141414; border:1px solid #2a2a2a; border-radius:3px; padding:14px 18px; margin:12px 0;">
        <div style="font-family:'IBM Plex Mono',monospace; font-size:0.65rem; color:#737373; margin-bottom:8px; letter-spacing:0.1em; text-transform:uppercase;">Published WM-811K Benchmarks (Reference)</div>
        <div style="font-size:0.8rem; color:#a3a3a3; font-family:'IBM Plex Mono',monospace; line-height:2;">
        CNN (WM-811K paper, 2019) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;→ ~97-98% accuracy (full 811K dataset)<br>
        ResNet-18 fine-tuned &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;→ ~96% accuracy<br>
        Feature + RF (baseline) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;→ ~89-91% accuracy<br>
        <span style="color:#f59e0b;">SemiSight CNN (5K synthetic) → comparable architecture, smaller dataset</span>
        </div>
        </div>
        """, unsafe_allow_html=True)

        # Training curves
        history = cnn_result["history"]
        col_a, col_b = st.columns(2)
        with col_a:
            fig_loss = go.Figure()
            fig_loss.add_trace(go.Scatter(
                y=history["train_loss"], mode="lines", name="Train Loss",
                line=dict(color="#f59e0b", width=2)
            ))
            fig_loss.add_trace(go.Scatter(
                y=history["val_loss"], mode="lines", name="Val Loss",
                line=dict(color="#06b6d4", width=2)
            ))
            fig_loss.update_layout(
                title="Training & Validation Loss",
                xaxis_title="Epoch", yaxis_title="Loss", height=300,
            )
            apply_theme(fig_loss)
            st.plotly_chart(fig_loss, use_container_width=True)

        with col_b:
            fig_acc = go.Figure()
            fig_acc.add_trace(go.Scatter(
                y=history["val_acc"], mode="lines", name="Val Accuracy",
                line=dict(color="#10b981", width=2)
            ))
            fig_acc.update_layout(
                title="Validation Accuracy",
                xaxis_title="Epoch", yaxis_title="Accuracy", height=300,
            )
            apply_theme(fig_acc)
            st.plotly_chart(fig_acc, use_container_width=True)

        # Confusion matrix
        st.markdown('<div class="sec-tag">// Confusion Matrix</div>', unsafe_allow_html=True)
        conf_mat = np.array(cnn_result["conf_matrix"])
        fig_cm = go.Figure(go.Heatmap(
            z=conf_mat,
            x=DEFECT_CLASSES,
            y=DEFECT_CLASSES,
            colorscale=[[0,"#141414"],[0.5,"#d97706"],[1,"#ef4444"]],
            text=conf_mat,
            texttemplate="%{text}",
            showscale=True,
        ))
        fig_cm.update_layout(
            title="CNN Confusion Matrix — Wafer Defect Classes",
            xaxis_title="Predicted", yaxis_title="Actual",
            height=450,
        )
        apply_theme(fig_cm)
        st.plotly_chart(fig_cm, use_container_width=True)

        # Per-class report
        st.markdown('<div class="sec-tag">// Per-Class Classification Report</div>', unsafe_allow_html=True)
        report_df = pd.DataFrame(cnn_result["report"]).T
        numeric_r = list(report_df.select_dtypes(include=[float]).columns)
        st.dataframe(
            report_df[numeric_r].style
                .highlight_max(axis=0, color="#1a2e1a")
                .format("{:.3f}"),
            use_container_width=True
        )

        # Model info
        st.markdown(f"""
        <div style="font-family:'IBM Plex Mono',monospace; font-size:0.72rem; color:#404040; background:#141414; border:1px solid #2a2a2a; padding:12px 16px; border-radius:3px;">
        Architecture: <span style="color:#a3a3a3">{cnn_result['architecture']}</span> &nbsp;|&nbsp;
        Parameters: <span style="color:#f59e0b">{cnn_result['params']:,}</span> &nbsp;|&nbsp;
        Device: <span style="color:#06b6d4">{cnn_result['device']}</span> &nbsp;|&nbsp;
        Epochs: <span style="color:#a3a3a3">{cnn_result['epochs']}</span>
        </div>
        """, unsafe_allow_html=True)

    elif not run_cnn:
        from data.loader import DEFECT_CLASSES as DC
        st.markdown(f"""
        <div style="background:#141414; border:1px solid #2a2a2a; border-radius:3px; padding:20px; text-align:center;">
        <div style="font-family:'IBM Plex Mono',monospace; font-size:0.75rem; color:#737373; margin-bottom:12px;">
        CLICK "TRAIN CNN" TO RUN RAW IMAGE CLASSIFICATION<br>
        ~60-120 SECONDS ON CPU · RUNS IN BROWSER
        </div>
        <div style="font-size:0.8rem; color:#404040;">
        9 defect classes: {" · ".join(DC)}
        </div>
        </div>
        """, unsafe_allow_html=True)

# ── TAB 8: BUSINESS IMPACT ───────────────────────────────────────────────────
with tab8:
    st.markdown('<div class="sec-tag">// Fab Revenue Impact Calculator</div>', unsafe_allow_html=True)

    xgb = results.get("xgboost", {})
    if xgb.get("success"):
        cm = xgb["metrics"]["Confusion Matrix"]
        tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]

        col1, col2 = st.columns(2)
        with col1:
            cost_per_wafer     = st.number_input("Cost per failed wafer ($)", value=500, step=50)
            wafers_per_day     = st.number_input("Wafers processed per day", value=2000, step=100)
            investigation_cost = st.number_input("Cost per false alarm ($)", value=200, step=50)
        with col2:
            failure_rate   = st.slider("Current failure rate (%)", 1.0, 15.0, 6.5, 0.5)
            detection_rate = st.slider("Detection capture rate (%)", 10, 90, 70) / 100

        daily_failures   = int(wafers_per_day * failure_rate / 100)
        failures_caught  = int(tp * detection_rate)
        revenue_saved    = failures_caught * cost_per_wafer
        false_alarm_cost = fp * investigation_cost
        net_benefit      = revenue_saved - false_alarm_cost
        annual_benefit   = net_benefit * 250

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Failures Caught",   f"{failures_caught:,}", f"missed: {fn:,}")
        c2.metric("Revenue Saved",     f"${revenue_saved:,.0f}")
        c3.metric("False Alarm Cost",  f"${false_alarm_cost:,.0f}")
        c4.metric("Net Daily Benefit", f"${net_benefit:,.0f}")

        st.markdown(f"""
        <div class="insight">
        📊 <b>Annual Projected Benefit: ${annual_benefit:,.0f}</b><br><br>
        Model detects <b>{tp:,}</b> failures of {tp+fn:,} in test set (<b>{tp/(tp+fn):.1%} recall</b>).
        At {failure_rate}% failure rate with {wafers_per_day:,} wafers/day,
        catching {detection_rate:.0%} of flagged failures saves
        <b>${revenue_saved:,.0f}/day</b> net of ${false_alarm_cost:,.0f}/day false alarm costs.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown('<div class="alert-crit">✗ RUN PIPELINE FIRST TO SEE BUSINESS IMPACT</div>', unsafe_allow_html=True)

st.divider()
st.markdown("""
<div style="font-family:'IBM Plex Mono',monospace; font-size:0.65rem; color:#404040; text-align:center; padding:8px;">
SemiSight · Built by <a href="https://github.com/BARA-LUCH" style="color:#f59e0b">Bara Luch</a> ·
ML Engineer & Data Scientist · Expected Apr 2026 ·
<a href="https://linkedin.com/in/bara-luch" style="color:#06b6d4">LinkedIn</a> ·
<a href="https://bara-luch.github.io" style="color:#06b6d4">Portfolio</a>
</div>
""", unsafe_allow_html=True)
