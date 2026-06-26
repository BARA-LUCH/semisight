"""
explainability/shap_analysis.py — SemiSight Root Cause Analysis
SHAP-based explainability for yield failure root cause ranking.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import warnings
warnings.filterwarnings("ignore")


def get_shap_values(model_result: dict, X_sample: np.ndarray,
                    feature_names: list, sample_size: int = 200) -> dict:
    """Compute SHAP values for yield predictor."""
    try:
        import shap
        model  = model_result["model"]
        n      = min(sample_size, len(X_sample))
        X_sub  = X_sample[:n]

        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sub)

        # For binary classification, shap_values may be list
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        return {
            "success":      True,
            "shap_values":  shap_values,
            "X_sample":     X_sub,
            "feature_names": feature_names,
            "expected_value": float(explainer.expected_value
                                    if not isinstance(explainer.expected_value, list)
                                    else explainer.expected_value[1]),
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def plot_feature_importance(shap_data: dict, top_n: int = 20) -> go.Figure:
    """Bar chart of top features by mean |SHAP|."""
    shap_values  = shap_data["shap_values"]
    feature_names = shap_data["feature_names"]

    mean_shap = np.abs(shap_values).mean(axis=0)
    top_idx   = np.argsort(mean_shap)[::-1][:top_n]

    names  = [feature_names[i] for i in top_idx]
    values = [mean_shap[i] for i in top_idx]

    # Color by process step
    colors = []
    step_colors = {
        "deposition": "#f59e0b",
        "etch":       "#06b6d4",
        "lithography": "#10b981",
        "cmp":        "#a78bfa",
        "inspection": "#ef4444",
        "param":      "#404040",
    }
    for name in names:
        color = "#4a6880"
        for step, c in step_colors.items():
            if step in name.lower():
                color = c
                break
        colors.append(color)

    fig = go.Figure(go.Bar(
        x=values[::-1], y=names[::-1],
        orientation="h",
        marker_color=colors[::-1],
        hovertemplate="<b>%{y}</b><br>Mean |SHAP|: %{x:.4f}<extra></extra>",
    ))
    fig.update_layout(
        title="Top Process Parameters Driving Yield Failures",
        xaxis_title="Mean |SHAP Value| (Impact on Failure Prediction)",
        height=500,
        plot_bgcolor="#141414",
        paper_bgcolor="#0e0e0e",
        font_color="#a3a3a3",
        xaxis=dict(gridcolor="#1f1f1f"),
        yaxis=dict(gridcolor="#1f1f1f"),
    )
    return fig


def plot_waterfall(shap_data: dict, run_idx: int = 0) -> go.Figure:
    """Waterfall chart for a single run's SHAP explanation."""
    shap_values   = shap_data["shap_values"]
    feature_names = shap_data["feature_names"]
    expected      = shap_data["expected_value"]

    sv = shap_values[run_idx]

    # Take top 12 by absolute value
    top_idx = np.argsort(np.abs(sv))[::-1][:12]
    names   = [feature_names[i] for i in top_idx]
    values  = sv[top_idx]

    colors = ["#ef4444" if v > 0 else "#10b981" for v in values]

    fig = go.Figure(go.Waterfall(
        orientation="h",
        measure=["relative"] * len(values) + ["total"],
        x=list(values) + [sum(values)],
        y=names + ["Final Prediction"],
        connector={"line": {"color": "#2a2a2a", "width": 1}},
        increasing={"marker": {"color": "#ef4444"}},
        decreasing={"marker": {"color": "#10b981"}},
        totals={"marker": {"color": "#f59e0b"}},
        hovertemplate="<b>%{y}</b><br>SHAP: %{x:.4f}<extra></extra>",
    ))
    fig.update_layout(
        title=f"Run #{run_idx} — Failure Risk Breakdown",
        xaxis_title="SHAP Value (← Reduces Risk | Increases Risk →)",
        height=450,
        plot_bgcolor="#141414",
        paper_bgcolor="#0e0e0e",
        font_color="#a3a3a3",
        xaxis=dict(gridcolor="#1f1f1f"),
    )
    return fig


def plot_yield_by_step(sql_results: dict) -> go.Figure:
    """Bar chart of failure rate by process step."""
    df = sql_results.get("yield_by_step", pd.DataFrame())
    if df.empty:
        return go.Figure()

    fig = go.Figure(go.Bar(
        x=df["process_step"],
        y=df["failure_pct"],
        marker_color=["#ef4444" if v > df["failure_pct"].mean() else "#f59e0b"
                      for v in df["failure_pct"]],
        hovertemplate="<b>%{x}</b><br>Failure Rate: %{y:.2f}%<extra></extra>",
    ))
    fig.add_hline(y=df["failure_pct"].mean(), line_dash="dash",
                  line_color="#737373", annotation_text="Average",
                  annotation_font_color="#737373")
    fig.update_layout(
        title="Failure Rate by Process Step",
        xaxis_title="Process Step",
        yaxis_title="Failure Rate (%)",
        height=350,
        plot_bgcolor="#141414",
        paper_bgcolor="#0e0e0e",
        font_color="#a3a3a3",
        xaxis=dict(gridcolor="#1f1f1f"),
        yaxis=dict(gridcolor="#1f1f1f"),
    )
    return fig


def plot_chamber_comparison(sql_results: dict) -> go.Figure:
    """Compare yield across chambers."""
    df = sql_results.get("chamber_yield", pd.DataFrame())
    if df.empty:
        return go.Figure()

    fig = go.Figure(go.Bar(
        x=df["chamber_id"],
        y=df["failure_pct"],
        marker_color="#06b6d4",
        hovertemplate="<b>Chamber %{x}</b><br>Failure: %{y:.2f}%<extra></extra>",
    ))
    fig.update_layout(
        title="Failure Rate by Chamber",
        xaxis_title="Chamber ID",
        yaxis_title="Failure Rate (%)",
        height=300,
        plot_bgcolor="#141414",
        paper_bgcolor="#0e0e0e",
        font_color="#a3a3a3",
        xaxis=dict(gridcolor="#1f1f1f"),
        yaxis=dict(gridcolor="#1f1f1f"),
    )
    return fig


def plot_weekly_trend(sql_results: dict) -> go.Figure:
    """Weekly yield trend line."""
    df = sql_results.get("weekly_trend", pd.DataFrame())
    if df.empty:
        return go.Figure()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["week"], y=df["yield_pct"],
        mode="lines+markers",
        name="Weekly Yield %",
        line=dict(color="#f59e0b", width=2),
        marker=dict(size=6, color="#f59e0b"),
    ))
    fig.add_hline(y=df["yield_pct"].mean(), line_dash="dash",
                  line_color="#737373", annotation_text="Average",
                  annotation_font_color="#737373")
    fig.update_layout(
        title="Weekly Yield Rate Trend",
        xaxis_title="Week",
        yaxis_title="Yield Rate (%)",
        height=300,
        plot_bgcolor="#141414",
        paper_bgcolor="#0e0e0e",
        font_color="#a3a3a3",
        xaxis=dict(gridcolor="#1f1f1f"),
        yaxis=dict(gridcolor="#1f1f1f"),
    )
    return fig


def plot_wafer_defect_distribution(wafer_df: pd.DataFrame) -> go.Figure:
    """Pie chart of wafer defect class distribution."""
    counts = wafer_df["defect_class"].value_counts()
    colors = ["#f59e0b", "#06b6d4", "#10b981", "#ef4444",
              "#a78bfa", "#fbbf24", "#34d399", "#f87171", "#404040"]

    fig = go.Figure(go.Pie(
        labels=counts.index,
        values=counts.values,
        marker_colors=colors[:len(counts)],
        hole=0.4,
        hovertemplate="<b>%{label}</b><br>Count: %{value}<br>%{percent}<extra></extra>",
    ))
    fig.update_layout(
        title="Wafer Defect Pattern Distribution",
        height=350,
        paper_bgcolor="#0e0e0e",
        font_color="#a3a3a3",
    )
    return fig


def get_yield_insight(results: dict, sql_results: dict) -> str:
    """Generate natural language insight from model results."""
    xgb_res = results.get("xgboost", {})
    if not xgb_res.get("success"):
        return "Model training failed — check data."

    auc    = xgb_res["metrics"]["ROC-AUC"]
    recall = xgb_res["metrics"]["Recall"]
    prec   = xgb_res["metrics"]["Precision"]

    # Find worst process step
    step_df = sql_results.get("yield_by_step", pd.DataFrame())
    worst_step = step_df.iloc[0]["process_step"] if not step_df.empty else "Unknown"
    worst_rate = step_df.iloc[0]["failure_pct"] if not step_df.empty else 0

    # Find worst chamber
    chamber_df = sql_results.get("chamber_yield", pd.DataFrame())
    worst_chamber = chamber_df.iloc[0]["chamber_id"] if not chamber_df.empty else "Unknown"

    insight = (
        f"XGBoost achieved ROC-AUC {auc:.3f} on imbalanced yield data "
        f"(Recall {recall:.1%}, Precision {prec:.1%}). "
        f"Highest failure rate in {worst_step} step ({worst_rate:.1f}%). "
        f"Chamber {worst_chamber} shows worst performance. "
        f"SHAP analysis identifies top process parameters — "
        f"address these to drive yield improvement."
    )
    return insight
