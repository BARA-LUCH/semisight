"""
spc/control_charts.py — SemiSight Statistical Process Control
X-bar charts, R-charts, Cpk, Western Electric rules.
Standard SPC methodology used in semiconductor fabs (SEMI E10, SPC-1).
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


# ── Control Chart Constants ───────────────────────────────────────────────────
# Standard A2, D3, D4 constants for X-bar/R charts by subgroup size
A2 = {2:1.880, 3:1.023, 4:0.729, 5:0.577, 6:0.483, 7:0.419, 8:0.373, 9:0.337, 10:0.308}
D3 = {2:0,     3:0,     4:0,     5:0,     6:0,     7:0.076, 8:0.136, 9:0.184, 10:0.223}
D4 = {2:3.267, 3:2.574, 4:2.282, 5:2.114, 6:2.004, 7:1.924, 8:1.864, 9:1.816, 10:1.777}
d2 = {2:1.128, 3:1.693, 4:2.059, 5:2.326, 6:2.534, 7:2.704, 8:2.847, 9:2.970, 10:3.078}


def compute_xbar_r(values: np.ndarray, subgroup_size: int = 5) -> dict:
    """
    Compute X-bar and R chart statistics.
    Returns control limits, centerlines, subgroup means/ranges.
    """
    values = values[~np.isnan(values)]
    n = len(values)
    if n < subgroup_size * 3:
        return None

    # Split into subgroups
    n_groups = n // subgroup_size
    trimmed  = values[:n_groups * subgroup_size]
    groups   = trimmed.reshape(n_groups, subgroup_size)

    xbars = groups.mean(axis=1)
    ranges = groups.max(axis=1) - groups.min(axis=1)

    xbar_bar = xbars.mean()
    r_bar    = ranges.mean()

    a2 = A2.get(subgroup_size, 0.577)
    d3 = D3.get(subgroup_size, 0)
    d4 = D4.get(subgroup_size, 2.114)

    # X-bar control limits
    ucl_x = xbar_bar + a2 * r_bar
    lcl_x = xbar_bar - a2 * r_bar

    # R chart control limits
    ucl_r = d4 * r_bar
    lcl_r = d3 * r_bar

    return {
        "xbars":     xbars,
        "ranges":    ranges,
        "xbar_bar":  xbar_bar,
        "r_bar":     r_bar,
        "ucl_x":     ucl_x,
        "lcl_x":     lcl_x,
        "ucl_r":     ucl_r,
        "lcl_r":     max(lcl_r, 0),
        "n_groups":  n_groups,
        "sigma_est": r_bar / d2.get(subgroup_size, 2.326),
    }


def compute_cpk(values: np.ndarray, usl: float = None, lsl: float = None,
                sigma: float = None) -> dict:
    """
    Compute process capability indices Cp and Cpk.
    If USL/LSL not provided, use ±3-sigma natural tolerance.
    """
    values = values[~np.isnan(values)]
    mu     = values.mean()
    s      = sigma if sigma else values.std()

    if usl is None:
        usl = mu + 3 * s
    if lsl is None:
        lsl = mu - 3 * s

    cp   = (usl - lsl) / (6 * s) if s > 0 else 0
    cpu  = (usl - mu) / (3 * s)  if s > 0 else 0
    cpl  = (mu - lsl) / (3 * s)  if s > 0 else 0
    cpk  = min(cpu, cpl)

    status = "CAPABLE" if cpk >= 1.33 else "MARGINAL" if cpk >= 1.0 else "INCAPABLE"

    return {
        "mu":    round(mu, 4),
        "sigma": round(s, 4),
        "usl":   round(usl, 4),
        "lsl":   round(lsl, 4),
        "cp":    round(cp, 4),
        "cpk":   round(cpk, 4),
        "cpu":   round(cpu, 4),
        "cpl":   round(cpl, 4),
        "status": status,
    }


def western_electric_violations(xbars: np.ndarray, cl: float,
                                  ucl: float, lcl: float) -> dict:
    """
    Detect Western Electric (WECO) rule violations.
    Standard 8 rules used in semiconductor SPC.
    """
    sigma = (ucl - cl) / 3
    violations = {
        "rule1": [],  # 1 point beyond 3σ
        "rule2": [],  # 2 of 3 beyond 2σ same side
        "rule3": [],  # 4 of 5 beyond 1σ same side
        "rule4": [],  # 8 consecutive same side of CL
        "rule5": [],  # 6 consecutive trending up or down
        "rule6": [],  # 15 consecutive within 1σ
    }
    n = len(xbars)

    for i in range(n):
        x = xbars[i]
        z = (x - cl) / sigma if sigma > 0 else 0

        # Rule 1: Beyond 3σ
        if abs(z) > 3:
            violations["rule1"].append(i)

        # Rule 2: 2 of 3 beyond 2σ same side
        if i >= 2:
            window = [(xbars[j] - cl) / sigma for j in range(i-2, i+1)] if sigma > 0 else [0,0,0]
            above  = sum(1 for w in window if w > 2)
            below  = sum(1 for w in window if w < -2)
            if above >= 2 or below >= 2:
                violations["rule2"].append(i)

        # Rule 4: 8 consecutive same side
        if i >= 7:
            window = xbars[i-7:i+1]
            if all(w > cl for w in window) or all(w < cl for w in window):
                violations["rule4"].append(i)

        # Rule 5: 6 consecutive trending
        if i >= 5:
            window = xbars[i-5:i+1]
            diffs  = np.diff(window)
            if all(d > 0 for d in diffs) or all(d < 0 for d in diffs):
                violations["rule5"].append(i)

    total = sum(len(v) for v in violations.values())
    return {"violations": violations, "total": total,
            "critical": len(violations["rule1"]) > 0}


def run_spc_analysis(df: pd.DataFrame, n_params: int = 10,
                     subgroup_size: int = 5) -> dict:
    """
    Run SPC analysis on top process parameters.
    Selects most variable parameters for charting.
    """
    param_cols = [c for c in df.columns if c.startswith("param_")][:n_params]

    results = {}
    for col in param_cols:
        vals = df[col].dropna().values
        if len(vals) < subgroup_size * 5:
            continue

        xr   = compute_xbar_r(vals, subgroup_size)
        if xr is None:
            continue

        cpk_data = compute_cpk(vals, sigma=xr["sigma_est"])
        weco     = western_electric_violations(
            xr["xbars"], xr["xbar_bar"], xr["ucl_x"], xr["lcl_x"])

        results[col] = {
            "xr":     xr,
            "cpk":    cpk_data,
            "weco":   weco,
            "values": vals,
        }

    return results


def plot_xbar_chart(param_name: str, spc_data: dict):
    """X-bar control chart for a single parameter."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    xr   = spc_data["xr"]
    weco = spc_data["weco"]
    cpk  = spc_data["cpk"]

    xbars   = xr["xbars"]
    n       = len(xbars)
    indices = list(range(n))

    # Color points by violation status
    colors = []
    all_violations = set()
    for v_list in weco["violations"].values():
        all_violations.update(v_list)
    for i in indices:
        if i in weco["violations"]["rule1"]:
            colors.append("#ef4444")   # Red — critical
        elif i in all_violations:
            colors.append("#f59e0b")   # Amber — warning
        else:
            colors.append("#10b981")   # Green — in control

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=["X-bar Chart (Subgroup Means)", "R Chart (Subgroup Ranges)"],
        vertical_spacing=0.15,
    )

    # X-bar chart
    fig.add_trace(go.Scatter(
        x=indices, y=xbars.tolist(),
        mode="lines+markers",
        name="Subgroup Mean",
        line=dict(color="#a3a3a3", width=1),
        marker=dict(color=colors, size=7, line=dict(width=1, color="#0e0e0e")),
        hovertemplate="Subgroup %{x}<br>Mean: %{y:.4f}<extra></extra>",
    ), row=1, col=1)

    # Control limits — X-bar
    for y_val, name, color, dash in [
        (xr["ucl_x"],   "UCL",  "#ef4444", "dash"),
        (xr["xbar_bar"],"CL",   "#f59e0b", "solid"),
        (xr["lcl_x"],   "LCL",  "#ef4444", "dash"),
        (xr["ucl_x"] - (xr["ucl_x"]-xr["xbar_bar"])/3*1, "1σ", "#2a2a2a", "dot"),
        (xr["lcl_x"] + (xr["xbar_bar"]-xr["lcl_x"])/3*1, "-1σ","#2a2a2a", "dot"),
        (xr["ucl_x"] - (xr["ucl_x"]-xr["xbar_bar"])/3*2, "2σ", "#363636", "dot"),
        (xr["lcl_x"] + (xr["xbar_bar"]-xr["lcl_x"])/3*2, "-2σ","#363636", "dot"),
    ]:
        fig.add_hline(y=y_val, line_dash=dash, line_color=color,
                      line_width=1 if "σ" in name else 1.5,
                      annotation_text=name if name in ["UCL","CL","LCL"] else "",
                      annotation_font_color=color,
                      annotation_font_size=10,
                      row=1, col=1)

    # R chart
    fig.add_trace(go.Scatter(
        x=indices, y=xr["ranges"].tolist(),
        mode="lines+markers",
        name="Subgroup Range",
        line=dict(color="#06b6d4", width=1),
        marker=dict(color="#06b6d4", size=5),
        hovertemplate="Subgroup %{x}<br>Range: %{y:.4f}<extra></extra>",
    ), row=2, col=1)

    for y_val, name, color in [
        (xr["ucl_r"],  "UCL", "#ef4444"),
        (xr["r_bar"],  "CL",  "#f59e0b"),
        (xr["lcl_r"],  "LCL", "#ef4444"),
    ]:
        fig.add_hline(y=y_val, line_dash="dash", line_color=color,
                      line_width=1.5,
                      annotation_text=name,
                      annotation_font_color=color,
                      annotation_font_size=10,
                      row=2, col=1)

    # Highlight violations on X-bar
    for idx in weco["violations"]["rule1"]:
        fig.add_vline(x=idx, line_color="rgba(239,68,68,0.3)", line_width=1, row=1, col=1)

    status_color = "#ef4444" if weco["critical"] else "#f59e0b" if weco["total"] > 0 else "#10b981"
    status_text  = "OUT OF CONTROL" if weco["critical"] else f"{weco['total']} VIOLATIONS" if weco["total"] > 0 else "IN CONTROL"

    fig.update_layout(
        title=f"{param_name} — X̄-R Chart &nbsp;|&nbsp; "
              f"<span style='color:{status_color}'>{status_text}</span> &nbsp;|&nbsp; "
              f"Cpk={cpk['cpk']:.3f} ({cpk['status']})",
        height=500,
        plot_bgcolor="#141414",
        paper_bgcolor="#0e0e0e",
        font=dict(family="IBM Plex Mono", color="#a3a3a3", size=11),
        showlegend=False,
        xaxis=dict(gridcolor="#1f1f1f", title="Subgroup #"),
        yaxis=dict(gridcolor="#1f1f1f", title="Mean"),
        xaxis2=dict(gridcolor="#1f1f1f", title="Subgroup #"),
        yaxis2=dict(gridcolor="#1f1f1f", title="Range"),
    )
    return fig


def plot_cpk_dashboard(spc_results: dict):
    """Cpk bar chart for all analyzed parameters."""
    import plotly.graph_objects as go
    params = list(spc_results.keys())
    cpks   = [spc_results[p]["cpk"]["cpk"] for p in params]
    status = [spc_results[p]["cpk"]["status"] for p in params]

    colors = []
    for cpk in cpks:
        if cpk >= 1.33:
            colors.append("#10b981")   # Green — capable
        elif cpk >= 1.0:
            colors.append("#f59e0b")   # Amber — marginal
        else:
            colors.append("#ef4444")   # Red — incapable

    # Short param names
    short_names = [p.replace("param_", "P") for p in params]

    fig = go.Figure(go.Bar(
        x=short_names,
        y=cpks,
        marker_color=colors,
        text=[f"{c:.3f}" for c in cpks],
        textposition="outside",
        textfont=dict(size=10, color="#a3a3a3"),
        hovertemplate="<b>%{x}</b><br>Cpk: %{y:.4f}<extra></extra>",
    ))

    # Reference lines
    fig.add_hline(y=1.33, line_dash="dash", line_color="#10b981",
                  annotation_text="Cpk=1.33 (Capable)",
                  annotation_font_color="#10b981", annotation_font_size=10)
    fig.add_hline(y=1.0,  line_dash="dash", line_color="#f59e0b",
                  annotation_text="Cpk=1.0 (Marginal)",
                  annotation_font_color="#f59e0b", annotation_font_size=10)

    n_cap  = sum(1 for c in cpks if c >= 1.33)
    n_marg = sum(1 for c in cpks if 1.0 <= c < 1.33)
    n_inc  = sum(1 for c in cpks if c < 1.0)

    fig.update_layout(
        title=f"Process Capability (Cpk) — {n_cap} Capable · {n_marg} Marginal · {n_inc} Incapable",
        xaxis_title="Parameter",
        yaxis_title="Cpk",
        height=380,
        plot_bgcolor="#141414",
        paper_bgcolor="#0e0e0e",
        font=dict(family="IBM Plex Mono", color="#a3a3a3", size=11),
        xaxis=dict(gridcolor="#1f1f1f"),
        yaxis=dict(gridcolor="#1f1f1f", range=[0, max(max(cpks)*1.2, 1.5)]),
    )
    return fig


def plot_violation_heatmap(spc_results: dict):
    """Heatmap of WECO rule violations per parameter."""
    import plotly.graph_objects as go
    params     = list(spc_results.keys())
    rule_names = ["Rule 1\n(3σ)", "Rule 2\n(2/3>2σ)", "Rule 4\n(8 run)",
                  "Rule 5\n(trend)", "Rule 6\n(15 in 1σ)"]
    rule_keys  = ["rule1", "rule2", "rule4", "rule5", "rule6"]

    matrix = []
    for p in params:
        weco = spc_results[p]["weco"]["violations"]
        row  = [len(weco.get(k, [])) for k in rule_keys]
        matrix.append(row)

    matrix = np.array(matrix)
    short  = [p.replace("param_", "P") for p in params]

    fig = go.Figure(go.Heatmap(
        z=matrix,
        x=rule_names,
        y=short,
        colorscale=[[0,"#141414"],[0.3,"#d97706"],[1,"#ef4444"]],
        text=matrix,
        texttemplate="%{text}",
        showscale=True,
        hovertemplate="<b>%{y}</b><br>%{x}: %{z} violations<extra></extra>",
    ))
    fig.update_layout(
        title="Western Electric Rule Violations Heatmap",
        height=max(300, len(params) * 28 + 100),
        plot_bgcolor="#141414",
        paper_bgcolor="#0e0e0e",
        font=dict(family="IBM Plex Mono", color="#a3a3a3", size=11),
        xaxis=dict(side="top"),
    )
    return fig


def get_spc_summary(spc_results: dict) -> dict:
    """Summary stats across all SPC results."""
    cpks      = [v["cpk"]["cpk"] for v in spc_results.values()]
    total_vio = sum(v["weco"]["total"] for v in spc_results.values())
    n_crit    = sum(1 for v in spc_results.values() if v["weco"]["critical"])
    n_incap   = sum(1 for c in cpks if c < 1.0)
    n_marg    = sum(1 for c in cpks if 1.0 <= c < 1.33)
    n_cap     = sum(1 for c in cpks if c >= 1.33)

    return {
        "n_params":       len(spc_results),
        "avg_cpk":        round(np.mean(cpks), 3),
        "min_cpk":        round(min(cpks), 3),
        "total_violations": total_vio,
        "critical_params":  n_crit,
        "capable":          n_cap,
        "marginal":         n_marg,
        "incapable":        n_incap,
    }
