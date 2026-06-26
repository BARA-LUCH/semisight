"""
features/engineer.py — SemiSight Feature Engineering
Handles missing values, feature selection, and process parameter engineering.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
import warnings
warnings.filterwarnings("ignore")


def prepare_secom_features(df: pd.DataFrame, variance_threshold: float = 0.01):
    """
    Full feature engineering pipeline for SECOM data:
    1. Extract process parameters
    2. Remove near-zero variance features
    3. Impute missing values (median)
    4. Scale features
    5. Engineer interaction features
    Returns X, y, feature_names, scaler
    """
    param_cols = [c for c in df.columns if c.startswith("param_")]
    X_raw = df[param_cols].copy()
    y = df["yield_pass"].values

    print(f"📊 Raw features: {X_raw.shape[1]}")

    # Step 1: Remove columns with >80% missing
    missing_pct = X_raw.isnull().mean()
    keep_cols   = missing_pct[missing_pct < 0.80].index.tolist()
    X_raw = X_raw[keep_cols]
    print(f"📊 After removing high-missing columns: {X_raw.shape[1]}")

    # Step 2: Impute missing values
    imputer = SimpleImputer(strategy="median")
    X_imp   = imputer.fit_transform(X_raw)
    X_imp   = pd.DataFrame(X_imp, columns=keep_cols)

    # Step 3: Remove near-zero variance features
    selector = VarianceThreshold(threshold=variance_threshold)
    X_sel    = selector.fit_transform(X_imp)
    sel_cols = [keep_cols[i] for i in range(len(keep_cols)) if selector.get_support()[i]]
    X_sel    = pd.DataFrame(X_sel, columns=sel_cols)
    print(f"📊 After variance filtering: {X_sel.shape[1]}")

    # Step 4: Engineer aggregate features per process step
    # Group parameters into 5 process steps
    n_params  = len(sel_cols)
    step_size = n_params // 5
    step_names = ["deposition", "etch", "lithography", "cmp", "inspection"]

    engineered = {}
    for s, name in enumerate(step_names):
        start = s * step_size
        end   = min(start + step_size, n_params)
        step_cols = sel_cols[start:end]
        if step_cols:
            engineered[f"{name}_mean"] = X_sel[step_cols].mean(axis=1)
            engineered[f"{name}_std"]  = X_sel[step_cols].std(axis=1)
            engineered[f"{name}_max"]  = X_sel[step_cols].max(axis=1)
            engineered[f"{name}_min"]  = X_sel[step_cols].min(axis=1)
            engineered[f"{name}_range"] = X_sel[step_cols].max(axis=1) - X_sel[step_cols].min(axis=1)

    eng_df = pd.DataFrame(engineered)
    X_final = pd.concat([X_sel, eng_df], axis=1)
    print(f"📊 After feature engineering: {X_final.shape[1]}")

    # Step 5: Scale
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_final)
    X_scaled = pd.DataFrame(X_scaled, columns=X_final.columns)

    return X_scaled.values, y, list(X_final.columns), scaler, imputer, selector, keep_cols


def prepare_wafer_features(df: pd.DataFrame):
    """Prepare wafer map features for classification."""
    feature_cols = [
        "defect_ratio", "n_defects", "mean_dist_center", "std_dist_center",
        "max_dist_center", "min_dist_center", "edge_defect_ratio",
        "quadrant_std", "q1_defects", "q2_defects", "q3_defects", "q4_defects",
        "center_defect_ratio", "defect_density"
    ]
    X = df[feature_cols].fillna(0).values
    y = df["class_idx"].values.astype(int)
    y_binary = df["is_defective"].values.astype(int)

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, y_binary, feature_cols, scaler


def get_top_failure_parameters(feature_names: list, shap_values: np.ndarray,
                                top_n: int = 20) -> pd.DataFrame:
    """Get top parameters by mean absolute SHAP value."""
    mean_shap = np.abs(shap_values).mean(axis=0)
    df = pd.DataFrame({
        "parameter":   feature_names,
        "importance":  mean_shap,
        "process_step": [_get_process_step(f) for f in feature_names],
    })
    return df.sort_values("importance", ascending=False).head(top_n).reset_index(drop=True)


def _get_process_step(param_name: str) -> str:
    """Map parameter name to process step."""
    if any(x in param_name for x in ["deposition", "dep"]):
        return "Deposition"
    elif "etch" in param_name:
        return "Etch"
    elif "litho" in param_name or "lithography" in param_name:
        return "Lithography"
    elif "cmp" in param_name:
        return "CMP"
    elif "insp" in param_name or "inspection" in param_name:
        return "Inspection"
    else:
        # Infer from param number
        try:
            num = int(param_name.split("_")[-1])
            steps = ["Deposition", "Etch", "Lithography", "CMP", "Inspection"]
            return steps[min(num // 120, 4)]
        except:
            return "Process"
