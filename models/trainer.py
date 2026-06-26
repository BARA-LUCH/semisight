"""
models/trainer.py — SemiSight ML Models
1. XGBoost Yield Predictor (imbalanced classification)
2. Isolation Forest Anomaly Detector
3. Wafer Defect Classifier (XGBoost multi-class)
4. Ensemble yield predictor
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    accuracy_score, confusion_matrix, classification_report,
    roc_curve, average_precision_score
)
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")


def compute_metrics(y_true, y_pred, y_prob) -> dict:
    """Compute full suite of classification metrics."""
    try:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        return {
            "Accuracy":        round(float(accuracy_score(y_true, y_pred)), 4),
            "F1 Score":        round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
            "Precision":       round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
            "Recall":          round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
            "ROC-AUC":         round(float(roc_auc_score(y_true, y_prob)), 4),
            "Avg Precision":   round(float(average_precision_score(y_true, y_prob)), 4),
            "Confusion Matrix": confusion_matrix(y_true, y_pred).tolist(),
            "ROC Curve":       {"fpr": fpr.tolist()[:500], "tpr": tpr.tolist()[:500]},
        }
    except Exception as e:
        return {"Error": str(e), "ROC-AUC": 0.5, "F1 Score": 0,
                "Accuracy": 0, "Precision": 0, "Recall": 0}


def train_yield_xgboost(X_train, X_test, y_train, y_test, feature_names) -> dict:
    """
    XGBoost yield predictor with scale_pos_weight for imbalanced data.
    Semiconductor yield data is highly imbalanced (~6.5% failure rate).
    """
    print("🤖 Training XGBoost Yield Predictor...")

    # Handle class imbalance
    n_pass = (y_train == 1).sum()
    n_fail = (y_train == 0).sum()
    scale_pos_weight = n_pass / max(n_fail, 1)

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="auc",
        random_state=42,
        n_jobs=-1,
        use_label_encoder=False,
    )

    # 5-fold stratified CV
    n_splits = min(5, int(min((y_train==0).sum(), (y_train==1).sum())))
    cv = StratifiedKFold(n_splits=max(2, n_splits), shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")

    model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              verbose=False)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test, y_pred, y_prob)
    metrics["CV ROC-AUC"] = round(float(cv_scores.mean()), 4)
    metrics["CV Std"]     = round(float(cv_scores.std()), 4)

    # Feature importance
    importance = dict(zip(feature_names, model.feature_importances_))
    top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:20]

    print(f"✅ XGBoost ROC-AUC: {metrics['ROC-AUC']:.4f} | CV: {metrics['CV ROC-AUC']:.4f} ± {metrics['CV Std']:.4f}")
    return {
        "name":        "XGBoost Yield Predictor",
        "model":       model,
        "metrics":     metrics,
        "cv_scores":   cv_scores.tolist(),
        "top_features": top_features,
        "y_pred":      y_pred,
        "y_prob":      y_prob,
        "success":     True,
    }


def train_isolation_forest(X_train, X_test, y_test) -> dict:
    """
    Isolation Forest for unsupervised anomaly detection.
    Detects process parameter anomalies without using yield labels.
    """
    print("🤖 Training Isolation Forest Anomaly Detector...")

    # Train only on passing runs (normal process)
    model = IsolationForest(
        n_estimators=200,
        contamination=0.065,  # ~6.5% expected anomaly rate
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train)

    # Predict: -1 = anomaly (failure), 1 = normal (pass)
    raw_pred  = model.predict(X_test)
    y_pred    = (raw_pred == -1).astype(int)  # 1 = predicted failure
    y_scores  = -model.score_samples(X_test)  # Higher = more anomalous

    # Normalize scores to [0, 1]
    y_scores = (y_scores - y_scores.min()) / (y_scores.max() - y_scores.min() + 1e-8)

    metrics = compute_metrics(y_test, y_pred, y_scores)

    print(f"✅ Isolation Forest ROC-AUC: {metrics['ROC-AUC']:.4f}")
    return {
        "name":    "Isolation Forest",
        "model":   model,
        "metrics": metrics,
        "y_pred":  y_pred,
        "y_prob":  y_scores,
        "success": True,
    }


def train_logistic_baseline(X_train, X_test, y_train, y_test) -> dict:
    """Logistic Regression baseline for comparison."""
    print("🤖 Training Logistic Regression Baseline...")

    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
        C=0.1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test, y_pred, y_prob)

    print(f"✅ Logistic Regression ROC-AUC: {metrics['ROC-AUC']:.4f}")
    return {
        "name":    "Logistic Regression",
        "model":   model,
        "metrics": metrics,
        "y_pred":  y_pred,
        "y_prob":  y_prob,
        "success": True,
    }


def train_wafer_classifier(X_train, X_test, y_train, y_test, class_names) -> dict:
    """XGBoost multi-class wafer defect pattern classifier."""
    print("🤖 Training Wafer Defect Classifier...")

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        use_label_encoder=False,
        eval_metric="mlogloss",
    )
    model.fit(X_train, y_train, verbose=False)

    y_pred = model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)

    print(f"✅ Wafer Classifier Accuracy: {acc:.4f}")
    return {
        "name":        "Wafer Defect Classifier",
        "model":       model,
        "accuracy":    round(acc, 4),
        "report":      report,
        "y_pred":      y_pred,
        "class_names": class_names,
        "success":     True,
    }


def train_all_models(X_train, X_test, y_train, y_test, feature_names,
                     X_wafer_train=None, X_wafer_test=None,
                     y_wafer_train=None, y_wafer_test=None,
                     wafer_classes=None) -> dict:
    """Train all models and return results dict."""
    from sklearn.model_selection import train_test_split

    results = {}
    successful = []

    # 1. XGBoost Yield Predictor
    try:
        results["xgboost"] = train_yield_xgboost(
            X_train, X_test, y_train, y_test, feature_names)
        successful.append("XGBoost")
    except Exception as e:
        results["xgboost"] = {"success": False, "error": str(e)}
        print(f"❌ XGBoost failed: {e}")

    # 2. Isolation Forest
    try:
        # Train IF only on normal (passing) runs
        X_normal = X_train[y_train == 1]
        results["isolation_forest"] = train_isolation_forest(
            X_normal, X_test, y_test)
        successful.append("Isolation Forest")
    except Exception as e:
        results["isolation_forest"] = {"success": False, "error": str(e)}
        print(f"❌ Isolation Forest failed: {e}")

    # 3. Logistic Regression baseline
    try:
        results["logistic"] = train_logistic_baseline(
            X_train, X_test, y_train, y_test)
        successful.append("Logistic Regression")
    except Exception as e:
        results["logistic"] = {"success": False, "error": str(e)}

    # 4. Wafer classifier (if data provided)
    if X_wafer_train is not None:
        try:
            results["wafer_classifier"] = train_wafer_classifier(
                X_wafer_train, X_wafer_test,
                y_wafer_train, y_wafer_test,
                wafer_classes)
            successful.append("Wafer Classifier")
        except Exception as e:
            results["wafer_classifier"] = {"success": False, "error": str(e)}

    # Build comparison table
    comparison_rows = []
    for key in ["xgboost", "logistic", "isolation_forest"]:
        r = results.get(key, {})
        if r.get("success"):
            m = r["metrics"]
            comparison_rows.append({
                "Model":       r["name"],
                "ROC-AUC":     m.get("ROC-AUC", 0),
                "F1 Score":    m.get("F1 Score", 0),
                "Precision":   m.get("Precision", 0),
                "Recall":      m.get("Recall", 0),
                "Accuracy":    m.get("Accuracy", 0),
            })
    results["comparison"]        = pd.DataFrame(comparison_rows).set_index("Model") if comparison_rows else pd.DataFrame()
    results["successful_models"] = successful
    results["train_size"]        = len(X_train)
    results["test_size"]         = len(X_test)
    results["fail_in_test"]      = int((y_test == 0).sum())

    return results
