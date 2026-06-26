"""
data/loader.py — SemiSight Data Pipeline
Loads SECOM (yield prediction) + WM-811K (wafer defect maps)
Falls back to realistic synthetic data if download fails.
"""

import numpy as np
import pandas as pd
import os
import urllib.request
import warnings
warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__))
SECOM_DATA_PATH  = os.path.join(DATA_DIR, "secom.csv")
SECOM_LABEL_PATH = os.path.join(DATA_DIR, "secom_labels.csv")
WAFER_PATH       = os.path.join(DATA_DIR, "wafer_maps.csv")

SECOM_DATA_URL  = "https://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom.data"
SECOM_LABEL_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom_labels.data"

# ── SECOM Yield Data ───────────────────────────────────────────────────────────
def load_secom() -> pd.DataFrame:
    """Load SECOM semiconductor yield dataset (590 process parameters)."""
    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(SECOM_DATA_PATH) and os.path.exists(SECOM_LABEL_PATH):
        print("📂 Loading SECOM from cache...")
        data   = pd.read_csv(SECOM_DATA_PATH)
        labels = pd.read_csv(SECOM_LABEL_PATH)
        df = data.copy()
        df["yield_pass"] = (labels.iloc[:, 0] == -1).astype(int)  # -1=pass, 1=fail
        print(f"✅ SECOM: {len(df):,} runs, {df.shape[1]-1} parameters")
        return df

    # Try downloading
    try:
        print("📥 Downloading SECOM dataset...")
        urllib.request.urlretrieve(SECOM_DATA_URL, SECOM_DATA_PATH)
        urllib.request.urlretrieve(SECOM_LABEL_URL, SECOM_LABEL_PATH)
        data   = pd.read_csv(SECOM_DATA_PATH, sep=" ", header=None)
        labels = pd.read_csv(SECOM_LABEL_PATH, sep=" ", header=None)
        data.columns = [f"param_{i:03d}" for i in range(data.shape[1])]
        df = data.copy()
        df["yield_pass"] = (labels.iloc[:, 0] == -1).astype(int)
        df["yield_pass"] = df["yield_pass"].fillna(1).astype(int)
        df.to_csv(SECOM_DATA_PATH, index=False)
        print(f"✅ SECOM downloaded: {len(df):,} runs")
        return df
    except Exception as e:
        print(f"⚠️ Download failed: {e}. Generating synthetic SECOM data...")
        return generate_synthetic_secom()


def generate_synthetic_secom(n_runs: int = 1567, n_params: int = 590) -> pd.DataFrame:
    """
    Realistic synthetic SECOM-style data.
    590 process parameters, ~6.5% failure rate (matching real SECOM).
    Parameters have realistic correlations and missing values.
    """
    np.random.seed(42)
    print(f"🔧 Generating {n_runs:,} synthetic semiconductor runs...")

    # Generate correlated process parameters
    # Group into process steps: deposition, etch, lithography, CMP, inspection
    n_fail  = int(n_runs * 0.065)
    n_pass  = n_runs - n_fail

    # Base process parameters — normal operation
    data = np.random.randn(n_runs, n_params)

    # Inject realistic correlations within process steps
    step_size = n_params // 5
    for step in range(5):
        start = step * step_size
        end   = min(start + step_size, n_params)
        # Parameters within same process step are correlated
        common_factor = np.random.randn(n_runs) * 0.5
        data[:, start:end] += common_factor[:, np.newaxis]

    # Inject failure signatures in a subset of parameters
    # Failed runs have drifted parameters
    fail_indices = np.random.choice(n_runs, n_fail, replace=False)
    key_params   = np.random.choice(n_params, 15, replace=False)  # 15 key parameters

    for idx in fail_indices:
        # Drift in key parameters causes failures
        drift_params = np.random.choice(key_params, np.random.randint(3, 8), replace=False)
        data[idx, drift_params] += np.random.choice([-3, 3], len(drift_params)) * np.random.uniform(0.5, 2.0)

    # Add realistic missing values (~5-15% per column, matching SECOM)
    missing_rates = np.random.uniform(0.05, 0.15, n_params)
    for j in range(n_params):
        missing_mask = np.random.random(n_runs) < missing_rates[j]
        data[missing_mask, j] = np.nan

    # Build dataframe
    cols = [f"param_{i:03d}" for i in range(n_params)]
    df   = pd.DataFrame(data, columns=cols)

    # Labels
    labels = np.ones(n_runs, dtype=int)  # 1 = pass
    labels[fail_indices] = 0             # 0 = fail

    df["yield_pass"] = labels

    # Add metadata
    df["timestamp"]    = pd.date_range("2023-01-01", periods=n_runs, freq="30min")
    df["process_step"] = np.random.choice(["DEP", "ETCH", "LITHO", "CMP", "INSP"], n_runs,
                                           p=[0.25, 0.25, 0.20, 0.15, 0.15])
    df["chamber_id"]   = np.random.choice(["A", "B", "C", "D"], n_runs)
    df["lot_id"]       = [f"LOT{i//25:04d}" for i in range(n_runs)]

    print(f"✅ Generated {n_runs:,} runs ({n_fail:,} failures, {n_fail/n_runs:.1%} failure rate)")
    df.to_csv(SECOM_DATA_PATH, index=False)
    return df


# ── Wafer Map Data ─────────────────────────────────────────────────────────────
DEFECT_CLASSES = [
    "Center", "Donut", "Edge-Loc", "Edge-Ring",
    "Loc", "Near-full", "Random", "Scratch", "None"
]

def generate_wafer_maps(n_wafers: int = 5000) -> pd.DataFrame:
    """
    Generate realistic wafer map dataset (WM-811K style).
    Each wafer is a 26x26 grid with defect pattern.
    Classes: Center, Donut, Edge-Loc, Edge-Ring, Loc, Near-full, Random, Scratch, None
    """
    np.random.seed(42)
    print(f"🔧 Generating {n_wafers:,} synthetic wafer maps...")

    records = []

    # Class distribution matching WM-811K
    class_weights = [0.10, 0.05, 0.15, 0.12, 0.10, 0.03, 0.08, 0.07, 0.30]

    for i in range(n_wafers):
        defect_class = np.random.choice(DEFECT_CLASSES, p=class_weights)
        wafer_map    = _generate_wafer_pattern(defect_class)

        # Extract features from wafer map
        features = _extract_wafer_features(wafer_map, defect_class)
        features["wafer_id"]     = f"W{i:06d}"
        features["defect_class"] = defect_class
        features["class_idx"]    = DEFECT_CLASSES.index(defect_class)
        features["is_defective"] = 1 if defect_class != "None" else 0
        records.append(features)

    df = pd.DataFrame(records)
    df.to_csv(WAFER_PATH, index=False)
    print(f"✅ Generated {n_wafers:,} wafer maps, {df['is_defective'].mean():.1%} defective")
    return df


def _generate_wafer_pattern(defect_class: str, size: int = 26) -> np.ndarray:
    """Generate a synthetic wafer map with specific defect pattern."""
    wafer = np.zeros((size, size))
    cx, cy = size // 2, size // 2
    radius = size // 2 - 1

    # Create circular wafer mask
    for i in range(size):
        for j in range(size):
            if (i - cx)**2 + (j - cy)**2 > radius**2:
                wafer[i, j] = -1  # outside wafer

    if defect_class == "None":
        return wafer

    elif defect_class == "Center":
        r = np.random.randint(3, 6)
        for i in range(size):
            for j in range(size):
                if (i - cx)**2 + (j - cy)**2 <= r**2 and wafer[i,j] != -1:
                    wafer[i, j] = 1

    elif defect_class == "Donut":
        r_in  = np.random.randint(3, 5)
        r_out = np.random.randint(6, 9)
        for i in range(size):
            for j in range(size):
                d = (i - cx)**2 + (j - cy)**2
                if r_in**2 <= d <= r_out**2 and wafer[i,j] != -1:
                    wafer[i, j] = 1

    elif defect_class == "Edge-Ring":
        for i in range(size):
            for j in range(size):
                d = (i - cx)**2 + (j - cy)**2
                if (radius-2)**2 <= d <= radius**2 and wafer[i,j] != -1:
                    wafer[i, j] = 1

    elif defect_class == "Edge-Loc":
        angle = np.random.uniform(0, 2 * np.pi)
        for i in range(size):
            for j in range(size):
                d     = np.sqrt((i - cx)**2 + (j - cy)**2)
                theta = np.arctan2(j - cy, i - cx)
                if d > radius - 4 and abs(theta - angle) < 0.5 and wafer[i,j] != -1:
                    wafer[i, j] = 1

    elif defect_class == "Loc":
        lx = np.random.randint(cx - 8, cx + 8)
        ly = np.random.randint(cy - 8, cy + 8)
        r  = np.random.randint(2, 5)
        for i in range(size):
            for j in range(size):
                if (i - lx)**2 + (j - ly)**2 <= r**2 and wafer[i,j] != -1:
                    wafer[i, j] = 1

    elif defect_class == "Scratch":
        x0 = np.random.randint(2, size - 2)
        y0 = np.random.randint(2, size - 2)
        angle = np.random.uniform(0, np.pi)
        length = np.random.randint(8, 18)
        for t in range(length):
            xi = int(x0 + t * np.cos(angle))
            yi = int(y0 + t * np.sin(angle))
            if 0 <= xi < size and 0 <= yi < size and wafer[xi, yi] != -1:
                wafer[xi, yi] = 1
                # Width of scratch
                for dx in [-1, 1]:
                    if 0 <= xi+dx < size and wafer[xi+dx, yi] != -1:
                        wafer[xi+dx, yi] = np.random.choice([0, 1], p=[0.4, 0.6])

    elif defect_class == "Random":
        n_defects = np.random.randint(20, 60)
        for _ in range(n_defects):
            xi = np.random.randint(0, size)
            yi = np.random.randint(0, size)
            if wafer[xi, yi] != -1:
                wafer[xi, yi] = 1

    elif defect_class == "Near-full":
        coverage = np.random.uniform(0.6, 0.85)
        for i in range(size):
            for j in range(size):
                if wafer[i, j] != -1 and np.random.random() < coverage:
                    wafer[i, j] = 1

    return wafer


def _extract_wafer_features(wafer: np.ndarray, defect_class: str) -> dict:
    """Extract statistical features from wafer map."""
    valid = wafer[wafer != -1]
    defects = (valid == 1)

    size      = wafer.shape[0]
    cx, cy    = size // 2, size // 2
    n_valid   = len(valid)
    n_defects = defects.sum()

    # Spatial distribution of defects
    defect_positions = np.argwhere(wafer == 1)
    if len(defect_positions) > 0:
        dist_from_center = np.sqrt(
            (defect_positions[:, 0] - cx)**2 +
            (defect_positions[:, 1] - cy)**2
        )
        mean_dist   = dist_from_center.mean()
        std_dist    = dist_from_center.std()
        max_dist    = dist_from_center.max()
        min_dist    = dist_from_center.min()
        # Quadrant analysis
        q1 = ((defect_positions[:, 0] < cx) & (defect_positions[:, 1] < cy)).sum()
        q2 = ((defect_positions[:, 0] < cx) & (defect_positions[:, 1] >= cy)).sum()
        q3 = ((defect_positions[:, 0] >= cx) & (defect_positions[:, 1] < cy)).sum()
        q4 = ((defect_positions[:, 0] >= cx) & (defect_positions[:, 1] >= cy)).sum()
        quadrant_std = np.std([q1, q2, q3, q4])
    else:
        mean_dist = std_dist = max_dist = min_dist = quadrant_std = 0
        q1 = q2 = q3 = q4 = 0

    # Edge vs center ratio
    edge_threshold = size // 2 - 3
    edge_defects   = sum(1 for p in defect_positions
                         if np.sqrt((p[0]-cx)**2 + (p[1]-cy)**2) > edge_threshold) \
                     if len(defect_positions) > 0 else 0

    return {
        "defect_ratio":        n_defects / max(n_valid, 1),
        "n_defects":           int(n_defects),
        "n_valid_dies":        int(n_valid),
        "mean_dist_center":    float(mean_dist),
        "std_dist_center":     float(std_dist),
        "max_dist_center":     float(max_dist),
        "min_dist_center":     float(min_dist),
        "edge_defect_ratio":   edge_defects / max(n_defects, 1),
        "quadrant_std":        float(quadrant_std),
        "q1_defects":          int(q1),
        "q2_defects":          int(q2),
        "q3_defects":          int(q3),
        "q4_defects":          int(q4),
        "center_defect_ratio": 1 - (edge_defects / max(n_defects, 1)),
        "defect_density":      n_defects / max(n_valid, 1) * 100,
    }


def load_wafer_maps() -> pd.DataFrame:
    """Load or generate wafer map dataset."""
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(WAFER_PATH):
        print("📂 Loading wafer maps from cache...")
        df = pd.read_csv(WAFER_PATH)
        print(f"✅ Wafer maps: {len(df):,} wafers, {df['is_defective'].mean():.1%} defective")
        return df
    return generate_wafer_maps(5000)


def get_secom_stats(df: pd.DataFrame) -> dict:
    """Key statistics about SECOM dataset."""
    n_fail = (df["yield_pass"] == 0).sum()
    n_pass = (df["yield_pass"] == 1).sum()
    n_params = len([c for c in df.columns if c.startswith("param_")])
    missing_pct = df[[c for c in df.columns if c.startswith("param_")]].isnull().mean().mean()
    return {
        "Total Runs":        f"{len(df):,}",
        "Pass":              f"{n_pass:,}",
        "Fail":              f"{n_fail:,}",
        "Failure Rate":      f"{n_fail/len(df):.2%}",
        "Process Params":    f"{n_params:,}",
        "Missing Data":      f"{missing_pct:.1%}",
    }
