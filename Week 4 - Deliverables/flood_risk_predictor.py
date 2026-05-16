#!/usr/bin/env python3
"""
=============================================================
  ALBAY FLOOD RISK PREDICTOR
  Rainfall-based flood risk prediction using SYNOP station data
  Target: Station 98444 (Albay/Legazpi)
  Supporting: Stations 98446, 98543, 98536, 98427, 98434, 98440
=============================================================

Usage:
    python flood_risk_predictor.py
    (Reads all CSVs from ./clean_dataset/)

Deliverables:
    1. Flood Risk Prediction  — Random Forest model
    2. Outlier Detection      — IQR + Z-score flagging of extreme rainfall events
"""

import os
import sys
import glob
import warnings
import textwrap
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from datetime import datetime

import joblib

# ── ML ─────────────────────────────────────────────────────────────────────
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, f1_score, precision_score, recall_score, accuracy_score,
    precision_recall_fscore_support,
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# ── Plotting ────────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap
import matplotlib.ticker as mticker

# ── Constants ───────────────────────────────────────────────────────────────
ALBAY_ID      = "98444"
DATASET_DIR   = "clean_dataset"
OUTPUT_DIR    = "flood_prediction_output"
RANDOM_STATE  = 42

# Stations and their roles
STATIONS = {
    "98444": "Albay (Legazpi)",
    "98446": "Catanduanes",
    "98543": "Masbate",
    "98536": "Romblon",
    "98427": "Tayabas",
    "98434": "Infanta",
    "98440": "Camarines Norte",
}

# Flood risk thresholds (mm / 3-hour period, Philippine standards)
RISK_THRESHOLDS = {
    "LOW":      0.0,
    "MODERATE": 0.75,   # moderate: 0.75–1.5 mm/3h
    "HIGH":     1.5,    # heavy:   1.5–3.0 mm/3h
    "EXTREME":  3.0,    # extreme: >3.0 mm/3h
}

# Rolling windows (in 3-hour steps)
WIN_12H = 4    # 12 hours
WIN_24H = 8    # 24 hours
WIN_48H = 16   # 48 hours

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_clean_dataset(directory: str) -> dict[str, pd.DataFrame]:
    """Load all CSV files from clean_dataset/, keyed by station_id."""
    if not os.path.isdir(directory):
        print(f"[ERROR] Directory not found: '{directory}'")
        print("  Please place your decoded SYNOP CSV files in a folder called 'clean_dataset'.")
        sys.exit(1)

    csv_files = glob.glob(os.path.join(directory, "*.csv"))
    if not csv_files:
        print(f"[ERROR] No CSV files found in '{directory}/'")
        sys.exit(1)

    print(f"\n{'='*60}")
    print("  ALBAY FLOOD RISK PREDICTOR — loading data")
    print(f"{'='*60}")
    print(f"  Found {len(csv_files)} file(s) in '{directory}/':\n")

    stations = {}
    for fpath in sorted(csv_files):
        try:
            df = pd.read_csv(fpath, low_memory=False)
            df.columns = df.columns.str.strip().str.lower()

            # Parse datetime
            if "date" in df.columns and "time" in df.columns:
                df["datetime"] = pd.to_datetime(
                    df["date"].astype(str) + " " + df["time"].astype(str),
                    errors="coerce"
                )
            elif "datetime" in df.columns:
                df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
            else:
                df["datetime"] = pd.NaT

            # Coerce numerics
            numeric_cols = ["temp","pressure","humidity","wind_speed",
                            "wind_dir","cloud_cover","visibility_m","rain_3h"]
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            # Identify station
            sid = None
            if "station_id" in df.columns:
                sid = str(df["station_id"].dropna().iloc[0]).strip() if not df["station_id"].dropna().empty else None
            if sid is None:
                sid = os.path.splitext(os.path.basename(fpath))[0]

            df["station_id"] = sid
            df.sort_values("datetime", inplace=True)
            df.reset_index(drop=True, inplace=True)

            stations[sid] = df
            name = STATIONS.get(sid, "Unknown")
            years = ""
            if df["datetime"].notna().any():
                y0 = df["datetime"].min().year
                y1 = df["datetime"].max().year
                years = f" | {y0}–{y1}"
            print(f"    [{sid}] {name:<25} — {len(df):>7,} records{years}")

        except Exception as e:
            print(f"  [WARN] Could not load {fpath}: {e}")

    print()
    return stations


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

def engineer_features(albay_df: pd.DataFrame,
                       supporting: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Build a rich feature matrix from Albay + surrounding stations.
    Each row = one 3-hour observation at Albay.

    Supporting stations each contribute individual features (not just aggregates)
    so the model can learn station-specific signals (e.g. Catanduanes rain as
    an upstream typhoon indicator for Albay).
    """
    df = albay_df.copy().set_index("datetime").sort_index()

    # ── Core Albay features ─────────────────────────────────────────────────
    core_cols = ["rain_3h","temp","pressure","humidity",
                 "wind_speed","wind_dir","cloud_cover","visibility_m"]
    for col in core_cols:
        if col not in df.columns:
            df[col] = np.nan

    # ── Missing data handling ───────────────────────────────────────────────
    # rain_3h is ~27% missing — fill with 0.0 (conservative no-rain assumption)
    # and add a flag so the model can distinguish imputed from real values
    df["rain_was_missing"] = df["rain_3h"].isna().astype(int)
    df["rain_3h"] = df["rain_3h"].fillna(0.0)
    # Winsorize: cap at 100mm/3h (corrected max is 98mm/3h — catch any residual outliers)
    df["rain_3h"] = df["rain_3h"].clip(upper=100.0)
    # Interpolate slow-moving variables (temp, pressure) for small gaps
    for col in ["temp", "pressure", "humidity"]:
        if col in df.columns:
            df[col] = df[col].interpolate(method="linear", limit=4)

    # ── Rolling rainfall windows ─────────────────────────────────────────────
    r = df["rain_3h"]
    df["rain_12h"] = r.rolling(WIN_12H, min_periods=1).sum()
    df["rain_24h"] = r.rolling(WIN_24H, min_periods=1).sum()
    df["rain_48h"] = r.rolling(WIN_48H, min_periods=1).sum()

    # ── Lagged Albay rainfall (explicit look-back signals) ───────────────────
    df["rain_lag_3h"]  = r.shift(1)   # what fell 3h ago
    df["rain_lag_6h"]  = r.shift(2)   # what fell 6h ago
    df["rain_lag_12h"] = r.shift(4)   # what fell 12h ago

    # ── Pressure tendency (drop = storm approaching) ─────────────────────────
    if "pressure" in df.columns:
        df["pressure_change_6h"]  = df["pressure"].diff(2)
        df["pressure_change_24h"] = df["pressure"].diff(WIN_24H)

    # ── Humidity × Rain interaction ──────────────────────────────────────────
    df["humidity_x_rain"] = df["humidity"] * r

    # ── Time features ────────────────────────────────────────────────────────
    idx = df.index
    df["hour"]              = idx.hour
    df["month"]             = idx.month
    df["is_monsoon"]        = idx.month.isin([6,7,8,9,10]).astype(int)
    df["is_typhoon_season"] = idx.month.isin([8,9,10,11]).astype(int)

    # ── Per-station individual features ─────────────────────────────────────
    # Each supporting station gets its own rain rolling windows and pressure
    # tendency — so the model can distinguish e.g. Catanduanes vs Masbate signals.
    all_support_rain3h = []   # kept for aggregate fallback

    for sid, sdf in supporting.items():
        s = sdf.copy()
        if "datetime" not in s.columns:
            continue
        s = s.set_index("datetime").sort_index()

        # Resample to 3h grid to align with Albay
        s_3h = s.resample("3h").mean(numeric_only=True)
        s_3h = s_3h.reindex(df.index)   # align index

        sid_short = sid  # e.g. "98446"

        if "rain_3h" in s.columns:
            df[f"rain_{sid_short}_was_missing"] = s_3h["rain_3h"].isna().astype(int)
            sr = s_3h["rain_3h"].fillna(0)
            df[f"rain_{sid_short}"]         = sr
            df[f"rain_{sid_short}_lag3h"]  = sr.shift(1)
            df[f"rain_{sid_short}_lag6h"]  = sr.shift(2)
            df[f"rain12h_{sid_short}"]     = sr.rolling(WIN_12H, min_periods=1).sum()
            df[f"rain24h_{sid_short}"]     = sr.rolling(WIN_24H, min_periods=1).sum()
            all_support_rain3h.append(sr.rename(sid_short))

        if "pressure" in s.columns:
            sp = s_3h["pressure"]
            df[f"pres_{sid_short}"]        = sp
            df[f"pres_chg6h_{sid_short}"]  = sp.diff(2)

        if "humidity" in s.columns:
            df[f"hum_{sid_short}"] = s_3h["humidity"]

    # ── Regional aggregates (kept as summary signals alongside per-station) ──
    if all_support_rain3h:
        agg = pd.concat(all_support_rain3h, axis=1)
        df["support_rain_mean"] = agg.mean(axis=1)
        df["support_rain_max"]  = agg.max(axis=1)

    df.reset_index(inplace=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: LABEL GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def assign_flood_risk(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign a 4-class flood risk label based on 24-hour accumulated rainfall
    and moisture / pressure context.

    Classes:
        0 – LOW
        1 – MODERATE
        2 – HIGH
        3 – EXTREME
    """
    r24 = df.get("rain_24h", pd.Series(0, index=df.index)).fillna(0)

    conditions = [
        r24 < 1.5,                        # LOW      (< 1.5 mm/24h)
        (r24 >= 1.5)  & (r24 < 5.0),      # MODERATE
        (r24 >= 5.0)  & (r24 < 10.0),     # HIGH
        r24 >= 10.0,                       # EXTREME
    ]
    labels = [0, 1, 2, 3]
    df["flood_risk"] = np.select(conditions, labels, default=0)

    label_map = {0:"LOW", 1:"MODERATE", 2:"HIGH", 3:"EXTREME"}
    df["flood_risk_label"] = df["flood_risk"].map(label_map)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: MODEL TRAINING
# ─────────────────────────────────────────────────────────────────────────────

# Base Albay features — always used if present
BASE_FEATURE_COLS = [
    "rain_3h",
    "rain_lag_3h","rain_lag_6h","rain_lag_12h",
    "temp","pressure","humidity","wind_speed","wind_dir",
    "cloud_cover","visibility_m",
    "pressure_change_6h","pressure_change_24h",
    "humidity_x_rain",
    "hour","month","is_monsoon","is_typhoon_season",
    "support_rain_mean","support_rain_max",
    "rain_was_missing",
]

# Prefixes used for dynamically generated per-station columns
PER_STATION_PREFIXES = [
    "rain_", "rain12h_", "rain24h_",
    "pres_", "pres_chg6h_",
    "hum_",
]

EXCLUDED_FEATURES = {"rain_12h", "rain_24h", "rain_48h", "rain_percentile"}

def build_model_data(df: pd.DataFrame):
    """
    Prepare X, y. Includes base Albay features + all per-station columns
    dynamically generated during feature engineering.
    """
    feature_cols = [c for c in BASE_FEATURE_COLS if c in df.columns]

    # Append per-station dynamically generated columns
    for col in df.columns:
        if col in feature_cols or col in EXCLUDED_FEATURES:
            continue
        if any(col.startswith(pfx) for pfx in PER_STATION_PREFIXES):
            feature_cols.append(col)

    X = df[feature_cols].copy()
    y = df["flood_risk"].copy()

    mask = y.notna()
    return X[mask], y[mask], feature_cols


def train_models(X, y, feature_names):
    """Train Random Forest (primary), GBT, and Logistic Regression.

    Uses temporal (not random) train/test split to prevent time-series leakage.
    """
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    models = {
        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(
                n_estimators=200,
                max_depth=12,
                min_samples_leaf=5,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ))
        ]),
        "Gradient Boost": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=RANDOM_STATE,
            ))
        ]),
        "Logistic Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=RANDOM_STATE,

            ))
        ]),
    }

    results = {}
    print(f"\n{'─'*60}")
    print("  MODEL TRAINING & EVALUATION")
    print(f"{'─'*60}")
    print(f"  Training samples : {len(X_train):,}")
    print(f"  Testing  samples : {len(X_test):,}")
    print(f"  Features used    : {len(feature_names)}\n")

    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        acc   = accuracy_score(y_test, y_pred)
        f1    = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        prec  = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec   = recall_score(y_test, y_pred, average="weighted", zero_division=0)

        try:
            y_proba = pipe.predict_proba(X_test)
            auc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="weighted")
        except Exception:
            auc = np.nan

        results[name] = {
            "pipe": pipe,
            "X_test": X_test,
            "y_test": y_test,
            "y_pred": y_pred,
            "acc": acc, "f1": f1, "prec": prec, "rec": rec, "auc": auc,
        }

        print(f"  [{name}]")
        print(f"    Accuracy     : {acc:.4f}")
        print(f"    F1 (wtd)     : {f1:.4f}")
        print(f"    AUC (OvR)    : {auc:.4f}")

        _, _, f1_per, _ = precision_recall_fscore_support(
            y_test, y_pred, labels=[0, 1, 2, 3], zero_division=0
        )
        print(f"    Per-class F1 : LOW={f1_per[0]:.4f}  MOD={f1_per[1]:.4f}  "
              f"HIGH={f1_per[2]:.4f}  EXT={f1_per[3]:.4f}")
        print()

    # TimeSeriesSplit cross-val on best model (RF)
    rf_pipe = models["Random Forest"]
    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = cross_val_score(rf_pipe, X, y, cv=tscv,
                                scoring="f1_weighted", n_jobs=-1)
    print(f"  Random Forest TimeSeries CV F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}\n")

    return results, X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: OUTLIER / EXTREME EVENT DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def detect_extreme_events(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flag extreme rainfall events using percentile thresholds (top 1%).
    P99 captures the rarest, most extreme rain events — a statistically
    sound definition of 'extreme' rather than the overly-permissive IQR.
    """
    rain = df["rain_3h"].fillna(0)
    p99  = rain.quantile(0.99)
    p995 = rain.quantile(0.995)

    df = df.copy()
    df["rain_percentile"] = rain.rank(pct=True)
    df["extreme_event"]   = rain >= p99

    df["extreme_tier"] = "normal"
    df.loc[df["extreme_event"], "extreme_tier"] = "extreme"
    df.loc[rain >= p995, "extreme_tier"] = "catastrophic"

    n_extreme   = df["extreme_event"].sum()
    n_catast    = (rain >= p995).sum()
    print(f"{'─'*60}")
    print("  OUTLIER / EXTREME EVENT DETECTION")
    print(f"{'─'*60}")
    print(f"  P99 threshold      : {p99:.2f} mm/3h")
    print(f"  P99.5 threshold    : {p995:.2f} mm/3h")
    print(f"  Extreme events     : {n_extreme:>6,}  ({100*n_extreme/len(df):.1f}%)")
    print(f"  Catastrophic events: {n_catast:>6,}  ({100*n_catast/len(df):.2f}%)")
    print(f"  Severe threshold   : ≥{RISK_THRESHOLDS['EXTREME']} mm/3h  →  {(rain >= RISK_THRESHOLDS['EXTREME']).sum()} events\n")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: VISUALIZATIONS
# ─────────────────────────────────────────────────────────────────────────────

COLORS = {
    "LOW":      "#2ecc71",
    "MODERATE": "#f1c40f",
    "HIGH":     "#e67e22",
    "EXTREME":  "#e74c3c",
}
RISK_COLORS = [COLORS[k] for k in ["LOW","MODERATE","HIGH","EXTREME"]]

def _save(fig, fname, output_dir):
    path = os.path.join(output_dir, fname)
    fig.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def plot_rainfall_timeline(df, output_dir):
    """Rainfall timeline with flood risk bands and extreme event markers."""
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), facecolor="#0d1117")
    fig.suptitle("Albay (Station 98444) — Rainfall Timeline & Flood Risk",
                 color="white", fontsize=16, fontweight="bold", y=0.98)

    bg = "#0d1117"
    for ax in axes:
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_color("#30363d")

    t = df["datetime"] if "datetime" in df.columns else df.index

    # Panel 1: 3h rainfall + extreme events
    ax = axes[0]
    rain = df["rain_3h"].fillna(0)
    ax.fill_between(t, rain, alpha=0.6, color="#58a6ff", label="Rain 3h (mm)")
    extreme_mask = df.get("extreme_event", pd.Series(False, index=df.index))
    ax.scatter(t[extreme_mask], rain[extreme_mask],
               color="#ff7b72", s=25, zorder=5, label="Extreme event", alpha=0.9)
    ax.set_ylabel("mm / 3h", color="white", fontsize=9)
    ax.legend(facecolor="#21262d", labelcolor="white", fontsize=8, loc="upper right")
    ax.set_title("3-Hour Rainfall", color="#8b949e", fontsize=10)

    # Panel 2: Accumulated 24h rain with risk bands
    ax = axes[1]
    r24 = df.get("rain_24h", rain.rolling(8, min_periods=1).sum())
    ax.fill_between(t, r24, alpha=0.5, color="#bc8cff", label="Rain 24h (mm)")
    for thresh, color, label in [
        (1.5,  "#f1c40f55", "Moderate"),
        (5.0,  "#e67e2255", "High"),
        (10.0, "#e74c3c55", "Extreme"),
    ]:
        ax.axhline(thresh, color=color.replace("55",""), linestyle="--",
                   linewidth=0.8, alpha=0.7, label=f"{label} ({thresh} mm)")
    ax.set_ylabel("mm / 24h", color="white", fontsize=9)
    ax.legend(facecolor="#21262d", labelcolor="white", fontsize=8,
              loc="upper right", ncol=2)
    ax.set_title("24-Hour Accumulated Rainfall", color="#8b949e", fontsize=10)

    # Panel 3: Flood risk class
    ax = axes[2]
    risk = df.get("flood_risk", pd.Series(0, index=df.index)).fillna(0)
    cmap = ListedColormap(RISK_COLORS)
    ax.scatter(t, risk, c=risk, cmap=cmap, vmin=0, vmax=3,
               s=4, alpha=0.7, linewidths=0)
    ax.set_yticks([0,1,2,3])
    ax.set_yticklabels(["LOW","MOD","HIGH","EXT"], color="white", fontsize=8)
    ax.set_ylabel("Flood Risk", color="white", fontsize=9)
    ax.set_title("Predicted Flood Risk Class", color="#8b949e", fontsize=10)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return _save(fig, "01_rainfall_timeline.png", output_dir)


def plot_model_comparison(results, output_dir):
    """Bar chart comparing model metrics."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor="#0d1117")
    fig.suptitle("Model Performance Comparison",
                 color="white", fontsize=14, fontweight="bold")

    names = list(results.keys())
    metrics = {
        "Accuracy":  [results[n]["acc"]  for n in names],
        "F1 (wtd)":  [results[n]["f1"]   for n in names],
        "AUC (OvR)": [results[n]["auc"]  for n in names],
    }

    bar_colors = ["#58a6ff", "#bc8cff", "#3fb950"]

    ax = axes[0]
    ax.set_facecolor("#161b22")
    for spine in ax.spines.values(): spine.set_color("#30363d")
    ax.tick_params(colors="white")

    x = np.arange(len(names))
    w = 0.25
    for i, (metric, vals) in enumerate(metrics.items()):
        bars = ax.bar(x + i*w, vals, w, label=metric,
                      color=bar_colors[i], alpha=0.85)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom",
                    color="white", fontsize=7.5)
    ax.set_xticks(x + w)
    ax.set_xticklabels([n.replace(" ", "\n") for n in names], color="white", fontsize=9)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score", color="white", fontsize=10)
    ax.set_title("Metric Comparison", color="#8b949e", fontsize=11)
    ax.legend(facecolor="#21262d", labelcolor="white", fontsize=8)

    # Confusion matrix for best model (RF)
    ax2 = axes[1]
    ax2.set_facecolor("#161b22")
    ax2.tick_params(colors="white")
    for spine in ax2.spines.values(): spine.set_color("#30363d")

    rf = results["Random Forest"]
    cm = confusion_matrix(rf["y_test"], rf["y_pred"])
    class_labels = ["LOW","MOD","HIGH","EXT"]
    im = ax2.imshow(cm, cmap="Blues", aspect="auto")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i,j] < cm.max()/2 else "black"
            ax2.text(j, i, str(cm[i,j]), ha="center", va="center",
                     color=color, fontsize=10, fontweight="bold")

    ax2.set_xticks(range(len(class_labels)))
    ax2.set_yticks(range(len(class_labels)))
    ax2.set_xticklabels(class_labels, color="white", fontsize=9)
    ax2.set_yticklabels(class_labels, color="white", fontsize=9)
    ax2.set_xlabel("Predicted", color="white", fontsize=10)
    ax2.set_ylabel("Actual", color="white", fontsize=10)
    ax2.set_title("Random Forest — Confusion Matrix", color="#8b949e", fontsize=11)

    plt.colorbar(im, ax=ax2)
    fig.tight_layout()
    return _save(fig, "02_model_comparison.png", output_dir)


def plot_feature_importance(results, feature_names, output_dir):
    """Horizontal bar chart of Random Forest feature importances."""
    rf_pipe = results["Random Forest"]["pipe"]
    rf_clf  = rf_pipe.named_steps["clf"]

    imp = rf_clf.feature_importances_
    idx = np.argsort(imp)[::-1]
    top_n = min(15, len(feature_names))
    idx = idx[:top_n][::-1]

    fig, ax = plt.subplots(figsize=(11, 7), facecolor="#0d1117")
    ax.set_facecolor("#161b22")
    ax.tick_params(colors="white")
    for spine in ax.spines.values(): spine.set_color("#30363d")

    bars = ax.barh(range(top_n), imp[idx], color="#58a6ff", alpha=0.8)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_names[i] for i in idx], color="white", fontsize=9)
    ax.set_xlabel("Importance", color="white", fontsize=10)
    ax.set_title("Random Forest — Top Feature Importances",
                 color="white", fontsize=13, fontweight="bold")

    for bar, val in zip(bars, imp[idx]):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                f"{val:.4f}", va="center", color="#8b949e", fontsize=8)

    fig.tight_layout()
    return _save(fig, "03_feature_importance.png", output_dir)


def plot_extreme_events(df, output_dir):
    """Scatter plot of extreme rainfall events with percentile distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), facecolor="#0d1117")
    fig.suptitle("Extreme Weather Events — Outlier Detection",
                 color="white", fontsize=14, fontweight="bold")

    t = df["datetime"] if "datetime" in df.columns else df.index
    rain = df["rain_3h"].fillna(0)
    extreme = df.get("extreme_event", pd.Series(False, index=df.index))
    pctile  = df.get("rain_percentile", pd.Series(0.0, index=df.index))

    # Panel 1: Scatter all events, highlight extremes
    ax = axes[0]
    ax.set_facecolor("#161b22")
    ax.tick_params(colors="white")
    for spine in ax.spines.values(): spine.set_color("#30363d")

    normal_mask = ~extreme
    ax.scatter(t[normal_mask], rain[normal_mask],
               c="#8b949e", s=3, alpha=0.3, label="Normal")
    ax.scatter(t[extreme], rain[extreme],
               c=df.loc[extreme, "flood_risk"].map(
                   lambda x: RISK_COLORS[int(x)] if not pd.isna(x) else "#ff7b72"
               ),
               s=30, alpha=0.9, zorder=5, label="Extreme event", edgecolors="white",
               linewidths=0.3)

    for thresh, label, color in [
        (RISK_THRESHOLDS["MODERATE"], "Moderate", "#f1c40f"),
        (RISK_THRESHOLDS["HIGH"],     "Heavy",    "#e67e22"),
        (RISK_THRESHOLDS["EXTREME"],  "Extreme",  "#e74c3c"),
    ]:
        ax.axhline(thresh, color=color, linestyle="--", linewidth=0.8,
                   alpha=0.8, label=f"{label} ({thresh} mm)")

    ax.set_ylabel("Rainfall (mm / 3h)", color="white", fontsize=10)
    ax.set_xlabel("Date", color="white", fontsize=10)
    ax.set_title("Extreme Rainfall Events Detected", color="#8b949e", fontsize=11)
    ax.legend(facecolor="#21262d", labelcolor="white", fontsize=8)

    # Panel 2: Percentile distribution
    ax2 = axes[1]
    ax2.set_facecolor("#161b22")
    ax2.tick_params(colors="white")
    for spine in ax2.spines.values(): spine.set_color("#30363d")

    ax2.scatter(t[normal_mask], pctile[normal_mask],
                c="#8b949e", s=3, alpha=0.3, label="Normal")
    ax2.scatter(t[extreme], pctile[extreme],
                c="#ff7b72", s=25, alpha=0.9, zorder=5, label="Outlier (P99+)")
    ax2.axhline(0.99, color="#e74c3c", linestyle="--", linewidth=1,
                label="P99 threshold")
    ax2.set_ylabel("Percentile", color="white", fontsize=10)
    ax2.set_xlabel("Date", color="white", fontsize=10)
    ax2.set_title("Rainfall Percentile Distribution", color="#8b949e", fontsize=11)
    ax2.legend(facecolor="#21262d", labelcolor="white", fontsize=8)

    fig.tight_layout()
    return _save(fig, "04_extreme_events.png", output_dir)


def plot_risk_distribution(df, output_dir):
    """Pie + monthly heatmap of flood risk distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor="#0d1117")
    fig.suptitle("Flood Risk Distribution — Albay",
                 color="white", fontsize=14, fontweight="bold")

    # Pie chart
    ax = axes[0]
    ax.set_facecolor("#0d1117")
    counts = df["flood_risk_label"].value_counts()
    order  = ["LOW","MODERATE","HIGH","EXTREME"]
    vals   = [counts.get(l, 0) for l in order]
    explode = [0, 0.05, 0.1, 0.15]

    wedges, texts, autotexts = ax.pie(
        vals, labels=order, colors=RISK_COLORS,
        autopct="%1.1f%%", startangle=140,
        explode=explode, pctdistance=0.85,
        wedgeprops=dict(edgecolor="#0d1117", linewidth=1.5)
    )
    for t in texts: t.set_color("white")
    for t in autotexts: t.set_color("white"); t.set_fontsize(9)
    ax.set_title("Overall Risk Class Distribution",
                 color="#8b949e", fontsize=11, pad=15)

    # Monthly heatmap
    ax2 = axes[1]
    ax2.set_facecolor("#161b22")
    ax2.tick_params(colors="white")
    for spine in ax2.spines.values(): spine.set_color("#30363d")

    if "datetime" in df.columns and "flood_risk" in df.columns:
        df2 = df.dropna(subset=["datetime","flood_risk"]).copy()
        df2["month"] = df2["datetime"].dt.month
        df2["year"]  = df2["datetime"].dt.year
        pivot = df2.groupby(["year","month"])["flood_risk"].mean().unstack(fill_value=0)

        cmap = matplotlib.colormaps.get_cmap("YlOrRd")
        im = ax2.imshow(pivot.values, cmap=cmap, aspect="auto",
                        vmin=0, vmax=3, interpolation="nearest")
        month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                       "Jul","Aug","Sep","Oct","Nov","Dec"]
        ax2.set_xticks(range(len(pivot.columns)))
        ax2.set_xticklabels([month_names[m-1] for m in pivot.columns],
                             color="white", fontsize=8, rotation=45)
        ax2.set_yticks(range(len(pivot.index)))
        ax2.set_yticklabels(pivot.index, color="white", fontsize=8)
        ax2.set_title("Mean Flood Risk by Year × Month",
                      color="#8b949e", fontsize=11)
        cb = plt.colorbar(im, ax=ax2, shrink=0.8)
        cb.set_ticks([0,1,2,3])
        cb.set_ticklabels(["LOW","MOD","HIGH","EXT"])
        cb.ax.yaxis.set_tick_params(color="white")
        plt.setp(plt.getp(cb.ax.axes, "yticklabels"), color="white")

    fig.tight_layout()
    return _save(fig, "05_risk_distribution.png", output_dir)


def plot_prediction_summary(df, results, output_dir):
    """A-at-a-glance dashboard tile."""
    rf  = results["Random Forest"]
    fig = plt.figure(figsize=(14, 8), facecolor="#0d1117")
    fig.suptitle("Albay Flood Risk Prediction — Summary Dashboard",
                 color="white", fontsize=15, fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    def styled_ax(pos):
        ax = fig.add_subplot(pos)
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="white")
        for s in ax.spines.values(): s.set_color("#30363d")
        return ax

    # KPI tiles
    kpis = [
        ("Accuracy",    f"{rf['acc']:.1%}",  "#58a6ff"),
        ("F1 Score",    f"{rf['f1']:.1%}",   "#bc8cff"),
        ("AUC (OvR)",   f"{rf['auc']:.4f}",  "#3fb950"),
    ]
    for i, (label, val, color) in enumerate(kpis):
        ax = styled_ax(gs[0, i])
        ax.text(0.5, 0.6, val, ha="center", va="center",
                color=color, fontsize=28, fontweight="bold",
                transform=ax.transAxes)
        ax.text(0.5, 0.2, label, ha="center", va="center",
                color="#8b949e", fontsize=11, transform=ax.transAxes)
        ax.set_xticks([]); ax.set_yticks([])

    # Per-class F1
    ax_cls = styled_ax(gs[1, :2])
    from sklearn.metrics import precision_recall_fscore_support
    prec, rec, f1s, sup = precision_recall_fscore_support(
        rf["y_test"], rf["y_pred"], labels=[0,1,2,3], zero_division=0
    )
    x = np.arange(4)
    labels = ["LOW","MOD","HIGH","EXT"]
    w = 0.25
    ax_cls.bar(x - w,   prec, w, color="#58a6ff", alpha=0.8, label="Precision")
    ax_cls.bar(x,       rec,  w, color="#3fb950", alpha=0.8, label="Recall")
    ax_cls.bar(x + w,   f1s,  w, color="#bc8cff", alpha=0.8, label="F1")
    ax_cls.set_xticks(x); ax_cls.set_xticklabels(labels, color="white")
    ax_cls.set_ylim(0, 1.15)
    ax_cls.set_title("Per-Class Metrics (Random Forest)", color="#8b949e", fontsize=10)
    ax_cls.legend(facecolor="#21262d", labelcolor="white", fontsize=8)

    # Class counts
    ax_cnt = styled_ax(gs[1, 2])
    counts = df["flood_risk_label"].value_counts()
    order  = ["LOW","MODERATE","HIGH","EXTREME"]
    c_vals = [counts.get(l, 0) for l in order]
    ax_cnt.bar(["LOW","MOD","HIGH","EXT"], c_vals,
               color=RISK_COLORS, alpha=0.85, edgecolor="#0d1117")
    ax_cnt.set_title("Records per Risk Class", color="#8b949e", fontsize=10)
    ax_cnt.set_ylabel("Count", color="white", fontsize=9)
    for i, v in enumerate(c_vals):
        ax_cnt.text(i, v + max(c_vals)*0.01, f"{v:,}",
                    ha="center", color="white", fontsize=8)

    return _save(fig, "00_summary_dashboard.png", output_dir)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7: CSV EXPORTS
# ─────────────────────────────────────────────────────────────────────────────

def export_predictions(df, output_dir):
    """Export labeled dataset with predictions."""
    cols = ["datetime","station_id","rain_3h","rain_12h","rain_24h","rain_48h",
            "temp","pressure","humidity","flood_risk","flood_risk_label",
            "extreme_event","extreme_tier","rain_percentile","rain_was_missing"]
    out_cols = [c for c in cols if c in df.columns]
    out = df[out_cols].copy()
    path = os.path.join(output_dir, "albay_flood_predictions.csv")
    out.to_csv(path, index=False)
    print(f"  Saved: {path}  ({len(out):,} rows)")
    return path

def export_extreme_events(df, output_dir):
    """Export only extreme events."""
    extreme_mask = df.get("extreme_event", pd.Series(False, index=df.index))
    extremes = df[extreme_mask].copy()
    cols = ["datetime","station_id","rain_3h","rain_percentile","flood_risk_label",
            "extreme_tier","temp","pressure","humidity","rain_was_missing"]
    out_cols = [c for c in cols if c in extremes.columns]
    path = os.path.join(output_dir, "extreme_events.csv")
    extremes[out_cols].to_csv(path, index=False)
    print(f"  Saved: {path}  ({len(extremes):,} extreme events)")
    return path

def export_model_metrics(results, output_dir):
    """Export model metrics to CSV."""
    rows = []
    for name, r in results.items():
        rows.append({
            "model": name,
            "accuracy": round(r["acc"], 4),
            "f1_weighted": round(r["f1"], 4),
            "precision_weighted": round(r["prec"], 4),
            "recall_weighted": round(r["rec"], 4),
            "auc_ovr": round(r["auc"], 4) if not np.isnan(r["auc"]) else "N/A",
        })
    path = os.path.join(output_dir, "model_metrics.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"  Saved: {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8: MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Load data
    stations = load_clean_dataset(DATASET_DIR)

    if ALBAY_ID not in stations:
        # Try to find any station that could be Albay
        print(f"[WARN] Station {ALBAY_ID} not found. Available: {list(stations.keys())}")
        albay_id = list(stations.keys())[0]
        print(f"       Using '{albay_id}' as primary station.")
    else:
        albay_id = ALBAY_ID

    albay_df  = stations[albay_id]
    support   = {k: v for k, v in stations.items() if k != albay_id}

    print(f"  Primary station  : [{albay_id}] {STATIONS.get(albay_id,'Unknown')}")
    print(f"  Supporting       : {len(support)} station(s)")

    # 2. Feature engineering
    print(f"\n{'─'*60}")
    print("  FEATURE ENGINEERING")
    print(f"{'─'*60}")
    feat_df = engineer_features(albay_df, support)
    feat_df = assign_flood_risk(feat_df)
    feat_df = detect_extreme_events(feat_df)

    print(f"  Feature matrix   : {len(feat_df):,} rows × {feat_df.shape[1]} columns")
    print(f"  Date range       : {feat_df['datetime'].min()} → {feat_df['datetime'].max()}")
    dist = feat_df["flood_risk_label"].value_counts()
    print(f"  Risk distribution:")
    for label in ["LOW","MODERATE","HIGH","EXTREME"]:
        cnt = dist.get(label, 0)
        pct = 100 * cnt / len(feat_df)
        bar = "█" * int(pct / 2)
        print(f"    {label:<10}: {cnt:>7,}  ({pct:5.1f}%)  {bar}")

    # 3. Model training
    X, y, feature_names = build_model_data(feat_df)
    results, X_tr, X_te, y_tr, y_te = train_models(X, y, feature_names)

    # 4. Plots
    print(f"\n{'─'*60}")
    print("  GENERATING CHARTS")
    print(f"{'─'*60}")
    plot_prediction_summary(feat_df, results, OUTPUT_DIR)
    plot_rainfall_timeline(feat_df, OUTPUT_DIR)
    plot_model_comparison(results, OUTPUT_DIR)
    plot_feature_importance(results, feature_names, OUTPUT_DIR)
    plot_extreme_events(feat_df, OUTPUT_DIR)
    plot_risk_distribution(feat_df, OUTPUT_DIR)

    # 5. Export CSVs
    print(f"\n{'─'*60}")
    print("  EXPORTING DATA")
    print(f"{'─'*60}")
    export_predictions(feat_df, OUTPUT_DIR)
    export_extreme_events(feat_df, OUTPUT_DIR)
    export_model_metrics(results, OUTPUT_DIR)

    # 6. Save trained model + feature names
    model_path = os.path.join(OUTPUT_DIR, "flood_model.joblib")
    joblib.dump({
        "model": results["Random Forest"]["pipe"],
        "feature_names": feature_names,
    }, model_path)
    size_kb = os.path.getsize(model_path) // 1024
    print(f"  Saved: {model_path}  ({size_kb} KB)")

    # 7. Final summary
    best = max(results.items(), key=lambda kv: kv[1]["f1"])
    print(f"\n{'='*60}")
    print("  DONE!")
    print(f"{'='*60}")
    print(f"  Best model  : {best[0]}  (F1 = {best[1]['f1']:.4f})")
    print(f"  Output dir  : ./{OUTPUT_DIR}/")
    print(f"\n  Outputs:")
    for fname in sorted(os.listdir(OUTPUT_DIR)):
        fpath = os.path.join(OUTPUT_DIR, fname)
        kb = os.path.getsize(fpath) // 1024
        print(f"    {fname:<45} {kb:>5} KB")
    print()


if __name__ == "__main__":
    main()
