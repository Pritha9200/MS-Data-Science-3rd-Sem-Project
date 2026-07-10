"""
================================================================================
  AI-ENABLED STUDENT PERFORMANCE PREDICTION PIPELINE
  Full end-to-end ML pipeline in a single file
================================================================================
  Stages:
    1. Data Ingestion & Validation
    2. Feature Engineering
    3. Preprocessing  (encode, split, scale)
    4. Model Training  (Logistic Regression, Decision Tree, Random Forest)
    5. Model Evaluation (metrics, confusion matrix, ROC, feature importance)
    6. Prediction Interface (predict a single student's grade)

  Usage:
    python student_performance_ml.py

  Requirements:
    pip install pandas numpy scikit-learn matplotlib
================================================================================
"""

import os
import json
import pickle
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.model_selection    import train_test_split
from sklearn.preprocessing      import StandardScaler, LabelEncoder
from sklearn.linear_model       import LogisticRegression
from sklearn.tree               import DecisionTreeClassifier
from sklearn.ensemble           import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics            import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, roc_curve, auc,
)

warnings.filterwarnings("ignore")


# ================================================================================
# GLOBAL CONFIGURATION
# ================================================================================

DATA_PATH    = "student_performance.csv"   # ← adjust path if needed
OUTPUT_DIR   = "outputs"                   # all plots / artefacts saved here

FEATURE_COLS = [
    "weekly_self_study_hours",
    "attendance_percentage",
    "class_participation",
    "total_score",
    "engagement_score",      # engineered
    "study_intensity",       # engineered
    "is_high_attendance",    # engineered
    "is_active_student",     # engineered
]
TARGET_COL   = "grade"
GRADE_ORDER  = ["A", "B", "C", "D", "F"]

SAMPLE_SIZE  = 100_000   # stratified sample for training (None = use all)
TEST_SIZE    = 0.20
VAL_SIZE     = 0.10
RANDOM_STATE = 42


# ================================================================================
# UTILITIES
# ================================================================================

def banner(title: str) -> None:
    """Print a section banner."""
    line = "=" * 70
    print(f"\n{line}\n  {title}\n{line}")


def sub(title: str) -> None:
    """Print a sub-section header."""
    print(f"\n  ── {title}")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# ================================================================================
# STAGE 1 — DATA INGESTION & VALIDATION
# ================================================================================

def ingest(path: str = DATA_PATH) -> pd.DataFrame:
    """
    Load the CSV, run schema / null / duplicate / range / label checks,
    and return a clean DataFrame ready for feature engineering.
    """
    banner("STAGE 1 — DATA INGESTION & VALIDATION")

    # ── Load ──────────────────────────────────────────────────────────────────
    print(f"\n  Loading: {path}")
    df = pd.read_csv(path)
    print(f"  Shape  : {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"  Columns: {list(df.columns)}")

    # ── Schema check ──────────────────────────────────────────────────────────
    sub("Schema check")
    required = ["weekly_self_study_hours", "attendance_percentage",
                "class_participation", "total_score", "grade"]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    print(f"  ✅ All required columns present")

    # ── Missing values ────────────────────────────────────────────────────────
    sub("Missing values")
    null_total = df.isnull().sum().sum()
    if null_total == 0:
        print("  ✅ No missing values")
    else:
        print(f"  ⚠️  {null_total:,} nulls found — dropping affected rows")
        df = df.dropna()

    # ── Duplicates ────────────────────────────────────────────────────────────
    sub("Duplicate rows")
    n_dupes = df.duplicated().sum()
    if n_dupes == 0:
        print("  ✅ No duplicates")
    else:
        print(f"  ⚠️  {n_dupes:,} duplicates removed")
        df = df.drop_duplicates()

    # ── Value-range check ─────────────────────────────────────────────────────
    sub("Value range check")
    bounds = {
        "weekly_self_study_hours": (0,   40),
        "attendance_percentage":   (0,  100),
        "class_participation":     (0,   10),
        "total_score":             (0,  100),
    }
    bad_mask = pd.Series(False, index=df.index)
    for col, (lo, hi) in bounds.items():
        out_of_range = (df[col] < lo) | (df[col] > hi)
        n_bad = out_of_range.sum()
        status = "✅ OK" if n_bad == 0 else f"⚠️  {n_bad:,} out-of-range"
        print(f"  {col:<30}: [{lo:>3}, {hi:>3}]  {status}")
        bad_mask |= out_of_range
    if bad_mask.any():
        df = df[~bad_mask]

    # ── Grade label check ─────────────────────────────────────────────────────
    sub("Grade label check")
    valid_grades = set(GRADE_ORDER)
    found_grades = set(df[TARGET_COL].unique())
    invalid = found_grades - valid_grades
    if invalid:
        print(f"  ⚠️  Removing rows with invalid grade labels: {invalid}")
        df = df[df[TARGET_COL].isin(valid_grades)]
    print(f"  ✅ Valid grades: {sorted(found_grades & valid_grades)}")

    # ── Summary ───────────────────────────────────────────────────────────────
    sub("Clean dataset summary")
    print(f"  Rows after cleaning: {len(df):,}")
    grade_counts = df[TARGET_COL].value_counts().reindex(GRADE_ORDER)
    print(f"\n  {'Grade':<7} {'Count':>9}  {'%':>6}  Distribution")
    for g, cnt in grade_counts.items():
        pct = cnt / len(df) * 100
        bar = "█" * int(pct / 2)
        print(f"  {g:<7} {cnt:>9,}  {pct:>5.2f}%  {bar}")

    return df


# ================================================================================
# STAGE 2 — FEATURE ENGINEERING
# ================================================================================

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create four new predictive features and return the enriched DataFrame.

    New columns
    -----------
    engagement_score   : 0–1 composite of attendance + participation
    study_intensity    : ordinal bucket of weekly study hours (0–6)
    is_high_attendance : 1 if attendance ≥ 90%, else 0
    is_active_student  : 1 if participation ≥ 7/10, else 0
    """
    banner("STAGE 2 — FEATURE ENGINEERING")

    df = df.copy()

    # engagement_score: equal-weight blend of attendance and participation
    df["engagement_score"] = (
        0.5 * df["attendance_percentage"] / 100.0
        + 0.5 * df["class_participation"] / 10.0
    ).round(6)

    # study_intensity: ordinal bucket (0 = 0–5 h, …, 6 = 30–40 h)
    df["study_intensity"] = (
        pd.cut(
            df["weekly_self_study_hours"],
            bins=[0, 5, 10, 15, 20, 25, 30, 40],
            labels=[0, 1, 2, 3, 4, 5, 6],
            include_lowest=True,
        )
        .astype(float)
        .fillna(0)
    )

    # Binary engagement flags
    df["is_high_attendance"] = (df["attendance_percentage"] >= 90).astype(int)
    df["is_active_student"]  = (df["class_participation"]   >= 7 ).astype(int)

    # ── Report ────────────────────────────────────────────────────────────────
    new_cols = ["engagement_score", "study_intensity",
                "is_high_attendance", "is_active_student"]

    print(f"\n  {'Feature':<25} {'Min':>7} {'Max':>7} {'Mean':>8}  Correlation→score")
    print("  " + "-" * 70)
    for col in new_cols:
        s  = df[col]
        r  = s.corr(df["total_score"])
        bar = ("+" if r >= 0 else "-") + "█" * int(abs(r) * 25)
        print(f"  {col:<25} {s.min():>7.3f} {s.max():>7.3f} "
              f"{s.mean():>8.4f}  r={r:+.4f} {bar}")

    print(f"\n  ✅ Dataset now has {df.shape[1]} columns")
    return df


# ================================================================================
# STAGE 3 — PREPROCESSING
# ================================================================================

def preprocess(df: pd.DataFrame) -> dict:
    """
    Encode target, stratified sample, train/val/test split, and scale features.

    Returns a dict with:
      X_train_sc, X_val_sc, X_test_sc   — StandardScaler-transformed arrays
      X_train_raw, X_val_raw, X_test_raw — unscaled (for tree models)
      y_train, y_val, y_test             — integer-encoded labels
      label_encoder, scaler              — fitted artefacts
      feature_names                      — list of feature column names
    """
    banner("STAGE 3 — PREPROCESSING")

    # ── Features & target ────────────────────────────────────────────────────
    X_raw = df[FEATURE_COLS].fillna(0).values
    y_raw = df[TARGET_COL].values

    # ── Label encoding ────────────────────────────────────────────────────────
    sub("Label encoding")
    le = LabelEncoder()
    le.fit(GRADE_ORDER)
    y_enc = le.transform(y_raw)
    print(f"  Mapping: { dict(zip(le.classes_, le.transform(le.classes_))) }")

    # ── Stratified sampling ───────────────────────────────────────────────────
    sub("Stratified sampling")
    if SAMPLE_SIZE and SAMPLE_SIZE < len(X_raw):
        X_s, _, y_s, _ = train_test_split(
            X_raw, y_enc,
            train_size=SAMPLE_SIZE,
            stratify=y_enc,
            random_state=RANDOM_STATE,
        )
        print(f"  Sampled {SAMPLE_SIZE:,} / {len(X_raw):,} records (stratified)")
    else:
        X_s, y_s = X_raw, y_enc
        print(f"  Using full dataset: {len(X_s):,} records")

    # ── Train / val / test split ──────────────────────────────────────────────
    sub("Train / Validation / Test split")
    X_tv, X_test, y_tv, y_test = train_test_split(
        X_s, y_s, test_size=TEST_SIZE, stratify=y_s, random_state=RANDOM_STATE
    )
    val_frac = VAL_SIZE / (1 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=val_frac, stratify=y_tv, random_state=RANDOM_STATE
    )
    total = len(X_s)
    print(f"  {'Split':<12} {'Samples':>8}  {'%':>6}")
    print(f"  {'Train':<12} {len(X_train):>8,}  {len(X_train)/total*100:>5.1f}%")
    print(f"  {'Validation':<12} {len(X_val):>8,}  {len(X_val)/total*100:>5.1f}%")
    print(f"  {'Test':<12} {len(X_test):>8,}  {len(X_test)/total*100:>5.1f}%")

    # Grade distribution in each split
    for split_name, y_split in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
        counts = np.bincount(y_split, minlength=len(GRADE_ORDER))
        dist   = "  ".join(f"{g}:{c:,}" for g, c in zip(GRADE_ORDER, counts))
        print(f"    {split_name}: {dist}")

    # ── StandardScaler ────────────────────────────────────────────────────────
    sub("StandardScaler (fit on train only)")
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc   = scaler.transform(X_val)
    X_test_sc  = scaler.transform(X_test)
    print(f"  ✅ Scaler fitted on {len(X_train):,} training samples")

    return {
        "X_train_sc":  X_train_sc,  "X_val_sc":  X_val_sc,  "X_test_sc":  X_test_sc,
        "X_train_raw": X_train,     "X_val_raw": X_val,     "X_test_raw": X_test,
        "y_train":     y_train,     "y_val":     y_val,     "y_test":     y_test,
        "label_encoder": le,
        "scaler":        scaler,
        "feature_names": FEATURE_COLS,
    }


# ================================================================================
# STAGE 4 — MODEL TRAINING
# ================================================================================

def train_models(data: dict) -> dict:
    """
    Train Logistic Regression, Decision Tree, and Random Forest.
    Returns a dict of { model_name → result_dict }.
    """
    banner("STAGE 4 — MODEL TRAINING")

    X_train_sc  = data["X_train_sc"]
    X_val_sc    = data["X_val_sc"]
    X_train_raw = data["X_train_raw"]
    X_val_raw   = data["X_val_raw"]
    y_train     = data["y_train"]
    y_val       = data["y_val"]

    # ── Class weight analysis ─────────────────────────────────────────────────
    sub("Class weight analysis")
    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    print(f"  {'Grade':<7} {'Samples':>9}  {'Weight':>8}  Note")
    for cls, w in zip(classes, weights):
        n    = (y_train == cls).sum()
        note = "↑ upweighted (minority)" if w > 1 else "↓ downweighted (majority)"
        print(f"  {GRADE_ORDER[cls]:<7} {n:>9,}  {w:>8.3f}  {note}")

    results = {}

    # ── Helper: fit, time, evaluate ───────────────────────────────────────────
    def fit_and_eval(name, model, X_tr, X_v, y_tr, y_v):
        print(f"\n  Training {name} …")
        t0 = time.time()
        model.fit(X_tr, y_tr)
        elapsed = round(time.time() - t0, 2)

        y_pred_val   = model.predict(X_v)
        y_pred_train = model.predict(X_tr)
        acc_v  = accuracy_score(y_v, y_pred_val)
        f1_v   = f1_score(y_v, y_pred_val, average="weighted")
        acc_tr = accuracy_score(y_tr, y_pred_train)

        print(f"    Time        : {elapsed}s")
        print(f"    Train  Acc  : {acc_tr:.4f}")
        print(f"    Val    Acc  : {acc_v:.4f}   Val F1: {f1_v:.4f}")

        proba = model.predict_proba(X_v) if hasattr(model, "predict_proba") else None
        return {
            "model":      model,
            "preds_val":  y_pred_val,
            "proba_val":  proba,
            "acc_val":    acc_v,
            "f1_val":     f1_v,
            "acc_train":  acc_tr,
            "time_s":     elapsed,
        }

    # ── 1. Logistic Regression ────────────────────────────────────────────────
    sub("Model 1 — Logistic Regression  (scaled features)")
    lr = LogisticRegression(
        C=1.0, max_iter=1000, class_weight="balanced",
        solver="lbfgs",
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    results["Logistic Regression"] = fit_and_eval(
        "Logistic Regression", lr,
        X_train_sc, X_val_sc, y_train, y_val
    )

    # ── 2. Decision Tree ──────────────────────────────────────────────────────
    sub("Model 2 — Decision Tree  (raw features)")
    dt = DecisionTreeClassifier(
        max_depth=15, min_samples_split=20, min_samples_leaf=10,
        class_weight="balanced", criterion="gini",
        random_state=RANDOM_STATE,
    )
    results["Decision Tree"] = fit_and_eval(
        "Decision Tree", dt,
        X_train_raw, X_val_raw, y_train, y_val
    )
    print(f"    Tree depth  : {dt.get_depth()}")
    print(f"    Num leaves  : {dt.get_n_leaves():,}")

    # ── 3. Random Forest ─────────────────────────────────────────────────────
    sub("Model 3 — Random Forest  ★ Recommended  (raw features)")
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=15, max_features="sqrt",
        min_samples_leaf=5, class_weight="balanced_subsample",
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    results["Random Forest"] = fit_and_eval(
        "Random Forest", rf,
        X_train_raw, X_val_raw, y_train, y_val
    )
    results["Random Forest"]["feature_importances"] = rf.feature_importances_

    # ── Training summary ──────────────────────────────────────────────────────
    sub("Training summary")
    print(f"\n  {'Model':<25} {'Val Acc':>9}  {'Val F1':>9}  {'Train Acc':>10}  {'Time':>6}")
    print("  " + "-" * 68)
    for name, r in results.items():
        star = " ★" if name == "Random Forest" else ""
        print(f"  {name:<25} {r['acc_val']:>9.4f}  {r['f1_val']:>9.4f}  "
              f"{r['acc_train']:>10.4f}  {r['time_s']:>5.1f}s{star}")

    best = max(results, key=lambda k: results[k]["f1_val"])
    print(f"\n  ★ Best model by validation F1: {best}")

    return results


# ================================================================================
# STAGE 5 — EVALUATION
# ================================================================================

def evaluate(data: dict, results: dict) -> None:
    """
    Full evaluation on the held-out test set:
      - Classification report per model
      - Confusion matrices
      - ROC curves (one-vs-rest)
      - Feature importance chart
      - Performance comparison dashboard
    All plots saved to OUTPUT_DIR/plots/
    """
    banner("STAGE 5 — MODEL EVALUATION  (held-out test set)")

    le          = data["label_encoder"]
    y_test      = data["y_test"]
    X_test_sc   = data["X_test_sc"]
    X_test_raw  = data["X_test_raw"]
    feat_names  = data["feature_names"]
    plot_dir    = os.path.join(OUTPUT_DIR, "plots")
    ensure_dir(plot_dir)

    GRADE_COLORS = {
        "A": "#4CAF50", "B": "#8BC34A",
        "C": "#FFC107", "D": "#FF9800", "F": "#F44336",
    }
    color_list = [GRADE_COLORS[g] for g in GRADE_ORDER]

    # ── Per-model test evaluation ─────────────────────────────────────────────
    test_inputs = {
        "Logistic Regression": X_test_sc,
        "Decision Tree":       X_test_raw,
        "Random Forest":       X_test_raw,
    }

    sub("Per-model classification report (test set)")
    final = {}
    for name, r in results.items():
        model  = r["model"]
        X_test = test_inputs[name]
        y_pred = model.predict(X_test)
        proba  = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
        acc    = accuracy_score(y_test, y_pred)
        f1     = f1_score(y_test, y_pred, average="weighted")
        report = classification_report(
            y_test, y_pred, target_names=le.classes_, output_dict=True
        )
        final[name] = {
            "preds": y_pred, "proba": proba,
            "acc": acc, "f1": f1, "report": report,
        }
        star = " ★" if name == "Random Forest" else ""
        print(f"\n  ── {name}{star}")
        print(f"     Test Accuracy : {acc:.4f}")
        print(f"     Test F1 (wtd) : {f1:.4f}")
        report_str = classification_report(y_test, y_pred, target_names=le.classes_)
        for line in report_str.splitlines():
            print(f"     {line}")

    # ── Confusion matrices ────────────────────────────────────────────────────
    sub("Confusion matrices")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), facecolor="#F8F9FA")
    fig.suptitle("Confusion Matrices — Test Set (%)", fontsize=15,
                 fontweight="bold", color="#1A237E")
    cmaps = ["Blues", "Oranges", "Greens"]
    for ax, (name, fd), cmap in zip(axes, final.items(), cmaps):
        cm     = confusion_matrix(y_test, fd["preds"])
        cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
        im = ax.imshow(cm_pct, cmap=cmap, vmin=0, vmax=100)
        for i in range(5):
            for j in range(5):
                ax.text(j, i, f"{cm_pct[i,j]:.1f}%",
                        ha="center", va="center", fontsize=9, fontweight="bold",
                        color="white" if cm_pct[i,j] > 55 else "black")
        ax.set_xticks(range(5)); ax.set_yticks(range(5))
        ax.set_xticklabels(GRADE_ORDER); ax.set_yticklabels(GRADE_ORDER)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        ax.set_title(name, fontweight="bold", fontsize=12)
        plt.colorbar(im, ax=ax, shrink=0.82)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "confusion_matrices.png"),
                dpi=140, bbox_inches="tight")
    plt.close()
    print("  Saved → outputs/plots/confusion_matrices.png")

    # ── ROC curves (RF, one-vs-rest) ──────────────────────────────────────────
    sub("ROC curves — Random Forest (one-vs-rest)")
    rf_proba = final["Random Forest"]["proba"]
    if rf_proba is not None:
        fig, ax = plt.subplots(figsize=(7, 6), facecolor="#F8F9FA")
        for i, (grade, color) in enumerate(zip(GRADE_ORDER, color_list)):
            y_bin = (y_test == i).astype(int)
            fpr, tpr, _ = roc_curve(y_bin, rf_proba[:, i])
            auc_val = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=color, lw=2.2,
                    label=f"Grade {grade}  (AUC = {auc_val:.4f})")
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random baseline")
        ax.set_xlabel("False Positive Rate", fontsize=11)
        ax.set_ylabel("True Positive Rate",  fontsize=11)
        ax.set_title("ROC Curves — Random Forest", fontweight="bold", fontsize=13)
        ax.legend(fontsize=9, loc="lower right")
        ax.set_facecolor("#FAFAFA"); ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "roc_curves.png"),
                    dpi=140, bbox_inches="tight")
        plt.close()
        print("  Saved → outputs/plots/roc_curves.png")

    # ── Feature importance (Random Forest) ───────────────────────────────────
    sub("Feature importance — Random Forest")
    fi      = results["Random Forest"]["feature_importances"]
    sort_ix = np.argsort(fi)
    fig, ax = plt.subplots(figsize=(8, 5), facecolor="#F8F9FA")
    colors  = plt.cm.RdYlGn(np.linspace(0.15, 0.85, len(fi)))
    bars = ax.barh(range(len(fi)), fi[sort_ix],
                   color=[colors[i] for i in range(len(fi))], edgecolor="white")
    ax.set_yticks(range(len(fi)))
    ax.set_yticklabels([feat_names[i].replace("_", " ").title() for i in sort_ix],
                       fontsize=10)
    for bar, val in zip(bars, fi[sort_ix]):
        ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=9, fontweight="bold")
    ax.set_xlabel("Importance Score"); ax.set_xlim(0, fi.max() * 1.18)
    ax.set_title("Random Forest — Feature Importance", fontweight="bold", fontsize=13)
    ax.set_facecolor("#FAFAFA"); ax.grid(axis="x", alpha=0.35)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "feature_importance.png"),
                dpi=140, bbox_inches="tight")
    plt.close()
    print("  Saved → outputs/plots/feature_importance.png")

    # ── Model comparison dashboard ────────────────────────────────────────────
    sub("Model comparison dashboard")
    fig = plt.figure(figsize=(16, 10), facecolor="#F8F9FA")
    fig.suptitle("Model Performance Dashboard — Test Set",
                 fontsize=16, fontweight="bold", color="#1A237E")
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

    model_names = list(final.keys())
    short_names = ["Log. Reg.", "Dec. Tree", "Rand. Forest"]
    accs = [final[n]["acc"] for n in model_names]
    f1s  = [final[n]["f1"]  for n in model_names]
    times = [results[n]["time_s"] for n in model_names]

    # Accuracy vs F1 bar
    ax1 = fig.add_subplot(gs[0, 0])
    x   = np.arange(3)
    b1  = ax1.bar(x - 0.2, accs, 0.35, label="Accuracy", color="#3F51B5", alpha=0.85)
    b2  = ax1.bar(x + 0.2, f1s,  0.35, label="F1 Score",  color="#E91E63", alpha=0.85)
    for b in [*b1, *b2]:
        ax1.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.0005,
                 f"{b.get_height():.4f}", ha="center", va="bottom", fontsize=7.5,
                 fontweight="bold")
    ax1.set_xticks(x); ax1.set_xticklabels(short_names, fontsize=8.5)
    ax1.set_ylim(0.96, 1.005); ax1.legend(fontsize=9)
    ax1.set_title("Accuracy vs F1 Score", fontweight="bold")
    ax1.set_facecolor("#FAFAFA"); ax1.grid(axis="y", alpha=0.35)
    ax1.spines[["top", "right"]].set_visible(False)

    # Training time
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.barh(short_names, times,
             color=["#5C6BC0", "#26A69A", "#EF5350"], alpha=0.85, edgecolor="white")
    for i, t in enumerate(times):
        ax2.text(t + 0.05, i, f"{t}s", va="center", fontsize=10, fontweight="bold")
    ax2.set_title("Training Time (seconds)", fontweight="bold")
    ax2.set_facecolor("#FAFAFA"); ax2.grid(axis="x", alpha=0.35)
    ax2.spines[["top", "right"]].set_visible(False)

    # Per-grade F1 comparison
    ax3 = fig.add_subplot(gs[0, 2])
    x   = np.arange(5)
    bw  = 0.25
    bar_colors = ["#3F51B5", "#009688", "#E91E63"]
    for idx, (name, sname, bc) in enumerate(
            zip(model_names, short_names, bar_colors)):
        f1_per_grade = [final[name]["report"][g]["f1-score"] for g in GRADE_ORDER]
        ax3.bar(x + (idx - 1) * bw, f1_per_grade, bw, label=sname,
                color=bc, alpha=0.80)
    ax3.set_xticks(x); ax3.set_xticklabels(GRADE_ORDER)
    ax3.set_ylim(0, 1.1); ax3.legend(fontsize=8)
    ax3.set_title("Per-Grade F1 Score", fontweight="bold")
    ax3.set_facecolor("#FAFAFA"); ax3.grid(axis="y", alpha=0.35)
    ax3.spines[["top", "right"]].set_visible(False)

    # RF confusion matrix (best model)
    ax4 = fig.add_subplot(gs[1, 0])
    cm_rf  = confusion_matrix(y_test, final["Random Forest"]["preds"])
    cm_pct = cm_rf.astype(float) / cm_rf.sum(axis=1, keepdims=True) * 100
    im = ax4.imshow(cm_pct, cmap="Greens", vmin=0, vmax=100)
    for i in range(5):
        for j in range(5):
            ax4.text(j, i, f"{cm_pct[i,j]:.1f}%",
                     ha="center", va="center", fontsize=8.5, fontweight="bold",
                     color="white" if cm_pct[i,j] > 55 else "black")
    ax4.set_xticks(range(5)); ax4.set_yticks(range(5))
    ax4.set_xticklabels(GRADE_ORDER); ax4.set_yticklabels(GRADE_ORDER)
    ax4.set_xlabel("Predicted"); ax4.set_ylabel("Actual")
    ax4.set_title("RF Confusion Matrix", fontweight="bold")
    plt.colorbar(im, ax=ax4, shrink=0.82)

    # Feature importance
    ax5 = fig.add_subplot(gs[1, 1])
    fi_sorted = fi[sort_ix]
    feat_sorted = [feat_names[i].replace("_", " ").title() for i in sort_ix]
    ax5.barh(feat_sorted, fi_sorted,
             color=plt.cm.RdYlGn(np.linspace(0.15, 0.85, len(fi))), edgecolor="white")
    for i, v in enumerate(fi_sorted):
        ax5.text(v + 0.002, i, f"{v:.3f}", va="center", fontsize=8.5, fontweight="bold")
    ax5.set_title("Feature Importance (RF)", fontweight="bold")
    ax5.set_facecolor("#FAFAFA"); ax5.grid(axis="x", alpha=0.35)
    ax5.spines[["top", "right"]].set_visible(False)

    # ROC (RF)
    ax6 = fig.add_subplot(gs[1, 2])
    if rf_proba is not None:
        for i, (grade, color) in enumerate(zip(GRADE_ORDER, color_list)):
            y_bin = (y_test == i).astype(int)
            fpr, tpr, _ = roc_curve(y_bin, rf_proba[:, i])
            ax6.plot(fpr, tpr, color=color, lw=2,
                     label=f"{grade} (AUC={auc(fpr,tpr):.3f})")
        ax6.plot([0,1],[0,1],"k--",lw=1,alpha=0.5)
        ax6.set_xlabel("FPR"); ax6.set_ylabel("TPR")
        ax6.set_title("ROC Curves (RF)", fontweight="bold")
        ax6.legend(fontsize=8, loc="lower right")
        ax6.set_facecolor("#FAFAFA"); ax6.grid(alpha=0.3)
        ax6.spines[["top", "right"]].set_visible(False)

    plt.savefig(os.path.join(plot_dir, "model_dashboard.png"),
                dpi=140, bbox_inches="tight", facecolor="#F8F9FA")
    plt.close()
    print("  Saved → outputs/plots/model_dashboard.png")

    # ── Save JSON metrics ─────────────────────────────────────────────────────
    metrics_out = {}
    for name, fd in final.items():
        metrics_out[name] = {
            "test_accuracy": round(fd["acc"], 6),
            "test_f1_weighted": round(fd["f1"], 6),
            "train_time_s": results[name]["time_s"],
        }
    metrics_out["best_model"] = max(final, key=lambda k: final[k]["f1"])
    with open(os.path.join(OUTPUT_DIR, "evaluation_metrics.json"), "w") as f:
        json.dump(metrics_out, f, indent=2)
    print(f"\n  ✅ Metrics saved → {OUTPUT_DIR}/evaluation_metrics.json")

    # ── Final leaderboard ─────────────────────────────────────────────────────
    sub("Final leaderboard")
    print(f"\n  {'Rank':<6} {'Model':<25} {'Test Acc':>10}  {'Test F1':>9}  {'Train Time':>10}")
    print("  " + "-" * 66)
    ranked = sorted(final.items(), key=lambda x: x[1]["f1"], reverse=True)
    for rank, (name, fd) in enumerate(ranked, 1):
        star = " ★ BEST" if rank == 1 else ""
        print(f"  {rank:<6} {name:<25} {fd['acc']:>10.4f}  {fd['f1']:>9.4f}  "
              f"{results[name]['time_s']:>8.1f}s{star}")


# ================================================================================
# STAGE 6 — PREDICTION INTERFACE
# ================================================================================

def predict_student(model, scaler: StandardScaler, le: LabelEncoder,
                    use_scaling: bool,
                    weekly_study_hours:     float,
                    attendance_percentage:  float,
                    class_participation:    float,
                    total_score:            float) -> dict:
    """
    Predict the grade for a single student.

    Parameters
    ----------
    model               : Trained sklearn classifier.
    scaler              : Fitted StandardScaler (used if use_scaling=True).
    le                  : Fitted LabelEncoder to decode predictions.
    use_scaling         : True for Logistic Regression, False for tree models.
    weekly_study_hours  : Hours of self-study per week  (0–40).
    attendance_percentage : Attendance rate in %          (50–100).
    class_participation : Class participation score      (0–10).
    total_score         : Cumulative score               (0–100).

    Returns
    -------
    dict with predicted_grade, confidence, all_probabilities.
    """
    # Derive engineered features
    engagement  = (0.5 * attendance_percentage / 100.0
                   + 0.5 * class_participation / 10.0)
    intensity   = min(6, int(weekly_study_hours / 5))   # simple bucket
    high_attend = int(attendance_percentage >= 90)
    active      = int(class_participation   >= 7)

    row = np.array([[weekly_study_hours, attendance_percentage,
                     class_participation, total_score,
                     engagement, intensity, high_attend, active]])

    if use_scaling:
        row = scaler.transform(row)

    pred_enc = model.predict(row)[0]
    pred_grade = le.inverse_transform([pred_enc])[0]
    proba = model.predict_proba(row)[0]

    return {
        "predicted_grade": pred_grade,
        "confidence":      round(float(proba[pred_enc]) * 100, 2),
        "all_probabilities": {
            g: round(float(p) * 100, 2)
            for g, p in zip(GRADE_ORDER, proba)
        },
    }


def demo_predictions(results: dict, data: dict) -> None:
    """Show example predictions for three hypothetical students."""
    banner("STAGE 6 — PREDICTION INTERFACE  (demo)")

    rf     = results["Random Forest"]["model"]
    scaler = data["scaler"]
    le     = data["label_encoder"]

    students = [
        {"name": "Alice (High performer)",
         "weekly_study_hours": 28, "attendance_percentage": 95,
         "class_participation": 9, "total_score": 94},
        {"name": "Bob (Average student)",
         "weekly_study_hours": 14, "attendance_percentage": 80,
         "class_participation": 6, "total_score": 73},
        {"name": "Charlie (At-risk student)",
         "weekly_study_hours": 3,  "attendance_percentage": 58,
         "class_participation": 2, "total_score": 38},
    ]

    for s in students:
        result = predict_student(
            model=rf, scaler=scaler, le=le, use_scaling=False,
            weekly_study_hours=s["weekly_study_hours"],
            attendance_percentage=s["attendance_percentage"],
            class_participation=s["class_participation"],
            total_score=s["total_score"],
        )
        print(f"\n  Student : {s['name']}")
        print(f"  Inputs  : Study={s['weekly_study_hours']}h/wk  "
              f"Attend={s['attendance_percentage']}%  "
              f"Partic={s['class_participation']}/10  "
              f"Score={s['total_score']}")
        print(f"  ► Predicted Grade : {result['predicted_grade']}")
        print(f"  ► Confidence      : {result['confidence']}%")
        print(f"  ► All probs       : {result['all_probabilities']}")


# ================================================================================
# SAVE MODELS
# ================================================================================

def save_models(results: dict, data: dict) -> None:
    """Serialise trained models, scaler, and label encoder to disk."""
    banner("SAVING MODELS & ARTEFACTS")
    art_dir = os.path.join(OUTPUT_DIR, "artifacts")
    ensure_dir(art_dir)

    for name, r in results.items():
        safe_name = name.lower().replace(" ", "_")
        path = os.path.join(art_dir, f"{safe_name}.pkl")
        with open(path, "wb") as f:
            pickle.dump(r["model"], f)
        print(f"  Saved {name:<25} → {path}")

    with open(os.path.join(art_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(data["scaler"], f)
    with open(os.path.join(art_dir, "label_encoder.pkl"), "wb") as f:
        pickle.dump(data["label_encoder"], f)
    print(f"  Saved scaler and label_encoder → {art_dir}/")
    print(f"\n  ✅ All artefacts saved")


# ================================================================================
# MAIN  —  run all stages in sequence
# ================================================================================

def main():
    start_total = time.time()

    ensure_dir(OUTPUT_DIR)
    ensure_dir(os.path.join(OUTPUT_DIR, "plots"))

    # Run pipeline
    df         = ingest(DATA_PATH)
    df         = engineer_features(df)
    data       = preprocess(df)
    results    = train_models(data)
    evaluate(data, results)
    save_models(results, data)
    demo_predictions(results, data)

    elapsed = round(time.time() - start_total, 1)
    banner(f"PIPELINE COMPLETE  —  total time: {elapsed}s")
    print(f"\n  Outputs written to: {os.path.abspath(OUTPUT_DIR)}/")
    print(f"  ├── plots/confusion_matrices.png")
    print(f"  ├── plots/roc_curves.png")
    print(f"  ├── plots/feature_importance.png")
    print(f"  ├── plots/model_dashboard.png")
    print(f"  ├── evaluation_metrics.json")
    print(f"  └── artifacts/  (random_forest.pkl, scaler.pkl, label_encoder.pkl …)")
    print()


if __name__ == "__main__":
    main()
