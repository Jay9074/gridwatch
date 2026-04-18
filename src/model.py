"""
GridWatch — src/model.py
==========================
Trains and evaluates ML models for outage risk prediction.

Models:
  1. Logistic Regression  (baseline)
  2. Random Forest        (interpretable ensemble)
  3. XGBoost              (best performer)

Key feature: SHAP explainability — shows WHY each prediction is made.
This separates junior from senior data science work.

Author: Jaykumar Patel
"""

import json
import pickle
import logging
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
warnings.filterwarnings("ignore")

from pathlib import Path
from datetime import datetime

from sklearn.model_selection  import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble         import RandomForestClassifier
from sklearn.linear_model     import LogisticRegression
from sklearn.preprocessing    import StandardScaler
from sklearn.metrics          import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    ConfusionMatrixDisplay, roc_curve, precision_recall_curve
)
from sklearn.pipeline         import Pipeline
from imblearn.over_sampling   import SMOTE
import xgboost as xgb
import shap

log = logging.getLogger(__name__)

BASE_DIR  = Path(__file__).parent.parent
PROC_DIR  = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ── Synthetic data (used until real data is ingested) ────────────
def _make_synthetic(n: int = 8_000, seed: int = 42) -> pd.DataFrame:
    """
    Realistic synthetic training data based on known Northeast
    outage patterns from DOE OE-417 historical analysis.
    Replace with real data after running data_ingestion.py.
    """
    rng = np.random.default_rng(seed)
    months = rng.integers(1, 13, n)
    seasons = np.where(months.isin([12,1,2]) if hasattr(months,'isin')
                       else np.isin(months,[12,1,2]),
                       "Winter",
                       np.where(np.isin(months,[3,4,5]), "Spring",
                       np.where(np.isin(months,[6,7,8]), "Summer", "Fall")))

    df = pd.DataFrame({
        "month":                    months,
        "quarter":                  (months - 1) // 3 + 1,
        "day_of_week":              rng.integers(0, 7, n),
        "is_weekend":               rng.integers(0, 2, n),
        "month_sin":                np.sin(2 * np.pi * months / 12),
        "month_cos":                np.cos(2 * np.pi * months / 12),
        "is_high_risk_month":       np.isin(months, [12,1,2,3]).astype(int),
        "season_risk_score":        np.where(np.isin(months,[12,1,2,3]), 3,
                                    np.where(np.isin(months,[9,10,11,6,7,8]), 2, 1)),
        "log_demand_loss":          rng.gamma(2, 2, n),
        "is_high_mw_loss":          rng.binomial(1, 0.15, n),
        "log_customers":            rng.gamma(3, 2, n),
        "state_risk_score":         rng.uniform(0.55, 0.85, n),
        "is_weather_caused":        rng.binomial(1, 0.58, n),
        "is_equipment_failure":     rng.binomial(1, 0.25, n),
        "is_cyber":                 rng.binomial(1, 0.04, n),
        "rolling_12mo_events":      rng.poisson(3, n),
        "ewma_customers":           rng.gamma(2, 15_000, n),
        "year":                     rng.integers(2015, 2025, n),
        "season_Winter":            np.isin(months,[12,1,2]).astype(int),
        "season_Spring":            np.isin(months,[3,4,5]).astype(int),
        "season_Summer":            np.isin(months,[6,7,8]).astype(int),
        "season_Fall":              np.isin(months,[9,10,11]).astype(int),
    })

    # Target — realistic correlations
    risk = (
        df["is_high_risk_month"]    * 2.0 +
        df["is_weather_caused"]     * 1.8 +
        df["is_high_mw_loss"]       * 1.5 +
        df["state_risk_score"]      * 1.2 +
        df["rolling_12mo_events"]   * 0.3 +
        df["log_demand_loss"]       * 0.4 +
        rng.normal(0, 0.8, n)
    )
    df["is_major_outage"] = (risk > 4.2).astype(int)
    log.info(f"Synthetic data: {n:,} rows | major outage rate: {df['is_major_outage'].mean():.1%}")
    return df


# ── Load data ────────────────────────────────────────────────────
def load_data(feature_cols: list = None) -> tuple:
    """Loads real processed data or falls back to synthetic."""
    proc_file = PROC_DIR / "doe_outages_northeast.csv"

    if proc_file.exists():
        log.info("Loading real processed data...")
        from feature_engineering import (
            load_processed_data, create_outage_features,
            create_weather_features, build_ml_dataset
        )
        doe, noaa = load_processed_data()
        doe_feat  = create_outage_features(doe)
        X, y, names = build_ml_dataset(doe_feat)
        return X, y, names
    else:
        log.info("Real data not found — using synthetic data for development.")
        df   = _make_synthetic()
        cols = [c for c in df.columns if c != "is_major_outage"]
        return df[cols], df["is_major_outage"], cols


# ── Train ────────────────────────────────────────────────────────
def train(X_tr, X_te, y_tr, y_te, feat_names: list) -> dict:
    """Trains all three models and returns metrics + objects."""

    # Handle class imbalance with SMOTE
    sm = SMOTE(random_state=42)
    X_bal, y_bal = sm.fit_resample(X_tr, y_tr)
    log.info(f"After SMOTE — {len(X_bal):,} training samples "
             f"(positive rate: {y_bal.mean():.1%})")

    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_bal)
    X_te_sc = scaler.transform(X_te)

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, C=1.0, random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=15, min_samples_split=5,
            class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=300, max_depth=7, learning_rate=0.08,
            subsample=0.85, colsample_bytree=0.85,
            scale_pos_weight=(y_bal==0).sum()/(y_bal==1).sum(),
            random_state=42, eval_metric="logloss",
            use_label_encoder=False, verbosity=0
        ),
    }

    results = {}
    for name, mdl in models.items():
        log.info(f"Training {name}...")

        use_scaled = name == "Logistic Regression"
        Xtr = X_tr_sc if use_scaled else X_bal
        Xte = X_te_sc if use_scaled else X_te
        ytr = y_bal

        # Cross-validation on training set
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(mdl, Xtr, ytr, cv=cv,
                                    scoring="roc_auc", n_jobs=-1)

        mdl.fit(Xtr, ytr)
        y_pred = mdl.predict(Xte)
        y_prob = mdl.predict_proba(Xte)[:, 1]

        metrics = {
            "accuracy":  round(float(accuracy_score(y_te, y_pred)),  4),
            "precision": round(float(precision_score(y_te, y_pred, zero_division=0)), 4),
            "recall":    round(float(recall_score(y_te, y_pred)),     4),
            "f1_score":  round(float(f1_score(y_te, y_pred)),         4),
            "roc_auc":   round(float(roc_auc_score(y_te, y_prob)),    4),
            "cv_roc_auc_mean": round(float(cv_scores.mean()), 4),
            "cv_roc_auc_std":  round(float(cv_scores.std()),  4),
        }

        results[name] = {
            "model": mdl, "scaler": scaler if use_scaled else None,
            "metrics": metrics, "y_pred": y_pred, "y_prob": y_prob
        }

        log.info(f"  Accuracy : {metrics['accuracy']:.4f}")
        log.info(f"  F1       : {metrics['f1_score']:.4f}")
        log.info(f"  ROC-AUC  : {metrics['roc_auc']:.4f}  "
                 f"(CV: {metrics['cv_roc_auc_mean']:.4f} ± {metrics['cv_roc_auc_std']:.4f})")

    return results


# ── SHAP Explainability ──────────────────────────────────────────
def compute_shap(model, X_sample: pd.DataFrame, feat_names: list,
                 model_name: str = "XGBoost"):
    """
    Computes SHAP values — the most important part for advanced DS.

    SHAP (SHapley Additive exPlanations) explains EXACTLY why
    the model predicted what it predicted for each sample.
    This is what separates junior from senior data scientists.
    """
    log.info(f"Computing SHAP values for {model_name}...")

    try:
        if "XGBoost" in model_name or "Forest" in model_name:
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.LinearExplainer(model, X_sample)

        shap_values = explainer.shap_values(X_sample)

        # Handle multi-class output from RF
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # class 1 (major outage)

        # ── Summary Plot (beeswarm) ──────────────────────────────
        fig, ax = plt.subplots(figsize=(10, 7))
        shap.summary_plot(
            shap_values, X_sample,
            feature_names=feat_names,
            show=False, plot_size=None
        )
        plt.title(f"SHAP Feature Impact — {model_name}\n"
                  f"(Each dot = one prediction | Color = feature value)",
                  fontsize=12, pad=12)
        plt.tight_layout()
        plt.savefig(MODEL_DIR / "shap_summary.png", dpi=150, bbox_inches="tight")
        plt.close()

        # ── Bar Plot (mean absolute SHAP) ────────────────────────
        mean_shap = pd.Series(
            np.abs(shap_values).mean(axis=0),
            index=feat_names
        ).sort_values(ascending=True).tail(15)

        fig, ax = plt.subplots(figsize=(10, 7))
        colors = ["#e63946" if v > mean_shap.quantile(0.75) else "#457b9d"
                  for v in mean_shap.values]
        mean_shap.plot(kind="barh", ax=ax, color=colors)
        ax.set_title(f"Mean |SHAP| — Top 15 Features ({model_name})",
                     fontsize=13, fontweight="bold")
        ax.set_xlabel("Mean |SHAP value| (impact on prediction)", fontsize=11)
        ax.spines[["top","right"]].set_visible(False)
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        plt.savefig(MODEL_DIR / "shap_bar.png", dpi=150, bbox_inches="tight")
        plt.close()

        log.info(f"SHAP plots saved → models/shap_summary.png, models/shap_bar.png")
        return shap_values, explainer

    except Exception as e:
        log.warning(f"SHAP computation failed: {e}")
        return None, None


# ── Save best model ──────────────────────────────────────────────
def save_best_model(results: dict, feat_names: list):
    """Saves best model (by F1) and all metrics to disk."""
    best_name = max(results, key=lambda k: results[k]["metrics"]["f1_score"])
    bundle = {
        "model":         results[best_name]["model"],
        "scaler":        results[best_name]["scaler"],
        "model_name":    best_name,
        "feature_names": feat_names,
        "trained_at":    datetime.now().isoformat(),
        "metrics":       results[best_name]["metrics"],
    }
    with open(MODEL_DIR / "outage_risk_model.pkl", "wb") as f:
        pickle.dump(bundle, f)

    # Save all metrics
    all_metrics = {
        name: data["metrics"] for name, data in results.items()
    }
    all_metrics["best_model"] = best_name
    all_metrics["trained_at"] = datetime.now().isoformat()

    with open(MODEL_DIR / "model_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    log.info(f"Best model saved: {best_name}  "
             f"(F1={results[best_name]['metrics']['f1_score']:.4f})")
    return best_name


# ── Evaluation plots ─────────────────────────────────────────────
def plot_evaluation(results: dict, y_te):
    """ROC curves and confusion matrix for all models."""
    colors = {"Logistic Regression":"#2a9d8f",
              "Random Forest":"#457b9d",
              "XGBoost":"#e63946"}

    # ROC curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    for name, data in results.items():
        fpr, tpr, _ = roc_curve(y_te, data["y_prob"])
        auc = data["metrics"]["roc_auc"]
        ax.plot(fpr, tpr, label=f"{name}  (AUC={auc:.3f})",
                color=colors[name], linewidth=2)
    ax.plot([0,1],[0,1],"k--", alpha=0.5, label="Random")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — All Models", fontweight="bold")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # Confusion matrix for best model
    ax = axes[1]
    best_name = max(results, key=lambda k: results[k]["metrics"]["f1_score"])
    cm = confusion_matrix(y_te, results[best_name]["y_pred"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["No Outage","Major Outage"],
                yticklabels=["No Outage","Major Outage"])
    ax.set_title(f"Confusion Matrix — {best_name}", fontweight="bold")
    ax.set_ylabel("Actual"); ax.set_xlabel("Predicted")

    plt.tight_layout()
    plt.savefig(MODEL_DIR / "model_evaluation.png", dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Evaluation plots saved → models/model_evaluation.png")


# ── Full pipeline ────────────────────────────────────────────────
def run_pipeline():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    log.info("=" * 55)
    log.info("GridWatch — ML Training Pipeline")
    log.info("=" * 55)

    X, y, feat_names = load_data()
    log.info(f"Dataset: {X.shape[0]:,} rows × {X.shape[1]} features")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    log.info(f"Train: {len(X_tr):,} | Test: {len(X_te):,}")

    results = train(X_tr, X_te, y_tr, y_te, feat_names)

    best_name = save_best_model(results, feat_names)
    plot_evaluation(results, y_te)

    # SHAP on best model (subsample for speed)
    best_model = results[best_name]["model"]
    sample_idx = np.random.choice(len(X_te), min(500, len(X_te)), replace=False)
    X_sample = pd.DataFrame(X_te.values[sample_idx],
                             columns=feat_names) if not isinstance(X_te, pd.DataFrame) \
               else X_te.iloc[sample_idx]
    compute_shap(best_model, X_sample, feat_names, best_name)

    # Final summary
    log.info("\n" + "=" * 55)
    log.info("✅ Training Complete")
    log.info("=" * 55)
    for name, data in results.items():
        m = data["metrics"]
        log.info(f"\n{name}:")
        log.info(f"  Accuracy  : {m['accuracy']:.4f}")
        log.info(f"  Precision : {m['precision']:.4f}")
        log.info(f"  Recall    : {m['recall']:.4f}")
        log.info(f"  F1        : {m['f1_score']:.4f}")
        log.info(f"  ROC-AUC   : {m['roc_auc']:.4f}")
        log.info(f"  CV AUC    : {m['cv_roc_auc_mean']:.4f} ± {m['cv_roc_auc_std']:.4f}")

    return results


if __name__ == "__main__":
    run_pipeline()
