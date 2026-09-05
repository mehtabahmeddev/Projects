"""
Advanced Model Training Pipeline
==================================
Uses ensemble ML + class balancing for production-grade fraud detection.
Techniques from: FICO Falcon, DataRobot, AWS Fraud Detector
"""

import os
import sys
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score,
    precision_recall_curve, roc_curve
)

# Add parent dir to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.features import create_features, get_feature_columns

warnings.filterwarnings("ignore")
os.makedirs("model", exist_ok=True)
os.makedirs("reports", exist_ok=True)


# ─── 1. LOAD & EXPAND DATA ───────────────────────────────────

def generate_training_data() -> pd.DataFrame:
    """
    Generate a realistic synthetic dataset with fraud patterns.
    In production this would be your real transaction DB.
    """
    np.random.seed(42)
    n = 2000

    # Legitimate transactions
    legit = pd.DataFrame({
        "user_id":       np.random.randint(1, 200, n),
        "amount":        np.random.normal(300, 80, n).clip(10),
        "avg_amount":    np.random.normal(280, 60, n).clip(50),
        "merchant":      np.random.choice(["Amazon","Daraz","Foodpanda","Netflix","Careem"], n),
        "location":      np.random.choice(["Karachi","Lahore","Islamabad","Multan","Faisalabad"], n),
        "prev_location": np.random.choice(["Karachi","Lahore","Islamabad","Multan","Faisalabad"], n),
        "device":        np.random.choice(["Mobile","Desktop","Tablet"], n, p=[0.6, 0.3, 0.1]),
        "prev_device":   np.random.choice(["Mobile","Desktop","Tablet"], n, p=[0.6, 0.3, 0.1]),
        "time_hour":     np.random.choice(range(8, 22), n),
        "txn_count_last_hour": np.random.randint(0, 3, n),
        "total_amount_today":  np.random.normal(800, 200, n).clip(0),
        "is_fraud":      0
    })

    # Fraudulent transactions (10% class — realistic imbalance)
    n_fraud = 200
    fraud = pd.DataFrame({
        "user_id":       np.random.randint(1, 200, n_fraud),
        "amount":        np.where(
                             np.random.random(n_fraud) < 0.8,
                             np.random.normal(4000, 1000, n_fraud).clip(500),
                             np.random.uniform(0.5, 5, n_fraud)
                         ),
        "avg_amount":    np.random.normal(280, 60, n_fraud).clip(50),
        "merchant":      np.random.choice(["Unknown","Crypto Exchange","Gift Cards","Anonymous","Wire Transfer"], n_fraud),
        "location":      np.random.choice(["London","Dubai","NewYork","North Korea","Russia","Iran"], n_fraud),
        "prev_location": np.random.choice(["Karachi","Lahore","Islamabad"], n_fraud),
        "device":        np.random.choice(["Web","Unknown","TOR","VPN"], n_fraud, p=[0.5,0.2,0.15,0.15]),
        "prev_device":   np.random.choice(["Mobile","Desktop"], n_fraud),
        "time_hour":     np.random.choice([0,1,2,3,4,23], n_fraud),
        "txn_count_last_hour": np.random.randint(4, 15, n_fraud),
        "total_amount_today":  np.random.normal(40000, 5000, n_fraud).clip(0),
        "is_fraud":      1
    })

    df = pd.concat([legit, fraud], ignore_index=True).sample(frac=1, random_state=42)
    return df


# ─── 2. TRAIN ────────────────────────────────────────────────

def train():
    print("=" * 60)
    print("  ADVANCED FRAUD DETECTION MODEL TRAINING")
    print("=" * 60)

    # Load or generate data
    csv_path = "data/transactions.csv"
    if os.path.exists(csv_path):
        base_df = pd.read_csv(csv_path)
        print(f"✅ Loaded CSV: {len(base_df)} rows")
    else:
        base_df = pd.DataFrame()

    # Merge with synthetic data
    synthetic = generate_training_data()
    if len(base_df) > 0:
        # Add missing columns to base_df
        for col in synthetic.columns:
            if col not in base_df.columns:
                base_df[col] = 0
        df = pd.concat([base_df, synthetic], ignore_index=True)
    else:
        df = synthetic

    print(f"📊 Total dataset size: {len(df)} transactions")
    print(f"🚨 Fraud rate: {df['is_fraud'].mean():.1%}")

    # Feature engineering
    df = create_features(df)

    # Feature selection
    feature_cols = get_feature_columns()
    available = [c for c in feature_cols if c in df.columns]
    X = df[available].fillna(0)
    y = df["is_fraud"]

    print(f"🔧 Features used: {len(available)}")

    # Train/test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # ── Ensemble Model ───────────────────────────────────────
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    gb = GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.08,
        max_depth=5,
        random_state=42
    )

    lr = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=500, random_state=42))
    ])

    ensemble = VotingClassifier(
        estimators=[("rf", rf), ("gb", gb), ("lr", lr)],
        voting="soft",
        weights=[3, 2, 1]   # RF gets most weight
    )

    print("\n🤖 Training ensemble model (RF + GBM + LR)...")
    ensemble.fit(X_train, y_train)

    # ── Evaluation ────────────────────────────────────────────
    y_pred  = ensemble.predict(X_test)
    y_proba = ensemble.predict_proba(X_test)[:, 1]

    print("\n📈 Model Performance:")
    print(classification_report(y_test, y_pred, target_names=["Legitimate","Fraud"]))
    print(f"ROC-AUC:              {roc_auc_score(y_test, y_proba):.4f}")
    print(f"Average Precision:    {average_precision_score(y_test, y_proba):.4f}")

    # Cross-validation
    cv_scores = cross_val_score(
        RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42),
        X, y, cv=StratifiedKFold(5), scoring="roc_auc"
    )
    print(f"5-Fold CV ROC-AUC:    {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # ── Save ──────────────────────────────────────────────────
    model_data = {
        "model":         ensemble,
        "feature_cols":  available,
        "roc_auc":       roc_auc_score(y_test, y_proba),
        "avg_precision": average_precision_score(y_test, y_proba),
    }
    joblib.dump(model_data, "model/advanced_model.pkl")
    print("\n✅ Advanced model saved → model/advanced_model.pkl")

    # Also save a simple RF for fallback
    simple_rf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
    simple_rf.fit(X_train[["amount","avg_amount","is_high_amount","is_night",
                             "is_new_location","is_new_device","is_unknown_merchant"]], y_train)
    joblib.dump(simple_rf, "model/model.pkl")
    print("✅ Fallback model saved → model/model.pkl")

    return model_data


if __name__ == "__main__":
    train()
