"""
Advanced Prediction Module
============================
Dual-layer scoring:
  Layer 1: ML Ensemble (RandomForest + GBM + LogisticRegression)
  Layer 2: Rule-based risk score (weighted flags)
Final score = weighted combination of both layers.

Approach used by: Visa Advanced Auth, Mastercard Decision Intelligence
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.features import create_features, get_feature_columns, get_flag_explanations
from config import HIGH_RISK_THRESHOLD, MEDIUM_RISK_THRESHOLD


# ─── Load Models ─────────────────────────────────────────────

def load_model():
    """Load the best available model (advanced → fallback)."""
    adv_path = "model/advanced_model.pkl"
    fallback  = "model/model.pkl"

    if os.path.exists(adv_path):
        data = joblib.load(adv_path)
        return data["model"], data.get("feature_cols", get_feature_columns()), True
    elif os.path.exists(fallback):
        model = joblib.load(fallback)
        basic_cols = ["amount","avg_amount","is_high_amount","is_night",
                      "is_new_location","is_new_device","is_unknown_merchant"]
        return model, basic_cols, False
    else:
        raise FileNotFoundError("No model found. Run src/train.py first.")


# ─── Prediction ───────────────────────────────────────────────

def predict_transaction(data: dict) -> dict:
    """
    Full prediction pipeline for a single transaction.

    Returns:
        dict with keys:
          - ml_score         : ML model probability (0-1)
          - rule_score       : Rule-based score (0-1)
          - final_score      : Combined risk score (0-1)
          - risk_level       : 'HIGH' | 'MEDIUM' | 'LOW'
          - is_fraud         : bool
          - flags            : list of triggered flag names
          - explanations     : list of human-readable reasons
          - feature_values   : dict of all computed features
    """
    model, feature_cols, is_advanced = load_model()

    df = pd.DataFrame([data])
    df = create_features(df)

    # ML Score
    available = [c for c in feature_cols if c in df.columns]
    X = df[available].fillna(0)

    ml_score  = model.predict_proba(X)[0][1]

    # Rule-based score
    rule_score = float(df["rule_based_score"].iloc[0]) if "rule_based_score" in df.columns else 0.0

    # Combined score (70% ML, 30% rules — like Stripe Radar)
    final_score = 0.70 * ml_score + 0.30 * rule_score

    # Risk level
    if final_score >= HIGH_RISK_THRESHOLD:
        risk_level = "HIGH"
    elif final_score >= MEDIUM_RISK_THRESHOLD:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    # Explanations
    explanations = get_flag_explanations(df.iloc[0])

    # Feature values for display
    flag_cols = [c for c in df.columns if c.startswith("is_")]
    feature_values = df[flag_cols].iloc[0].to_dict()

    return {
        "ml_score":      round(ml_score, 4),
        "rule_score":    round(rule_score, 4),
        "final_score":   round(final_score, 4),
        "risk_level":    risk_level,
        "is_fraud":      final_score >= HIGH_RISK_THRESHOLD,
        "flags":         [k for k, v in feature_values.items() if v],
        "explanations":  explanations,
        "feature_values": feature_values,
        "total_flags":   int(df["total_flags"].iloc[0]) if "total_flags" in df.columns else 0,
        "is_advanced":   is_advanced,
    }


# ─── CLI Demo ─────────────────────────────────────────────────

if __name__ == "__main__":
    samples = [
        {
            "name": "Normal Transaction",
            "amount": 300, "avg_amount": 280,
            "merchant": "Amazon", "location": "Karachi",
            "device": "Mobile", "prev_location": "Karachi",
            "prev_device": "Mobile", "time_hour": 14,
        },
        {
            "name": "Suspicious Transaction",
            "amount": 8000, "avg_amount": 280,
            "merchant": "Unknown", "location": "Dubai",
            "device": "Unknown", "prev_location": "Karachi",
            "prev_device": "Mobile", "time_hour": 2,
        },
        {
            "name": "Card Testing (Micro)",
            "amount": 1, "avg_amount": 500,
            "merchant": "Unknown", "location": "NewYork",
            "device": "Web", "prev_location": "Lahore",
            "prev_device": "Mobile", "time_hour": 3,
        },
    ]

    for sample in samples:
        name = sample.pop("name")
        result = predict_transaction(sample)
        print(f"\n{'─'*50}")
        print(f"  {name}")
        print(f"{'─'*50}")
        print(f"  ML Score:    {result['ml_score']:.2%}")
        print(f"  Rule Score:  {result['rule_score']:.2%}")
        print(f"  Final Score: {result['final_score']:.2%}")
        print(f"  Risk Level:  {result['risk_level']}")
        print(f"  Flags ({result['total_flags']}): {', '.join(result['flags'][:4]) or 'None'}")
