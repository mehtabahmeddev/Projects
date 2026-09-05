"""
Advanced Feature Engineering for Fraud Detection
=================================================
Features inspired by top fraud detection systems:
  - PayPal Risk Engine
  - Stripe Radar
  - Visa Advanced Authorization
  - Mastercard Decision Intelligence
  - Sift Science
  - FICO Falcon
  - IBM Safer Payments
  - AWS Fraud Detector
  - Feedzai
  - DataRobot Fraud Platform

Feature Categories:
  1. Amount-Based Features
  2. Time-Based Features
  3. Geographic Features
  4. Device & Channel Features
  5. Merchant Features
  6. Velocity Features
  7. Behavioral Anomaly Features
  8. Risk Score Computation
"""

import pandas as pd
import numpy as np
from config import (
    HIGH_AMOUNT_MULTIPLIER, EXTREME_AMOUNT_MULTIPLIER,
    MICRO_TRANSACTION_LIMIT, NIGHT_HOURS_START, NIGHT_HOURS_END,
    HIGH_RISK_COUNTRIES, KNOWN_SAFE_CITIES,
    HIGH_RISK_DEVICES, TRUSTED_MERCHANTS, HIGH_RISK_MERCHANTS,
    FEATURE_WEIGHTS
)

# ─────────────────────────────────────────────────────────────
# 1. AMOUNT-BASED FEATURES
# ─────────────────────────────────────────────────────────────

def add_amount_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    PayPal / Stripe-style amount anomaly detection.
    """
    avg = df["avg_amount"].replace(0, 1)  # avoid div by zero

    # Ratio of current to average spend
    df["amount_ratio"] = df["amount"] / avg

    # Z-score style deviation
    df["amount_deviation"] = (df["amount"] - avg) / avg.clip(lower=1)

    # Is the amount suspiciously high?
    df["is_high_amount"] = (df["amount"] > avg * HIGH_AMOUNT_MULTIPLIER).astype(int)

    # Extreme outlier (5x average — strong fraud signal)
    df["is_extreme_amount"] = (df["amount"] > avg * EXTREME_AMOUNT_MULTIPLIER).astype(int)

    # Micro-transaction test (card testing pattern)
    df["is_micro_transaction"] = (df["amount"] < MICRO_TRANSACTION_LIMIT).astype(int)

    # Round-number amounts (common in fraud)
    df["is_round_amount"] = (df["amount"] % 100 == 0).astype(int)

    return df


# ─────────────────────────────────────────────────────────────
# 2. TIME-BASED FEATURES
# ─────────────────────────────────────────────────────────────

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Time anomaly detection (FICO Falcon / Visa-style).
    Late-night transactions carry significantly higher risk.
    """
    hour = df["time_hour"]

    # Night transaction (highest fraud window globally)
    df["is_night"] = ((hour >= NIGHT_HOURS_START) & (hour <= NIGHT_HOURS_END)).astype(int)

    # Early morning suspicious window
    df["is_early_morning"] = ((hour >= 1) & (hour <= 4)).astype(int)

    # Weekend flag (if day_of_week available)
    if "day_of_week" in df.columns:
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    else:
        df["is_weekend"] = 0

    # Time risk score: 0-1
    df["time_risk_score"] = np.where(
        (hour >= 1) & (hour <= 4), 1.0,
        np.where((hour >= 23) | (hour == 0), 0.7,
        np.where((hour >= 5) & (hour <= 7), 0.3, 0.1))
    )

    return df


# ─────────────────────────────────────────────────────────────
# 3. GEOGRAPHIC FEATURES
# ─────────────────────────────────────────────────────────────

def add_geo_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mastercard Decision Intelligence & Visa Authorization style.
    Location mismatch is one of the strongest fraud signals.
    """
    # New location (changed from previous)
    df["is_new_location"] = (df["location"] != df["prev_location"]).astype(int)

    # High-risk country
    df["is_high_risk_country"] = df["location"].isin(HIGH_RISK_COUNTRIES).astype(int)

    # Known safe city
    df["is_known_safe_city"] = df["location"].isin(KNOWN_SAFE_CITIES).astype(int)

    # Impossible travel: prev_location was safe city, now foreign country
    df["is_impossible_travel"] = (
        df["prev_location"].isin(KNOWN_SAFE_CITIES) &
        ~df["location"].isin(KNOWN_SAFE_CITIES) &
        df["is_night"]
    ).astype(int)

    # International transaction flag
    domestic_cities = KNOWN_SAFE_CITIES
    df["is_international"] = (~df["location"].isin(domestic_cities)).astype(int)

    return df


# ─────────────────────────────────────────────────────────────
# 4. DEVICE & CHANNEL FEATURES
# ─────────────────────────────────────────────────────────────

def add_device_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sift Science / AWS Fraud Detector style device fingerprinting.
    """
    # New device used
    df["is_new_device"] = (df["device"] != df["prev_device"]).astype(int)

    # High-risk device type
    df["is_high_risk_device"] = df["device"].isin(HIGH_RISK_DEVICES).astype(int)

    # Web-based (higher fraud rate than mobile/POS)
    df["is_web_channel"] = df["device"].isin(["Web", "Desktop", "Laptop"]).astype(int)

    # Simultaneous device + location change (strong signal)
    df["device_location_change"] = (df["is_new_device"] & df["is_new_location"]).astype(int)

    return df


# ─────────────────────────────────────────────────────────────
# 5. MERCHANT FEATURES
# ─────────────────────────────────────────────────────────────

def add_merchant_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feedzai / IBM Safer Payments style merchant risk scoring.
    """
    df["is_unknown_merchant"] = (~df["merchant"].isin(TRUSTED_MERCHANTS)).astype(int)
    df["is_trusted_merchant"]  = df["merchant"].isin(TRUSTED_MERCHANTS).astype(int)
    df["is_high_risk_merchant"] = df["merchant"].isin(HIGH_RISK_MERCHANTS).astype(int)

    # High-amount transaction at unknown merchant (combined risk)
    df["high_amount_unknown_merchant"] = (
        df["is_high_amount"] & df["is_unknown_merchant"]
    ).astype(int)

    return df


# ─────────────────────────────────────────────────────────────
# 6. VELOCITY FEATURES (Simulated)
# ─────────────────────────────────────────────────────────────

def add_velocity_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Stripe Radar / DataRobot style velocity checks.
    In production these would query a real-time DB.
    Here we simulate using transaction count columns if available.
    """
    if "txn_count_last_hour" in df.columns:
        df["is_rapid_succession"] = (df["txn_count_last_hour"] > 3).astype(int)
        df["txn_velocity_score"]  = (df["txn_count_last_hour"] / 5).clip(0, 1)
    else:
        df["is_rapid_succession"] = 0
        df["txn_velocity_score"]  = 0.0

    if "total_amount_today" in df.columns:
        df["is_daily_limit_near"] = (df["total_amount_today"] > 40_000).astype(int)
    else:
        df["is_daily_limit_near"] = 0

    return df


# ─────────────────────────────────────────────────────────────
# 7. COMPOSITE RISK SCORE (Rule-Based Layer)
# ─────────────────────────────────────────────────────────────

def compute_rule_based_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Weighted rule-based fraud score (0-1 scale).
    This augments the ML model — dual-layer approach like Visa/Mastercard.
    """
    score = pd.Series(0.0, index=df.index)

    for feature, weight in FEATURE_WEIGHTS.items():
        if feature in df.columns:
            score += df[feature] * weight

    # Normalize to [0, 1]
    max_score = max(score.max(), 1)
    df["rule_based_score"] = (score / max_score).clip(0, 1)

    # Anomaly flags count
    flag_cols = [c for c in df.columns if c.startswith("is_") and c in FEATURE_WEIGHTS]
    df["total_flags"] = df[flag_cols].sum(axis=1)

    return df


# ─────────────────────────────────────────────────────────────
# 8. MAIN PIPELINE
# ─────────────────────────────────────────────────────────────

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full feature engineering pipeline.
    Call this on any raw transaction DataFrame.
    """
    df = df.copy()

    df = add_amount_features(df)
    df = add_time_features(df)
    df = add_geo_features(df)
    df = add_device_features(df)
    df = add_merchant_features(df)
    df = add_velocity_features(df)
    df = compute_rule_based_score(df)

    return df


def get_feature_columns() -> list:
    """Return the exact feature columns the ML model expects."""
    return [
        "amount",
        "avg_amount",
        "amount_ratio",
        "amount_deviation",
        "is_high_amount",
        "is_extreme_amount",
        "is_micro_transaction",
        "is_round_amount",
        "is_night",
        "is_early_morning",
        "time_risk_score",
        "is_new_location",
        "is_high_risk_country",
        "is_impossible_travel",
        "is_international",
        "is_new_device",
        "is_high_risk_device",
        "device_location_change",
        "is_unknown_merchant",
        "is_high_risk_merchant",
        "high_amount_unknown_merchant",
        "is_rapid_succession",
        "txn_velocity_score",
        "rule_based_score",
        "total_flags",
    ]


def get_flag_explanations(df_row: pd.Series) -> list:
    """
    Return human-readable fraud reasons for a single transaction row.
    Used in the UI explanation panel.
    """
    explanations = []

    checks = {
        "is_extreme_amount":            ("🚨", "Extreme amount — over 5x your usual spending"),
        "is_high_amount":               ("💰", "Unusually high transaction amount"),
        "is_micro_transaction":         ("🔍", "Micro-transaction — possible card testing"),
        "is_round_amount":              ("🔢", "Round-number amount (common fraud pattern)"),
        "is_night":                     ("🌙", "Late-night transaction (high-risk window)"),
        "is_early_morning":             ("⚠️", "Very early morning — peak fraud hour"),
        "is_impossible_travel":         ("✈️", "Impossible travel detected"),
        "is_high_risk_country":         ("🌍", "Transaction from high-risk country"),
        "is_international":             ("🌐", "International transaction"),
        "is_new_location":              ("📍", "New or changed location"),
        "device_location_change":       ("📱📍", "Both device AND location changed simultaneously"),
        "is_new_device":                ("📱", "New or unrecognized device"),
        "is_high_risk_device":          ("💻", "High-risk device type"),
        "is_high_risk_merchant":        ("🏴", "High-risk merchant category"),
        "is_unknown_merchant":          ("🏪", "Unknown or unverified merchant"),
        "high_amount_unknown_merchant": ("🚩", "Large amount at unverified merchant"),
        "is_rapid_succession":          ("⚡", "Multiple transactions in short time"),
    }

    for col, (icon, msg) in checks.items():
        if col in df_row.index and df_row[col]:
            explanations.append(f"{icon} {msg}")

    return explanations
