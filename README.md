# 🛡️ Advanced AI Fraud Detection System

Enterprise-grade fraud detection system inspired by the world's top fraud platforms.

---

## 🌟 Features Added (vs Original)

### Original System
- 5 basic features (amount, night, location, device, merchant)
- Single RandomForest model
- Simple UI

### Advanced System
| Feature | Original | Advanced |
|---|---|---|
| Features | 5 | **25+** |
| Models | RandomForest | **Ensemble (RF + GBM + LR)** |
| Scoring Layers | 1 | **2 (ML + Rule-Based)** |
| Risk Levels | 3 | **3 + Custom thresholds** |
| Batch Analysis | ❌ | ✅ |
| Audit Logging | ❌ | ✅ |
| Velocity Detection | ❌ | ✅ |
| Impossible Travel | ❌ | ✅ |
| Country Risk | ❌ | ✅ |
| Card Testing Detection | ❌ | ✅ |
| Transaction History | ❌ | ✅ |
| Recommended Action | ❌ | ✅ |
| CSV Export | ❌ | ✅ |

---

## 📊 Feature Categories (Inspired By Top Systems)

### 1. Amount Features (PayPal / Stripe)
- `is_high_amount` — 2.5x above average
- `is_extreme_amount` — 5x above average (strong fraud signal)
- `is_micro_transaction` — Card testing detection (< PKR 5)
- `is_round_amount` — Round numbers = common fraud pattern
- `amount_ratio` — Current / average ratio
- `amount_deviation` — Z-score style deviation

### 2. Time Features (FICO Falcon)
- `is_night` — Midnight to 5am
- `is_early_morning` — 1am–4am peak fraud window
- `time_risk_score` — Continuous 0–1 risk score
- `is_weekend` — Weekend flag

### 3. Geographic Features (Visa AI / Mastercard DI)
- `is_new_location` — Location changed from previous
- `is_high_risk_country` — Sanctioned/high-risk countries
- `is_impossible_travel` — Changed city + night = suspicious
- `is_international` — Non-domestic transaction

### 4. Device Features (Sift Science / AWS Fraud Detector)
- `is_new_device` — Device changed from previous
- `is_high_risk_device` — TOR, VPN, Unknown
- `device_location_change` — Both device AND location changed (strong signal)
- `is_web_channel` — Web > Mobile for fraud

### 5. Merchant Features (Feedzai / IBM Safer Payments)
- `is_unknown_merchant` — Not in trusted list
- `is_high_risk_merchant` — Crypto, wire transfer, gift cards
- `high_amount_unknown_merchant` — Combined risk signal

### 6. Velocity Features (Stripe Radar / DataRobot)
- `is_rapid_succession` — Many transactions in 1 hour
- `txn_velocity_score` — Continuous velocity score
- `is_daily_limit_near` — Near daily spend limit

### 7. Composite Score
- `rule_based_score` — Weighted rule engine score
- `total_flags` — Count of triggered risk flags
- `final_score` — 70% ML + 30% Rules (like Stripe)

---

## 🚀 How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Train advanced model (recommended first time)
cd src
python train.py
cd ..

# Launch app
streamlit run app.py
```

---

## 📁 Project Structure

```
fraud_advanced/
├── app.py                    # Main Streamlit UI (Advanced)
├── config.py                 # All settings & thresholds
├── requirements.txt
├── README.md
├── model/
│   ├── model.pkl             # Original fallback model
│   └── advanced_model.pkl    # New ensemble model (after training)
├── data/
│   └── transactions.csv      # Training data
├── src/
│   ├── features.py           # 25+ feature engineering
│   ├── train.py              # Ensemble training pipeline
│   └── predict.py            # Dual-layer prediction
├── utils/
│   └── logger.py             # Audit trail logging
└── logs/
    └── transactions_analyzed.csv   # Auto-generated log
```

---

## 🔬 Model Architecture

```
Transaction Input
      │
      ▼
Feature Engineering (25+ features)
      │
      ├──► ML Layer (70% weight)
      │         RandomForest (weight 3)
      │       + GradientBoosting (weight 2)
      │       + LogisticRegression (weight 1)
      │         → Soft Voting Ensemble
      │
      ├──► Rule Layer (30% weight)
      │         Weighted flag scoring
      │         FEATURE_WEIGHTS dict
      │
      ▼
  Final Score = 0.70 × ML + 0.30 × Rules
      │
      ▼
  Risk Decision → BLOCK / STEP-UP AUTH / APPROVE
```

---

## 📞 References (Top 10 Systems Studied)
1. **PayPal Risk Engine** — Amount anomaly + velocity
2. **Stripe Radar** — ML + rule combination, velocity
3. **Visa Advanced Authorization** — Geographic + time risk
4. **Mastercard Decision Intelligence** — Real-time AI scoring
5. **Sift Science** — Device fingerprinting + behavior
6. **FICO Falcon** — Time-series behavioral analysis
7. **IBM Safer Payments** — Merchant risk + network analysis
8. **AWS Fraud Detector** — Ensemble models + rule engine
9. **Feedzai** — Real-time streaming fraud detection
10. **DataRobot Fraud Platform** — AutoML + explainability
