# ============================================================
# FRAUD DETECTION SYSTEM — ADVANCED CONFIG
# Inspired by: PayPal, Stripe, Visa, Mastercard, Sift, etc.
# ============================================================

# ── Model paths ──────────────────────────────────────────────
MODEL_PATH = "model/model.pkl"
ADVANCED_MODEL_PATH = "model/advanced_model.pkl"

# ── Risk Thresholds ──────────────────────────────────────────
HIGH_RISK_THRESHOLD   = 0.75
MEDIUM_RISK_THRESHOLD = 0.40
LOW_RISK_THRESHOLD    = 0.20

# ── Velocity Limits (per user, per time window) ──────────────
MAX_TXN_PER_HOUR    = 5
MAX_TXN_PER_DAY     = 20
MAX_AMOUNT_PER_DAY  = 50_000   # PKR
VELOCITY_WINDOW_MIN = 60       # minutes

# ── Amount Thresholds ─────────────────────────────────────────
HIGH_AMOUNT_MULTIPLIER   = 2.5   # 2.5x avg = suspicious
EXTREME_AMOUNT_MULTIPLIER = 5.0  # 5x avg = very suspicious
MICRO_TRANSACTION_LIMIT  = 5.0   # Testing with tiny amounts

# ── Geographic Risk ───────────────────────────────────────────
HIGH_RISK_COUNTRIES = [
    "North Korea", "Iran", "Venezuela", "Myanmar",
    "Cuba", "Syria", "Russia"
]

KNOWN_SAFE_CITIES = [
    "Karachi", "Lahore", "Islamabad", "Rawalpindi",
    "Peshawar", "Quetta", "Multan", "Faisalabad",
    "Hyderabad", "Sialkot"
]

# ── Device Risk ───────────────────────────────────────────────
HIGH_RISK_DEVICES = ["Unknown", "TOR", "VPN", "Proxy"]
SAFE_DEVICES      = ["Mobile", "Desktop", "Laptop", "Tablet", "POS"]

# ── Merchant Risk ─────────────────────────────────────────────
TRUSTED_MERCHANTS = [
    "Amazon", "Daraz", "Foodpanda", "Careem",
    "Netflix", "Uber", "AliExpress", "Shopify",
    "eBay", "Google", "Apple"
]

HIGH_RISK_MERCHANTS = [
    "Unknown", "Crypto Exchange", "Wire Transfer",
    "Gift Cards", "Anonymous"
]

# ── Time Risk ─────────────────────────────────────────────────
NIGHT_HOURS_START = 0    # midnight
NIGHT_HOURS_END   = 5    # 5am

# ── Feature Weights (for rule-based score) ────────────────────
FEATURE_WEIGHTS = {
    "is_high_amount":          0.20,
    "is_extreme_amount":       0.35,
    "is_micro_transaction":    0.10,
    "is_night":                0.10,
    "is_new_location":         0.10,
    "is_high_risk_country":    0.30,
    "is_new_device":           0.10,
    "is_high_risk_device":     0.25,
    "is_unknown_merchant":     0.15,
    "is_high_risk_merchant":   0.25,
    "is_rapid_succession":     0.20,
    "is_impossible_travel":    0.40,
    "amount_deviation_score":  0.15,
}

# ── Logging ───────────────────────────────────────────────────
LOG_FILE        = "logs/transactions.log"
REPORT_DIR      = "reports/"
LOG_LEVEL       = "INFO"

# ── App Settings ──────────────────────────────────────────────
APP_TITLE       = "🛡️ Advanced AI Fraud Detection System"
APP_SUBTITLE    = "Enterprise-Grade Real-Time Transaction Risk Analyzer"
