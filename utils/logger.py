"""
Transaction Logger & Audit Trail
=================================
Every transaction analyzed is logged for audit/compliance.
Approach: PCI-DSS compliant logging (like PayPal / bank systems)
"""

import os
import csv
import json
import logging
from datetime import datetime

LOG_DIR = "logs"
REPORT_DIR = "reports"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# Setup Python logger
logging.basicConfig(
    filename=os.path.join(LOG_DIR, "fraud_system.log"),
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

TXN_CSV = os.path.join(LOG_DIR, "transactions_analyzed.csv")
CSV_HEADERS = [
    "timestamp", "amount", "merchant", "location", "device",
    "ml_score", "rule_score", "final_score", "risk_level",
    "is_fraud", "total_flags", "flags"
]


def log_transaction(txn_data: dict, result: dict):
    """
    Write transaction + result to CSV log file.
    """
    row = {
        "timestamp":   datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "amount":      txn_data.get("amount", ""),
        "merchant":    txn_data.get("merchant", ""),
        "location":    txn_data.get("location", ""),
        "device":      txn_data.get("device", ""),
        "ml_score":    result.get("ml_score", ""),
        "rule_score":  result.get("rule_score", ""),
        "final_score": result.get("final_score", ""),
        "risk_level":  result.get("risk_level", ""),
        "is_fraud":    result.get("is_fraud", ""),
        "total_flags": result.get("total_flags", 0),
        "flags":       "|".join(result.get("flags", [])),
    }

    file_exists = os.path.exists(TXN_CSV)
    with open(TXN_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    # Also log to Python logger
    level = logging.WARNING if result.get("is_fraud") else logging.INFO
    logging.log(level, f"TXN | Score={result.get('final_score'):.2%} | Risk={result.get('risk_level')} | Merchant={txn_data.get('merchant')} | Amount={txn_data.get('amount')}")


def get_transaction_history() -> list:
    """Read all logged transactions from CSV."""
    if not os.path.exists(TXN_CSV):
        return []
    rows = []
    with open(TXN_CSV, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def get_risk_level(prob: float) -> str:
    if prob > 0.75:
        return "HIGH"
    elif prob > 0.40:
        return "MEDIUM"
    return "LOW"
