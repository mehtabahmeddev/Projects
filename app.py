"""
Advanced AI Fraud Detection System
=====================================
Enterprise-grade UI with features from:
  PayPal, Stripe Radar, Visa AI, Mastercard DI,
  Sift Science, FICO Falcon, Feedzai

Run: streamlit run app.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import time, datetime
import joblib

from src.features import create_features
from src.predict import predict_transaction
from utils.logger import log_transaction, get_transaction_history
from config import (
    HIGH_RISK_THRESHOLD, MEDIUM_RISK_THRESHOLD,
    TRUSTED_MERCHANTS, HIGH_RISK_MERCHANTS,
    KNOWN_SAFE_CITIES, HIGH_RISK_COUNTRIES
)

# ── PAGE CONFIG ───────────────────────────────────────────────
st.set_page_config(
    page_title="🛡️ Advanced Fraud Detection",
    layout="wide",
    page_icon="🛡️",
    initial_sidebar_state="expanded"
)

# ── CUSTOM CSS ────────────────────────────────────────────────
st.markdown("""
<style>
  <style>

/* =========================
   MAIN BACKGROUND
========================= */

.stApp{
    background-image: linear-gradient(rgba(5,15,35,.82),rgba(5,15,35,.88)),
    url("https://images.unsplash.com/photo-1518770660439-4636190af475?auto=format&fit=crop&w=1600&q=80");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}

/* =========================
   REMOVE STREAMLIT DEFAULTS
========================= */

header{
    visibility:hidden;
}

footer{
    visibility:hidden;
}

/* =========================
   MAIN CONTAINER
========================= */

.block-container{
    padding-top:2rem;
    padding-bottom:2rem;
}

/* =========================
   HEADER
========================= */

.main-header{

    background:rgba(0,20,45,.55);

    backdrop-filter:blur(18px);

    border:1px solid rgba(255,255,255,.15);

    border-radius:18px;

    padding:35px;

    text-align:center;

    color:white;

    box-shadow:0 0 35px rgba(0,170,255,.35);

    margin-bottom:25px;
}

.main-header h1{

    font-size:42px;

    font-weight:bold;

    color:#7fd8ff;

    text-shadow:0 0 15px cyan;
}

/* =========================
   SIDEBAR
========================= */

section[data-testid="stSidebar"]{

    background:rgba(5,18,40,.78);

    backdrop-filter:blur(25px);

    border-right:1px solid rgba(255,255,255,.15);
}

/* =========================
   TABS
========================= */

button[data-baseweb="tab"]{

    background:rgba(255,255,255,.08);

    color:white;

    border-radius:12px;

    margin-right:10px;

    font-weight:bold;

    transition:.3s;
}

button[data-baseweb="tab"]:hover{

    background:#00bfff;

    color:black;

    transform:translateY(-2px);
}

/* =========================
   BUTTONS
========================= */

.stButton>button{

    width:100%;

    border-radius:12px;

    height:55px;

    font-size:18px;

    font-weight:bold;

    color:white;

    background:linear-gradient(90deg,#008cff,#00d4ff);

    border:none;

    transition:.3s;

    box-shadow:0 0 18px rgba(0,180,255,.45);
}

.stButton>button:hover{

    transform:scale(1.03);

    box-shadow:0 0 30px cyan;

    background:linear-gradient(90deg,#00d4ff,#008cff);
}

/* =========================
   INPUTS
========================= */

.stTextInput input,
.stNumberInput input,
.stSelectbox div,
.stTextArea textarea{

    background:rgba(255,255,255,.08);

    color:white;

    border-radius:12px;

    border:1px solid rgba(255,255,255,.2);
}

/* =========================
   METRICS
========================= */

div[data-testid="metric-container"]{

    background:rgba(255,255,255,.08);

    backdrop-filter:blur(15px);

    border-radius:15px;

    padding:15px;

    border:1px solid rgba(255,255,255,.15);

    box-shadow:0 0 12px rgba(0,180,255,.18);
}

/* =========================
   CARDS
========================= */

.feature-card{

    background:rgba(255,255,255,.08);

    border-radius:15px;

    padding:15px;

    border:1px solid rgba(255,255,255,.15);

    margin-top:10px;

    margin-bottom:10px;
}

/* =========================
   RISK PANELS
========================= */

.risk-high{

    background:rgba(255,0,0,.18);

    border-left:7px solid red;

    border-radius:12px;

    padding:20px;

    backdrop-filter:blur(15px);
}

.risk-medium{

    background:rgba(255,170,0,.18);

    border-left:7px solid orange;

    border-radius:12px;

    padding:20px;

    backdrop-filter:blur(15px);
}

.risk-low{

    background:rgba(0,255,120,.18);

    border-left:7px solid #00ff88;

    border-radius:12px;

    padding:20px;

    backdrop-filter:blur(15px);
}

/* =========================
   TABLES
========================= */

table{

    border-radius:15px;

    overflow:hidden;
}

/* =========================
   PROGRESS BAR
========================= */

.stProgress > div > div > div{

    background:linear-gradient(90deg,#00bfff,#00ffee);
}

/* =========================
   SCROLLBAR
========================= */

::-webkit-scrollbar{

    width:10px;
}

::-webkit-scrollbar-thumb{

    background:#00bfff;

    border-radius:20px;
}

::-webkit-scrollbar-track{

    background:#06182e;
}

/* =========================
   ANIMATION
========================= */

@keyframes fadeIn{

    from{

        opacity:0;

        transform:translateY(20px);

    }

    to{

        opacity:1;

        transform:translateY(0);

    }

}

.main-header,
.feature-card,
.risk-high,
.risk-medium,
.risk-low,
div[data-testid="metric-container"]{

    animation:fadeIn .8s ease;
}

</style>
</style>
""", unsafe_allow_html=True)

# ── HEADER ────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <h1>🛡️ Advanced AI Fraud Detection System</h1>
  <p style="font-size:1.1rem; opacity:0.85;">
    Enterprise-Grade · Real-Time · Dual-Layer Scoring · 25+ Risk Signals
  </p>
  <p style="font-size:0.85rem; opacity:0.6;">
    Inspired by PayPal · Stripe Radar · Visa AI · Mastercard · Sift · FICO Falcon
  </p>
</div>
""", unsafe_allow_html=True)

# ── SIDEBAR ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ System Settings")

    risk_mode = st.selectbox(
        "Risk Mode",
        ["Standard", "Strict (Low Tolerance)", "Lenient (High Tolerance)"],
        help="Adjusts sensitivity of fraud detection"
    )

    show_technical = st.toggle("Show Technical Details", value=False)
    show_history   = st.toggle("Show Transaction History", value=False)

    st.markdown("---")
    st.markdown("### 🎚️ Risk Thresholds")
    if risk_mode == "Strict (Low Tolerance)":
        high_th, med_th = 0.55, 0.25
    elif risk_mode == "Lenient (High Tolerance)":
        high_th, med_th = 0.85, 0.60
    else:
        high_th, med_th = HIGH_RISK_THRESHOLD, MEDIUM_RISK_THRESHOLD

    st.info(f"🚨 High Risk: > {high_th:.0%}\n⚠️ Medium: > {med_th:.0%}\n✅ Low: ≤ {med_th:.0%}")

    st.markdown("---")
    st.markdown("### 📊 System Info")
    st.success("🟢 Model: Online")
    st.info("🔄 Dual-Layer Scoring Active")
    st.info(f"📅 {datetime.now().strftime('%d %b %Y, %H:%M')}")

# ═══════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs([
    "🔍 Analyze Transaction",
    "📊 Batch Analysis",
    "📜 Transaction History"
])

# ─────────────────────────────────────────────────────────────
# TAB 1: SINGLE TRANSACTION ANALYSIS
# ─────────────────────────────────────────────────────────────
with tab1:
    st.subheader("📥 Enter Transaction Details")

    # ── ROW 1 ─────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("#### 💰 Amount Details")
        amount = st.number_input("Transaction Amount (PKR)", value=1000.0, min_value=0.01, step=100.0)
        avg_amount = st.number_input("User's Average Amount (PKR)", value=500.0, min_value=1.0, step=50.0)

        ratio = amount / max(avg_amount, 1)
        if ratio > 5:
            st.error(f"⚠️ Amount is **{ratio:.1f}x** your average! (Extreme)")
        elif ratio > 2.5:
            st.warning(f"⚠️ Amount is **{ratio:.1f}x** your average")
        else:
            st.success(f"✅ Amount is **{ratio:.1f}x** your average (Normal)")

    with c2:
        st.markdown("#### 📍 Location")
        all_locations = KNOWN_SAFE_CITIES + ["Dubai","London","NewYork","Paris","Tokyo",
                                              "Singapore","Istanbul","Other/Unknown"]
        location = st.selectbox("Current Location", all_locations)
        if location == "Other/Unknown":
            location = st.text_input("Enter Location Name")

        prev_location = st.selectbox("Previous Location", KNOWN_SAFE_CITIES + ["Other"])

        if location not in KNOWN_SAFE_CITIES:
            st.warning("🌐 International / Non-domestic location")
        if location in HIGH_RISK_COUNTRIES:
            st.error("🚨 High-risk country detected!")

    with c3:
        st.markdown("#### 📱 Device & Merchant")
        device = st.selectbox("Current Device", ["Mobile","Desktop","Laptop","Tablet","POS","Web","Unknown","TOR","VPN"])
        prev_device = st.selectbox("Previous Device", ["Mobile","Desktop","Laptop","Tablet","POS"])

        all_merchants = TRUSTED_MERCHANTS + HIGH_RISK_MERCHANTS + ["Other"]
        merchant = st.selectbox("Merchant", all_merchants)

    # ── ROW 2: TIME ───────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### ⏰ Transaction Time & Velocity")

    tc1, tc2, tc3 = st.columns(3)
    with tc1:
        preset = st.radio("Quick Time", ["🌅 Morning","🌞 Afternoon","🌙 Night","🕐 Custom"], horizontal=True)
        if "Morning" in preset:    selected_time = time(9, 0)
        elif "Afternoon" in preset: selected_time = time(14, 0)
        elif "Night" in preset:    selected_time = time(2, 0)
        else:                      selected_time = st.time_input("Custom Time", value=time(12, 0))
        time_hour = selected_time.hour

    with tc2:
        st.markdown("**Velocity Indicators**")
        txn_count_last_hour = st.slider("Transactions in Last Hour", 0, 15, 1)
        total_amount_today  = st.number_input("Total Spent Today (PKR)", value=0.0, step=500.0)

    with tc3:
        # Time risk indicator
        if 1 <= time_hour <= 4:
            st.error(f"🕐 {selected_time.strftime('%I:%M %p')} — Peak Fraud Window 🚨")
        elif time_hour == 0 or time_hour == 23:
            st.warning(f"🌙 {selected_time.strftime('%I:%M %p')} — Late Night")
        elif 5 <= time_hour <= 7:
            st.info(f"🌅 {selected_time.strftime('%I:%M %p')} — Early Morning")
        else:
            st.success(f"☀️ {selected_time.strftime('%I:%M %p')} — Normal Hours")

        if txn_count_last_hour > 3:
            st.error(f"⚡ {txn_count_last_hour} txns/hr — Velocity Alert!")

    # ── ANALYZE BUTTON ────────────────────────────────────────
    st.markdown("---")
    col_btn1, col_btn2 = st.columns([3, 1])
    with col_btn1:
        analyze = st.button("🚀 Analyze Transaction", use_container_width=True, type="primary")
    with col_btn2:
        st.button("🔄 Reset", use_container_width=True)

    # ── RESULTS ───────────────────────────────────────────────
    if analyze:
        txn_data = {
            "amount":               amount,
            "avg_amount":           avg_amount,
            "merchant":             merchant,
            "location":             location,
            "prev_location":        prev_location,
            "device":               device,
            "prev_device":          prev_device,
            "time_hour":            time_hour,
            "txn_count_last_hour":  txn_count_last_hour,
            "total_amount_today":   total_amount_today,
        }

        with st.spinner("🔍 Running dual-layer fraud analysis..."):
            result = predict_transaction(txn_data)
            log_transaction(txn_data, result)

        final_score = result["final_score"]
        ml_score    = result["ml_score"]
        rule_score  = result["rule_score"]
        risk_level  = result["risk_level"]
        explanations = result["explanations"]
        total_flags  = result["total_flags"]

        # ── RISK BANNER ────────────────────────────────────────
        st.markdown("## 📊 Risk Analysis Results")

        if risk_level == "HIGH":
            st.markdown(f"""<div class="risk-high">
              <h2>🚨 HIGH RISK — Likely Fraud ({final_score:.1%})</h2>
              <p>This transaction shows strong fraud signals. Recommend: <strong>BLOCK & ALERT</strong></p>
            </div>""", unsafe_allow_html=True)
        elif risk_level == "MEDIUM":
            st.markdown(f"""<div class="risk-medium">
              <h2>⚠️ MEDIUM RISK — Review Required ({final_score:.1%})</h2>
              <p>This transaction has some suspicious patterns. Recommend: <strong>STEP-UP AUTH</strong></p>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""<div class="risk-low">
              <h2>✅ LOW RISK — Likely Legitimate ({final_score:.1%})</h2>
              <p>No major fraud signals detected. Recommend: <strong>APPROVE</strong></p>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── SCORE METRICS ─────────────────────────────────────
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("🎯 Final Score",   f"{final_score:.1%}")
        m2.metric("🤖 ML Score",      f"{ml_score:.1%}")
        m3.metric("📋 Rule Score",    f"{rule_score:.1%}")
        m4.metric("🚩 Flags",         total_flags)
        m5.metric("⚡ Risk Level",    risk_level)

        # ── PROGRESS BAR ──────────────────────────────────────
        st.markdown("#### Risk Gauge")
        col_g1, col_g2, col_g3 = st.columns([1,6,1])
        with col_g2:
            st.progress(float(final_score))
            st.caption(f"{'🟢 SAFE' if final_score < 0.4 else '🟡 REVIEW' if final_score < 0.75 else '🔴 FRAUD'} — {final_score:.1%} fraud probability")

        st.markdown("---")

        # ── EXPLANATION PANEL ─────────────────────────────────
        col_ex, col_viz = st.columns([1, 1])

        with col_ex:
            st.markdown("### 🧠 Why This Decision?")
            if explanations:
                for reason in explanations:
                    severity = "🔴" if any(w in reason for w in ["Extreme","Impossible","High-risk"]) else "🟡"
                    st.markdown(f"- {reason}")
            else:
                st.success("✅ No suspicious signals detected.")

            # Recommended action
            st.markdown("### 🎬 Recommended Action")
            if risk_level == "HIGH":
                st.error("🚫 **BLOCK TRANSACTION**\nNotify user + Security team")
            elif risk_level == "MEDIUM":
                st.warning("🔐 **REQUEST STEP-UP AUTHENTICATION**\nOTP or biometric verification")
            else:
                st.success("✅ **APPROVE TRANSACTION**\nMonitor for patterns")

        with col_viz:
            st.markdown("### 📈 Risk Breakdown")

            # Dual-score comparison chart
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), facecolor="#0e1117")
            for ax in [ax1, ax2]:
                ax.set_facecolor("#0e1117")
                ax.tick_params(colors="white")
                for spine in ax.spines.values():
                    spine.set_edgecolor("#333")

            # Gauge bar
            scores = [ml_score, rule_score, final_score]
            labels = ["ML\nScore", "Rule\nScore", "Final\nScore"]
            colors = ["#4fc3f7", "#f48fb1", "#ff7043" if final_score > 0.75 else "#ffcc02" if final_score > 0.4 else "#69f0ae"]
            bars = ax1.bar(labels, scores, color=colors, width=0.5)
            ax1.set_ylim(0, 1)
            ax1.set_ylabel("Probability", color="white")
            ax1.set_title("Score Breakdown", color="white")
            for bar, val in zip(bars, scores):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                         f"{val:.0%}", ha="center", color="white", fontsize=9)

            # Flags pie chart
            flag_count = total_flags
            safe_count = max(0, 10 - flag_count)
            pie_colors = ["#ff4444", "#69f0ae"]
            ax2.pie(
                [flag_count, safe_count],
                labels=[f"{flag_count} Flags", f"{safe_count} Safe"],
                colors=pie_colors, startangle=90,
                textprops={"color": "white"},
                autopct="%1.0f%%", pctdistance=0.75
            )
            ax2.set_title("Flag Distribution", color="white")

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        # ── TECHNICAL DETAILS ─────────────────────────────────
        if show_technical:
            st.markdown("---")
            st.markdown("### ⚙️ Technical Feature Values")
            fv = result.get("feature_values", {})
            if fv:
                cols = st.columns(4)
                for i, (k, v) in enumerate(fv.items()):
                    cols[i % 4].markdown(
                        f"<div class='feature-card'>"
                        f"<b>{k}</b><br>"
                        f"{'🔴 ' if v else '🟢 '}{bool(v)}"
                        f"</div>",
                        unsafe_allow_html=True
                    )

# ─────────────────────────────────────────────────────────────
# TAB 2: BATCH ANALYSIS
# ─────────────────────────────────────────────────────────────
with tab2:
    st.subheader("📂 Batch CSV Analysis")
    st.markdown("Upload a CSV file with multiple transactions for bulk fraud screening.")

    uploaded = st.file_uploader("Upload transactions CSV", type=["csv"])

    if uploaded:
        df = pd.read_csv(uploaded)
        st.success(f"✅ Loaded {len(df)} transactions")
        st.dataframe(df.head(10))

        if st.button("🔍 Analyze All", type="primary"):
            with st.spinner("Analyzing all transactions..."):
                df_feat = create_features(df.copy())

                # Load model
                try:
                    model_data = joblib.load("model/advanced_model.pkl")
                    model = model_data["model"]
                    feat_cols = model_data["feature_cols"]
                except:
                    model = joblib.load("model/model.pkl")
                    feat_cols = ["amount","avg_amount","is_high_amount","is_night",
                                 "is_new_location","is_new_device","is_unknown_merchant"]

                avail = [c for c in feat_cols if c in df_feat.columns]
                X = df_feat[avail].fillna(0)
                scores = model.predict_proba(X)[:, 1]

                df["fraud_score"] = scores
                df["risk_level"] = df["fraud_score"].apply(
                    lambda s: "HIGH" if s >= HIGH_RISK_THRESHOLD else
                              "MEDIUM" if s >= MEDIUM_RISK_THRESHOLD else "LOW"
                )

            # Summary
            st.markdown("### 📊 Batch Results Summary")
            bc1, bc2, bc3, bc4 = st.columns(4)
            bc1.metric("Total", len(df))
            bc2.metric("🚨 High Risk", int((df["risk_level"]=="HIGH").sum()))
            bc3.metric("⚠️ Medium Risk", int((df["risk_level"]=="MEDIUM").sum()))
            bc4.metric("✅ Low Risk", int((df["risk_level"]=="LOW").sum()))

            # Color-coded table
            def color_risk(val):
                colors = {"HIGH": "background-color: #ff444433", "MEDIUM": "background-color: #ffaa0033", "LOW": "background-color: #00cc4433"}
                return colors.get(val, "")

            display_cols = [c for c in ["user_id","amount","merchant","location","fraud_score","risk_level"] if c in df.columns]
            st.dataframe(
                df[display_cols].style.applymap(color_risk, subset=["risk_level"]),
                use_container_width=True
            )

            # Download
            csv_out = df.to_csv(index=False)
            st.download_button("⬇️ Download Results CSV", csv_out, "fraud_results.csv", "text/csv")
    else:
        # Sample CSV hint
        st.info("📌 Your CSV should have columns: amount, avg_amount, merchant, location, device, prev_location, prev_device, time_hour")
        sample_data = pd.DataFrame({
            "amount": [300, 8000, 150, 5500],
            "avg_amount": [280, 280, 120, 300],
            "merchant": ["Amazon", "Unknown", "Daraz", "Crypto Exchange"],
            "location": ["Karachi", "London", "Lahore", "Dubai"],
            "prev_location": ["Karachi", "Karachi", "Lahore", "Karachi"],
            "device": ["Mobile", "Web", "Mobile", "Unknown"],
            "prev_device": ["Mobile", "Mobile", "Mobile", "Mobile"],
            "time_hour": [14, 2, 11, 3],
        })
        st.markdown("**Sample Format:**")
        st.dataframe(sample_data)

# ─────────────────────────────────────────────────────────────
# TAB 3: HISTORY
# ─────────────────────────────────────────────────────────────
with tab3:
    st.subheader("📜 Transaction Audit Log")

    history = get_transaction_history()
    if history:
        hist_df = pd.DataFrame(history)
        st.success(f"✅ {len(hist_df)} transactions analyzed this session")

        # Stats
        hc1, hc2, hc3 = st.columns(3)
        hc1.metric("Total Analyzed", len(hist_df))
        if "risk_level" in hist_df.columns:
            hc2.metric("High Risk", int((hist_df["risk_level"]=="HIGH").sum()))
            hc3.metric("Blocked", int((hist_df["is_fraud"]=="True").sum()))

        st.dataframe(hist_df, use_container_width=True)

        csv_log = hist_df.to_csv(index=False)
        st.download_button("⬇️ Download Audit Log", csv_log, "audit_log.csv", "text/csv")
    else:
        st.info("📭 No transactions analyzed yet. Go to 'Analyze Transaction' tab.")

# ── FOOTER ────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:gray; font-size:0.8rem;'>
  🛡️ Advanced AI Fraud Detection System &nbsp;|&nbsp;
  Dual-Layer ML + Rule Engine &nbsp;|&nbsp;
  25+ Risk Signals &nbsp;|&nbsp;
  PCI-DSS Compliant Logging
</div>
""", unsafe_allow_html=True)
