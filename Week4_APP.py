"""
Customer Churn Intelligence Platform
====================================
Production-grade Streamlit application for churn-risk scoring, explanation,
scenario simulation, retention economics and portfolio-level monitoring.

Expected artifacts (same folder):
    churn_model.joblib   - fitted sklearn-compatible estimator/pipeline with predict_proba
    model_meta.json      - {"feature_columns": [...], "threshold": float, "model_name": str,
                            "cv_auc": float, "cv_auc_std": float}
"""
from __future__ import annotations

import io
import json
import logging
import os
import time
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

# ----------------------------------------------------------------------------
# 1. LOGGING & PAGE CONFIG
# ----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("churn_platform")

st.set_page_config(
    page_title="Churn Intelligence Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
        .block-container { padding-top: 2rem; padding-bottom: 3rem; }
        .kpi {
            background: linear-gradient(145deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02));
            border: 1px solid rgba(255,255,255,0.10);
            border-radius: 14px; padding: 18px 20px; height: 100%;
            box-shadow: 0 4px 14px rgba(0,0,0,0.18);
            transition: transform .2s ease, border-color .2s ease;
        }
        .kpi:hover { transform: translateY(-2px); border-color: rgba(255,255,255,0.25); }
        .kpi .label { font-size: .72rem; letter-spacing: .08em; text-transform: uppercase; opacity: .65; }
        .kpi .value { font-size: 1.9rem; font-weight: 700; line-height: 1.25; margin-top: 4px; }
        .kpi .sub   { font-size: .8rem; opacity: .6; margin-top: 2px; }
        .tier-pill  { display:inline-block; padding: 4px 12px; border-radius: 999px; font-weight: 600; font-size: .85rem; }
        .caption-bar { font-size: .85rem; color: #8892b0; margin: -6px 0 18px 0; }
        .note { font-size: .82rem; opacity: .7; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# 2. SCHEMA & CONSTANTS
# ----------------------------------------------------------------------------
RAW_COLS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod", "MonthlyCharges", "TotalCharges",
]
SERVICES = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
PAYMENTS = ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
CONTRACTS = ["Month-to-month", "One year", "Two year"]
INTERNET = ["Fiber optic", "DSL", "No"]

COLORS = {"high": "#ef4444", "mid": "#f59e0b", "low": "#22c55e", "accent": "#61afef", "muted": "#8892b0"}

# Reference ("typical, low-friction") customer used for occlusion-based attribution.
REFERENCE = {
    "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No", "tenure": 32,
    "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "DSL",
    **{s: "No" for s in SERVICES},
    "Contract": "One year", "PaperlessBilling": "No", "PaymentMethod": "Bank transfer (automatic)",
    "MonthlyCharges": 65.0, "TotalCharges": 32 * 65.0,
}

ATTRIBUTION_GROUPS = {
    "Tenure": ["tenure"],
    "Monthly charges": ["MonthlyCharges"],
    "Contract type": ["Contract"],
    "Payment method": ["PaymentMethod"],
    "Paperless billing": ["PaperlessBilling"],
    "Internet & add-on services": ["InternetService"] + SERVICES,
    "Phone service": ["PhoneService", "MultipleLines"],
    "Senior citizen": ["SeniorCitizen"],
    "Partner & dependents": ["Partner", "Dependents"],
    "Gender": ["gender"],
}

PRESETS = {
    "Custom": None,
    "High-risk archetype": dict(tenure=3, contract="Month-to-month", monthly=95.0, internet="Fiber optic",
                                services=[], phone="Yes", multi="Yes", payment="Electronic check", paperless="Yes"),
    "Loyal archetype": dict(tenure=60, contract="Two year", monthly=55.0, internet="DSL",
                            services=["OnlineSecurity", "TechSupport", "OnlineBackup"], phone="Yes", multi="No",
                            payment="Credit card (automatic)", paperless="No"),
}

# ----------------------------------------------------------------------------
# 3. ARTIFACT LOADING
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model artifacts...")
def load_artifacts():
    try:
        if not (os.path.exists("churn_model.joblib") and os.path.exists("model_meta.json")):
            logger.error("Model artifacts missing from working directory.")
            return None, None
        mdl = joblib.load("churn_model.joblib")
        with open("model_meta.json") as fh:
            mt = json.load(fh)
        logger.info("Model artifacts loaded.")
        return mdl, mt
    except Exception as exc:  # noqa: BLE001
        logger.critical("Artifact loading failed: %s", exc)
        return None, None


model, meta = load_artifacts()
if model is None or meta is None:
    st.error("### 🚨 Boot failure: model artifacts not found")
    st.info("Place `churn_model.joblib` and `model_meta.json` in the project root and reload.")
    st.stop()

COLS = meta["feature_columns"]
DEFAULT_THRESHOLD = float(meta.get("threshold", 0.5))

# ----------------------------------------------------------------------------
# 4. INFERENCE & ANALYTICS HELPERS
# ----------------------------------------------------------------------------
def prepare_serving_input(raw: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode and align to the training feature space (unseen -> 0)."""
    x = pd.get_dummies(raw[RAW_COLS])
    return x.reindex(columns=COLS, fill_value=0).astype(float)


def predict(raw: pd.DataFrame) -> np.ndarray:
    t0 = time.perf_counter()
    try:
        out = model.predict_proba(prepare_serving_input(raw))[:, 1]
    except Exception as exc:  # noqa: BLE001
        logger.exception("Inference failed")
        st.error(f"Inference error: {exc}")
        return np.zeros(len(raw))
    st.session_state["latency_ms"] = (time.perf_counter() - t0) * 1000
    return out


def tier_of(p: float, t: float) -> tuple[str, str, str]:
    """Return (label, colour, recommended action)."""
    if p >= t:
        return "HIGH RISK", COLORS["high"], "Priority outbound retention call + tailored offer"
    if p >= t / 2:
        return "WATCHLIST", COLORS["mid"], "Enrol in loyalty / nudge campaign; review in 30 days"
    return "STABLE", COLORS["low"], "No action - maintain standard service cadence"


def tier_series(p: pd.Series, t: float) -> pd.Series:
    return pd.cut(p, bins=[-1, t / 2, t, 2], labels=["Stable", "Watchlist", "High risk"])


def finalize_total(df: pd.DataFrame) -> pd.DataFrame:
    df["TotalCharges"] = df["tenure"] * df["MonthlyCharges"]
    return df


def fix_dependencies(row: dict) -> dict:
    """Keep internet/phone dependent fields internally consistent."""
    if row["InternetService"] == "No":
        for s in SERVICES:
            row[s] = "No internet service"
    else:
        for s in SERVICES:
            if row[s] == "No internet service":
                row[s] = "No"
    if row["PhoneService"] == "No":
        row["MultipleLines"] = "No phone service"
    elif row["MultipleLines"] == "No phone service":
        row["MultipleLines"] = "No"
    row["TotalCharges"] = row["tenure"] * row["MonthlyCharges"]
    return row


@st.cache_data(show_spinner=False)
def compute_attributions(customer_json: str) -> pd.DataFrame:
    """Occlusion attribution: delta in P(churn) when a feature group is set to the reference value."""
    cust = json.loads(customer_json)
    rows = [cust]
    for cols in ATTRIBUTION_GROUPS.values():
        r = dict(cust)
        for c in cols:
            r[c] = REFERENCE[c]
        rows.append(fix_dependencies(r))
    p = predict(pd.DataFrame(rows))
    base = p[0]
    out = pd.DataFrame({"Factor": list(ATTRIBUTION_GROUPS), "Contribution": base - p[1:]})
    return out.sort_values("Contribution", key=np.abs, ascending=True).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def sensitivity_tenure(customer_json: str) -> pd.DataFrame:
    cust = json.loads(customer_json)
    frames = []
    for ctr in CONTRACTS:
        for ten in range(0, 73, 2):
            r = dict(cust, Contract=ctr, tenure=ten)
            frames.append(fix_dependencies(r) | {"_c": ctr})
    df = pd.DataFrame(frames)
    df["p"] = predict(df)
    return df[["_c", "tenure", "p"]].rename(columns={"_c": "Contract"})


@st.cache_data(show_spinner=False)
def sensitivity_price(customer_json: str) -> pd.DataFrame:
    cust = json.loads(customer_json)
    frames = []
    for net in INTERNET:
        for m in np.arange(18, 121, 3.0):
            r = dict(cust, InternetService=net, MonthlyCharges=float(m))
            frames.append(fix_dependencies(r) | {"_n": net})
    df = pd.DataFrame(frames)
    df["p"] = predict(df)
    return df[["_n", "MonthlyCharges", "p"]].rename(columns={"_n": "Internet"})


@st.cache_data(show_spinner=False)
def sensitivity_grid(customer_json: str) -> pd.DataFrame:
    cust = json.loads(customer_json)
    rows = []
    for ten in range(0, 73, 6):
        for m in range(20, 121, 10):
            rows.append(fix_dependencies(dict(cust, tenure=ten, MonthlyCharges=float(m))) | {"_t": ten, "_m": m})
    df = pd.DataFrame(rows)
    df["p"] = predict(df)
    return df.pivot(index="_m", columns="_t", values="p")


def build_levers(cust: dict) -> list[tuple[str, dict]]:
    """Actionable retention levers applicable to this customer."""
    levers = []
    if cust["Contract"] == "Month-to-month":
        levers.append(("Upgrade to 1-year contract", {"Contract": "One year"}))
        levers.append(("Upgrade to 2-year contract", {"Contract": "Two year"}))
    elif cust["Contract"] == "One year":
        levers.append(("Upgrade to 2-year contract", {"Contract": "Two year"}))
    if cust["PaymentMethod"] in ("Electronic check", "Mailed check"):
        levers.append(("Move to automatic card payment", {"PaymentMethod": "Credit card (automatic)"}))
        levers.append(("Move to automatic bank transfer", {"PaymentMethod": "Bank transfer (automatic)"}))
    if cust["InternetService"] != "No":
        for s, label in [("TechSupport", "Add tech support"), ("OnlineSecurity", "Add online security"),
                         ("OnlineBackup", "Add online backup"), ("DeviceProtection", "Add device protection")]:
            if cust[s] == "No":
                levers.append((label, {s: "Yes"}))
    if cust["PaperlessBilling"] == "Yes":
        levers.append(("Switch to paper billing", {"PaperlessBilling": "No"}))
    if cust["MonthlyCharges"] > 30:
        levers.append(("10% loyalty discount", {"MonthlyCharges": round(cust["MonthlyCharges"] * 0.9, 2)}))
    return levers


def synth_portfolio(n: int = 600, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    net = rng.choice(INTERNET, n, p=[0.44, 0.34, 0.22])
    ten = rng.integers(0, 73, n)
    monthly = np.where(net == "Fiber optic", rng.uniform(70, 110, n),
                       np.where(net == "DSL", rng.uniform(40, 72, n), rng.uniform(18, 30, n))).round(2)
    phone = rng.choice(["Yes", "No"], n, p=[0.9, 0.1])
    df = pd.DataFrame({
        "customerID": [f"DEMO-{i:05d}" for i in range(n)],
        "gender": rng.choice(["Female", "Male"], n),
        "SeniorCitizen": rng.choice([0, 1], n, p=[0.84, 0.16]),
        "Partner": rng.choice(["Yes", "No"], n),
        "Dependents": rng.choice(["Yes", "No"], n, p=[0.3, 0.7]),
        "tenure": ten, "PhoneService": phone,
        "MultipleLines": np.where(phone == "No", "No phone service", rng.choice(["Yes", "No"], n)),
        "InternetService": net,
        "Contract": rng.choice(CONTRACTS, n, p=[0.55, 0.21, 0.24]),
        "PaperlessBilling": rng.choice(["Yes", "No"], n, p=[0.6, 0.4]),
        "PaymentMethod": rng.choice(PAYMENTS, n, p=[0.34, 0.23, 0.22, 0.21]),
        "MonthlyCharges": monthly, "TotalCharges": (ten * monthly).round(2),
    })
    for s in SERVICES:
        df[s] = np.where(net == "No", "No internet service", rng.choice(["Yes", "No"], n))
    return df


def validate_and_clean(df: pd.DataFrame) -> tuple[pd.DataFrame | None, list[str], list[str]]:
    errors, warnings = [], []
    missing = [c for c in RAW_COLS if c not in df.columns]
    if missing:
        return None, [f"Missing required columns: {', '.join(missing)}"], warnings
    df = df.copy()
    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].astype(str).str.strip()
    for c in ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    n_tc = int(df["TotalCharges"].isna().sum())
    if n_tc:
        df["TotalCharges"] = df["TotalCharges"].fillna(df["tenure"] * df["MonthlyCharges"])
        warnings.append(f"{n_tc} blank TotalCharges values imputed as tenure x MonthlyCharges.")
    n_bad = int(df[["tenure", "MonthlyCharges", "SeniorCitizen"]].isna().any(axis=1).sum())
    if n_bad:
        df = df.dropna(subset=["tenure", "MonthlyCharges", "SeniorCitizen"])
        warnings.append(f"{n_bad} rows dropped (non-numeric tenure / MonthlyCharges / SeniorCitizen).")
    oor = int(((df["tenure"] < 0) | (df["tenure"] > 120) | (df["MonthlyCharges"] < 0)).sum())
    if oor:
        warnings.append(f"{oor} rows contain out-of-range numeric values (kept; review recommended).")
    cats = {"Contract": CONTRACTS, "PaymentMethod": PAYMENTS, "InternetService": INTERNET}
    for c, allowed in cats.items():
        bad = int((~df[c].isin(allowed)).sum())
        if bad:
            warnings.append(f"{bad} rows have unseen values in '{c}' (encoded as all-zero for that field).")
    if df.empty:
        errors.append("No valid rows remain after cleaning.")
    return (None if errors else df), errors, warnings


def style(fig: go.Figure, height: int = 380, title: str | None = None) -> go.Figure:
    fig.update_layout(
        template="plotly_dark", height=height, title=title,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=50 if title else 20, b=10),
        font=dict(family="Inter, sans-serif"), legend=dict(orientation="h", y=-0.2),
    )
    return fig


def show(fig: go.Figure) -> None:
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})


def kpi(col, label: str, value: str, sub: str = "", color: str | None = None) -> None:
    style_attr = f" style='color:{color};'" if color else ""
    col.markdown(
        f"<div class='kpi'><div class='label'>{label}</div><div class='value'{style_attr}>{value}</div>"
        f"<div class='sub'>{sub}</div></div>",
        unsafe_allow_html=True,
    )


def feature_importance() -> pd.DataFrame | None:
    est = model.steps[-1][1] if hasattr(model, "steps") else model
    vals = None
    if hasattr(est, "feature_importances_"):
        vals = np.asarray(est.feature_importances_, dtype=float)
    elif hasattr(est, "coef_"):
        vals = np.abs(np.ravel(est.coef_))
    if vals is None or len(vals) != len(COLS):
        return None
    return pd.DataFrame({"Feature": COLS, "Importance": vals}).sort_values("Importance", ascending=False)


def gain_curve(y: np.ndarray, p: np.ndarray) -> pd.DataFrame:
    order = np.argsort(-p)
    ys = y[order]
    cum = np.cumsum(ys) / max(ys.sum(), 1)
    pct = np.arange(1, len(ys) + 1) / len(ys)
    return pd.DataFrame({"pct": pct, "gain": cum, "lift": cum / pct})


# ----------------------------------------------------------------------------
# 5. SIDEBAR - CUSTOMER PROFILE & BUSINESS ASSUMPTIONS
# ----------------------------------------------------------------------------
def apply_preset() -> None:
    p = PRESETS.get(st.session_state["preset"])
    if p:
        for k, v in p.items():
            st.session_state[f"w_{k}"] = v


with st.sidebar:
    st.title("⚙️ Control Panel")
    st.selectbox("Quick-load profile", list(PRESETS), key="preset", on_change=apply_preset)

    st.markdown("#### Customer profile")
    tenure = st.slider("Tenure (months)", 0, 72, 12, key="w_tenure")
    contract = st.selectbox("Contract", CONTRACTS, key="w_contract")
    monthly = st.number_input("Monthly charges (USD)", 18.0, 120.0, 75.0, step=1.0, key="w_monthly")
    internet = st.selectbox("Internet service", INTERNET, key="w_internet")
    services: list[str] = []
    if internet != "No":
        services = st.multiselect("Add-on services", SERVICES, default=["OnlineSecurity"], key="w_services")
    phone = st.radio("Phone service", ["Yes", "No"], horizontal=True, key="w_phone")
    multi = "No phone service"
    if phone == "Yes":
        multi = st.radio("Multiple lines", ["No", "Yes"], horizontal=True, key="w_multi")
    payment = st.selectbox("Payment method", PAYMENTS, key="w_payment")
    paperless = st.radio("Paperless billing", ["Yes", "No"], horizontal=True, key="w_paperless")

    with st.expander("👥 Demographics"):
        gender = st.radio("Gender", ["Female", "Male"], horizontal=True)
        senior = st.checkbox("Senior citizen")
        partner = st.checkbox("Has partner")
        dependents = st.checkbox("Has dependents")

    st.divider()
    st.markdown("#### Decision policy")
    threshold = st.slider("Risk threshold (t*)", 0.05, 0.95, float(np.clip(DEFAULT_THRESHOLD, 0.05, 0.95)), 0.05,
                          help="Customers at or above t* are flagged HIGH RISK. Watchlist starts at t*/2.")

    with st.expander("💰 Retention economics"):
        horizon = st.slider("Value horizon (months)", 6, 36, 12)
        offer_cost = st.number_input("Cost per retention action (USD)", 0.0, 500.0, 25.0, step=5.0)
        success_rate = st.slider("Offer success rate", 0.05, 0.90, 0.30, 0.05)

    st.divider()
    st.caption(f"Model: **{meta.get('model_name', 'n/a')}**")
    if "latency_ms" in st.session_state:
        st.caption(f"Last inference latency: {st.session_state['latency_ms']:.1f} ms")

customer = fix_dependencies({
    "gender": gender, "SeniorCitizen": int(senior), "Partner": "Yes" if partner else "No",
    "Dependents": "Yes" if dependents else "No", "tenure": tenure, "PhoneService": phone,
    "MultipleLines": multi, "InternetService": internet, "Contract": contract,
    "PaperlessBilling": paperless, "PaymentMethod": payment, "MonthlyCharges": float(monthly),
    "TotalCharges": 0.0,
    **{s: ("Yes" if s in services else "No") for s in SERVICES},
})
customer_df = pd.DataFrame([customer])
customer_json = json.dumps(customer, sort_keys=True)

p_churn = float(predict(customer_df)[0])
tier_label, tier_color, tier_action = tier_of(p_churn, threshold)
value_at_stake = monthly * horizon
expected_loss = p_churn * value_at_stake
net_gain = p_churn * success_rate * value_at_stake - offer_cost

# ----------------------------------------------------------------------------
# 6. HEADER
# ----------------------------------------------------------------------------
st.title("🛡️ Customer Churn Intelligence Platform")
st.markdown(
    f"<div class='caption-bar'>Engine: <b>{meta.get('model_name', 'n/a')}</b> &nbsp;|&nbsp; "
    f"Cross-validated AUC: <b>{meta.get('cv_auc', float('nan')):.3f} ± {meta.get('cv_auc_std', float('nan')):.3f}</b> "
    f"&nbsp;|&nbsp; Features: <b>{len(COLS)}</b> &nbsp;|&nbsp; Operating threshold: <b>{threshold:.2f}</b></div>",
    unsafe_allow_html=True,
)

tabs = st.tabs([
    "🎯 Customer Risk", "🧠 Explainability", "🔬 Sensitivity Lab",
    "💰 Retention Economics", "📁 Portfolio Scoring", "📘 Model Card",
])

# ----------------------------------------------------------------------------
# TAB 1 - CUSTOMER RISK
# ----------------------------------------------------------------------------
with tabs[0]:
    c1, c2, c3, c4 = st.columns(4)
    kpi(c1, "Churn probability", f"{p_churn:.1%}", f"threshold {threshold:.0%}", COLORS["accent"])
    kpi(c2, "Risk tier", tier_label, "decision band", tier_color)
    kpi(c3, "Expected revenue loss", f"${expected_loss:,.0f}", f"over {horizon} months (p x value)", COLORS["mid"])
    kpi(c4, "Net value of intervention", f"${net_gain:,.0f}", "p x success x value - cost",
        COLORS["low"] if net_gain > 0 else COLORS["high"])

    st.markdown("<br>", unsafe_allow_html=True)
    g1, g2 = st.columns([1, 1.25])
    with g1:
        gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta", value=p_churn * 100,
            number={"suffix": "%", "font": {"size": 44}},
            delta={"reference": threshold * 100, "increasing": {"color": COLORS["high"]},
                   "decreasing": {"color": COLORS["low"]}, "suffix": " pts vs t*"},
            gauge={
                "axis": {"range": [0, 100]}, "bar": {"color": tier_color, "thickness": 0.3},
                "steps": [
                    {"range": [0, threshold * 50], "color": "rgba(34,197,94,0.25)"},
                    {"range": [threshold * 50, threshold * 100], "color": "rgba(245,158,11,0.25)"},
                    {"range": [threshold * 100, 100], "color": "rgba(239,68,68,0.25)"},
                ],
                "threshold": {"line": {"color": "white", "width": 3}, "thickness": 0.8, "value": threshold * 100},
            },
        ))
        show(style(gauge, 320, "Churn risk gauge"))
        st.info(f"**Recommended action:** {tier_action}")
        if net_gain < 0 and tier_label != "STABLE":
            st.warning("Intervention is not cost-effective under the current economic assumptions.")

    with g2:
        attr = compute_attributions(customer_json)
        top = attr.tail(5).iloc[::-1]
        fig = go.Figure(go.Bar(
            x=top["Contribution"] * 100, y=top["Factor"], orientation="h",
            marker_color=[COLORS["high"] if v > 0 else COLORS["low"] for v in top["Contribution"]],
            text=[f"{v * 100:+.1f} pts" for v in top["Contribution"]], textposition="outside",
        ))
        fig.update_xaxes(title="Impact on churn probability (percentage points)", zeroline=True)
        fig.update_yaxes(autorange="reversed")
        show(style(fig, 320, "Top 5 risk drivers (vs. reference customer)"))

    st.subheader("🔮 Retention what-if simulator")
    st.markdown("<p class='note'>Each row simulates one actionable change applied to this customer. "
                "The final row applies every lever together.</p>", unsafe_allow_html=True)
    levers = build_levers(customer)
    if levers:
        sim_rows, combined = [], dict(customer)
        for _, change in levers:
            r = dict(customer)
            r.update(change)
            sim_rows.append(fix_dependencies(r))
        # Combined plan: best contract + best payment + all add-ons (no stacking of conflicting options)
        for label, change in levers:
            if label in ("Upgrade to 2-year contract", "Move to automatic card payment") or label.startswith("Add") \
                    or label == "Switch to paper billing":
                combined.update(change)
        sim_rows.append(fix_dependencies(combined))
        p_sim = predict(pd.DataFrame(sim_rows))
        labels = [l for l, _ in levers] + ["⭐ Combined plan"]
        res = pd.DataFrame({
            "Action": labels, "New risk": p_sim, "Change (pts)": (p_sim - p_churn) * 100,
        })
        costs = [offer_cost] * len(res)
        res["Net value (USD)"] = (p_churn - p_sim) * value_at_stake - np.array(costs)
        res = res.sort_values("Change (pts)").reset_index(drop=True)

        w1, w2 = st.columns([1.2, 1])
        with w1:
            st.dataframe(
                res, hide_index=True, use_container_width=True,
                column_config={
                    "New risk": st.column_config.ProgressColumn("New risk", format="%.1f%%", min_value=0, max_value=1),
                    "Change (pts)": st.column_config.NumberColumn(format="%+.1f"),
                    "Net value (USD)": st.column_config.NumberColumn(format="$%.0f"),
                },
            )
        with w2:
            fig = go.Figure(go.Bar(
                x=res["Change (pts)"], y=res["Action"], orientation="h",
                marker_color=[COLORS["low"] if v < 0 else COLORS["high"] for v in res["Change (pts)"]],
            ))
            fig.update_yaxes(autorange="reversed")
            fig.update_xaxes(title="Δ churn probability (pts)")
            show(style(fig, 360))
    else:
        st.success("No further actionable levers - this customer is already on the most retentive configuration.")

    st.divider()
    a, b, _ = st.columns([1, 1, 3])
    if a.button("📌 Save snapshot"):
        st.session_state.setdefault("history", []).append({
            "Time": datetime.now().strftime("%H:%M:%S"), "Tenure": tenure, "Contract": contract,
            "Monthly": monthly, "Internet": internet, "Payment": payment,
            "P(churn)": p_churn, "Tier": tier_label,
        })
    report = {"generated_at": datetime.now().isoformat(timespec="seconds"), "profile": customer,
              "churn_probability": p_churn, "tier": tier_label, "threshold": threshold,
              "recommended_action": tier_action, "expected_loss_usd": expected_loss,
              "net_value_of_intervention_usd": net_gain}
    b.download_button("⬇️ Report (JSON)", json.dumps(report, indent=2, default=str), "customer_risk_report.json",
                      "application/json")
    hist = st.session_state.get("history", [])
    if hist:
        hdf = pd.DataFrame(hist)
        st.markdown("**Saved scenario snapshots**")
        st.dataframe(hdf, hide_index=True, use_container_width=True, column_config={
            "P(churn)": st.column_config.ProgressColumn("P(churn)", format="%.1f%%", min_value=0, max_value=1)})
        if len(hdf) > 1:
            fig = px.line(hdf, x=hdf.index + 1, y="P(churn)", markers=True)
            fig.add_hline(y=threshold, line_dash="dash", line_color=COLORS["high"], annotation_text="t*")
            fig.update_xaxes(title="Snapshot #")
            show(style(fig, 260, "Risk across saved snapshots"))

# ----------------------------------------------------------------------------
# TAB 2 - EXPLAINABILITY
# ----------------------------------------------------------------------------
with tabs[1]:
    st.subheader("Local explanation - why this customer?")
    st.markdown(
        "<p class='note'>Occlusion attribution: each factor group is replaced by a typical reference value and "
        "the change in predicted churn probability is measured. Positive = pushes risk up. "
        "This is model-agnostic and approximates, but is not identical to, Shapley values.</p>",
        unsafe_allow_html=True)
    attr = compute_attributions(customer_json)
    ref_p = float(predict(pd.DataFrame([fix_dependencies(dict(REFERENCE))]))[0])
    e1, e2 = st.columns([1.4, 1])
    with e1:
        fig = go.Figure(go.Bar(
            x=attr["Contribution"] * 100, y=attr["Factor"], orientation="h",
            marker_color=[COLORS["high"] if v > 0 else COLORS["low"] for v in attr["Contribution"]],
            text=[f"{v * 100:+.1f}" for v in attr["Contribution"]], textposition="outside"))
        fig.update_xaxes(title="Contribution (percentage points)", zeroline=True)
        show(style(fig, 460, "All factor contributions"))
    with e2:
        wf = attr.sort_values("Contribution", key=np.abs, ascending=False)
        fig = go.Figure(go.Waterfall(
            orientation="v", measure=["absolute"] + ["relative"] * len(wf) + ["total"],
            x=["Reference"] + wf["Factor"].tolist() + ["This customer"],
            y=[ref_p * 100] + (wf["Contribution"] * 100).tolist() + [0],
            connector={"line": {"color": COLORS["muted"]}},
            increasing={"marker": {"color": COLORS["high"]}}, decreasing={"marker": {"color": COLORS["low"]}},
            totals={"marker": {"color": COLORS["accent"]}}))
        fig.update_layout(showlegend=False)
        fig.update_xaxes(tickangle=-40)
        fig.update_yaxes(title="P(churn) %")
        show(style(fig, 460, "Approximate walk from reference to customer"))
        st.caption("Waterfall is indicative: with interacting features, individual effects do not sum exactly.")

    st.subheader("Global drivers - what does the model rely on?")
    fi = feature_importance()
    if fi is not None:
        top = fi.head(20).iloc[::-1]
        fig = go.Figure(go.Bar(x=top["Importance"], y=top["Feature"], orientation="h", marker_color=COLORS["accent"]))
        fig.update_xaxes(title="Importance (|coef| or impurity-based)")
        show(style(fig, 520, "Top 20 model features"))
    else:
        st.info("Global importances are unavailable for this estimator type "
                "(requires `feature_importances_` or `coef_` on the final estimator).")

    st.subheader("Customer vs. reference profile")
    cmp_df = pd.DataFrame({"Field": RAW_COLS, "This customer": [customer[c] for c in RAW_COLS],
                           "Reference": [REFERENCE[c] for c in RAW_COLS]}).astype(str)
    st.dataframe(cmp_df, hide_index=True, use_container_width=True)

# ----------------------------------------------------------------------------
# TAB 3 - SENSITIVITY LAB
# ----------------------------------------------------------------------------
with tabs[2]:
    st.markdown("<p class='note'>All curves vary one or two inputs while holding the rest of the "
                "sidebar profile fixed. The dashed marker is the current customer.</p>", unsafe_allow_html=True)
    s1, s2 = st.columns(2)
    with s1:
        df = sensitivity_tenure(customer_json)
        fig = px.line(df, x="tenure", y="p", color="Contract",
                      color_discrete_sequence=[COLORS["high"], COLORS["mid"], COLORS["low"]])
        fig.add_vline(x=tenure, line_dash="dash", line_color="white")
        fig.add_hline(y=threshold, line_dash="dot", line_color=COLORS["muted"], annotation_text="t*")
        fig.update_yaxes(tickformat=".0%", title="P(churn)")
        fig.update_xaxes(title="Tenure (months)")
        show(style(fig, 380, "Churn risk vs. tenure by contract"))
    with s2:
        df = sensitivity_price(customer_json)
        fig = px.line(df, x="MonthlyCharges", y="p", color="Internet",
                      color_discrete_sequence=[COLORS["high"], COLORS["accent"], COLORS["low"]])
        fig.add_vline(x=monthly, line_dash="dash", line_color="white")
        fig.add_hline(y=threshold, line_dash="dot", line_color=COLORS["muted"], annotation_text="t*")
        fig.update_yaxes(tickformat=".0%", title="P(churn)")
        fig.update_xaxes(title="Monthly charges (USD)")
        show(style(fig, 380, "Churn risk vs. price by internet type"))

    grid = sensitivity_grid(customer_json)
    fig = go.Figure(go.Heatmap(
        z=grid.values * 100, x=grid.columns, y=grid.index, colorscale="RdYlGn_r", zmin=0, zmax=100,
        colorbar=dict(title="P(churn) %"), hovertemplate="Tenure %{x}m, $%{y}<br>%{z:.1f}%<extra></extra>"))
    fig.add_trace(go.Scatter(x=[tenure], y=[monthly], mode="markers",
                             marker=dict(size=14, color="white", symbol="x", line=dict(width=2)), name="Current"))
    fig.update_xaxes(title="Tenure (months)")
    fig.update_yaxes(title="Monthly charges (USD)")
    show(style(fig, 440, "Risk surface: tenure × monthly charges"))

    st.subheader("Contract × payment method comparison")
    rows = [fix_dependencies(dict(customer, Contract=c, PaymentMethod=p)) | {"_c": c, "_p": p}
            for c in CONTRACTS for p in PAYMENTS]
    cp = pd.DataFrame(rows)
    cp["p"] = predict(cp)
    fig = px.bar(cp, x="_p", y="p", color="_c", barmode="group", labels={"_p": "Payment method", "p": "P(churn)",
                                                                       "_c": "Contract"},
                 color_discrete_sequence=[COLORS["high"], COLORS["mid"], COLORS["low"]])
    fig.update_yaxes(tickformat=".0%")
    show(style(fig, 360))

# ----------------------------------------------------------------------------
# PORTFOLIO DATA (shared by economics + batch tabs)
# ----------------------------------------------------------------------------
def current_portfolio() -> tuple[pd.DataFrame | None, str]:
    if "portfolio" in st.session_state:
        return st.session_state["portfolio"], st.session_state.get("portfolio_src", "uploaded")
    return None, ""


# ----------------------------------------------------------------------------
# TAB 5 - PORTFOLIO SCORING (built before tab 4 so economics can use it)
# ----------------------------------------------------------------------------
with tabs[4]:
    st.subheader("Batch scoring & portfolio monitoring")
    u1, u2, u3 = st.columns([2, 1, 1])
    upload = u1.file_uploader("Upload customer CSV (optional `Churn` column enables model evaluation)", type=["csv"])
    template = synth_portfolio(3)[["customerID"] + RAW_COLS]
    u2.markdown("<br>", unsafe_allow_html=True)
    u2.download_button("⬇️ CSV template", template.to_csv(index=False), "churn_template.csv", "text/csv")
    u3.markdown("<br>", unsafe_allow_html=True)
    if u3.button("🎲 Load demo portfolio (600)"):
        st.session_state["portfolio_raw"] = synth_portfolio(600)
        st.session_state["portfolio_src"] = "synthetic demo data"
        st.session_state.pop("portfolio", None)
    if upload is not None:
        try:
            st.session_state["portfolio_raw"] = pd.read_csv(upload)
            st.session_state["portfolio_src"] = f"upload: {upload.name}"
            st.session_state.pop("portfolio", None)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not read CSV: {exc}")

    raw = st.session_state.get("portfolio_raw")
    if raw is None:
        st.info("Upload a CSV or load the demo portfolio to begin.")
    else:
        clean, errs, warns = validate_and_clean(raw)
        for e in errs:
            st.error(e)
        for w in warns:
            st.warning(w)
        if clean is not None:
            with st.spinner("Scoring portfolio..."):
                scored = clean.copy()
                scored["churn_probability"] = predict(scored)
            scored["tier"] = tier_series(scored["churn_probability"], threshold)
            scored["flagged"] = scored["churn_probability"] >= threshold
            scored["value_at_stake"] = scored["MonthlyCharges"] * horizon
            scored["expected_loss"] = scored["churn_probability"] * scored["value_at_stake"]
            st.session_state["portfolio"] = scored
            src = st.session_state.get("portfolio_src", "")
            st.caption(f"Source: {src} — {len(scored):,} customers scored.")

            k = st.columns(5)
            kpi(k[0], "Customers", f"{len(scored):,}")
            kpi(k[1], "Mean risk", f"{scored['churn_probability'].mean():.1%}", color=COLORS["accent"])
            kpi(k[2], "Flagged high-risk", f"{scored['flagged'].mean():.1%}", f"{int(scored['flagged'].sum()):,} accounts",
                COLORS["high"])
            kpi(k[3], "Expected churners", f"{scored['churn_probability'].sum():,.0f}")
            kpi(k[4], "Revenue at risk", f"${scored['expected_loss'].sum():,.0f}", f"{horizon}-month horizon", COLORS["mid"])
            st.markdown("<br>", unsafe_allow_html=True)

            r1, r2 = st.columns([1.5, 1])
            with r1:
                fig = px.histogram(scored, x="churn_probability", nbins=40, color="tier",
                                   color_discrete_map={"Stable": COLORS["low"], "Watchlist": COLORS["mid"],
                                                       "High risk": COLORS["high"]},
                                   category_orders={"tier": ["Stable", "Watchlist", "High risk"]})
                fig.add_vline(x=threshold, line_dash="dash", line_color="white", annotation_text="t*")
                fig.update_xaxes(title="Predicted churn probability", tickformat=".0%")
                show(style(fig, 360, "Risk score distribution"))
            with r2:
                cnt = scored["tier"].value_counts().reindex(["Stable", "Watchlist", "High risk"]).fillna(0)
                fig = go.Figure(go.Pie(labels=cnt.index.astype(str), values=cnt.values, hole=0.55,
                                       marker_colors=[COLORS["low"], COLORS["mid"], COLORS["high"]]))
                show(style(fig, 360, "Tier mix"))

            r3, r4 = st.columns(2)
            with r3:
                g = scored.groupby("Contract", observed=True)["churn_probability"].mean().reset_index()
                fig = px.bar(g, x="Contract", y="churn_probability", color="Contract",
                             color_discrete_sequence=[COLORS["high"], COLORS["mid"], COLORS["low"]])
                fig.update_yaxes(tickformat=".0%", title="Mean P(churn)")
                fig.update_layout(showlegend=False)
                show(style(fig, 330, "Mean risk by contract"))
            with r4:
                g = scored.groupby("PaymentMethod", observed=True)["churn_probability"].mean().reset_index()
                fig = px.bar(g, x="churn_probability", y="PaymentMethod", orientation="h",
                             color_discrete_sequence=[COLORS["accent"]])
                fig.update_xaxes(tickformat=".0%", title="Mean P(churn)")
                show(style(fig, 330, "Mean risk by payment method"))

            r5, r6 = st.columns(2)
            with r5:
                scored["tenure_band"] = pd.cut(scored["tenure"], [-1, 6, 12, 24, 48, 100],
                                               labels=["0-6", "7-12", "13-24", "25-48", "49+"])
                g = scored.groupby("tenure_band", observed=True)["churn_probability"].mean().reset_index()
                fig = px.line(g, x="tenure_band", y="churn_probability", markers=True)
                fig.update_yaxes(tickformat=".0%", title="Mean P(churn)")
                fig.update_xaxes(title="Tenure band (months)")
                show(style(fig, 330, "Risk by tenure band"))
            with r6:
                samp = scored.sample(min(len(scored), 3000), random_state=1)
                fig = px.scatter(samp, x="tenure", y="MonthlyCharges", color="churn_probability",
                                 color_continuous_scale="RdYlGn_r", range_color=(0, 1), opacity=0.7)
                show(style(fig, 330, "Customer map: tenure vs. charges"))

            st.markdown("#### 🎯 Priority retention list")
            n_top = st.slider("Show top N by expected revenue loss", 10, 200, 25, 5)
            topn = scored.sort_values("expected_loss", ascending=False).head(n_top)
            show_cols = [c for c in ["customerID", "tenure", "Contract", "InternetService", "PaymentMethod",
                                     "MonthlyCharges", "churn_probability", "expected_loss", "tier"] if c in topn.columns]
            st.dataframe(topn[show_cols], hide_index=True, use_container_width=True, column_config={
                "churn_probability": st.column_config.ProgressColumn("P(churn)", format="%.1f%%", min_value=0, max_value=1),
                "expected_loss": st.column_config.NumberColumn("Expected loss", format="$%.0f"),
                "MonthlyCharges": st.column_config.NumberColumn(format="$%.2f")})
            out_csv = scored.drop(columns=["tenure_band"], errors="ignore").to_csv(index=False)
            st.download_button("⬇️ Download full scored portfolio", out_csv, "scored_portfolio.csv", "text/csv")

            # ---- Model evaluation (only when ground truth available)
            if "Churn" in raw.columns:
                st.divider()
                st.subheader("📈 Model evaluation on labelled data")
                yraw = raw.loc[scored.index, "Churn"]
                y = yraw.astype(str).str.strip().str.lower().map({"yes": 1, "no": 0, "1": 1, "0": 0, "true": 1, "false": 0})
                valid = y.notna()
                if valid.sum() < 20 or y[valid].nunique() < 2:
                    st.warning("Not enough valid, two-class `Churn` labels to evaluate.")
                else:
                    yv = y[valid].astype(int).values
                    pv = scored.loc[valid[valid].index, "churn_probability"].values
                    pred = (pv >= threshold).astype(int)
                    tn, fp, fn, tp = confusion_matrix(yv, pred, labels=[0, 1]).ravel()
                    prec = tp / max(tp + fp, 1)
                    rec = tp / max(tp + fn, 1)
                    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
                    m = st.columns(6)
                    kpi(m[0], "ROC-AUC", f"{roc_auc_score(yv, pv):.3f}")
                    kpi(m[1], "PR-AUC", f"{average_precision_score(yv, pv):.3f}")
                    kpi(m[2], "Precision", f"{prec:.1%}", f"@ t*={threshold:.2f}")
                    kpi(m[3], "Recall", f"{rec:.1%}")
                    kpi(m[4], "F1", f"{f1:.3f}")
                    kpi(m[5], "Brier", f"{brier_score_loss(yv, pv):.3f}", "lower is better")
                    st.markdown("<br>", unsafe_allow_html=True)

                    v1, v2, v3 = st.columns(3)
                    with v1:
                        fpr, tpr, _ = roc_curve(yv, pv)
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="Model", line=dict(color=COLORS["accent"], width=3)))
                        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance",
                                                 line=dict(dash="dash", color=COLORS["muted"])))
                        fig.update_xaxes(title="False positive rate")
                        fig.update_yaxes(title="True positive rate")
                        show(style(fig, 340, "ROC curve"))
                    with v2:
                        pr, rc, _ = precision_recall_curve(yv, pv)
                        fig = go.Figure(go.Scatter(x=rc, y=pr, mode="lines", line=dict(color=COLORS["mid"], width=3)))
                        fig.add_hline(y=yv.mean(), line_dash="dash", line_color=COLORS["muted"], annotation_text="Base rate")
                        fig.update_xaxes(title="Recall")
                        fig.update_yaxes(title="Precision")
                        show(style(fig, 340, "Precision-recall curve"))
                    with v3:
                        cm = np.array([[tn, fp], [fn, tp]])
                        fig = go.Figure(go.Heatmap(z=cm, x=["Pred: stay", "Pred: churn"], y=["Actual: stay", "Actual: churn"],
                                                   text=cm, texttemplate="%{text}", colorscale="Blues", showscale=False))
                        fig.update_yaxes(autorange="reversed")
                        show(style(fig, 340, "Confusion matrix"))

                    w1, w2, w3 = st.columns(3)
                    with w1:
                        n_bins = 8 if len(yv) >= 400 else 5
                        frac, mean_p = calibration_curve(yv, pv, n_bins=n_bins, strategy="quantile")
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=mean_p, y=frac, mode="lines+markers", name="Model"))
                        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect",
                                                 line=dict(dash="dash", color=COLORS["muted"])))
                        fig.update_xaxes(title="Mean predicted probability")
                        fig.update_yaxes(title="Observed churn rate")
                        show(style(fig, 340, "Calibration curve"))
                    with w2:
                        gc = gain_curve(yv, pv)
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=gc["pct"], y=gc["gain"], name="Model", line=dict(color=COLORS["low"], width=3)))
                        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Random", line=dict(dash="dash", color=COLORS["muted"])))
                        fig.update_xaxes(title="Share of customers contacted", tickformat=".0%")
                        fig.update_yaxes(title="Share of churners captured", tickformat=".0%")
                        show(style(fig, 340, "Cumulative gain"))
                    with w3:
                        fig = go.Figure(go.Scatter(x=gc["pct"].values[::max(len(gc) // 100, 1)],
                                                   y=gc["lift"].values[::max(len(gc) // 100, 1)],
                                                   line=dict(color=COLORS["accent"], width=3)))
                        fig.add_hline(y=1, line_dash="dash", line_color=COLORS["muted"])
                        fig.update_xaxes(title="Share of customers contacted", tickformat=".0%")
                        fig.update_yaxes(title="Lift")
                        show(style(fig, 340, "Lift curve"))

                    ts = np.linspace(0.05, 0.95, 19)
                    sweep = []
                    for t in ts:
                        pr_ = (pv >= t).astype(int)
                        tp_ = int(((pr_ == 1) & (yv == 1)).sum())
                        fp_ = int(((pr_ == 1) & (yv == 0)).sum())
                        fn_ = int(((pr_ == 0) & (yv == 1)).sum())
                        p_ = tp_ / max(tp_ + fp_, 1)
                        r_ = tp_ / max(tp_ + fn_, 1)
                        sweep.append({"threshold": t, "precision": p_, "recall": r_, "f1": 2 * p_ * r_ / max(p_ + r_, 1e-9)})
                    sw = pd.DataFrame(sweep)
                    fig = px.line(sw.melt("threshold"), x="threshold", y="value", color="variable", markers=True)
                    fig.add_vline(x=threshold, line_dash="dash", line_color="white")
                    fig.update_yaxes(title="Score")
                    show(style(fig, 340, "Threshold sweep: precision / recall / F1"))

# ----------------------------------------------------------------------------
# TAB 4 - RETENTION ECONOMICS
# ----------------------------------------------------------------------------
with tabs[3]:
    st.subheader("Retention campaign economics")
    st.markdown(
        "<p class='note'>Expected net benefit of contacting a customer = P(churn) × success rate × customer value − "
        "action cost. Adjust assumptions in the sidebar. Calibrated probabilities are assumed.</p>",
        unsafe_allow_html=True)
    port, src = current_portfolio()
    if port is None:
        port = synth_portfolio(600)
        port["churn_probability"] = predict(port)
        src = "synthetic demo data (load/upload data in the Portfolio tab to use your own)"
    st.caption(f"Portfolio used: {src}")

    value = (port["MonthlyCharges"] * horizon).values
    p = port["churn_probability"].values
    ts = np.linspace(0.02, 0.98, 49)
    profits, contacted = [], []
    for t in ts:
        sel = p >= t
        profits.append(float((p[sel] * success_rate * value[sel] - offer_cost).sum()))
        contacted.append(int(sel.sum()))
    best_i = int(np.argmax(profits))
    best_t, best_profit = float(ts[best_i]), profits[best_i]
    cur_sel = p >= threshold
    cur_profit = float((p[cur_sel] * success_rate * value[cur_sel] - offer_cost).sum())

    e = st.columns(4)
    kpi(e[0], "Profit-optimal threshold", f"{best_t:.2f}", f"vs. current {threshold:.2f}", COLORS["accent"])
    kpi(e[1], "Max campaign net benefit", f"${best_profit:,.0f}", f"{contacted[best_i]:,} customers contacted", COLORS["low"])
    kpi(e[2], "Net benefit at current t*", f"${cur_profit:,.0f}", f"{int(cur_sel.sum()):,} contacted",
        COLORS["low"] if cur_profit >= 0 else COLORS["high"])
    kpi(e[3], "Uplift from re-tuning", f"${best_profit - cur_profit:,.0f}", "optimal − current", COLORS["mid"])
    st.markdown("<br>", unsafe_allow_html=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ts, y=profits, mode="lines", name="Net benefit", fill="tozeroy",
                             line=dict(color=COLORS["low"], width=3)))
    fig.add_vline(x=best_t, line_dash="dash", line_color=COLORS["accent"], annotation_text="optimal")
    fig.add_vline(x=threshold, line_dash="dash", line_color="white", annotation_text="current t*")
    fig.update_xaxes(title="Decision threshold")
    fig.update_yaxes(title="Expected campaign net benefit (USD)")
    show(style(fig, 380, "Profit curve vs. threshold"))

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure(go.Scatter(x=ts, y=contacted, mode="lines", line=dict(color=COLORS["mid"], width=3)))
        fig.add_vline(x=threshold, line_dash="dash", line_color="white")
        fig.update_xaxes(title="Decision threshold")
        fig.update_yaxes(title="Customers contacted")
        show(style(fig, 320, "Campaign size vs. threshold"))
    with c2:
        be = offer_cost / max(success_rate * float(np.mean(value)), 1e-9)
        st.markdown("##### Sensitivity of optimal profit to assumptions")
        sr_grid = np.arange(0.1, 0.71, 0.1)
        cost_grid = np.array([5, 15, 25, 50, 100])
        z = np.zeros((len(cost_grid), len(sr_grid)))
        for i, cc in enumerate(cost_grid):
            for j, ss in enumerate(sr_grid):
                z[i, j] = max(float(((p[p >= t] * ss * value[p >= t]) - cc).sum()) for t in ts)
        fig = go.Figure(go.Heatmap(z=z, x=[f"{s:.0%}" for s in sr_grid], y=[f"${c}" for c in cost_grid],
                                   colorscale="Viridis", colorbar=dict(title="USD"),
                                   hovertemplate="Success %{x}, cost %{y}<br>$%{z:,.0f}<extra></extra>"))
        fig.update_xaxes(title="Offer success rate")
        fig.update_yaxes(title="Cost per action")
        show(style(fig, 320))
        st.caption(f"Break-even churn probability for an average customer ≈ {min(be, 1):.1%}.")

# ----------------------------------------------------------------------------
# TAB 6 - MODEL CARD
# ----------------------------------------------------------------------------
with tabs[5]:
    st.subheader("Model card")
    m1, m2 = st.columns([1, 1])
    with m1:
        st.markdown("##### Metadata")
        summary = {k: v for k, v in meta.items() if k != "feature_columns"}
        summary["n_input_features (after encoding)"] = len(COLS)
        summary["estimator_class"] = type(model.steps[-1][1] if hasattr(model, "steps") else model).__name__
        st.json(summary)
        st.markdown("##### Intended use")
        st.markdown(
            "- Rank customers by short-term churn risk to prioritise retention outreach.\n"
            "- Support what-if analysis of contract, billing and service levers.\n"
            "- Not intended for automated adverse decisions affecting customers' access to service."
        )
    with m2:
        st.markdown("##### Limitations & responsible use")
        st.markdown(
            "- Trained on a single historical telecom dataset; performance may drift as offers, pricing and "
            "competition change - monitor calibration and re-train periodically.\n"
            "- Predictions are **associational, not causal**: the what-if simulator shows how the *model* responds, "
            "not guaranteed behavioural effects. Validate levers with A/B tests.\n"
            "- Demographic inputs (gender, senior citizen) can encode bias. Audit group-level error rates before "
            "operational use and consider excluding them.\n"
            "- `TotalCharges` is approximated as tenure × monthly charges in the single-customer view."
        )
        st.markdown("##### Input schema")
        st.dataframe(pd.DataFrame({"Column": RAW_COLS, "Role": ["numeric" if c in ("tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen") else "categorical" for c in RAW_COLS]}),
                     hide_index=True, use_container_width=True, height=250)

    fi = feature_importance()
    if fi is not None:
        fig = px.bar(fi.head(15).iloc[::-1], x="Importance", y="Feature", orientation="h",
                     color_discrete_sequence=[COLORS["accent"]])
        show(style(fig, 420, "Top 15 model features"))

    with st.expander("Encoded feature list"):
        st.write(COLS)

    st.markdown("##### System health")
    h = st.columns(3)
    kpi(h[0], "Artifacts", "OK", "model + metadata loaded", COLORS["low"])
    kpi(h[1], "Inference latency", f"{st.session_state.get('latency_ms', 0):.1f} ms", "last call")
    kpi(h[2], "Session snapshots", str(len(st.session_state.get("history", []))))
