import json
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score, brier_score_loss, confusion_matrix,
    precision_recall_curve, roc_auc_score, roc_curve,
)

import config as cfg
import engine
import ui
from config import COLORS

st.set_page_config(
    page_title="Churn Intelligence | Pakistan",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)
ui.inject_css()
ui.reset_keys()

if not engine.load():
    st.error("### Model files not found")
    st.info("Put `churn_model.joblib` and `model_meta.json` in the same folder as `app.py` and refresh.")
    st.stop()

META = engine.META
COLS = engine.COLS
DEFAULT_T = float(META.get("threshold", 0.5))


# ======================================================================
# state helpers
# ======================================================================
def load_demo():
    clean, _, notes = engine.validate_and_clean(engine.synth_portfolio(1500))
    st.session_state.update(
        data=clean,
        data_name="Demo portfolio (synthetic, 1,500 customers)",
        data_notes=notes,
        data_id=st.session_state.get("data_id", 0) + 1,
    )


def apply_preset():
    p = cfg.PRESETS.get(st.session_state["preset"])
    if not p:
        return
    for k in ["tenure", "contract", "internet", "services", "phone", "multi", "payment", "paperless"]:
        st.session_state[f"w_{k}"] = p[k]
    cur = st.session_state.get("cur", "PKR")
    fx = st.session_state.get("fx", cfg.DEFAULT_FX)
    st.session_state["monthly_usd"] = p["monthly"]
    st.session_state[f"w_monthly_{cur}"] = p["monthly"] * (fx if cur == "PKR" else 1.0)


if "data" not in st.session_state:
    load_demo()

st.session_state.setdefault("w_tenure", 12)
st.session_state.setdefault("w_services", ["OnlineSecurity"])

# ======================================================================
# sidebar
# ======================================================================
with st.sidebar:
    st.markdown("## 🛡️ Churn Intelligence")
    st.caption("Telecom retention analytics, built for the Pakistan market")

    with st.expander("🌐 Display settings", expanded=True):
        cur = st.selectbox("Currency", ["PKR", "USD"], key="cur")
        fx = st.number_input(
            "USD → PKR rate", 100.0, 600.0, cfg.DEFAULT_FX, 1.0, key="fx", disabled=(cur == "USD"),
            help="Editable. Set this to the current interbank / open-market rate you want to use.",
        )
        chart_style = st.radio("Chart style", ["Dark", "Light"], horizontal=True, key="chart_style")
    ui.configure(cur, fx, chart_style.lower())
    scale, pre = ui.scale(), ui.prefix()
    is_pkr = cur == "PKR"

    with st.expander("🎯 Decision policy & economics", expanded=True):
        threshold = st.slider(
            "Risk threshold (t*)", 0.05, 0.95, float(np.clip(DEFAULT_T, 0.05, 0.95)), 0.05,
            help="Customers at or above t* are HIGH RISK. The watchlist starts at half of t*.",
        )
        horizon = st.slider("Value horizon (months)", 6, 36, 12)
        cost_in = st.number_input(
            f"Cost per retention action ({cur})", 0.0, 200000.0 if is_pkr else 700.0,
            2500.0 if is_pkr else 9.0, 250.0 if is_pkr else 1.0, key=f"cost_{cur}",
        )
        cost_usd = cost_in / scale
        success = st.slider("Offer success rate", 0.05, 0.90, 0.30, 0.05)

    with st.expander("👤 Customer profile", expanded=True):
        st.selectbox("Quick-load profile", list(cfg.PRESETS), key="preset", on_change=apply_preset)
        tenure = st.slider("Tenure (months)", 0, 72, key="w_tenure")
        contract = st.selectbox("Contract", cfg.CONTRACTS, key="w_contract")

        # the bill is stored in USD underneath, the widget shows whichever currency is selected
        lo, hi = cfg.BILL_MIN_USD * scale, cfg.BILL_MAX_USD * scale
        bill_key = f"w_monthly_{cur}"
        sig = (cur, round(float(fx), 4))
        if st.session_state.get("_bill_sig") != sig:
            now_usd = st.session_state.get("monthly_usd", 75.0)
            st.session_state[bill_key] = float(np.clip(now_usd * scale, lo, hi))
            st.session_state["_bill_sig"] = sig
        bill_in = st.number_input(f"Monthly bill ({cur})", lo, hi, step=100.0 if is_pkr else 1.0, key=bill_key)
        monthly_usd = bill_in / scale
        st.session_state["monthly_usd"] = monthly_usd

        internet = st.selectbox("Internet service", cfg.INTERNET, key="w_internet")
        services = []
        if internet != "No":
            services = st.multiselect("Add-on services", cfg.SERVICES, key="w_services")
        phone = st.radio("Phone service", ["Yes", "No"], horizontal=True, key="w_phone")
        multi = "No phone service"
        if phone == "Yes":
            multi = st.radio("Multiple lines", ["No", "Yes"], horizontal=True, key="w_multi")
        payment = st.selectbox("Payment method", cfg.PAYMENTS, key="w_payment")
        paperless = st.radio("Paperless billing", ["Yes", "No"], horizontal=True, key="w_paperless")
        gender = st.radio("Gender", ["Female", "Male"], horizontal=True)
        senior = st.checkbox("Senior citizen")
        partner = st.checkbox("Has partner")
        dependents = st.checkbox("Has dependents")

    with st.expander("📂 Data source", expanded=False):
        upload = st.file_uploader("Upload customer CSV", type=["csv"], key="uploader",
                                  help="Works with the original Telco churn CSV. A 'City' column adds the Pakistan map.")
        if upload is not None and st.session_state.get("up_sig") != (upload.name, upload.size):
            st.session_state["up_sig"] = (upload.name, upload.size)
            try:
                clean, errs, notes = engine.validate_and_clean(pd.read_csv(upload))
                if errs:
                    for e in errs:
                        st.error(e)
                else:
                    st.session_state.update(
                        data=clean, data_name=f"Uploaded: {upload.name}", data_notes=notes,
                        data_id=st.session_state.get("data_id", 0) + 1,
                    )
            except Exception as e:
                st.error(f"Could not read that file: {e}")
        b1, b2 = st.columns(2)
        b1.button("Use demo data", on_click=load_demo, width="stretch")
        b2.download_button("Template", engine.template_csv(), "churn_template.csv", "text/csv",
                           width="stretch")
        st.caption(f"Current: {st.session_state['data_name']}")

    scored = engine.score(st.session_state["data"])
    did = st.session_state.get("data_id", 0)

    with st.expander("🔎 Portfolio filters", expanded=False):
        flt = {"cats": {}}
        if "City" in scored.columns:
            flt["cats"]["City"] = st.multiselect("City", sorted(scored["City"].astype(str).unique()), key=f"f_city_{did}")
        flt["cats"]["Contract"] = st.multiselect("Contract type", cfg.CONTRACTS, key=f"f_contract_{did}")
        flt["cats"]["InternetService"] = st.multiselect("Internet service", cfg.INTERNET, key=f"f_net_{did}")
        flt["cats"]["PaymentMethod"] = st.multiselect("Payment method", cfg.PAYMENTS, key=f"f_pay_{did}")
        flt["cats"]["tier"] = st.multiselect("Risk tier", cfg.TIERS, key=f"f_tier_{did}")

        t_lo, t_hi = int(scored["tenure"].min()), int(scored["tenure"].max())
        if t_lo < t_hi:
            flt["tenure"] = st.slider("Tenure range (months)", t_lo, t_hi, (t_lo, t_hi), key=f"f_ten_{did}")
        else:
            flt["tenure"] = (t_lo, t_hi)

        b_lo = float(np.floor(scored["MonthlyCharges"].min() * scale))
        b_hi = float(np.ceil(scored["MonthlyCharges"].max() * scale))
        if b_lo < b_hi:
            flt["bill"] = st.slider(f"Monthly bill range ({cur})", b_lo, b_hi, (b_lo, b_hi),
                                    step=100.0 if is_pkr else 1.0, key=f"f_bill_{cur}_{int(fx)}_{did}")
        else:
            flt["bill"] = (b_lo, b_hi)

        flt["senior"] = st.radio("Senior citizens", ["All", "Only seniors", "Exclude seniors"], key=f"f_sen_{did}")
        flt["min_p"] = st.slider("Minimum churn probability", 0.0, 1.0, 0.0, 0.05, key=f"f_minp_{did}")

    st.divider()
    st.caption(f"Model: {META.get('model_name', 'n/a')}")
    if "latency_ms" in st.session_state:
        st.caption(f"Last inference: {st.session_state['latency_ms']:.1f} ms")


# ======================================================================
# portfolio frame (adds business columns, then applies the filters)
# ======================================================================
base = scored.copy()
base["tier"] = engine.tier_series(base["churn_probability"], threshold)
base["flagged"] = base["churn_probability"] >= threshold
base["value_usd"] = base["MonthlyCharges"] * horizon
base["exp_loss_usd"] = base["churn_probability"] * base["value_usd"]
base["net_usd"] = base["churn_probability"] * success * base["value_usd"] - cost_usd
base["bill"] = base["MonthlyCharges"] * scale
base["exp_loss"] = base["exp_loss_usd"] * scale
base["tenure_band"] = pd.cut(base["tenure"], [-1, 6, 12, 24, 48, 1000],
                             labels=["0-6 m", "7-12 m", "13-24 m", "25-48 m", "49+ m"])
base["Senior"] = np.where(base["SeniorCitizen"] == 1, "Senior", "Non-senior")

_edges = [0, 35, 55, 75, 95]
_names = ([f"under {pre}{_edges[1] * scale:,.0f}"]
          + [f"{pre}{_edges[i] * scale:,.0f} to {_edges[i + 1] * scale:,.0f}" for i in range(1, 4)]
          + [f"{pre}{_edges[4] * scale:,.0f}+"])
base["bill_band"] = pd.cut(base["MonthlyCharges"], [-1, 35, 55, 75, 95, 1e9], labels=_names)

mask = pd.Series(True, index=base.index)
for col, chosen in flt["cats"].items():
    if chosen:
        mask &= base[col].astype(str).isin(chosen)
mask &= base["tenure"].between(*flt["tenure"])
mask &= base["bill"].between(flt["bill"][0] - 0.5, flt["bill"][1] + 0.5)
mask &= base["churn_probability"] >= flt["min_p"]
if flt["senior"] == "Only seniors":
    mask &= base["SeniorCitizen"] == 1
elif flt["senior"] == "Exclude seniors":
    mask &= base["SeniorCitizen"] == 0
view = base[mask]
st.sidebar.info(f"Showing **{len(view):,}** of **{len(base):,}** customers")

# the single customer being simulated from the sidebar
customer = engine.fix_dependencies({
    "gender": gender, "SeniorCitizen": int(senior), "Partner": "Yes" if partner else "No",
    "Dependents": "Yes" if dependents else "No", "tenure": tenure, "PhoneService": phone,
    "MultipleLines": multi, "InternetService": internet, "Contract": contract,
    "PaperlessBilling": paperless, "PaymentMethod": payment, "MonthlyCharges": float(monthly_usd),
    "TotalCharges": 0.0, **{s: ("Yes" if s in services else "No") for s in cfg.SERVICES},
})
cust_json = json.dumps(customer, sort_keys=True)
p_churn = float(engine.predict(pd.DataFrame([customer]))[0])
tier_label, tier_color, tier_action = engine.tier_of(p_churn, threshold)
cust_value_usd = monthly_usd * horizon
cust_loss_usd = p_churn * cust_value_usd
cust_net_usd = p_churn * success * cust_value_usd - cost_usd

# ======================================================================
# header
# ======================================================================
auc_txt = f"{META.get('cv_auc', float('nan')):.3f} ± {META.get('cv_auc_std', float('nan')):.3f}"
ui.hero(
    "Customer Churn Intelligence",
    "Score customers, understand why they might leave, and plan retention campaigns that pay for themselves.",
    [
        f"Model: {META.get('model_name', 'n/a')}",
        f"CV AUC {auc_txt}",
        f"Threshold {threshold:.2f}",
        f"Currency: PKR @ {fx:.0f}" if is_pkr else "Currency: USD",
        f"{len(view):,} / {len(base):,} customers in view",
    ],
)

tabs = st.tabs([
    "🏠 Overview", "🎯 Customer Risk", "🧠 Explainability", "🔬 Sensitivity Lab",
    "🧩 Segments", "📣 Campaign Planner", "🗂️ Data & Evaluation", "📘 Model Card",
])


# ======================================================================
# tab 1 - overview
# ======================================================================
def make_insights(df):
    out = []
    by_contract = df.groupby("Contract", observed=True)["churn_probability"].mean()
    if len(by_contract) > 1:
        out.append(("📄", f"<b>{by_contract.idxmax()}</b> customers are the riskiest contract group at "
                          f"<b>{by_contract.max():.0%}</b> average risk, against {by_contract.min():.0%} for {by_contract.idxmin()}."))
    losses = df["exp_loss_usd"].sort_values(ascending=False)
    if losses.sum() > 0:
        k = max(int(len(losses) * 0.1), 1)
        out.append(("🎯", f"The top 10% of customers carry <b>{losses.head(k).sum() / losses.sum():.0%}</b> "
                          "of the revenue at risk, so a focused campaign covers most of the exposure."))
    by_pay = df.groupby("PaymentMethod", observed=True)["churn_probability"].mean()
    if len(by_pay) > 1:
        out.append(("💳", f"Customers paying by <b>{by_pay.idxmax().lower()}</b> churn most "
                          f"(<b>{by_pay.max():.0%}</b>), while {by_pay.idxmin().lower()} users sit at {by_pay.min():.0%}."))
    if "City" in df.columns and df["City"].nunique() > 1:
        by_city = df.groupby("City")["exp_loss_usd"].sum().sort_values(ascending=False)
        out.append(("📍", f"<b>{by_city.index[0]}</b> has the largest revenue at risk "
                          f"({ui.money(by_city.iloc[0], True)}), followed by {by_city.index[1]}."))
    else:
        new = df[df["tenure"] <= 6]
        if len(new):
            out.append(("🆕", f"Customers in their first 6 months make up {len(new) / len(df):.0%} of the base "
                              f"with an average risk of <b>{new['churn_probability'].mean():.0%}</b>."))
    return out[:4]


def tab_overview(df):
    if df.empty:
        st.warning("No customers match the current filters.")
        return

    k = st.columns(6)
    ui.kpi(k[0], "Customers", f"{len(df):,}", f"of {len(base):,} in portfolio")
    ui.kpi(k[1], "Average churn risk", f"{df['churn_probability'].mean():.1%}", "mean model probability", COLORS["violet"])
    ui.kpi(k[2], "High-risk accounts", f"{int(df['flagged'].sum()):,}", f"{df['flagged'].mean():.1%} of customers", COLORS["high"])
    ui.kpi(k[3], "Monthly revenue", ui.money(df["MonthlyCharges"].sum(), True), "recurring, in view", COLORS["low"])
    ui.kpi(k[4], "Revenue in high-risk", ui.money(df.loc[df["flagged"], "MonthlyCharges"].sum(), True), "monthly, flagged accounts", COLORS["mid"])
    ui.kpi(k[5], "Revenue at risk", ui.money(df["exp_loss_usd"].sum(), True), f"expected over {horizon} months", COLORS["high"])
    st.markdown("<br>", unsafe_allow_html=True)

    notes = make_insights(df)
    if notes:
        cols = st.columns(len(notes))
        for c, (icon, text) in zip(cols, notes):
            ui.insight(c, icon, text)
    st.markdown("<br>", unsafe_allow_html=True)

    a, b = st.columns([1.6, 1])
    with a:
        fig = px.histogram(df, x="churn_probability", nbins=40, color="tier",
                           color_discrete_map={"Stable": COLORS["low"], "Watchlist": COLORS["mid"], "High risk": COLORS["high"]},
                           category_orders={"tier": cfg.TIERS})
        fig.add_vline(x=threshold, line_dash="dash", line_color="white", annotation_text="t*")
        fig.update_xaxes(title="Predicted churn probability", tickformat=".0%")
        fig.update_yaxes(title="Customers")
        ui.show(fig, 350, "How risky is the base?")
    with b:
        cnt = df["tier"].value_counts().reindex(cfg.TIERS).fillna(0)
        fig = go.Figure(go.Pie(labels=cnt.index.astype(str), values=cnt.values, hole=0.6,
                               marker_colors=[COLORS["low"], COLORS["mid"], COLORS["high"]], sort=False))
        fig.add_annotation(text=f"<b>{len(df):,}</b><br>customers", showarrow=False, font=dict(size=15))
        ui.show(fig, 350, "Risk tiers")

    c, d = st.columns(2)
    with c:
        srt = df.sort_values("exp_loss_usd", ascending=False)
        total = srt["exp_loss_usd"].sum()
        if total > 0:
            cum = srt["exp_loss_usd"].cumsum().values / total
            share = np.arange(1, len(srt) + 1) / len(srt)
            n80 = share[np.searchsorted(cum, 0.8)] if cum[-1] >= 0.8 else 1.0
            fig = go.Figure(go.Scatter(x=share, y=cum, mode="lines", fill="tozeroy",
                                       line=dict(color=COLORS["accent"], width=3)))
            fig.add_hline(y=0.8, line_dash="dot", line_color=COLORS["muted"])
            fig.add_vline(x=n80, line_dash="dot", line_color=COLORS["muted"])
            fig.update_xaxes(title="Share of customers (highest exposure first)", tickformat=".0%")
            fig.update_yaxes(title="Share of revenue at risk", tickformat=".0%")
            ui.show(fig, 360, f"Concentration: 80% of the exposure sits in the top {n80:.0%} of customers")
    with d:
        t = df.assign(loss_pos=df["exp_loss"].clip(lower=0.01))
        fig = px.treemap(t, path=["Contract", "InternetService"], values="loss_pos",
                         color="churn_probability", color_continuous_scale="RdYlGn_r", range_color=(0, 1))
        fig.update_traces(hovertemplate="%{label}<br>Revenue at risk: %{value:,.0f}<extra></extra>")
        fig.update_coloraxes(colorbar_title="Risk")
        ui.show(fig, 360, f"Where the revenue at risk lives ({cur})")

    st.markdown("---")
    if "City" in df.columns:
        g = df.groupby("City").agg(customers=("churn_probability", "size"), avg_risk=("churn_probability", "mean"),
                                   loss=("exp_loss", "sum")).reset_index()
        g["lat"] = g["City"].map(lambda c: cfg.CITIES.get(c, (np.nan, np.nan, 0))[0])
        g["lon"] = g["City"].map(lambda c: cfg.CITIES.get(c, (np.nan, np.nan, 0))[1])
        m, r = st.columns([1.3, 1])
        geo = g.dropna(subset=["lat", "lon"])
        with m:
            if len(geo):
                fig = px.scatter_geo(geo, lat="lat", lon="lon", size="customers", color="avg_risk",
                                     hover_name="City", hover_data={"lat": False, "lon": False, "customers": True,
                                                                    "avg_risk": ":.1%"},
                                     color_continuous_scale="RdYlGn_r", size_max=42, scope="asia")
                fig.update_geos(fitbounds="locations", showcountries=True, countrycolor="#666", showland=True,
                                landcolor="rgba(120,120,120,0.18)", bgcolor="rgba(0,0,0,0)", showocean=False,
                                showframe=False)
                fig.update_coloraxes(colorbar_title="Avg risk")
                ui.show(fig, 430, "Churn risk across Pakistan")
            else:
                st.info("None of the city names in your data matched the built-in coordinates.")
        with r:
            top = g.sort_values("loss", ascending=True).tail(12)
            fig = go.Figure(go.Bar(x=top["loss"], y=top["City"], orientation="h", marker_color=COLORS["mid"],
                                   text=[ui.money(v / scale, True) for v in top["loss"]], textposition="outside"))
            fig.update_xaxes(title=f"Revenue at risk ({cur})")
            ui.show(fig, 430, "Revenue at risk by city")
    else:
        g = df.groupby("tenure_band", observed=True)["churn_probability"].mean().reset_index()
        fig = px.line(g, x="tenure_band", y="churn_probability", markers=True)
        fig.update_yaxes(tickformat=".0%", title="Average risk")
        ui.show(fig, 340, "Risk by tenure band (add a City column to your data to unlock the map)")


with tabs[0]:
    tab_overview(view)


# ======================================================================
# tab 2 - single customer
# ======================================================================
def tab_customer():
    k = st.columns(4)
    ui.kpi(k[0], "Churn probability", f"{p_churn:.1%}", f"threshold {threshold:.0%}", COLORS["accent"])
    ui.kpi(k[1], "Risk tier", tier_label, "decision band", tier_color)
    ui.kpi(k[2], "Expected loss", ui.money(cust_loss_usd), f"over {horizon} months", COLORS["mid"])
    ui.kpi(k[3], "Net value of a retention call", ui.money(cust_net_usd), "p × success × value − cost",
           COLORS["low"] if cust_net_usd > 0 else COLORS["high"])
    st.markdown("<br>", unsafe_allow_html=True)

    g1, g2 = st.columns([1, 1.25])
    with g1:
        gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta", value=p_churn * 100,
            number={"suffix": "%", "font": {"size": 42}},
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
        ui.show(gauge, 320, "Churn risk")
        health = int(round((1 - p_churn) * 100))
        st.progress(health / 100, text=f"Customer health score: {health}/100")
    with g2:
        attr = engine.attributions(cust_json)
        top = attr.tail(6).iloc[::-1]
        fig = go.Figure(go.Bar(
            x=top["Contribution"] * 100, y=top["Factor"], orientation="h",
            marker_color=[COLORS["high"] if v > 0 else COLORS["low"] for v in top["Contribution"]],
            text=[f"{v * 100:+.1f} pts" for v in top["Contribution"]], textposition="outside"))
        fig.update_yaxes(autorange="reversed")
        fig.update_xaxes(title="Effect on churn probability (pts)", zeroline=True)
        ui.show(fig, 360, "What is driving this customer's risk")

    ui.section("🔮 What-if simulator",
               "Each row applies one realistic retention action to this customer. The last row combines everything that doesn't conflict.")
    levers = engine.retention_levers(customer)
    if not levers:
        st.success("This customer is already on the most retentive setup, nothing left to try.")
    else:
        rows = [engine.fix_dependencies({**customer, **change}) for _, change in levers]
        rows.append(engine.fix_dependencies(engine.combined_plan(customer, levers)))
        p_new = engine.predict(pd.DataFrame(rows))
        res = pd.DataFrame({
            "Action": [n for n, _ in levers] + ["⭐ Combined plan"],
            "New risk": p_new,
            "Change (pts)": (p_new - p_churn) * 100,
            "Net value": ((p_churn - p_new) * cust_value_usd - cost_usd) * scale,
        }).sort_values("Change (pts)").reset_index(drop=True)

        best = res.iloc[0]
        if best["Change (pts)"] < 0:
            st.markdown(
                f"<div class='action-card'>✅ <b>Best next action: {best['Action']}</b><br>"
                f"Cuts churn risk from {p_churn:.1%} to {best['New risk']:.1%} "
                f"({best['Change (pts)']:+.1f} pts), worth about {pre}{best['Net value']:,.0f} after the cost of the action.</div>",
                unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

        w1, w2 = st.columns([1.2, 1])
        with w1:
            ui.table(res, column_config={
                "New risk": st.column_config.ProgressColumn("New risk", format="%.1f%%", min_value=0, max_value=1),
                "Change (pts)": st.column_config.NumberColumn(format="%+.1f"),
                "Net value": st.column_config.NumberColumn(f"Net value ({cur})", format=f"{pre}%.0f"),
            })
        with w2:
            fig = go.Figure(go.Bar(x=res["Change (pts)"], y=res["Action"], orientation="h",
                                   marker_color=[COLORS["low"] if v < 0 else COLORS["high"] for v in res["Change (pts)"]]))
            fig.update_yaxes(autorange="reversed")
            fig.update_xaxes(title="Change in churn probability (pts)")
            ui.show(fig, 360)

    st.divider()
    c1, c2, _ = st.columns([1, 1, 3])
    if c1.button("📌 Save snapshot"):
        st.session_state.setdefault("history", []).append({
            "Time": datetime.now().strftime("%H:%M:%S"), "Tenure": tenure, "Contract": contract,
            f"Bill ({cur})": round(bill_in), "Internet": internet, "Payment": payment,
            "P(churn)": p_churn, "Tier": tier_label,
        })
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"), "profile": customer,
        "churn_probability": round(p_churn, 4), "tier": tier_label, "threshold": threshold,
        "recommended_action": tier_action, "currency": cur, "fx_rate": fx if is_pkr else None,
        "expected_loss": round(cust_loss_usd * scale), "net_value_of_call": round(cust_net_usd * scale),
    }
    c2.download_button("⬇️ JSON report", json.dumps(report, indent=2), "customer_report.json", "application/json")

    hist = st.session_state.get("history", [])
    if hist:
        h = pd.DataFrame(hist)
        st.markdown("**Saved snapshots**")
        ui.table(h, column_config={
            "P(churn)": st.column_config.ProgressColumn("P(churn)", format="%.1f%%", min_value=0, max_value=1)})
        if len(h) > 1:
            fig = px.line(h, x=h.index + 1, y="P(churn)", markers=True)
            fig.add_hline(y=threshold, line_dash="dash", line_color=COLORS["high"], annotation_text="t*")
            fig.update_xaxes(title="Snapshot #")
            fig.update_yaxes(tickformat=".0%")
            ui.show(fig, 260, "Risk across saved snapshots")


with tabs[1]:
    tab_customer()


# ======================================================================
# tab 3 - explainability
# ======================================================================
def tab_explain():
    ui.section("Why this customer?",
               "Each group of features is swapped for a typical customer's value and we measure how much the risk moves. "
               "It is model-agnostic and similar in spirit to SHAP, but not the same thing.")
    attr = engine.attributions(cust_json)
    ref_p = engine.reference_probability()

    a, b = st.columns([1.3, 1])
    with a:
        fig = go.Figure(go.Bar(
            x=attr["Contribution"] * 100, y=attr["Factor"], orientation="h",
            marker_color=[COLORS["high"] if v > 0 else COLORS["low"] for v in attr["Contribution"]],
            text=[f"{v * 100:+.1f}" for v in attr["Contribution"]], textposition="outside"))
        fig.update_xaxes(title="Contribution (percentage points)", zeroline=True)
        ui.show(fig, 460, "All factors")
    with b:
        wf = attr.sort_values("Contribution", key=np.abs, ascending=False)
        fig = go.Figure(go.Waterfall(
            orientation="v", measure=["absolute"] + ["relative"] * len(wf) + ["total"],
            x=["Typical customer"] + wf["Factor"].tolist() + ["This customer"],
            y=[ref_p * 100] + (wf["Contribution"] * 100).tolist() + [0],
            connector={"line": {"color": COLORS["muted"]}},
            increasing={"marker": {"color": COLORS["high"]}}, decreasing={"marker": {"color": COLORS["low"]}},
            totals={"marker": {"color": COLORS["accent"]}}))
        fig.update_xaxes(tickangle=-40)
        fig.update_yaxes(title="Churn probability (%)")
        ui.show(fig, 460, "From a typical customer to this one")
        st.caption("Indicative only: when features interact the pieces don't add up exactly.")

    ui.section("What does the model rely on overall?")
    fi = engine.feature_importance()
    if fi is None:
        st.info("Global importances need `feature_importances_` or `coef_` on the final estimator.")
    else:
        top = fi.head(20).iloc[::-1]
        fig = go.Figure(go.Bar(x=top["Importance"], y=top["Feature"], orientation="h", marker_color=COLORS["accent"]))
        fig.update_xaxes(title="Importance")
        ui.show(fig, 520, "Top 20 features")

    ui.section("This customer vs. a typical one")
    cmp = pd.DataFrame({
        "Field": cfg.RAW_COLS,
        "This customer": [customer[c] for c in cfg.RAW_COLS],
        "Typical customer": [cfg.REFERENCE[c] for c in cfg.RAW_COLS],
    }).astype(str)
    ui.table(cmp)


with tabs[2]:
    tab_explain()


# ======================================================================
# tab 4 - sensitivity lab
# ======================================================================
def tab_sensitivity():
    st.markdown("<p class='note'>Every chart changes one or two inputs and keeps the rest of the sidebar profile fixed. "
                "The white marker is the current customer.</p>", unsafe_allow_html=True)
    s1, s2 = st.columns(2)
    with s1:
        d = engine.curve_tenure(cust_json)
        fig = px.line(d, x="tenure", y="p", color="Contract",
                      color_discrete_sequence=[COLORS["high"], COLORS["mid"], COLORS["low"]])
        fig.add_vline(x=tenure, line_dash="dash", line_color="white")
        fig.add_hline(y=threshold, line_dash="dot", line_color=COLORS["muted"], annotation_text="t*")
        fig.update_yaxes(tickformat=".0%", title="Churn probability")
        fig.update_xaxes(title="Tenure (months)")
        ui.show(fig, 380, "Risk vs. tenure, by contract")
    with s2:
        d = engine.curve_price(cust_json)
        d["bill"] = d["usd"] * scale
        fig = px.line(d, x="bill", y="p", color="Internet",
                      color_discrete_sequence=[COLORS["high"], COLORS["accent"], COLORS["low"]])
        fig.add_vline(x=bill_in, line_dash="dash", line_color="white")
        fig.add_hline(y=threshold, line_dash="dot", line_color=COLORS["muted"], annotation_text="t*")
        fig.update_yaxes(tickformat=".0%", title="Churn probability")
        fig.update_xaxes(title=f"Monthly bill ({cur})")
        ui.show(fig, 380, "Risk vs. bill, by internet type")

    grid = engine.risk_surface(cust_json)
    fig = go.Figure(go.Heatmap(
        z=grid.values * 100, x=grid.columns, y=grid.index * scale, colorscale="RdYlGn_r", zmin=0, zmax=100,
        colorbar=dict(title="Risk %"), hovertemplate="Tenure %{x} m<br>Bill %{y:,.0f}<br>Risk %{z:.1f}%<extra></extra>"))
    fig.add_trace(go.Scatter(x=[tenure], y=[bill_in], mode="markers", name="Current",
                             marker=dict(size=15, color="white", symbol="x", line=dict(width=2))))
    fig.update_xaxes(title="Tenure (months)")
    fig.update_yaxes(title=f"Monthly bill ({cur})")
    ui.show(fig, 440, "Risk surface: tenure × bill")

    cp = engine.contract_payment_grid(customer)
    fig = px.bar(cp, x="_p", y="p", color="_c", barmode="group",
                 labels={"_p": "Payment method", "p": "Churn probability", "_c": "Contract"},
                 color_discrete_sequence=[COLORS["high"], COLORS["mid"], COLORS["low"]])
    fig.update_yaxes(tickformat=".0%")
    ui.show(fig, 360, "Contract × payment method")


with tabs[3]:
    tab_sensitivity()


# ======================================================================
# tab 5 - segments
# ======================================================================
SEGMENTS = {
    "Contract": "Contract", "Internet service": "InternetService", "Payment method": "PaymentMethod",
    "Tenure band": "tenure_band", "Bill band": "bill_band", "Paperless billing": "PaperlessBilling",
    "Senior status": "Senior", "Gender": "gender", "Risk tier": "tier",
}
if "City" in base.columns:
    SEGMENTS = {"City": "City", **SEGMENTS}


def tab_segments(df):
    if df.empty:
        st.warning("No customers match the current filters.")
        return
    seg_name = st.selectbox("Break the portfolio down by", list(SEGMENTS), key="seg_main")
    col = SEGMENTS[seg_name]

    g = df.groupby(col, observed=True).agg(
        customers=("churn_probability", "size"), avg_risk=("churn_probability", "mean"),
        loss=("exp_loss", "sum")).reset_index()
    g[col] = g[col].astype(str)

    a, b = st.columns(2)
    with a:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=g[col], y=g["loss"], name=f"Revenue at risk ({cur})", marker_color=COLORS["accent"]),
                      secondary_y=False)
        fig.add_trace(go.Scatter(x=g[col], y=g["avg_risk"], name="Average risk", mode="lines+markers",
                                 line=dict(color=COLORS["high"], width=3)), secondary_y=True)
        fig.update_yaxes(title_text=f"Revenue at risk ({cur})", secondary_y=False)
        fig.update_yaxes(title_text="Average risk", tickformat=".0%", secondary_y=True)
        ui.show(fig, 380, f"Exposure and risk by {seg_name.lower()}")
    with b:
        order = g.sort_values("avg_risk", ascending=False)[col].tolist()
        t = df.assign(_seg=df[col].astype(str))
        fig = px.box(t, x="_seg", y="churn_probability", color="_seg", points=False,
                     category_orders={"_seg": order})
        fig.update_layout(showlegend=False)
        fig.update_xaxes(title="")
        fig.update_yaxes(tickformat=".0%", title="Churn probability")
        ui.show(fig, 380, f"Risk spread within each {seg_name.lower()}")

    ui.section("Cross-segment heatmap", "Pick two dimensions to see where risk concentrates. Cells show average risk and customer count.")
    h1, h2 = st.columns(2)
    row_name = h1.selectbox("Rows", list(SEGMENTS), index=0, key="seg_row")
    col_name = h2.selectbox("Columns", list(SEGMENTS), index=min(2, len(SEGMENTS) - 1), key="seg_col")
    if row_name == col_name:
        st.info("Choose two different dimensions.")
    else:
        t = df.assign(_r=df[SEGMENTS[row_name]].astype(str), _c=df[SEGMENTS[col_name]].astype(str))
        risk = t.pivot_table(index="_r", columns="_c", values="churn_probability", aggfunc="mean")
        cnt = t.pivot_table(index="_r", columns="_c", values="churn_probability", aggfunc="size").reindex_like(risk)
        text = np.where(risk.notna(), (risk * 100).round(0).astype("Int64").astype(str) + "%<br>n=" + cnt.fillna(0).astype(int).astype(str), "")
        fig = go.Figure(go.Heatmap(z=risk.values * 100, x=risk.columns, y=risk.index, text=text, texttemplate="%{text}",
                                   colorscale="RdYlGn_r", zmin=0, zmax=100, colorbar=dict(title="Risk %")))
        ui.show(fig, 420, f"{row_name} × {col_name}")

    c, d = st.columns(2)
    with c:
        t = df.assign(loss_pos=df["exp_loss"].clip(lower=0.01), tier_s=df["tier"].astype(str))
        fig = px.sunburst(t, path=["Contract", "InternetService", "tier_s"], values="loss_pos",
                          color="churn_probability", color_continuous_scale="RdYlGn_r", range_color=(0, 1))
        ui.show(fig, 430, "Contract → internet → tier (sized by revenue at risk)")
    with d:
        samp = df.sample(min(len(df), 2500), random_state=3)
        fig = px.scatter(samp, x="tenure", y="bill", color="churn_probability", color_continuous_scale="RdYlGn_r",
                         range_color=(0, 1), opacity=0.7)
        fig.update_xaxes(title="Tenure (months)")
        fig.update_yaxes(title=f"Monthly bill ({cur})")
        ui.show(fig, 430, "Every customer: tenure vs. bill")

    ui.section("Hot-spot segments", "Combinations of contract, internet service and payment method with the highest average risk.")
    min_n = st.slider("Minimum customers per segment", 1, 100, 10)
    hs = df.groupby(["Contract", "InternetService", "PaymentMethod"], observed=True).agg(
        customers=("churn_probability", "size"), avg_risk=("churn_probability", "mean"),
        revenue_at_risk=("exp_loss", "sum")).reset_index()
    hs = hs[hs["customers"] >= min_n].sort_values("avg_risk", ascending=False).head(12)
    if hs.empty:
        st.info("No segment is big enough, lower the minimum.")
    else:
        ui.table(hs, column_config={
            "avg_risk": st.column_config.ProgressColumn("Average risk", format="%.1f%%", min_value=0, max_value=1),
            "revenue_at_risk": st.column_config.NumberColumn(f"Revenue at risk ({cur})", format=f"{pre}%.0f"),
        })


with tabs[4]:
    tab_segments(view)


# ======================================================================
# tab 6 - campaign planner
# ======================================================================
def tab_campaign(df):
    if df.empty:
        st.warning("No customers match the current filters.")
        return
    ui.section("Plan a retention campaign",
               "Set a budget and capacity, pick how to rank customers, and see what the campaign is expected to return. "
               "Cost per action and success rate come from the sidebar.")

    p1, p2, p3 = st.columns(3)
    budget_in = p1.number_input(f"Campaign budget ({cur})", 0.0, 5e9 if is_pkr else 2e7,
                                500000.0 if is_pkr else 1800.0, 50000.0 if is_pkr else 500.0, key=f"budget_{cur}")
    capacity = p2.slider("Max customers the team can contact", 10, 2000, 300, 10)
    strategy = p3.selectbox("Rank customers by", ["Highest net benefit", "Highest churn risk", "Highest revenue at risk"])
    only_positive = st.checkbox("Skip customers whose expected return is below the cost of contacting them", value=True)

    sort_col = {"Highest net benefit": "net_usd", "Highest churn risk": "churn_probability",
                "Highest revenue at risk": "exp_loss_usd"}[strategy]
    ranked = df.sort_values(sort_col, ascending=False)
    if only_positive:
        ranked = ranked[ranked["net_usd"] > 0]

    budget_usd = budget_in / scale
    by_budget = int(budget_usd // cost_usd) if cost_usd > 0 else len(ranked)
    n_plan = max(min(by_budget, capacity, len(ranked)), 0)
    chosen = ranked.head(n_plan)

    spend = n_plan * cost_usd
    saved = float((chosen["churn_probability"] * success * chosen["value_usd"]).sum())
    net = saved - spend
    roi = net / spend if spend > 0 else 0.0
    coverage = chosen["churn_probability"].sum() / max(df["churn_probability"].sum(), 1e-9)

    k = st.columns(5)
    ui.kpi(k[0], "Customers contacted", f"{n_plan:,}", "limited by budget / capacity" if n_plan in (by_budget, capacity) else "all eligible")
    ui.kpi(k[1], "Campaign spend", ui.money(spend, True), f"{ui.money(cost_usd)} per contact", COLORS["mid"])
    ui.kpi(k[2], "Expected revenue saved", ui.money(saved, True), f"{success:.0%} success rate", COLORS["low"])
    ui.kpi(k[3], "Net benefit", ui.money(net, True), f"ROI {roi:+.0%}", COLORS["low"] if net >= 0 else COLORS["high"])
    ui.kpi(k[4], "Churn exposure covered", f"{coverage:.0%}", "of expected churners in view", COLORS["violet"])
    st.markdown("<br>", unsafe_allow_html=True)

    a, b = st.columns(2)
    with a:
        stages = ["Customers in view", "Flagged high-risk", "Selected for outreach", "Expected to be retained"]
        vals = [len(df), int(df["flagged"].sum()), n_plan, float((chosen["churn_probability"] * success).sum())]
        fig = go.Figure(go.Funnel(y=stages, x=[round(v, 1) for v in vals], textinfo="value+percent initial",
                                  marker=dict(color=[COLORS["accent"], COLORS["high"], COLORS["mid"], COLORS["low"]])))
        ui.show(fig, 380, "Campaign funnel")
    with b:
        cum = (ranked["net_usd"].cumsum() * scale).values
        x = np.arange(1, len(cum) + 1)
        fig = go.Figure(go.Scatter(x=x, y=cum, mode="lines", line=dict(color=COLORS["low"], width=3), fill="tozeroy"))
        if n_plan > 0 and len(cum):
            fig.add_trace(go.Scatter(x=[n_plan], y=[cum[min(n_plan, len(cum)) - 1]], mode="markers", name="Plan",
                                     marker=dict(size=13, color="white", line=dict(width=2))))
        fig.update_xaxes(title="Customers contacted (in ranked order)")
        fig.update_yaxes(title=f"Cumulative net benefit ({cur})")
        ui.show(fig, 380, "Diminishing returns along the ranked list")

    ui.section("Threshold economics", "What happens to total net benefit if the whole high-risk list is contacted at different thresholds.")
    p = df["churn_probability"].values
    value = df["value_usd"].values
    ts = np.linspace(0.02, 0.98, 49)
    profits, sizes = [], []
    for t in ts:
        sel = p >= t
        profits.append(float((p[sel] * success * value[sel] - cost_usd).sum()) * scale)
        sizes.append(int(sel.sum()))
    best = int(np.argmax(profits))
    cur_sel = p >= threshold
    cur_profit = float((p[cur_sel] * success * value[cur_sel] - cost_usd).sum()) * scale
    ceiling = float(np.clip(p * success * value - cost_usd, 0, None).sum()) * scale

    e = st.columns(4)
    ui.kpi(e[0], "Best threshold", f"{ts[best]:.2f}", f"your t* is {threshold:.2f}", COLORS["accent"])
    ui.kpi(e[1], "Net benefit at best t*", ui.money(profits[best] / scale, True), f"{sizes[best]:,} contacted", COLORS["low"])
    ui.kpi(e[2], "Net benefit at your t*", ui.money(cur_profit / scale, True), f"{int(cur_sel.sum()):,} contacted",
           COLORS["low"] if cur_profit >= 0 else COLORS["high"])
    ui.kpi(e[3], "Theoretical ceiling", ui.money(ceiling / scale, True), "contact only positive-return customers", COLORS["violet"])
    st.markdown("<br>", unsafe_allow_html=True)

    c, d = st.columns(2)
    with c:
        fig = go.Figure(go.Scatter(x=ts, y=profits, mode="lines", fill="tozeroy", line=dict(color=COLORS["low"], width=3)))
        fig.add_vline(x=ts[best], line_dash="dash", line_color=COLORS["accent"], annotation_text="best")
        fig.add_vline(x=threshold, line_dash="dash", line_color="white", annotation_text="yours")
        fig.update_xaxes(title="Decision threshold")
        fig.update_yaxes(title=f"Net benefit ({cur})")
        ui.show(fig, 340, "Net benefit vs. threshold")
    with d:
        costs_usd = np.array([2, 5, 10, 20, 40], dtype=float)
        succ = np.arange(0.1, 0.71, 0.1)
        z = np.array([[np.clip(p * s * value - c, 0, None).sum() * scale for s in succ] for c in costs_usd])
        fig = go.Figure(go.Heatmap(z=z, x=[f"{s:.0%}" for s in succ], y=[ui.money(c) for c in costs_usd],
                                   colorscale="Viridis", colorbar=dict(title=cur),
                                   hovertemplate="Success %{x}<br>Cost %{y}<br>%{z:,.0f}<extra></extra>"))
        fig.update_xaxes(title="Offer success rate")
        fig.update_yaxes(title="Cost per contact")
        ui.show(fig, 340, "How sensitive is the result to your assumptions?")

    ui.section("Call list", "Rule-based talking points. The model decides who to call, these suggestions only help the agent start the conversation.")
    if chosen.empty:
        st.info("Nothing selected with the current budget / capacity / filters.")
        return
    lst = chosen.copy()
    lst.insert(0, "Priority", np.arange(1, len(lst) + 1))
    lst["Suggested approach"] = lst.apply(engine.suggest_action, axis=1)
    lst["Net benefit"] = lst["net_usd"] * scale
    id_col = ["customerID"] if "customerID" in lst.columns else []
    city_col = ["City"] if "City" in lst.columns else []
    show_cols = ["Priority"] + id_col + city_col + ["tenure", "Contract", "bill", "churn_probability",
                                                    "exp_loss", "Net benefit", "Suggested approach"]
    out = lst[show_cols].rename(columns={"bill": f"Bill ({cur})", "exp_loss": f"Revenue at risk ({cur})",
                                         "Net benefit": f"Net benefit ({cur})"})
    ui.table(out.head(100), column_config={
        "churn_probability": st.column_config.ProgressColumn("P(churn)", format="%.1f%%", min_value=0, max_value=1),
        f"Bill ({cur})": st.column_config.NumberColumn(format=f"{pre}%.0f"),
        f"Revenue at risk ({cur})": st.column_config.NumberColumn(format=f"{pre}%.0f"),
        f"Net benefit ({cur})": st.column_config.NumberColumn(format=f"{pre}%.0f"),
    })
    st.download_button("⬇️ Download full call list (CSV)", out.to_csv(index=False), "call_list.csv", "text/csv")


with tabs[5]:
    tab_campaign(view)


# ======================================================================
# tab 7 - data & evaluation
# ======================================================================
def slice_report(d, y, col):
    rows = []
    for name, sub in d.groupby(col, observed=True):
        ys = y.loc[sub.index]
        flagged = sub["flagged"]
        tp = int((flagged & (ys == 1)).sum())
        fp = int((flagged & (ys == 0)).sum())
        fn = int((~flagged & (ys == 1)).sum())
        rows.append({
            col: str(name), "customers": len(sub), "actual churn": ys.mean(),
            "predicted risk": sub["churn_probability"].mean(), "flagged": flagged.mean(),
            "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
            "AUC": roc_auc_score(ys, sub["churn_probability"]) if ys.nunique() == 2 else np.nan,
        })
    return pd.DataFrame(rows)


def evaluation_section(d):
    labels = engine.parse_labels(d["Churn"])
    ok = labels.notna()
    d = d[ok]
    y = labels[ok].astype(int)
    if len(d) < 20 or y.nunique() < 2:
        st.warning("Need at least 20 rows with both churn and non-churn labels to evaluate.")
        return
    yv, pv = y.values, d["churn_probability"].values
    pred = (pv >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(yv, pred, labels=[0, 1]).ravel()
    prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)

    ui.section("📈 Model evaluation", "Calculated on the customers currently in view, so you can also evaluate a single segment by filtering.")
    m = st.columns(6)
    ui.kpi(m[0], "ROC-AUC", f"{roc_auc_score(yv, pv):.3f}")
    ui.kpi(m[1], "PR-AUC", f"{average_precision_score(yv, pv):.3f}", f"base rate {yv.mean():.1%}", COLORS["violet"])
    ui.kpi(m[2], "Precision", f"{prec:.1%}", f"at t* = {threshold:.2f}", COLORS["mid"])
    ui.kpi(m[3], "Recall", f"{rec:.1%}", "churners caught", COLORS["low"])
    ui.kpi(m[4], "F1", f"{f1:.3f}")
    ui.kpi(m[5], "Brier score", f"{brier_score_loss(yv, pv):.3f}", "lower is better", COLORS["high"])
    st.markdown("<br>", unsafe_allow_html=True)

    v1, v2, v3 = st.columns(3)
    with v1:
        fpr, tpr, _ = roc_curve(yv, pv)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, name="Model", line=dict(color=COLORS["accent"], width=3)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Chance", line=dict(dash="dash", color=COLORS["muted"])))
        fig.update_xaxes(title="False positive rate")
        fig.update_yaxes(title="True positive rate")
        ui.show(fig, 340, "ROC curve")
    with v2:
        pr, rc, _ = precision_recall_curve(yv, pv)
        fig = go.Figure(go.Scatter(x=rc, y=pr, line=dict(color=COLORS["mid"], width=3)))
        fig.add_hline(y=yv.mean(), line_dash="dash", line_color=COLORS["muted"], annotation_text="base rate")
        fig.update_xaxes(title="Recall")
        fig.update_yaxes(title="Precision")
        ui.show(fig, 340, "Precision-recall")
    with v3:
        cm = np.array([[tn, fp], [fn, tp]])
        fig = go.Figure(go.Heatmap(z=cm, x=["Predicted stay", "Predicted churn"], y=["Actually stayed", "Actually churned"],
                                   text=cm, texttemplate="%{text}", colorscale="Blues", showscale=False))
        fig.update_yaxes(autorange="reversed")
        ui.show(fig, 340, "Confusion matrix")

    w1, w2, w3 = st.columns(3)
    with w1:
        frac, mean_p = calibration_curve(yv, pv, n_bins=8 if len(yv) >= 400 else 5, strategy="quantile")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=mean_p, y=frac, mode="lines+markers", name="Model"))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Perfect", line=dict(dash="dash", color=COLORS["muted"])))
        fig.update_xaxes(title="Predicted probability")
        fig.update_yaxes(title="Observed churn rate")
        ui.show(fig, 340, "Calibration")
    gt = engine.gain_table(yv, pv)
    step = max(len(gt) // 120, 1)
    with w2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=gt["share"][::step], y=gt["gain"][::step], name="Model", line=dict(color=COLORS["low"], width=3)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Random", line=dict(dash="dash", color=COLORS["muted"])))
        fig.update_xaxes(title="Share of customers contacted", tickformat=".0%")
        fig.update_yaxes(title="Share of churners captured", tickformat=".0%")
        ui.show(fig, 340, "Cumulative gain")
    with w3:
        fig = go.Figure(go.Scatter(x=gt["share"][::step], y=gt["lift"][::step], line=dict(color=COLORS["violet"], width=3)))
        fig.add_hline(y=1, line_dash="dash", line_color=COLORS["muted"])
        fig.update_xaxes(title="Share of customers contacted", tickformat=".0%")
        fig.update_yaxes(title="Lift")
        ui.show(fig, 340, "Lift")

    sweep = []
    for t in np.linspace(0.05, 0.95, 19):
        pr_ = pv >= t
        tp_, fp_ = int((pr_ & (yv == 1)).sum()), int((pr_ & (yv == 0)).sum())
        fn_ = int((~pr_ & (yv == 1)).sum())
        p_, r_ = tp_ / max(tp_ + fp_, 1), tp_ / max(tp_ + fn_, 1)
        sweep.append({"threshold": t, "precision": p_, "recall": r_, "f1": 2 * p_ * r_ / max(p_ + r_, 1e-9)})
    fig = px.line(pd.DataFrame(sweep).melt("threshold"), x="threshold", y="value", color="variable", markers=True,
                  color_discrete_sequence=[COLORS["mid"], COLORS["low"], COLORS["violet"]])
    fig.add_vline(x=threshold, line_dash="dash", line_color="white")
    fig.update_yaxes(title="Score")
    ui.show(fig, 340, "Threshold sweep")

    ui.section("Fairness check by group", "Compare error rates across groups before trusting the model on all of them.")
    group_cols = [c for c in ["gender", "Senior", "Contract", "City"] if c in d.columns]
    pick = st.selectbox("Group by", group_cols, key="fair_group")
    rep = slice_report(d, y, pick)
    ui.table(rep, column_config={
        c: st.column_config.NumberColumn(format="%.3f")
        for c in ["actual churn", "predicted risk", "flagged", "precision", "recall", "AUC"]
    })
    if pick in ("gender", "Senior") and len(rep) > 1:
        gap = rep["recall"].max() - rep["recall"].min()
        if gap > 0.1:
            st.warning(f"Recall differs by {gap:.0%} between groups. Worth investigating before operational use.")


def tab_data(d):
    ui.section("Data source")
    s = st.columns(4)
    ui.kpi(s[0], "Rows (full file)", f"{len(base):,}", st.session_state["data_name"][:34])
    ui.kpi(s[1], "Rows in view", f"{len(d):,}", "after filters", COLORS["violet"])
    ui.kpi(s[2], "Columns", f"{base.shape[1]}", "including derived", COLORS["mid"])
    ui.kpi(s[3], "Has churn labels", "Yes" if "Churn" in base.columns else "No", "enables evaluation",
           COLORS["low"] if "Churn" in base.columns else COLORS["muted"])
    st.markdown("<br>", unsafe_allow_html=True)

    for n in st.session_state.get("data_notes", []):
        st.warning(n)

    q = pd.DataFrame({
        "Column": cfg.RAW_COLS,
        "Type": [str(scored[c].dtype) for c in cfg.RAW_COLS],
        "Missing": [int(scored[c].isna().sum()) for c in cfg.RAW_COLS],
        "Unique values": [int(scored[c].nunique()) for c in cfg.RAW_COLS],
    })
    c1, c2 = st.columns([1, 1.4])
    with c1:
        st.markdown("**Schema check**")
        ui.table(q, height=380)
    with c2:
        st.markdown("**Scored data preview**")
        cols = [c for c in ["customerID", "City", "tenure", "Contract", "InternetService", "PaymentMethod", "bill",
                            "churn_probability", "tier"] if c in d.columns]
        prev = d[cols].head(200).rename(columns={"bill": f"Bill ({cur})"})
        ui.table(prev, height=380, column_config={
            "churn_probability": st.column_config.ProgressColumn("P(churn)", format="%.1f%%", min_value=0, max_value=1),
            f"Bill ({cur})": st.column_config.NumberColumn(format=f"{pre}%.0f")})
    if not d.empty:
        out = d.drop(columns=["bill", "exp_loss", "tenure_band", "bill_band", "Senior"], errors="ignore")
        st.download_button("⬇️ Download scored customers in view (CSV)", out.to_csv(index=False),
                           "scored_customers.csv", "text/csv")

    if "Churn" in d.columns and not d.empty:
        st.divider()
        evaluation_section(d)
    elif "Churn" not in base.columns:
        st.info("Upload a CSV with a `Churn` column (Yes/No) to unlock ROC, calibration, lift and fairness analysis. "
                "The original Telco churn dataset works as is.")


with tabs[6]:
    tab_data(view)


# ======================================================================
# tab 8 - model card
# ======================================================================
def tab_card():
    a, b = st.columns(2)
    with a:
        st.markdown("##### Metadata")
        info = {k: v for k, v in META.items() if k != "feature_columns"}
        info["encoded_features"] = len(COLS)
        info["estimator"] = engine.estimator_name()
        st.json(info)
        st.markdown("##### Intended use")
        st.markdown(
            "- Rank customers by short-term churn risk so retention teams can prioritise.\n"
            "- Explore how contract, billing and service changes move the model's prediction.\n"
            "- Not meant for automated decisions that limit a customer's access to service."
        )
    with b:
        st.markdown("##### Limitations")
        st.markdown(
            "- **Currency:** the model was trained on a USD telecom price list. PKR is a display conversion at the rate you set, "
            "so bills in a typical Pakistani tariff may fall outside what the model saw. Retrain on local data before real use.\n"
            "- **Associational, not causal:** the what-if tool shows how the *model* reacts. Test real offers with an A/B experiment.\n"
            "- **Bias:** gender and senior-citizen are model inputs. Use the fairness check in the Data tab and consider dropping them.\n"
            "- **Drift:** monitor calibration regularly and retrain when pricing, competition or customer mix changes.\n"
            "- **City** is only used for filtering and the map, the model never sees it."
        )

    fi = engine.feature_importance()
    if fi is not None:
        fig = px.bar(fi.head(15).iloc[::-1], x="Importance", y="Feature", orientation="h",
                     color_discrete_sequence=[COLORS["accent"]])
        ui.show(fig, 420, "Top 15 features")

    with st.expander("Encoded feature list"):
        st.write(COLS)

    h = st.columns(3)
    ui.kpi(h[0], "Model files", "Loaded", "joblib + metadata", COLORS["low"])
    ui.kpi(h[1], "Last inference", f"{st.session_state.get('latency_ms', 0):.1f} ms", "single call", COLORS["accent"])
    ui.kpi(h[2], "Saved snapshots", str(len(st.session_state.get("history", []))), "this session", COLORS["violet"])


with tabs[7]:
    tab_card()
