# -*- coding: utf-8 -*-
"""
CredResolve Analytics - Machine Learning Pipeline
Trains churn prediction and MRR forecasting models.
Run from credresolve-analytics/ root: python src/ml_models.py
"""
import base64
import json
import os
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import xgboost as xgb
from prophet import Prophet
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED)

# ── Output directories ─────────────────────────────────────────────────────
Path("output/models").mkdir(parents=True, exist_ok=True)
Path("output/reports").mkdir(parents=True, exist_ok=True)
Path("data/processed/analytics_outputs").mkdir(parents=True, exist_ok=True)

A   = "data/processed/analytics_outputs/"
P   = "data/processed/"
SEP = "=" * 65

# ── Shared color palette ───────────────────────────────────────────────────
DARK    = "#1a1a2e"
PRIMARY = "#2E86AB"
SUCCESS = "#28A745"
WARNING = "#FFC107"
DANGER  = "#DC3545"
TEXT    = "#e0e0e0"

_CHART_STYLE = dict(
    paper_bgcolor=DARK, plot_bgcolor="#0f3460",
    font=dict(color=TEXT, size=11),
    margin=dict(l=55, r=20, t=55, b=50),
)


def apply_dark(fig):
    fig.update_layout(**_CHART_STYLE)
    fig.update_layout(
        xaxis=dict(gridcolor="#1a3a5c", linecolor="#2a4a6a", tickfont=dict(color=TEXT)),
        yaxis=dict(gridcolor="#1a3a5c", linecolor="#2a4a6a", tickfont=dict(color=TEXT)),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT)),
    )
    return fig


def save_chart(fig, png_path, width=800, height=450):
    try:
        img_bytes = fig.to_image(format="png", width=width, height=height, scale=2)
        Path(png_path).write_bytes(img_bytes)
        b64 = base64.b64encode(img_bytes).decode("utf-8")
        print(f"  Saved: {png_path}  ({len(img_bytes)//1024} KB)")
        return f"data:image/png;base64,{b64}"
    except Exception as exc:
        print(f"  [WARN] Could not export {png_path}: {exc}")
        return ""


def format_currency(v):
    v = float(v)
    if v >= 1_000_000: return f"${v/1_000_000:.2f}M"
    if v >= 1_000:     return f"${v/1_000:.1f}K"
    return f"${v:.0f}"


# =============================================================================
# SECTION 2 -- Load Data
# =============================================================================
print(f"\n{SEP}\n  SECTION 2 -- Load Data\n{SEP}")

ca     = pd.read_csv(A + "customer_analytics_enriched.csv", parse_dates=["signup_date"])
mrr_ts = pd.read_csv(A + "mrr_timeseries.csv")

with open(A + "kpi_summary.json", encoding="utf-8") as _f:
    kpi = json.load(_f)

print(f"  customer_analytics_enriched: {len(ca):,} rows  {len(ca.columns)} cols")
print(f"  mrr_timeseries:              {len(mrr_ts):,} rows")


# =============================================================================
# SECTION 3 -- Feature Engineering
# =============================================================================
print(f"\n{SEP}\n  SECTION 3 -- Feature Engineering\n{SEP}")

df = ca.copy()

# Target
df["churn_label"] = (df["status"] == "Churned").astype(int)

# Fill missing numeric features
df["total_tickets"]          = df["total_tickets"].fillna(0)
df["avg_satisfaction_score"] = df["avg_satisfaction_score"].fillna(
    df["avg_satisfaction_score"].mean()
)
df["total_usage_count"]      = df["total_usage_count"].fillna(0)
df["features_used"]          = df["distinct_features_used"].fillna(0)
df["days_since_last_access"] = df["days_since_last_access"].fillna(999)
df["engagement_score"]       = df["engagement_score"].fillna(0)
df["churn_risk_score"]       = df["churn_risk_score"].fillna(0)

# Encode categoricals
label_encoders: dict = {}
for src_col, enc_col in [
    ("segment",   "segment_encoded"),
    ("region",    "region_encoded"),
    ("plan_type", "plan_encoded"),
]:
    le = LabelEncoder()
    df[enc_col] = le.fit_transform(df[src_col].fillna("Unknown"))
    label_encoders[src_col] = le

# Derived interaction features
df["revenue_per_day"]    = df["total_revenue"] / (df["tenure_days"] + 1)
df["tickets_per_month"]  = df["total_tickets"]  / (df["tenure_days"] / 30 + 1)
df["usage_per_day"]      = df["total_usage_count"] / (df["tenure_days"] + 1)
df["has_low_engagement"] = (df["engagement_score"] < 40).astype(int)
df["is_basic_plan"]      = (df["plan_type"] == "Basic").astype(int)
df["is_smb"]             = (df["segment"] == "SMB").astype(int)

FEATURES = [
    "tenure_days", "total_revenue", "total_invoices", "total_tickets",
    "avg_satisfaction_score", "total_usage_count", "features_used",
    "days_since_last_access", "engagement_score", "churn_risk_score",
    "segment_encoded", "region_encoded", "plan_encoded",
    "revenue_per_day", "tickets_per_month", "usage_per_day",
    "has_low_engagement", "is_basic_plan", "is_smb",
]

model_df = df[FEATURES + ["churn_label"]].dropna()
n_total   = len(model_df)
n_churned = int(model_df["churn_label"].sum())
n_active  = n_total - n_churned

print(f"  Modeling rows: {n_total:,}  (after dropna)")
print(f"  Class distribution:")
print(f"    Churned (1): {n_churned:,}  ({n_churned/n_total*100:.1f}%)")
print(f"    Active  (0): {n_active:,}  ({n_active/n_total*100:.1f}%)")

X = model_df[FEATURES].values
y = model_df["churn_label"].values


# =============================================================================
# SECTION 4 -- Train/Test Split
# =============================================================================
print(f"\n{SEP}\n  SECTION 4 -- Train/Test Split\n{SEP}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED
)
print(f"  X_train: {X_train.shape}   y_train: {y_train.shape}")
print(f"  X_test:  {X_test.shape}    y_test:  {y_test.shape}")


# =============================================================================
# SECTION 5 -- Train 3 Models & Compare
# =============================================================================
print(f"\n{SEP}\n  SECTION 5 -- Train Models\n{SEP}")

results: dict = {}

# ── Model A: Logistic Regression ─────────────────────────────────────────
print("  [A] Logistic Regression...")
scaler         = StandardScaler()
X_tr_sc        = scaler.fit_transform(X_train)
X_te_sc        = scaler.transform(X_test)
lr             = LogisticRegression(max_iter=1000, random_state=SEED)
lr.fit(X_tr_sc, y_train)
lr_proba       = lr.predict_proba(X_te_sc)[:, 1]
lr_pred        = lr.predict(X_te_sc)
results["Logistic Regression"] = {
    "model": lr, "scaler": scaler,
    "accuracy": float((lr_pred == y_test).mean()),
    "auc":      float(roc_auc_score(y_test, lr_proba)),
    "proba":    lr_proba, "pred": lr_pred,
}
print(f"      Accuracy {results['Logistic Regression']['accuracy']:.3f}  "
      f"AUC-ROC {results['Logistic Regression']['auc']:.3f}")

# ── Model B: Random Forest ────────────────────────────────────────────────
print("  [B] Random Forest...")
rf       = RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1)
rf.fit(X_train, y_train)
rf_proba = rf.predict_proba(X_test)[:, 1]
rf_pred  = rf.predict(X_test)
results["Random Forest"] = {
    "model": rf,
    "accuracy":    float((rf_pred == y_test).mean()),
    "auc":         float(roc_auc_score(y_test, rf_proba)),
    "proba":       rf_proba, "pred": rf_pred,
    "importances": rf.feature_importances_,
}
print(f"      Accuracy {results['Random Forest']['accuracy']:.3f}  "
      f"AUC-ROC {results['Random Forest']['auc']:.3f}")

# ── Model C: XGBoost ─────────────────────────────────────────────────────
print("  [C] XGBoost...")
xgb_clf = xgb.XGBClassifier(
    n_estimators=200, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    random_state=SEED, eval_metric="logloss", verbosity=0,
)
xgb_clf.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
xgb_proba = xgb_clf.predict_proba(X_test)[:, 1]
xgb_pred  = xgb_clf.predict(X_test)
results["XGBoost"] = {
    "model": xgb_clf,
    "accuracy":    float((xgb_pred == y_test).mean()),
    "auc":         float(roc_auc_score(y_test, xgb_proba)),
    "proba":       xgb_proba, "pred": xgb_pred,
    "importances": xgb_clf.feature_importances_,
}
print(f"      Accuracy {results['XGBoost']['accuracy']:.3f}  "
      f"AUC-ROC {results['XGBoost']['auc']:.3f}")

# ── Comparison table ──────────────────────────────────────────────────────
print(f"\n  {'Model':<25} {'Accuracy':>10} {'AUC-ROC':>10}")
print(f"  {'-'*25} {'-'*10} {'-'*10}")
for name, r in results.items():
    print(f"  {name:<25} {r['accuracy']:>10.3f} {r['auc']:>10.3f}")

best_name  = max(results, key=lambda k: results[k]["auc"])
best_r     = results[best_name]
best_model = best_r["model"]
best_proba = best_r["proba"]
best_pred  = best_r["pred"]

print(f"\n  Best model: {best_name}  (AUC-ROC = {best_r['auc']:.3f})")
print(f"\n  Classification report ({best_name}):")
print(classification_report(y_test, best_pred, target_names=["Active", "Churned"]))


# =============================================================================
# SECTION 6 -- Generate Predictions on All Customers
# =============================================================================
print(f"\n{SEP}\n  SECTION 6 -- Predict on All Customers\n{SEP}")

X_full = df[FEATURES].fillna(0).values
if best_name == "Logistic Regression":
    X_full_in = scaler.transform(X_full)
else:
    X_full_in = X_full

all_proba = best_model.predict_proba(X_full_in)[:, 1]
all_pred  = (all_proba >= 0.5).astype(int)

ca_pred = ca.copy()
ca_pred["churn_probability"] = all_proba
ca_pred["churn_prediction"]  = all_pred


def _ml_risk_tier(row) -> str | None:
    if row["status"] != "Active":
        return None
    p = row["churn_probability"]
    if p >= 0.75: return "Critical"
    if p >= 0.50: return "High"
    if p >= 0.30: return "Medium"
    return "Low"


ca_pred["ml_risk_tier"] = ca_pred.apply(_ml_risk_tier, axis=1)

# Save full predictions
ca_pred.to_csv(A + "customer_predictions.csv", index=False)
print(f"  Saved: {A}customer_predictions.csv  ({len(ca_pred):,} rows)")

# Top 50 highest-risk active customers
active_pred = ca_pred[ca_pred["status"] == "Active"].copy()
_top_cols   = ["customer_id", "segment", "region", "plan_type",
               "total_revenue", "churn_probability", "ml_risk_tier",
               "engagement_score", "tenure_days"]
top_50 = (active_pred.sort_values("churn_probability", ascending=False)
          .head(50)[_top_cols])
top_50.to_csv(A + "top_churn_risk.csv", index=False)
print(f"  Saved: {A}top_churn_risk.csv  (top 50 highest-risk active)")

# Risk tier distribution
tier_counts = active_pred["ml_risk_tier"].value_counts().to_dict()
print(f"\n  Active customers by ML risk tier:")
for tier in ["Critical", "High", "Medium", "Low"]:
    cnt = int(tier_counts.get(tier, 0))
    pct = cnt / len(active_pred) * 100 if active_pred.shape[0] > 0 else 0
    print(f"    {tier:<10}: {cnt:>5,}  ({pct:.1f}%)")

print(f"\n  Avg churn probability by segment:")
for seg in ["Enterprise", "Mid-Market", "SMB"]:
    avg = float(active_pred[active_pred["segment"] == seg]["churn_probability"].mean())
    print(f"    {seg:<15}: {avg:.3f}")

print(f"\n  Top 10 highest-risk active customers:")
print(f"  {'Customer ID':<15} {'Segment':<15} {'Churn Prob':>12}")
print(f"  {'-'*15} {'-'*15} {'-'*12}")
for _, row in top_50.head(10).iterrows():
    print(f"  {row['customer_id']:<15} {row['segment']:<15} {row['churn_probability']:>12.3f}")


# =============================================================================
# SECTION 7 -- Feature Importance Chart
# =============================================================================
print(f"\n{SEP}\n  SECTION 7 -- Feature Importance Chart\n{SEP}")

if "importances" in best_r:
    importances = best_r["importances"]
else:
    importances = np.abs(best_model.coef_[0])

imp_df = (pd.DataFrame({"feature": FEATURES, "importance": importances})
          .sort_values("importance", ascending=False))
imp_df.to_csv("output/models/feature_importance.csv", index=False)
print("  Saved: output/models/feature_importance.csv")

imp_sorted = imp_df.sort_values("importance", ascending=True)
fig_imp = go.Figure(go.Bar(
    x=imp_sorted["importance"],
    y=imp_sorted["feature"],
    orientation="h",
    marker=dict(
        color=imp_sorted["importance"],
        colorscale=[[0.0, "#1a3a5c"], [1.0, PRIMARY]],
        showscale=False,
    ),
    hovertemplate="%{y}: %{x:.5f}<extra></extra>",
))
fig_imp.update_layout(
    title=f"Churn Prediction -- Feature Importance ({best_name})",
    xaxis_title="Importance Score",
    height=560,
)
apply_dark(fig_imp)
save_chart(fig_imp, "output/reports/feature_importance.png", width=800, height=560)


# =============================================================================
# SECTION 8 -- ROC Curve
# =============================================================================
print(f"\n{SEP}\n  SECTION 8 -- ROC Curve\n{SEP}")

fpr, tpr, _ = roc_curve(y_test, best_proba)
auc_val     = best_r["auc"]

fig_roc = go.Figure()
fig_roc.add_trace(go.Scatter(
    x=fpr, y=tpr, mode="lines",
    name=f"{best_name} (AUC = {auc_val:.3f})",
    line=dict(color=SUCCESS, width=2.5),
    fill="tozeroy", fillcolor="rgba(40, 167, 69, 0.10)",
))
fig_roc.add_trace(go.Scatter(
    x=[0, 1], y=[0, 1], mode="lines", name="Random Baseline",
    line=dict(color="#556b7a", dash="dash", width=1.5),
))
fig_roc.add_annotation(
    x=0.62, y=0.32,
    text=f"AUC = {auc_val:.3f}",
    showarrow=False,
    font=dict(color=SUCCESS, size=15, family="monospace"),
    bgcolor="rgba(0,0,0,0.55)", bordercolor=SUCCESS, borderwidth=1,
)
fig_roc.update_layout(
    title=f"ROC Curve -- Churn Prediction ({best_name})",
    xaxis_title="False Positive Rate",
    yaxis_title="True Positive Rate",
    xaxis_range=[-0.02, 1.02],
    yaxis_range=[-0.02, 1.05],
)
apply_dark(fig_roc)
save_chart(fig_roc, "output/reports/roc_curve.png")


# =============================================================================
# SECTION 9 -- Prophet Revenue Forecasting
# =============================================================================
print(f"\n{SEP}\n  SECTION 9 -- Prophet Revenue Forecasting\n{SEP}")

prophet_df = (mrr_ts[["invoice_month", "total_mrr"]]
              .rename(columns={"invoice_month": "ds", "total_mrr": "y"})
              .copy())
prophet_df["ds"] = pd.to_datetime(prophet_df["ds"])
prophet_df = (prophet_df.sort_values("ds")
              .drop_duplicates(subset="ds")
              .loc[lambda d: d["y"] > 0]
              .reset_index(drop=True))

print(f"  Training observations: {len(prophet_df)}")
print(f"  Range: {prophet_df['ds'].iloc[0].strftime('%Y-%m')} "
      f"to {prophet_df['ds'].iloc[-1].strftime('%Y-%m')}")

m_prophet = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=False,
    daily_seasonality=False,
    changepoint_prior_scale=0.05,
    seasonality_prior_scale=10,
    interval_width=0.95,
)
m_prophet.fit(prophet_df)
print("  Prophet model fitted.")

try:
    future = m_prophet.make_future_dataframe(periods=6, freq="MS")
except Exception:
    try:
        future = m_prophet.make_future_dataframe(periods=6, freq="ME")
    except Exception:
        future = m_prophet.make_future_dataframe(periods=6, freq="M")

forecast      = m_prophet.predict(future)
last_hist_dt  = prophet_df["ds"].max()
hist_fc       = forecast[forecast["ds"] <= last_hist_dt].copy()
future_fc     = forecast[forecast["ds"] > last_hist_dt].copy()

# Save CSVs
forecast.to_csv("output/models/mrr_forecast.csv", index=False)
(future_fc[["ds", "yhat", "yhat_lower", "yhat_upper"]]
 .to_csv("output/models/mrr_forecast_future.csv", index=False))
print("  Saved: output/models/mrr_forecast.csv")
print("  Saved: output/models/mrr_forecast_future.csv")

# Forecast chart
fig_fc = go.Figure()
fig_fc.add_trace(go.Scatter(
    x=prophet_df["ds"], y=prophet_df["y"],
    mode="lines", name="Historical MRR",
    line=dict(color=PRIMARY, width=2.5),
))
fig_fc.add_trace(go.Scatter(
    x=hist_fc["ds"], y=hist_fc["yhat"],
    mode="lines", name="Model Fit",
    line=dict(color="#6ab7d4", width=1.5, dash="dash"),
))
fig_fc.add_trace(go.Scatter(
    x=future_fc["ds"], y=future_fc["yhat"],
    mode="lines+markers", name="Forecast",
    line=dict(color=SUCCESS, width=2.5),
    marker=dict(size=6, color=SUCCESS),
))
# Confidence band
x_band = list(future_fc["ds"]) + list(future_fc["ds"])[::-1]
y_band = list(future_fc["yhat_upper"]) + list(future_fc["yhat_lower"])[::-1]
fig_fc.add_trace(go.Scatter(
    x=x_band, y=y_band,
    fill="toself",
    fillcolor="rgba(40, 167, 69, 0.15)",
    line=dict(color="rgba(0,0,0,0)"),
    name="95% Confidence",
))
# Forecast start annotation
fig_fc.add_vline(
    x=last_hist_dt.strftime("%Y-%m-%d"),
    line_dash="dash", line_color="#7a8a9a", line_width=1.5,
)
fig_fc.add_annotation(
    x=future_fc["ds"].iloc[0].strftime("%Y-%m-%d"),
    y=float(future_fc["yhat"].max()) * 0.94,
    text="Forecast Period",
    showarrow=False,
    font=dict(color=SUCCESS, size=12),
    bgcolor="rgba(0,0,0,0.55)", bordercolor=SUCCESS, borderwidth=1,
)
fig_fc.update_layout(
    title="MRR Forecast -- Next 6 Months (Prophet)",
    xaxis_title="Month",
    yaxis_title="Monthly Recurring Revenue",
    yaxis_tickprefix="$", yaxis_tickformat=",",
    height=450,
)
apply_dark(fig_fc)
save_chart(fig_fc, "output/reports/mrr_forecast.png", width=900, height=450)

# Forecast table
print(f"\n  {'Month':<12} {'Forecast MRR':>16} {'Lower (95%)':>14} {'Upper (95%)':>14}")
print(f"  {'-'*12} {'-'*16} {'-'*14} {'-'*14}")
for _, row in future_fc.iterrows():
    mo = row["ds"].strftime("%b %Y")
    print(f"  {mo:<12} {format_currency(row['yhat']):>16} "
          f"{format_currency(row['yhat_lower']):>14} "
          f"{format_currency(row['yhat_upper']):>14}")


# =============================================================================
# SECTION 10 -- Save Models & Summary JSON
# =============================================================================
print(f"\n{SEP}\n  SECTION 10 -- Save Models & Summary\n{SEP}")

with open("output/models/churn_model.pkl", "wb") as fh:
    pickle.dump(best_model, fh)
print("  Saved: output/models/churn_model.pkl")

with open("output/models/label_encoders.pkl", "wb") as fh:
    pickle.dump(label_encoders, fh)
print("  Saved: output/models/label_encoders.pkl")

# Derived summary stats
last_mrr_val    = float(prophet_df["y"].iloc[-1])
jul_row         = future_fc.iloc[0]
dec_row         = future_fc.iloc[-1]
forecast_total  = float(future_fc["yhat"].sum())
forecast_lower  = float(future_fc["yhat_lower"].sum())
forecast_upper  = float(future_fc["yhat_upper"].sum())
expected_growth = (float(dec_row["yhat"]) - last_mrr_val) / last_mrr_val * 100

hr_mask    = active_pred["ml_risk_tier"].isin(["High", "Critical"])
hr_count   = int(hr_mask.sum())
hr_revenue = float(active_pred.loc[hr_mask, "total_revenue"].sum())

ml_summary = {
    "churn_model": {
        "best_model":              best_name,
        "auc_roc":                 round(best_r["auc"], 4),
        "accuracy":                round(best_r["accuracy"], 4),
        "total_customers_scored":  int(len(ca_pred)),
        "active_customers_scored": int(len(active_pred)),
        "critical_risk_count":     int(tier_counts.get("Critical", 0)),
        "high_risk_count":         int(tier_counts.get("High", 0)),
        "medium_risk_count":       int(tier_counts.get("Medium", 0)),
        "low_risk_count":          int(tier_counts.get("Low", 0)),
        "avg_churn_prob_smb":         round(float(
            active_pred[active_pred["segment"] == "SMB"]["churn_probability"].mean()), 4),
        "avg_churn_prob_midmarket":   round(float(
            active_pred[active_pred["segment"] == "Mid-Market"]["churn_probability"].mean()), 4),
        "avg_churn_prob_enterprise":  round(float(
            active_pred[active_pred["segment"] == "Enterprise"]["churn_probability"].mean()), 4),
    },
    "revenue_forecast": {
        "model":               "Prophet",
        "last_historical_mrr": round(last_mrr_val, 2),
        "forecast_jul_2024":   round(float(jul_row["yhat"]), 2),
        "forecast_dec_2024":   round(float(dec_row["yhat"]), 2),
        "forecast_6m_total":   round(forecast_total, 2),
        "forecast_6m_lower":   round(forecast_lower, 2),
        "forecast_6m_upper":   round(forecast_upper, 2),
        "expected_growth_pct": round(expected_growth, 4),
    },
}

with open(A + "ml_summary.json", "w", encoding="utf-8") as fh:
    json.dump(ml_summary, fh, indent=2)
print(f"  Saved: {A}ml_summary.json")

# Final printed summary
print(f"\n  --- MODEL PERFORMANCE ---")
print(f"  {'Model':<25} {'Accuracy':>10} {'AUC-ROC':>10}")
print(f"  {'-'*25} {'-'*10} {'-'*10}")
for name, r in results.items():
    tag = " <-- BEST" if name == best_name else ""
    print(f"  {name:<25} {r['accuracy']:>10.3f} {r['auc']:>10.3f}{tag}")

print(f"\n  --- RISK TIER DISTRIBUTION (Active) ---")
for tier in ["Critical", "High", "Medium", "Low"]:
    cnt = int(tier_counts.get(tier, 0))
    pct = cnt / len(active_pred) * 100 if len(active_pred) > 0 else 0
    print(f"  {tier:<10}: {cnt:>5,}  ({pct:.1f}%)")

print(f"\n  --- 6-MONTH REVENUE FORECAST ---")
print(f"  Current MRR (Jun 2024): {format_currency(last_mrr_val)}")
print(f"  Forecast Jul 2024:      {format_currency(float(jul_row['yhat']))}")
print(f"  Forecast Dec 2024:      {format_currency(float(dec_row['yhat']))}")
print(f"  6-month total:          {format_currency(forecast_total)}")
print(f"  Expected growth:        {expected_growth:+.1f}%")

print(f"\n  --- REVENUE AT RISK ---")
print(f"  High + Critical customers: {hr_count:,}")
print(f"  Total revenue at risk:     {format_currency(hr_revenue)}")


# =============================================================================
# SECTION 11 -- ML Insight Strings
# =============================================================================
print(f"\n{SEP}\n  SECTION 11 -- ML Insights\n{SEP}")


def get_ml_insights_text() -> dict:
    total_active   = int(len(active_pred))
    critical_count = int(tier_counts.get("Critical", 0))
    high_count     = int(tier_counts.get("High", 0))
    auc            = best_r["auc"]
    jul_fc         = float(jul_row["yhat"])
    dec_fc         = float(dec_row["yhat"])
    dec_lo         = float(dec_row["yhat_lower"])
    dec_hi         = float(dec_row["yhat_upper"])

    ml_churn_insight = (
        f"Machine learning analysis of {total_active:,} active customers identified "
        f"{critical_count} at critical churn risk and {high_count} at high risk. "
        f"{best_name} model (AUC: {auc:.3f}) predicts "
        f"${hr_revenue:,.0f} in revenue at risk over the next 90 days."
    )

    ml_forecast_insight = (
        f"Prophet time-series model forecasts MRR of {format_currency(jul_fc)} for July 2024, "
        f"growing to {format_currency(dec_fc)} by December 2024 -- a {expected_growth:.1f}% "
        f"increase over current MRR of {format_currency(last_mrr_val)}. "
        f"95% confidence interval ranges from {format_currency(dec_lo)} to {format_currency(dec_hi)}."
    )

    return {
        "ml_churn_insight":    ml_churn_insight,
        "ml_forecast_insight": ml_forecast_insight,
    }


insights = get_ml_insights_text()
print(f"  ML Churn Insight:")
print(f"  {insights['ml_churn_insight']}")
print(f"\n  ML Forecast Insight:")
print(f"  {insights['ml_forecast_insight']}")

print(f"\n{SEP}")
print("  ML pipeline complete.")
print(f"  Models  -> output/models/  (churn_model.pkl, label_encoders.pkl, *.csv)")
print(f"  Charts  -> output/reports/ (feature_importance.png, roc_curve.png, mrr_forecast.png)")
print(f"  Data    -> {A}  (customer_predictions.csv, top_churn_risk.csv, ml_summary.json)")
print(f"{SEP}\n")
