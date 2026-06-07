"""
Analytics pipeline for credresolve-analytics.
Loads processed CSVs and warehouse.db, computes advanced analytics,
saves outputs to data/processed/analytics_outputs/.
Run from the credresolve-analytics/ root directory.
"""
import json
import os
import sqlite3
import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

PROCESSED_DIR = "data/processed/"
OUT_DIR       = "data/processed/analytics_outputs/"
DB_PATH       = "data/warehouse.db"

os.makedirs(OUT_DIR, exist_ok=True)

SEP = "=" * 65


def save(df: pd.DataFrame, name: str) -> str:
    path = os.path.join(OUT_DIR, name)
    df.to_csv(path, index=False)
    return path


def add_zscore(series: pd.Series, threshold: float = 1.5):
    """Return (z_score series, is_anomaly bool series)."""
    filled = series.fillna(series.mean())
    zs = pd.Series(stats.zscore(filled), index=series.index).round(4)
    return zs, (zs.abs() > threshold)


# =============================================================================
# SECTION 1 -- Load
# =============================================================================
print(f"\n{SEP}\n  SECTION 1 -- Loading processed data\n{SEP}")

p = PROCESSED_DIR

cust      = pd.read_csv(p + "customers_clean.csv",     parse_dates=["signup_date"])
subs      = pd.read_csv(p + "subscriptions_clean.csv", parse_dates=["invoice_date"])
mrr_m     = pd.read_csv(p + "mrr_monthly.csv")
mrr_seg   = pd.read_csv(p + "mrr_by_segment.csv")
mrr_reg   = pd.read_csv(p + "mrr_by_region.csv")
churn_seg = pd.read_csv(p + "churn_by_segment.csv")
churn_reg = pd.read_csv(p + "churn_by_region.csv")
churn_mon = pd.read_csv(p + "churn_monthly.csv")
camp_chan  = pd.read_csv(p + "campaign_by_channel.csv")
csat_mon  = pd.read_csv(p + "csat_monthly.csv")
tix_vol   = pd.read_csv(p + "ticket_volume_monthly.csv")
feat_adop = pd.read_csv(p + "feature_adoption.csv")
mau_mon   = pd.read_csv(p + "mau_monthly.csv")
cohort    = pd.read_csv(p + "cohort_retention.csv")
ca        = pd.read_csv(p + "customer_analytics.csv",
                        parse_dates=["signup_date", "latest_invoice_date",
                                     "last_product_access_date", "churn_date"])
camps     = pd.read_csv(p + "marketing_clean.csv",
                        parse_dates=["start_date", "end_date"])

for name, df in [("customers_clean", cust), ("subscriptions_clean", subs),
                  ("customer_analytics", ca), ("cohort_retention", cohort),
                  ("marketing_clean", camps)]:
    print(f"  {name:<30} {len(df):>8,} rows")


# =============================================================================
# SECTION 2 -- MRR Time-Series Analysis
# =============================================================================
print(f"\n{SEP}\n  SECTION 2 -- MRR Time-Series Analysis\n{SEP}")

total_mrr_ts = (
    mrr_m.groupby("invoice_month")["mrr"]
    .sum().reset_index().rename(columns={"mrr": "total_mrr"})
    .sort_values("invoice_month").reset_index(drop=True)
)

total_mrr_ts["mom_change"]         = total_mrr_ts["total_mrr"].diff().fillna(0)
total_mrr_ts["mom_pct_change"]     = (total_mrr_ts["total_mrr"].pct_change().fillna(0) * 100).round(4)
total_mrr_ts["rolling_3m_avg"]     = total_mrr_ts["total_mrr"].rolling(3, min_periods=1).mean()
total_mrr_ts["cumulative_revenue"] = total_mrr_ts["total_mrr"].cumsum()

mrr_first           = total_mrr_ts["total_mrr"].iloc[0]
mrr_last            = total_mrr_ts["total_mrr"].iloc[-1]
overall_mrr_growth  = (mrr_last - mrr_first) / mrr_first * 100

total_mrr_ts["z_score"], total_mrr_ts["is_anomaly"] = add_zscore(total_mrr_ts["total_mrr"])

# saved again after NRR is appended in Section 3
save(total_mrr_ts, "mrr_timeseries.csv")

print(f"  Months in series:     {len(total_mrr_ts)}")
print(f"  First month MRR:      ${mrr_first:,.2f}")
print(f"  Latest month MRR:     ${mrr_last:,.2f}")
print(f"  Overall MRR growth:   {overall_mrr_growth:.1f}%")
print(f"  MRR anomaly months:   {total_mrr_ts['is_anomaly'].sum()}")


# =============================================================================
# SECTION 3 -- Churn & Retention Analysis
# =============================================================================
print(f"\n{SEP}\n  SECTION 3 -- Churn & Retention Analysis\n{SEP}")

sorted_months = sorted(subs["invoice_month"].unique())

# Pre-build {month: set(customer_ids)} for fast lookups
cust_per_month = {
    m: set(subs.loc[subs["invoice_month"] == m, "customer_id"])
    for m in sorted_months
}

# First invoice month per customer (used to identify "new" customers)
first_inv_dict = subs.groupby("customer_id")["invoice_month"].min().to_dict()

# A) Churn reconstruction from subscriptions
churn_ts_rows = []
for i, month in enumerate(sorted_months):
    curr     = cust_per_month[month]
    prev     = cust_per_month[sorted_months[i - 1]] if i > 0 else set()
    active   = len(curr)
    new_custs = len({c for c in curr if first_inv_dict.get(c) == month})
    churned  = len(prev - curr)
    churn_rate = (churned / len(prev) * 100) if prev else 0.0
    churn_ts_rows.append({
        "invoice_month":     month,
        "active_customers":  active,
        "new_customers":     new_custs,
        "churned_customers": churned,
        "churn_rate_pct":    round(churn_rate, 4),
    })

churn_ts = pd.DataFrame(churn_ts_rows)

# B) Retention rate
churn_ts["retention_rate_pct"] = (100 - churn_ts["churn_rate_pct"]).round(4)
churn_ts["rolling_3m_churn"]   = churn_ts["churn_rate_pct"].rolling(3, min_periods=1).mean()
churn_ts["mom_churn_change"]   = churn_ts["churn_rate_pct"].diff().fillna(0)
churn_ts["z_score"], churn_ts["is_anomaly"] = add_zscore(churn_ts["churn_rate_pct"])

# C) Net Revenue Retention per month
cust_mrr_map = mrr_m.set_index(["invoice_month", "customer_id"])["mrr"].to_dict()

nrr_vals = []
for i, month in enumerate(sorted_months):
    if i == 0:
        nrr_vals.append(100.0)
        continue
    prev_month = sorted_months[i - 1]
    prev_custs = cust_per_month[prev_month]
    start_mrr  = sum(cust_mrr_map.get((prev_month, c), 0.0) for c in prev_custs)
    end_mrr    = sum(cust_mrr_map.get((month, c), 0.0)      for c in prev_custs)
    nrr_vals.append(round(end_mrr / start_mrr * 100, 4) if start_mrr > 0 else 0.0)

churn_ts["nrr_pct"] = nrr_vals

# Merge NRR into MRR timeseries and re-save both
total_mrr_ts = total_mrr_ts.merge(
    churn_ts[["invoice_month", "nrr_pct"]], on="invoice_month", how="left"
)
save(total_mrr_ts, "mrr_timeseries.csv")
save(churn_ts,     "churn_timeseries.csv")

avg_churn   = churn_ts["churn_rate_pct"].iloc[1:].mean()
avg_nrr     = pd.Series(nrr_vals[1:]).mean()

print(f"  Churn timeseries rows:  {len(churn_ts)}")
print(f"  Avg monthly churn:      {avg_churn:.2f}%")
print(f"  Avg NRR:                {avg_nrr:.1f}%")
print(f"  Churn anomaly months:   {churn_ts['is_anomaly'].sum()}")


# =============================================================================
# SECTION 4 -- Customer Segmentation Analytics
# =============================================================================
print(f"\n{SEP}\n  SECTION 4 -- Customer Segmentation Analytics\n{SEP}")

# A) Revenue distribution
ca_sorted = ca.sort_values("total_revenue", ascending=False).reset_index(drop=True)
ca_sorted["revenue_rank"]       = ca_sorted.index + 1
ca_sorted["revenue_percentile"] = ((1 - ca_sorted.index / len(ca_sorted)) * 100).round(2)

total_rev              = ca_sorted["total_revenue"].sum()
ca_sorted["rev_cumsum"] = ca_sorted["total_revenue"].cumsum()
pareto_count           = (ca_sorted["rev_cumsum"] / total_rev <= 0.80).sum()
pareto_pct             = pareto_count / len(ca_sorted) * 100
top20_idx              = max(1, int(len(ca_sorted) * 0.20))
top20_rev_share        = ca_sorted.head(top20_idx)["total_revenue"].sum() / total_rev * 100

def rev_stats(df, col):
    return df.groupby(col)["total_revenue"].agg(
        mean="mean", median="median", total="sum", std="std"
    ).round(2).reset_index()

revenue_by_segment = rev_stats(ca, "segment")
revenue_by_region  = rev_stats(ca, "region")
revenue_by_plan    = rev_stats(ca, "plan_type")

print(f"  Pareto: {pareto_pct:.1f}% of customers -> 80% of revenue")
print(f"  Top 20% customers revenue share: {top20_rev_share:.1f}%")

# B) CLV (active customers only)
CHURN_PROB_MONTHLY = {
    "SMB":        30.2 / 12 / 100,
    "Mid-Market": 18.5 / 12 / 100,
    "Enterprise":  8.8 / 12 / 100,
}

ca_active = ca[ca["status"] == "Active"].copy()
ca_active["monthly_revenue"]     = ca_active["total_revenue"] / ca_active["tenure_days"].clip(lower=1) * 30
ca_active["monthly_churn_prob"]  = ca_active["segment"].map(CHURN_PROB_MONTHLY)
ca_active["clv"]                 = ca_active["monthly_revenue"] / ca_active["monthly_churn_prob"]
clv_cap                          = ca_active["clv"].quantile(0.99)
ca_active["clv"]                 = ca_active["clv"].clip(upper=clv_cap).round(2)

clv_by_segment = (
    ca_active.groupby("segment")["clv"]
    .agg(avg_clv="mean", median_clv="median").round(2).reset_index()
)
clv_by_region = (
    ca_active.groupby("region")["clv"]
    .agg(avg_clv="mean", median_clv="median").round(2).reset_index()
)

print(f"\n  CLV by segment:")
print(clv_by_segment.to_string(index=False))

# C) High-risk customers (active, score >= 40)
def _risk_tier(score):
    if score >= 75: return "Critical"
    if score >= 50: return "High"
    return "Medium"

high_risk = (
    ca[(ca["status"] == "Active") & (ca["churn_risk_score"] >= 40)]
    .copy()
    .sort_values(["churn_risk_score", "total_revenue"], ascending=[False, False])
)
high_risk["risk_tier"] = high_risk["churn_risk_score"].apply(_risk_tier)
save(high_risk, "high_risk_customers.csv")

# D) Top 100 active customers by revenue
top_customers = (
    ca[ca["status"] == "Active"]
    .sort_values("total_revenue", ascending=False)
    .head(100)
    [["customer_id", "segment", "region", "plan_type",
      "total_revenue", "tenure_days", "churn_risk_score"]]
)
save(top_customers, "top_customers.csv")

print(f"\n  High-risk active (score >= 40): {len(high_risk):,}")
print(f"  Critical risk   (score >= 75): {(high_risk['churn_risk_score'] >= 75).sum():,}")
print(f"  High risk       (score >= 50): {(high_risk['churn_risk_score'] >= 50).sum():,}")


# =============================================================================
# SECTION 5 -- Cohort Analysis
# =============================================================================
print(f"\n{SEP}\n  SECTION 5 -- Cohort Analysis\n{SEP}")

cohort_pivot = cohort.pivot(
    index="cohort", columns="period_number", values="retention_pct"
).fillna(0)
cohort_pivot.columns = [f"period_{int(c)}" for c in cohort_pivot.columns]

cohort_size_map         = cohort[cohort["period_number"] == 0].set_index("cohort")["cohort_size"]
cohort_pivot["cohort_size"] = cohort_size_map

save(cohort_pivot.reset_index(), "cohort_heatmap.csv")

avg_curve = (
    cohort.groupby("period_number")["retention_pct"]
    .mean().reset_index().rename(columns={"retention_pct": "avg_retention_pct"})
)
save(avg_curve, "avg_retention_curve.csv")

if "period_3" in cohort_pivot.columns:
    best_cohort  = str(cohort_pivot["period_3"].idxmax())
    worst_cohort = str(cohort_pivot["period_3"].idxmin())
else:
    best_cohort  = "N/A"
    worst_cohort = "N/A"

print(f"  Cohort pivot shape:    {cohort_pivot.shape}")
print(f"  Best period-3 cohort:  {best_cohort}")
print(f"  Worst period-3 cohort: {worst_cohort}")


# =============================================================================
# SECTION 6 -- Marketing Analytics
# =============================================================================
print(f"\n{SEP}\n  SECTION 6 -- Marketing Analytics\n{SEP}")

# A) Campaign performance
camps_sorted = camps.sort_values("roi", ascending=False).reset_index(drop=True)
save(camps_sorted.head(10),               "top_campaigns.csv")
save(camps_sorted.tail(10).sort_values("roi").reset_index(drop=True), "worst_campaigns.csv")

funnel = camps.groupby("channel").agg(
    total_impressions=("impressions", "sum"),
    total_clicks=     ("clicks",      "sum"),
    total_leads=      ("leads",       "sum"),
    total_conversions=("conversions", "sum"),
).reset_index()
funnel["ctr_pct"]           = (funnel["total_clicks"] / funnel["total_impressions"].replace(0, np.nan) * 100).round(4)
funnel["lead_rate_pct"]     = (funnel["total_leads"]  / funnel["total_clicks"].replace(0, np.nan) * 100).round(4)
funnel["conv_rate_pct"]     = (funnel["total_conversions"] / funnel["total_leads"].replace(0, np.nan) * 100).round(4)
funnel["funnel_efficiency"] = (funnel["total_conversions"] / funnel["total_impressions"].replace(0, np.nan) * 100).round(6)
save(funnel, "marketing_funnel.csv")

# B) Campaign ROI time series
camps["start_month"] = camps["start_date"].dt.to_period("M").astype(str)
roi_monthly = (
    camps.groupby(["start_month", "channel"])["roi"]
    .mean().reset_index().rename(columns={"roi": "avg_roi"})
)
save(roi_monthly, "campaign_roi_monthly.csv")

# C) Segment targeting effectiveness
seg_eff = camps.groupby("target_segment").agg(
    avg_roi=          ("roi",               "mean"),
    total_conversions=("conversions",       "sum"),
    total_revenue=    ("revenue_generated", "sum"),
).round(2).reset_index()
save(seg_eff, "campaign_by_segment_analytics.csv")

best_channel    = camp_chan.sort_values("mean_roi", ascending=False).iloc[0]["channel"]
total_conv_all  = int(camps["conversions"].sum())
total_rev_camps = float(camps["revenue_generated"].sum())

print(f"  Best ROI channel:    {best_channel}")
print(f"  Top campaign:        {camps_sorted.iloc[0]['campaign_id']}  ROI={camps_sorted.iloc[0]['roi']:.1f}%")
print(f"  Total conversions:   {total_conv_all:,}")


# =============================================================================
# SECTION 7 -- Support Analytics (from SQLite)
# =============================================================================
print(f"\n{SEP}\n  SECTION 7 -- Support Analytics\n{SEP}")

conn = sqlite3.connect(DB_PATH)

tix = pd.read_sql(
    "SELECT * FROM support_clean", conn,
    parse_dates=["created_date", "resolved_date"]
)

# A) Resolution time trends
res_monthly = (
    tix[tix["status"] == "Resolved"]
    .groupby("ticket_month")["resolution_time_hours"]
    .mean().reset_index()
    .rename(columns={"resolution_time_hours": "avg_resolution_hours"})
    .sort_values("ticket_month").reset_index(drop=True)
)
res_monthly["rolling_3m_avg"]    = res_monthly["avg_resolution_hours"].rolling(3, min_periods=1).mean()
res_monthly["z_score"], res_monthly["is_anomaly"] = add_zscore(res_monthly["avg_resolution_hours"])
save(res_monthly, "resolution_time_monthly.csv")

# B) CSAT trends
csat_ts = csat_mon.sort_values("ticket_month").reset_index(drop=True)
csat_ts["mom_change"]     = csat_ts["avg_csat"].diff().fillna(0)
csat_ts["rolling_3m_avg"] = csat_ts["avg_csat"].rolling(3, min_periods=1).mean()
csat_ts["low_csat_flag"]  = csat_ts["avg_csat"] < 3.0
save(csat_ts, "csat_timeseries.csv")

# C) Category x priority breakdown
cat_analysis = pd.read_sql("""
    SELECT
        category,
        priority,
        COUNT(*)                   AS ticket_count,
        AVG(resolution_time_hours) AS avg_resolution_hours,
        AVG(satisfaction_score)    AS avg_satisfaction
    FROM support_clean
    GROUP BY category, priority
    ORDER BY category, priority
""", conn)
save(cat_analysis, "support_category_analysis.csv")

# D) Tickets per customer by segment
tickets_by_seg = pd.read_sql("""
    SELECT
        c.segment,
        COUNT(t.ticket_id)                                                  AS total_tickets,
        COUNT(DISTINCT t.customer_id)                                       AS customers_with_tickets,
        CAST(COUNT(t.ticket_id) AS FLOAT) / COUNT(DISTINCT t.customer_id)  AS avg_tickets_per_customer
    FROM support_clean t
    JOIN customers_clean c ON t.customer_id = c.customer_id
    GROUP BY c.segment
""", conn)
save(tickets_by_seg, "tickets_by_segment.csv")

# KPI support queries -- use fixed report date 2024-06-01 (data boundary)
open_tix_count = pd.read_sql(
    "SELECT COUNT(*) AS n FROM support_clean WHERE status='Open'", conn
).iloc[0]["n"]
last30_count = pd.read_sql(
    "SELECT COUNT(*) AS n FROM support_clean "
    "WHERE created_date >= date('2024-06-01', '-30 days')", conn
).iloc[0]["n"]
avg_res_hours = pd.read_sql(
    "SELECT AVG(resolution_time_hours) AS v FROM support_clean WHERE status='Resolved'", conn
).iloc[0]["v"]

conn.close()

print(f"  Resolution time rows:    {len(res_monthly)}")
print(f"  Overall avg resolution:  {float(avg_res_hours):.1f} hrs")
print(f"  Open tickets:            {int(open_tix_count):,}")
print(f"  CSAT low months (< 3.0): {csat_ts['low_csat_flag'].sum()}")


# =============================================================================
# SECTION 8 -- Product Usage Analytics
# =============================================================================
print(f"\n{SEP}\n  SECTION 8 -- Product Usage Analytics\n{SEP}")

conn = sqlite3.connect(DB_PATH)
usg = pd.read_sql(
    "SELECT * FROM product_usage_clean", conn,
    parse_dates=["last_access_date"]
)
conn.close()

# A) Feature adoption monthly pivot
feat_monthly = (
    usg.groupby(["access_month", "feature"])["customer_id"]
    .nunique().reset_index().rename(columns={"customer_id": "unique_customers"})
)
feat_pivot = feat_monthly.pivot(
    index="access_month", columns="feature", values="unique_customers"
).fillna(0).reset_index()
save(feat_pivot, "feature_adoption_monthly.csv")

# B) Usage intensity by segment
usage_by_seg = (
    usg.groupby(["segment", "feature"]).agg(
        mean_usage_count=    ("usage_count",              "mean"),
        mean_session_minutes=("session_duration_minutes", "mean"),
    ).round(3).reset_index()
)
save(usage_by_seg, "usage_by_segment.csv")

# C) Engagement score per customer
def _engagement_score(row) -> float:
    feat_used  = row["distinct_features_used"] if pd.notna(row["distinct_features_used"]) else 0.0
    usage_cnt  = row["total_usage_count"]       if pd.notna(row["total_usage_count"])       else 0.0
    dsa        = row["days_since_last_access"]  if pd.notna(row["days_since_last_access"])  else 30.0
    feat_score    = (feat_used / 7) * 40
    usage_score   = (min(usage_cnt, 100) / 100) * 40
    recency_score = (max(0.0, 30.0 - dsa) / 30.0) * 20
    return round(feat_score + usage_score + recency_score, 2)

def _engagement_cat(score: float) -> str:
    if score >= 70: return "Highly Engaged"
    if score >= 40: return "Moderately Engaged"
    return "At Risk"

ca_enrich = ca.copy()
ca_enrich["engagement_score"]    = ca_enrich.apply(_engagement_score, axis=1)
ca_enrich["engagement_category"] = ca_enrich["engagement_score"].apply(_engagement_cat)

# Add CLV from Section 4 (active customers only; churned get NaN)
clv_map = ca_active.set_index("customer_id")["clv"].to_dict()
ca_enrich["clv"] = ca_enrich["customer_id"].map(clv_map)

save(ca_enrich, "customer_analytics_enriched.csv")

highly_engaged = (ca_enrich["engagement_category"] == "Highly Engaged").sum()
at_risk_eng    = (ca_enrich["engagement_category"] == "At Risk").sum()

print(f"  Feature adoption pivot: {feat_pivot.shape}")
print(f"  Usage by segment rows:  {len(usage_by_seg)}")
print(f"  Highly Engaged:         {highly_engaged:,}")
print(f"  At Risk (engagement):   {at_risk_eng:,}")


# =============================================================================
# SECTION 9 -- KPI Summary JSON
# =============================================================================
print(f"\n{SEP}\n  SECTION 9 -- KPI Summary JSON\n{SEP}")

# MRR
latest_mrr  = float(total_mrr_ts["total_mrr"].iloc[-1])
prev_mrr    = float(total_mrr_ts["total_mrr"].iloc[-2]) if len(total_mrr_ts) > 1 else latest_mrr
mrr_mom_pct = (latest_mrr - prev_mrr) / prev_mrr * 100 if prev_mrr != 0 else 0.0
rolling_3m  = float(total_mrr_ts["rolling_3m_avg"].iloc[-1])

ytd = total_mrr_ts[total_mrr_ts["invoice_month"].str.startswith("2024")]
ytd_growth = (
    (ytd["total_mrr"].iloc[-1] - ytd["total_mrr"].iloc[0]) / ytd["total_mrr"].iloc[0] * 100
    if len(ytd) >= 2 else 0.0
)

# Churn
overall_churn_rate = ca["status"].eq("Churned").mean() * 100
latest_churn_rate  = float(churn_ts["churn_rate_pct"].iloc[-1])
prev_churn_rate    = float(churn_ts["churn_rate_pct"].iloc[-2]) if len(churn_ts) > 1 else latest_churn_rate
churn_mom          = latest_churn_rate - prev_churn_rate
latest_nrr         = float(churn_ts["nrr_pct"].dropna().iloc[-1])
avg_tenure_active  = float(ca[ca["status"] == "Active"]["tenure_days"].mean())

# Revenue
total_lifetime_rev = float(subs["final_amount"].sum())
avg_rev_per_cust   = total_lifetime_rev / len(ca)

# Risk (from full enriched table, active only)
active_mask        = ca_enrich["status"] == "Active"
high_risk_count    = int((active_mask & (ca_enrich["churn_risk_score"] >= 50)).sum())
critical_risk_count = int((active_mask & (ca_enrich["churn_risk_score"] >= 75)).sum())
high_risk_rev      = float(ca_enrich[active_mask & (ca_enrich["churn_risk_score"] >= 50)]["total_revenue"].sum())

# Product
mau_latest        = int(mau_mon.sort_values("access_month").iloc[-1]["monthly_active_users"])
avg_feat_per_cust = float(ca_enrich["distinct_features_used"].mean())
top_feat_row      = feat_adop.sort_values("unique_customers", ascending=False).iloc[0]

kpi = {
    "report_date": "2024-06-01",
    "mrr": {
        "current":        round(latest_mrr, 2),
        "previous":       round(prev_mrr, 2),
        "mom_change_pct": round(mrr_mom_pct, 4),
        "ytd_growth_pct": round(ytd_growth, 4),
        "3m_rolling_avg": round(rolling_3m, 2),
    },
    "churn": {
        "overall_rate_pct":        round(overall_churn_rate, 2),
        "smb_rate_pct":            30.2,
        "midmarket_rate_pct":      18.5,
        "enterprise_rate_pct":     8.8,
        "latest_monthly_rate_pct": round(latest_churn_rate, 4),
        "mom_change_pct":          round(churn_mom, 4),
    },
    "retention": {
        "overall_rate_pct": round(100 - overall_churn_rate, 2),
        "nrr_pct":          round(latest_nrr, 2),
        "avg_tenure_days":  round(avg_tenure_active, 1),
    },
    "revenue": {
        "total_lifetime":                    round(total_lifetime_rev, 2),
        "avg_revenue_per_customer":          round(avg_rev_per_cust, 2),
        "top_20pct_customers_revenue_share": round(top20_rev_share, 2),
    },
    "support": {
        "avg_resolution_hours": round(float(avg_res_hours), 2),
        "avg_csat":             3.63,
        "open_tickets":         int(open_tix_count),
        "tickets_last_30_days": int(last30_count),
    },
    "marketing": {
        "avg_roi_pct":            round(float(camps["roi"].mean()), 2),
        "best_channel":           best_channel,
        "total_conversions":      total_conv_all,
        "total_campaign_revenue": round(total_rev_camps, 2),
    },
    "product": {
        "top_feature":               str(top_feat_row["feature"]),
        "top_feature_adoption_pct":  round(float(top_feat_row["adoption_pct"]), 1),
        "avg_features_per_customer": round(avg_feat_per_cust, 2),
        "mau_latest":                mau_latest,
    },
    "risk": {
        "high_risk_customers":        high_risk_count,
        "critical_risk_customers":    critical_risk_count,
        "high_risk_revenue_at_stake": round(high_risk_rev, 2),
    },
}

kpi_path = os.path.join(OUT_DIR, "kpi_summary.json")
with open(kpi_path, "w") as f:
    json.dump(kpi, f, indent=2)

print(f"  Saved: {kpi_path}")
for section, vals in kpi.items():
    if isinstance(vals, dict):
        print(f"\n  [{section}]")
        for k, v in vals.items():
            print(f"    {k:<38} {v}")
    else:
        print(f"  {section}: {vals}")


# =============================================================================
# SECTION 10 -- Print Analytics Summary
# =============================================================================
print(f"\n{SEP}")
print("  ANALYTICS SUMMARY")
print(f"{SEP}")

print(f"\n[MRR]")
print(f"  Current MRR:        ${kpi['mrr']['current']:>14,.2f}")
print(f"  Previous month:     ${kpi['mrr']['previous']:>14,.2f}")
print(f"  MoM change:         {kpi['mrr']['mom_change_pct']:>+13.2f}%")
print(f"  YTD growth (2024):  {kpi['mrr']['ytd_growth_pct']:>+13.2f}%")
print(f"  3M rolling avg:     ${kpi['mrr']['3m_rolling_avg']:>14,.2f}")
print(f"  Overall growth:     {overall_mrr_growth:>+13.1f}%  (first -> last month)")

print(f"\n[CHURN & RETENTION]")
print(f"  Overall churn:      {kpi['churn']['overall_rate_pct']:>13.2f}%")
print(f"  Enterprise:         {kpi['churn']['enterprise_rate_pct']:>13.1f}%")
print(f"  Mid-Market:         {kpi['churn']['midmarket_rate_pct']:>13.1f}%")
print(f"  SMB:                {kpi['churn']['smb_rate_pct']:>13.1f}%")
print(f"  Latest NRR:         {kpi['retention']['nrr_pct']:>13.1f}%")
print(f"  Avg active tenure:  {kpi['retention']['avg_tenure_days']:>13.1f} days")

print(f"\n[REVENUE]")
print(f"  Total lifetime:     ${kpi['revenue']['total_lifetime']:>14,.2f}")
print(f"  Avg per customer:   ${kpi['revenue']['avg_revenue_per_customer']:>14,.2f}")
print(f"  Pareto insight:     {pareto_pct:.1f}% of customers = 80% of revenue")
print(f"  Top 20% rev share:  {kpi['revenue']['top_20pct_customers_revenue_share']:.1f}%")

print(f"\n[MRR ANOMALY MONTHS]")
mrr_anom = total_mrr_ts[total_mrr_ts["is_anomaly"]].sort_values(
    "z_score", key=lambda s: s.abs(), ascending=False
)
if len(mrr_anom):
    for _, row in mrr_anom.head(5).iterrows():
        print(f"  {row['invoice_month']}  MRR=${row['total_mrr']:>12,.0f}  z={row['z_score']:+.2f}")
else:
    print("  None detected (z-score threshold 1.5)")

print(f"\n[CHURN ANOMALY MONTHS]")
churn_anom = churn_ts[churn_ts["is_anomaly"]].sort_values(
    "z_score", key=lambda s: s.abs(), ascending=False
)
if len(churn_anom):
    for _, row in churn_anom.head(5).iterrows():
        print(f"  {row['invoice_month']}  churn={row['churn_rate_pct']:.2f}%  z={row['z_score']:+.2f}")
else:
    print("  None detected (z-score threshold 1.5)")

print(f"\n[CAMPAIGNS -- TOP 3 BY ROI]")
for _, r in camps_sorted.head(3).iterrows():
    print(f"  {r['campaign_id']}  {r['channel']:<20} ROI={r['roi']:>9.1f}%")

print(f"\n[CAMPAIGNS -- BOTTOM 3 BY ROI]")
for _, r in camps_sorted.tail(3).sort_values("roi").iterrows():
    print(f"  {r['campaign_id']}  {r['channel']:<20} ROI={r['roi']:>9.1f}%")

print(f"\n[COHORT RETENTION]")
print(f"  Best period-3 cohort:  {best_cohort}")
print(f"  Worst period-3 cohort: {worst_cohort}")
print(f"  Average retention by period:")
for _, row in avg_curve.iterrows():
    bar = "#" * max(0, int(row["avg_retention_pct"] / 5))
    print(f"    Period {int(row['period_number'])}: {row['avg_retention_pct']:>5.1f}%  {bar}")

print(f"\n[FEATURE ADOPTION TREND  (first 3 vs last 3 months)]")
sorted_access_months = sorted(feat_monthly["access_month"].unique())
first3_set = set(sorted_access_months[:3])
last3_set  = set(sorted_access_months[-3:])
for feat in sorted(feat_monthly["feature"].unique()):
    early = feat_monthly[(feat_monthly["access_month"].isin(first3_set)) &
                         (feat_monthly["feature"] == feat)]["unique_customers"].mean()
    late  = feat_monthly[(feat_monthly["access_month"].isin(last3_set)) &
                         (feat_monthly["feature"] == feat)]["unique_customers"].mean()
    if   late < early * 0.95: trend = "DECLINING"
    elif late > early * 1.05: trend = "growing  "
    else:                     trend = "stable   "
    print(f"  {feat:<18}  early={early:>6.0f}  late={late:>6.0f}  [{trend}]")

print(f"\n[RISK SUMMARY]")
print(f"  High-risk active  (>= 50):  {kpi['risk']['high_risk_customers']:,}")
print(f"  Critical-risk     (>= 75):  {kpi['risk']['critical_risk_customers']:,}")
print(f"  Revenue at stake:           ${kpi['risk']['high_risk_revenue_at_stake']:,.2f}")

print(f"\n[SUPPORT & PRODUCT]")
print(f"  Avg resolution time:  {kpi['support']['avg_resolution_hours']:.1f} hrs")
print(f"  Overall CSAT:         {kpi['support']['avg_csat']:.2f} / 5.00")
print(f"  Open tickets:         {kpi['support']['open_tickets']:,}")
print(f"  Tickets (last 30d):   {kpi['support']['tickets_last_30_days']:,}")
print(f"  Top feature:          {kpi['product']['top_feature']} ({kpi['product']['top_feature_adoption_pct']}% adoption)")
print(f"  MAU (latest month):   {kpi['product']['mau_latest']:,}")

# Count saved files
saved_files = [f for f in os.listdir(OUT_DIR) if f.endswith(".csv") or f.endswith(".json")]
print(f"\n{SEP}")
print(f"  Analytics complete -- {len(saved_files)} files saved to {OUT_DIR}")
print(f"{SEP}\n")
