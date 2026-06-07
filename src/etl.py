"""
ETL pipeline for credresolve-analytics.
Reads raw CSVs -> cleans & transforms -> saves processed CSVs + SQLite warehouse.
Run from the credresolve-analytics/ root directory.
"""

import os
import sqlite3
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# -- Paths ---------------------------------------------------------------------
RAW_DIR       = "data/raw/"
PROCESSED_DIR = "data/processed/"
DB_PATH       = "data/warehouse.db"

os.makedirs(PROCESSED_DIR, exist_ok=True)

TODAY = pd.Timestamp.today().normalize()
SEP   = "=" * 65


# ===============================================================================
# SECTION 1 -- Load raw CSVs
# ===============================================================================
print(f"\n{SEP}\n  SECTION 1 -- Loading raw CSVs\n{SEP}")

cust  = pd.read_csv(RAW_DIR + "customers.csv")
subs  = pd.read_csv(RAW_DIR + "subscriptions.csv")
churn = pd.read_csv(RAW_DIR + "churn_retention.csv")
camps = pd.read_csv(RAW_DIR + "marketing_campaigns.csv")
tix   = pd.read_csv(RAW_DIR + "support_tickets.csv")
usg   = pd.read_csv(RAW_DIR + "product_usage.csv")

for name, df in [("customers", cust), ("subscriptions", subs), ("churn_retention", churn),
                  ("marketing_campaigns", camps), ("support_tickets", tix), ("product_usage", usg)]:
    print(f"  {name:<25} {len(df):>8,} rows")


# ===============================================================================
# SECTION 2 -- Clean customers
# ===============================================================================
print(f"\n{SEP}\n  SECTION 2 -- Clean customers\n{SEP}")

cust["signup_date"] = pd.to_datetime(cust["signup_date"])

for col in cust.select_dtypes("object").columns:
    cust[col] = cust[col].str.strip()

cust["customer_id"]   = cust["customer_id"].astype(str).str.upper().str.strip()
cust["tenure_days"]   = (TODAY - cust["signup_date"]).dt.days.clip(lower=0)
cust["signup_month"]  = cust["signup_date"].dt.to_period("M").astype(str)
cust["signup_cohort"] = cust["signup_date"].apply(lambda d: f"{d.year}-Q{d.quarter}")

cust = cust.drop_duplicates(subset="customer_id", keep="first").reset_index(drop=True)

print(f"  Unique customers: {len(cust):,}")
print("\n  Status distribution:")
status_counts = cust["status"].value_counts().reset_index()
status_counts.columns = ["Status", "Count"]
print(status_counts.to_string(index=False))


# ===============================================================================
# SECTION 3 -- Clean subscriptions + MRR tables
# ===============================================================================
print(f"\n{SEP}\n  SECTION 3 -- Clean subscriptions & MRR\n{SEP}")

subs["invoice_date"]  = pd.to_datetime(subs["invoice_date"])
subs["amount"]        = subs["amount"].astype(float)
subs["final_amount"]  = subs["final_amount"].astype(float)
subs["discount_pct"]  = subs["discount_pct"].astype(float)
subs["customer_id"]   = subs["customer_id"].astype(str)
subs["invoice_id"]    = subs["invoice_id"].astype(str)

subs = subs.drop_duplicates(subset="invoice_id", keep="first").reset_index(drop=True)

subs["invoice_month"]   = subs["invoice_date"].dt.to_period("M").astype(str)
subs["invoice_year"]    = subs["invoice_date"].dt.year.astype(int)
subs["invoice_quarter"] = subs["invoice_date"].apply(lambda d: f"{d.year}-Q{d.quarter}")

subs = subs.merge(
    cust[["customer_id", "segment", "region", "plan_type"]],
    on="customer_id", how="left"
)

mrr_monthly = (
    subs.groupby(["customer_id", "invoice_month"])["final_amount"]
    .sum().reset_index().rename(columns={"final_amount": "mrr"})
)
mrr_by_segment = (
    subs.groupby(["invoice_month", "segment"])["final_amount"]
    .sum().reset_index().rename(columns={"final_amount": "mrr"})
)
mrr_by_region = (
    subs.groupby(["invoice_month", "region"])["final_amount"]
    .sum().reset_index().rename(columns={"final_amount": "mrr"})
)

print(f"  Subscriptions cleaned:  {len(subs):>8,}")
print(f"  MRR monthly records:    {len(mrr_monthly):>8,}")
print(f"  MRR by segment rows:    {len(mrr_by_segment):>8,}")
print(f"  MRR by region rows:     {len(mrr_by_region):>8,}")


# ===============================================================================
# SECTION 4 -- Clean churn_retention + churn rates
# ===============================================================================
print(f"\n{SEP}\n  SECTION 4 -- Clean churn & churn rates\n{SEP}")

churn["churn_date"]         = pd.to_datetime(churn["churn_date"])
churn["last_activity_date"] = pd.to_datetime(churn["last_activity_date"])
churn["days_since_last_activity"] = (
    churn["churn_date"] - churn["last_activity_date"]
).dt.days

churn = churn.merge(
    cust[["customer_id", "segment", "region", "plan_type", "signup_date"]],
    on="customer_id", how="left"
)
churn["churn_month"] = churn["churn_date"].dt.to_period("M").astype(str)


def _churn_rate_table(groupby_col: str) -> pd.DataFrame:
    total   = cust.groupby(groupby_col)["customer_id"].count().rename("total_customers")
    churned = (
        cust[cust["status"] == "Churned"]
        .groupby(groupby_col)["customer_id"].count().rename("churned_customers")
    )
    tbl = pd.concat([total, churned], axis=1).fillna(0).reset_index()
    tbl["churned_customers"] = tbl["churned_customers"].astype(int)
    tbl["churn_rate_pct"] = (
        tbl["churned_customers"] / tbl["total_customers"] * 100
    ).round(2)
    return tbl


churn_by_segment = _churn_rate_table("segment")
churn_by_region  = _churn_rate_table("region")
churn_by_plan    = _churn_rate_table("plan_type")   # computed, not in save list

# Monthly churn rate: churns per month / unique active customers that month
monthly_churns = (
    churn.groupby("churn_month")["customer_id"].count()
    .reset_index().rename(columns={"churn_month": "month_year", "customer_id": "churns"})
)
monthly_active_subs = (
    subs.groupby("invoice_month")["customer_id"].nunique()
    .reset_index().rename(columns={"invoice_month": "month_year", "customer_id": "active_customers"})
)
churn_monthly = monthly_churns.merge(monthly_active_subs, on="month_year", how="left")
churn_monthly["monthly_churn_rate_pct"] = (
    churn_monthly["churns"] / churn_monthly["active_customers"] * 100
).round(4)

print(f"  Churned customer records: {len(churn):>6,}")
print(f"  Monthly churn rows:       {len(churn_monthly):>6,}")


# ===============================================================================
# SECTION 5 -- Clean marketing_campaigns
# ===============================================================================
print(f"\n{SEP}\n  SECTION 5 -- Clean marketing campaigns\n{SEP}")

camps["start_date"] = pd.to_datetime(camps["start_date"])
camps["end_date"]   = pd.to_datetime(camps["end_date"])

for col in ["impressions", "clicks", "leads", "conversions", "cost"]:
    camps[col] = pd.to_numeric(camps[col], errors="coerce").fillna(0).astype(int)
camps["revenue_generated"] = pd.to_numeric(camps["revenue_generated"], errors="coerce").fillna(0.0)

camps["ctr"] = np.where(camps["impressions"] > 0, camps["clicks"] / camps["impressions"], 0.0)
camps["roi"] = np.where(
    camps["cost"] > 0,
    (camps["revenue_generated"] - camps["cost"]) / camps["cost"] * 100,
    0.0,
)
camps["campaign_duration_days"] = (camps["end_date"] - camps["start_date"]).dt.days
camps["cost_per_lead"] = np.where(
    camps["leads"] > 0, camps["cost"] / camps["leads"], np.nan
)
camps["cost_per_conversion"] = np.where(
    camps["conversions"] > 0, camps["cost"] / camps["conversions"], np.nan
)

_perf_agg = dict(
    total_impressions=("impressions",        "sum"),
    total_leads=      ("leads",              "sum"),
    total_conversions=("conversions",        "sum"),
    total_cost=       ("cost",               "sum"),
    total_revenue=    ("revenue_generated",  "sum"),
    mean_roi=         ("roi",                "mean"),
)
campaign_by_channel  = camps.groupby("channel").agg(**_perf_agg).reset_index()
campaign_by_segment  = camps.groupby("target_segment").agg(**_perf_agg).reset_index()  # not saved

print(f"  Campaigns cleaned: {len(camps):,}")


# ===============================================================================
# SECTION 6 -- Clean support_tickets
# ===============================================================================
print(f"\n{SEP}\n  SECTION 6 -- Clean support tickets\n{SEP}")

tix["created_date"]          = pd.to_datetime(tix["created_date"])
tix["resolved_date"]         = pd.to_datetime(tix["resolved_date"], errors="coerce")
tix["resolution_time_hours"] = pd.to_numeric(tix["resolution_time_hours"], errors="coerce")
tix["satisfaction_score"]    = pd.to_numeric(tix["satisfaction_score"],    errors="coerce")

for col in ["category", "priority", "status"]:
    tix[col] = tix[col].str.strip()

tix["ticket_month"] = tix["created_date"].dt.to_period("M").astype(str)

avg_resolution_by_category = (
    tix.dropna(subset=["resolution_time_hours"])
    .groupby("category")["resolution_time_hours"]
    .mean().reset_index()
    .rename(columns={"resolution_time_hours": "avg_resolution_hours"})
)
csat_monthly = (
    tix.dropna(subset=["satisfaction_score"])
    .groupby("ticket_month")["satisfaction_score"]
    .mean().reset_index()
    .rename(columns={"satisfaction_score": "avg_csat"})
)
csat_by_category = (
    tix.dropna(subset=["satisfaction_score"])
    .groupby("category")["satisfaction_score"]
    .mean().reset_index()
    .rename(columns={"satisfaction_score": "avg_csat"})
)
ticket_volume_monthly = (
    tix.groupby("ticket_month")["ticket_id"]
    .count().reset_index()
    .rename(columns={"ticket_id": "ticket_count"})
)

print(f"  Tickets cleaned:       {len(tix):>7,}")
print(f"  CSAT monthly rows:     {len(csat_monthly):>7,}")
print(f"  Ticket volume rows:    {len(ticket_volume_monthly):>7,}")


# ===============================================================================
# SECTION 7 -- Clean product_usage
# ===============================================================================
print(f"\n{SEP}\n  SECTION 7 -- Clean product usage\n{SEP}")

usg["last_access_date"]         = pd.to_datetime(usg["last_access_date"])
usg["usage_count"]              = usg["usage_count"].astype(int)
usg["session_duration_minutes"] = usg["session_duration_minutes"].astype(float)
usg["access_month"]             = usg["last_access_date"].dt.to_period("M").astype(str)

usg = usg.merge(
    cust[["customer_id", "segment", "region", "plan_type"]],
    on="customer_id", how="left"
)

n_customers = cust["customer_id"].nunique()

feature_adoption = (
    usg.groupby("feature")["customer_id"].nunique()
    .reset_index().rename(columns={"customer_id": "unique_customers"})
)
feature_adoption["adoption_pct"] = (
    feature_adoption["unique_customers"] / n_customers * 100
).round(2)

feature_adoption_by_segment = (
    usg.groupby(["feature", "segment"])["customer_id"].nunique()
    .reset_index().rename(columns={"customer_id": "unique_customers"})
)
feature_adoption_by_region = (
    usg.groupby(["feature", "region"])["customer_id"].nunique()
    .reset_index().rename(columns={"customer_id": "unique_customers"})
)
mau_monthly = (
    usg.groupby("access_month")["customer_id"].nunique()
    .reset_index().rename(columns={"customer_id": "monthly_active_users"})
)
avg_session_by_feature = (
    usg.groupby("feature")["session_duration_minutes"]
    .mean().reset_index()
    .rename(columns={"session_duration_minutes": "avg_session_minutes"})
)

print(f"  Usage records cleaned: {len(usg):>8,}")
print(f"  MAU monthly rows:      {len(mau_monthly):>8,}")
print(f"  Features tracked:      {feature_adoption['feature'].nunique():>8,}")


# ===============================================================================
# SECTION 8 -- Cohort Retention
# ===============================================================================
print(f"\n{SEP}\n  SECTION 8 -- Cohort retention analysis\n{SEP}")

# Integer quarter key: year*4 + (quarter_1) enables simple offset arithmetic
subs_q_key = subs["invoice_date"].dt.year * 4 + subs["invoice_date"].dt.quarter - 1

# Pre-build customer -> set(active quarter keys) for O(1) membership tests
customer_active_quarters = defaultdict(set)
for cid, qk in zip(subs["customer_id"], subs_q_key):
    customer_active_quarters[cid].add(int(qk))

cohort_rows = []
for cohort_label in sorted(cust["signup_cohort"].unique()):
    cohort_custs = cust.loc[cust["signup_cohort"] == cohort_label, "customer_id"].values
    cohort_size  = len(cohort_custs)
    year_s, q_s  = cohort_label.split("-Q")
    base_q_key   = int(year_s) * 4 + int(q_s) - 1

    for period in range(7):
        target_q = base_q_key + period
        active   = sum(
            1 for c in cohort_custs
            if target_q in customer_active_quarters.get(c, set())
        )
        cohort_rows.append({
            "cohort":           cohort_label,
            "period_number":    period,
            "active_customers": active,
            "cohort_size":      cohort_size,
            "retention_pct":    round(active / cohort_size * 100, 2) if cohort_size > 0 else 0.0,
        })

cohort_retention = pd.DataFrame(cohort_rows)

print(f"  Cohort retention rows: {len(cohort_retention):,}")
print(f"  Unique cohorts:        {cohort_retention['cohort'].nunique()}")
print("\n  Period-0 retention snapshot (cohort size check):")
p0 = cohort_retention[cohort_retention["period_number"] == 0][["cohort", "cohort_size", "retention_pct"]]
print(p0.to_string(index=False))


# ===============================================================================
# SECTION 9 -- Master Analytical Table: customer_analytics
# ===============================================================================
print(f"\n{SEP}\n  SECTION 9 -- Building customer_analytics master table\n{SEP}")

rev_agg = (
    subs.groupby("customer_id").agg(
        total_revenue=      ("final_amount", "sum"),
        total_invoices=     ("invoice_id",   "count"),
        latest_invoice_date=("invoice_date", "max"),
    ).reset_index()
)

tix_agg = (
    tix.groupby("customer_id").agg(
        total_tickets=         ("ticket_id",          "count"),
        avg_satisfaction_score=("satisfaction_score", "mean"),
    ).reset_index()
)

usg_agg = (
    usg.groupby("customer_id").agg(
        total_usage_count=     ("usage_count",              "sum"),
        distinct_features_used=("feature",                  "nunique"),
        last_product_access_date=("last_access_date",       "max"),
    ).reset_index()
)
usg_agg["days_since_last_access"] = (TODAY - usg_agg["last_product_access_date"]).dt.days

churn_info = churn[["customer_id", "churn_date", "churn_reason"]].copy()

ca = cust.copy()
for agg_df in [rev_agg, tix_agg, usg_agg, churn_info]:
    ca = ca.merge(agg_df, on="customer_id", how="left")

for col in ["total_revenue", "total_invoices", "total_tickets",
            "total_usage_count", "distinct_features_used"]:
    ca[col] = ca[col].fillna(0)


def _churn_risk(row) -> float:
    if row["status"] == "Churned":
        return np.nan
    score = 0
    # No recent product engagement
    if pd.isna(row["days_since_last_access"]) or row["days_since_last_access"] > 30:
        score += 20
    # Very low total usage
    if row["total_usage_count"] < 10:
        score += 20
    # Poor satisfaction signal
    if not pd.isna(row["avg_satisfaction_score"]) and row["avg_satisfaction_score"] < 3.0:
        score += 15
    # High support burden
    if row["total_tickets"] > 5:
        score += 15
    # Very new customer
    if row["tenure_days"] < 90:
        score += 10
    # Lowest-tier plan
    if row["plan_type"] == "Basic":
        score += 10
    return float(max(0, min(100, score)))


ca["churn_risk_score"] = ca.apply(_churn_risk, axis=1)

active_risk = ca.loc[ca["status"] == "Active", "churn_risk_score"]
print(f"  customer_analytics rows:       {len(ca):,}")
print(f"  Columns:                       {len(ca.columns)}")
print(f"  Avg churn risk score (Active): {active_risk.mean():.1f}")
print(f"  High-risk active customers:    {(active_risk >= 50).sum():,}  (score >= 50)")


# ===============================================================================
# SECTION 10 -- Save processed CSVs and SQLite warehouse
# ===============================================================================
print(f"\n{SEP}\n  SECTION 10 -- Saving outputs\n{SEP}")

output_tables = {
    "customers_clean":       cust,
    "subscriptions_clean":   subs,
    "churn_clean":           churn,
    "marketing_clean":       camps,
    "support_clean":         tix,
    "product_usage_clean":   usg,
    "mrr_monthly":           mrr_monthly,
    "mrr_by_segment":        mrr_by_segment,
    "mrr_by_region":         mrr_by_region,
    "churn_by_segment":      churn_by_segment,
    "churn_by_region":       churn_by_region,
    "churn_monthly":         churn_monthly,
    "campaign_by_channel":   campaign_by_channel,
    "csat_monthly":          csat_monthly,
    "ticket_volume_monthly": ticket_volume_monthly,
    "feature_adoption":      feature_adoption,
    "mau_monthly":           mau_monthly,
    "cohort_retention":      cohort_retention,
    "customer_analytics":    ca,
}

# -- CSVs ----------------------------------------------------------------------
print("\n  Writing CSVs -> data/processed/")
for name, df in output_tables.items():
    path = os.path.join(PROCESSED_DIR, f"{name}.csv")
    df.to_csv(path, index=False)
    kb = os.path.getsize(path) / 1024
    print(f"    {name:<30}  {len(df):>8,} rows   {kb:>7.1f} KB")

# -- SQLite --------------------------------------------------------------------
print(f"\n  Writing SQLite warehouse -> {DB_PATH}")


def _prep_for_sql(df: pd.DataFrame) -> pd.DataFrame:
    """Convert datetime64 columns to ISO strings so SQLite stores them as TEXT."""
    df = df.copy()
    for col in df.select_dtypes(include=["datetime64[ns]"]).columns:
        df[col] = df[col].dt.strftime("%Y-%m-%d")   # NaT -> NaN -> NULL in SQLite
    return df


conn = sqlite3.connect(DB_PATH)
for name, df in output_tables.items():
    _prep_for_sql(df).to_sql(name, conn, if_exists="replace", index=False)
conn.close()

db_mb = Path(DB_PATH).stat().st_size / 1024 / 1024
print(f"  SQLite written: {db_mb:.1f} MB  ({len(output_tables)} tables)")

# -- Final Summary -------------------------------------------------------------
print(f"\n{SEP}")
print("  FINAL SUMMARY")
print(f"{SEP}")

print("\n>>  Top 5 customers by total revenue:")
top5 = (
    ca[["customer_id", "name", "segment", "plan_type", "total_revenue"]]
    .sort_values("total_revenue", ascending=False)
    .head(5)
)
top5["total_revenue"] = top5["total_revenue"].apply(lambda x: f"${x:,.2f}")
print(top5.to_string(index=False))

print("\n>>  Churn rate by segment:")
seg_display = churn_by_segment[
    ["segment", "total_customers", "churned_customers", "churn_rate_pct"]
].copy()
seg_display["churn_rate_pct"] = seg_display["churn_rate_pct"].apply(lambda x: f"{x:.2f}%")
print(seg_display.to_string(index=False))

print("\n>>  Top 3 features by adoption:")
top3 = feature_adoption.sort_values("unique_customers", ascending=False).head(3).copy()
top3["adoption_pct"] = top3["adoption_pct"].apply(lambda x: f"{x:.1f}%")
print(top3.to_string(index=False))

overall_csat = tix["satisfaction_score"].dropna().mean()
print(f"\n>>  Average CSAT score overall: {overall_csat:.2f} / 5.00")

latest_month = mrr_by_segment["invoice_month"].max()
latest_mrr   = mrr_by_segment.loc[
    mrr_by_segment["invoice_month"] == latest_month, "mrr"
].sum()
print(f"\n>>  MRR for latest month ({latest_month}): ${latest_mrr:,.2f}")

print(f"\n{SEP}")
print(f"  ETL complete -- {len(output_tables)} tables saved to CSV + SQLite")
print(f"{SEP}\n")
