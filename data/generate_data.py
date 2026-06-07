import os
import random
import numpy as np
import pandas as pd
from faker import Faker
from datetime import datetime, timedelta

np.random.seed(42)
random.seed(42)
fake = Faker()
Faker.seed(42)

RAW_DIR = os.path.join(os.path.dirname(__file__), "raw")
os.makedirs(RAW_DIR, exist_ok=True)

START_DATE = datetime(2021, 1, 1)
END_DATE = datetime(2024, 6, 1)


def random_date(start, end):
    return start + timedelta(days=random.randint(0, (end - start).days))


# ──────────────────────────────────────────────
# Dataset 1: customers.csv
# ──────────────────────────────────────────────
def generate_customers(n=10_000):
    segments = ["Enterprise", "Mid-Market", "SMB"]
    seg_weights = [0.15, 0.30, 0.55]
    regions = ["North", "South", "East", "West", "International"]
    reg_weights = [0.25, 0.20, 0.20, 0.20, 0.15]
    plans = ["Basic", "Pro", "Enterprise"]
    plan_weights = [0.40, 0.35, 0.25]
    churn_prob = {"Enterprise": 0.08, "Mid-Market": 0.18, "SMB": 0.30}

    rows = []
    for i in range(1, n + 1):
        segment = random.choices(segments, weights=seg_weights)[0]
        plan = random.choices(plans, weights=plan_weights)[0]
        region = random.choices(regions, weights=reg_weights)[0]
        signup = random_date(START_DATE, END_DATE)
        churned = random.random() < churn_prob[segment]
        rows.append({
            "customer_id": f"CUST_{i:05d}",
            "name": fake.name(),
            "signup_date": signup.date(),
            "segment": segment,
            "region": region,
            "plan_type": plan,
            "status": "Churned" if churned else "Active",
            "email": fake.email(),
        })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RAW_DIR, "customers.csv"), index=False)
    return df


# ──────────────────────────────────────────────
# Dataset 2: subscriptions.csv
# ──────────────────────────────────────────────
def generate_subscriptions(customers_df):
    plan_means = {"Basic": 99, "Pro": 299, "Enterprise": 999}
    plan_sigma = {"Basic": 0.15, "Pro": 0.15, "Enterprise": 0.12}
    discount_vals = [0, 10, 20]
    discount_weights = [0.70, 0.20, 0.10]

    rows = []
    invoice_num = 1

    for _, cust in customers_df.iterrows():
        signup = datetime.combine(cust["signup_date"], datetime.min.time())
        is_churned = cust["status"] == "Churned"
        plan = cust["plan_type"]

        # generate monthly invoices from signup up to END_DATE
        current = signup.replace(day=1)
        invoices = []
        while current <= END_DATE:
            invoices.append(current)
            # advance one month
            month = current.month + 1
            year = current.year + (month - 1) // 12
            month = ((month - 1) % 12) + 1
            current = current.replace(year=year, month=month)

        for idx, inv_date in enumerate(invoices):
            is_last = idx == len(invoices) - 1
            mean = plan_means[plan]
            sigma = plan_sigma[plan]
            amount = float(np.random.lognormal(np.log(mean), sigma))
            amount = round(amount + np.random.uniform(-5, 5), 2)
            discount = random.choices(discount_vals, weights=discount_weights)[0]
            final = round(amount * (1 - discount / 100), 2)

            if is_churned and is_last:
                renewal = "Churned"
            elif is_churned and not is_last:
                renewal = "Renewed"
            else:
                renewal = "Renewed"

            rows.append({
                "invoice_id": f"INV_{invoice_num:06d}",
                "customer_id": cust["customer_id"],
                "invoice_date": inv_date.date(),
                "amount": amount,
                "renewal_status": renewal,
                "discount_pct": discount,
                "final_amount": final,
            })
            invoice_num += 1

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RAW_DIR, "subscriptions.csv"), index=False)
    return df


# ──────────────────────────────────────────────
# Dataset 3: churn_retention.csv
# ──────────────────────────────────────────────
def generate_churn_retention(customers_df, subscriptions_df):
    churn_reasons = ["Price", "Competitor", "Product Gap", "Support Issues", "Business Closure"]
    reason_weights = [0.30, 0.25, 0.20, 0.15, 0.10]

    churned = customers_df[customers_df["status"] == "Churned"]["customer_id"].tolist()

    last_inv = (
        subscriptions_df[subscriptions_df["customer_id"].isin(churned)]
        .groupby("customer_id")["invoice_date"]
        .max()
        .reset_index()
        .rename(columns={"invoice_date": "last_invoice_date"})
    )

    rows = []
    for _, row in last_inv.iterrows():
        last_inv_dt = datetime.combine(row["last_invoice_date"], datetime.min.time())
        max_churn = min(last_inv_dt + timedelta(days=90), END_DATE)
        if max_churn <= last_inv_dt:
            max_churn = last_inv_dt + timedelta(days=30)
        churn_date = random_date(last_inv_dt + timedelta(days=1), max_churn)
        days_before = random.randint(7, 60)
        last_activity = churn_date - timedelta(days=days_before)
        rows.append({
            "customer_id": row["customer_id"],
            "churn_date": churn_date.date(),
            "churn_reason": random.choices(churn_reasons, weights=reason_weights)[0],
            "last_activity_date": last_activity.date(),
            "days_since_last_activity": days_before,
        })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RAW_DIR, "churn_retention.csv"), index=False)
    return df


# ──────────────────────────────────────────────
# Dataset 4: marketing_campaigns.csv
# ──────────────────────────────────────────────
def generate_marketing_campaigns(n=500):
    channels = ["Email", "Paid Search", "Social Media", "Webinar", "Content Marketing"]
    chan_weights = [0.30, 0.25, 0.20, 0.15, 0.10]
    segments = ["Enterprise", "Mid-Market", "SMB", "All"]
    deal_size = {"Enterprise": 15000, "Mid-Market": 5000, "SMB": 1200, "All": 4000}

    rows = []
    camp_start = datetime(2023, 1, 1)
    camp_end = datetime(2024, 3, 1)

    for i in range(1, n + 1):
        channel = random.choices(channels, weights=chan_weights)[0]
        segment = random.choices(segments)[0]
        impressions = random.randint(1000, 500_000)
        ctr = round(random.uniform(0.01, 0.15), 4)
        clicks = int(impressions * ctr)
        lead_rate = random.uniform(0.05, 0.25)
        leads = int(clicks * lead_rate)
        conv_rate = random.uniform(0.05, 0.20)
        conversions = int(leads * conv_rate)
        cost = random.randint(500, 50_000)
        revenue = conversions * deal_size[segment] * random.uniform(0.8, 1.2)
        revenue = round(revenue, 2)
        roi = round((revenue - cost) / cost * 100, 2) if cost > 0 else 0.0
        start = random_date(camp_start, camp_end)
        end = start + timedelta(days=random.randint(7, 60))
        rows.append({
            "campaign_id": f"CAMP_{i:03d}",
            "channel": channel,
            "target_segment": segment,
            "impressions": impressions,
            "ctr": ctr,
            "clicks": clicks,
            "leads": leads,
            "conversions": conversions,
            "cost": cost,
            "revenue_generated": revenue,
            "roi": roi,
            "start_date": start.date(),
            "end_date": end.date(),
        })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RAW_DIR, "marketing_campaigns.csv"), index=False)
    return df


# ──────────────────────────────────────────────
# Dataset 5: support_tickets.csv
# ──────────────────────────────────────────────
def generate_support_tickets(customers_df, n=20_000):
    categories = ["Billing", "Technical", "Feature Request", "Account", "Bug Report"]
    cat_weights = [0.25, 0.30, 0.20, 0.15, 0.10]
    priorities = ["Low", "Medium", "High", "Critical"]
    pri_weights = [0.30, 0.40, 0.20, 0.10]
    resolution_hours = {
        "Low": (24, 72),
        "Medium": (8, 48),
        "High": (2, 24),
        "Critical": (1, 12),
    }
    # weight active customers higher for tickets
    active = customers_df[customers_df["status"] == "Active"]["customer_id"].tolist()
    churned = customers_df[customers_df["status"] == "Churned"]["customer_id"].tolist()
    all_ids = active * 3 + churned  # active 3x more likely

    ticket_start = datetime(2022, 1, 1)

    rows = []
    for i in range(1, n + 1):
        cust_id = random.choice(all_ids)
        category = random.choices(categories, weights=cat_weights)[0]
        priority = random.choices(priorities, weights=pri_weights)[0]
        created = random_date(ticket_start, END_DATE)
        lo, hi = resolution_hours[priority]
        res_hours = random.randint(lo, hi)
        # 5% open, 100% of Critical occasionally stays open
        is_open = random.random() < 0.05
        resolved = None if is_open else (created + timedelta(hours=res_hours)).date()

        # satisfaction: base score influenced by resolution speed and category
        base_score = 5 - (res_hours / hi) * 2
        if category in ("Billing", "Bug Report"):
            base_score -= 0.5
        sat = int(round(max(1, min(5, base_score + random.uniform(-0.5, 0.5)))))

        rows.append({
            "ticket_id": f"TKT_{i:06d}",
            "customer_id": cust_id,
            "category": category,
            "priority": priority,
            "created_date": created.date(),
            "resolved_date": resolved,
            "resolution_time_hours": res_hours if not is_open else None,
            "satisfaction_score": sat if not is_open else None,
            "status": "Open" if is_open else "Resolved",
        })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RAW_DIR, "support_tickets.csv"), index=False)
    return df


# ──────────────────────────────────────────────
# Dataset 6: product_usage.csv
# ──────────────────────────────────────────────
def generate_product_usage(customers_df, n=50_000):
    features = ["Dashboard", "Analytics", "Reports", "API", "Integrations", "Alerts", "Export"]
    feat_weights = [0.25, 0.20, 0.18, 0.12, 0.10, 0.08, 0.07]
    usage_start = datetime(2023, 1, 1)

    # plan → usage multiplier
    plan_mult = {"Basic": 1, "Pro": 2, "Enterprise": 4}

    cust_plan = customers_df.set_index("customer_id")["plan_type"].to_dict()
    all_ids = customers_df["customer_id"].tolist()

    rows = []
    for i in range(1, n + 1):
        cust_id = random.choice(all_ids)
        plan = cust_plan.get(cust_id, "Basic")
        mult = plan_mult[plan]
        feature = random.choices(features, weights=feat_weights)[0]
        usage_count = int(np.random.poisson(lam=10 * mult))
        usage_count = max(1, usage_count)
        last_access = random_date(usage_start, END_DATE)
        session_minutes = round(float(np.random.lognormal(mean=3.0, sigma=0.8)), 1)
        session_minutes = max(1.0, min(180.0, session_minutes))
        rows.append({
            "usage_id": f"USG_{i:06d}",
            "customer_id": cust_id,
            "feature": feature,
            "usage_count": usage_count,
            "last_access_date": last_access.date(),
            "session_duration_minutes": session_minutes,
            "month_year": last_access.strftime("%Y-%m"),
        })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RAW_DIR, "product_usage.csv"), index=False)
    return df


# ──────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────
def print_summary(datasets: dict):
    sep = "=" * 60
    print(f"\n{sep}")
    print("  DATA GENERATION SUMMARY")
    print(sep)

    for name, df in datasets.items():
        path = os.path.join(RAW_DIR, name)
        size_kb = os.path.getsize(path) / 1024
        nulls = df.isnull().sum()
        null_cols = nulls[nulls > 0].to_dict()

        print(f"\n{name}")
        print(f"  Rows      : {len(df):,}")
        print(f"  Columns   : {len(df.columns)}")
        print(f"  File size : {size_kb:.1f} KB")
        if null_cols:
            print(f"  Nulls     : {null_cols}")
        else:
            print("  Nulls     : none")
        print("  First 3 rows:")
        print(df.head(3).to_string(index=False))

    print(f"\n{sep}\n")


if __name__ == "__main__":
    print("Generating customers...")
    customers = generate_customers()

    print("Generating subscriptions (this may take a moment)...")
    subscriptions = generate_subscriptions(customers)

    print("Generating churn/retention records...")
    churn = generate_churn_retention(customers, subscriptions)

    print("Generating marketing campaigns...")
    campaigns = generate_marketing_campaigns()

    print("Generating support tickets...")
    tickets = generate_support_tickets(customers)

    print("Generating product usage...")
    usage = generate_product_usage(customers)

    datasets = {
        "customers.csv": customers,
        "subscriptions.csv": subscriptions,
        "churn_retention.csv": churn,
        "marketing_campaigns.csv": campaigns,
        "support_tickets.csv": tickets,
        "product_usage.csv": usage,
    }

    print_summary(datasets)
    print("All datasets saved to:", RAW_DIR)
