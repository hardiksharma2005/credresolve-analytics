# CredResolve Analytics Platform
### Senior Data Analyst Assignment Submission

A production-grade SaaS analytics platform featuring multi-source data integration,
interactive executive dashboards, ML-powered churn prediction, and automated reporting.

---

## 📊 What's Built

| Component | Technology | Description |
|-----------|-----------|-------------|
| Data Pipeline | Python, pandas | 6-source ETL → SQLite warehouse |
| Analytics Engine | scipy, statsmodels | KPIs, cohorts, anomaly detection |
| Interactive Dashboard | Plotly Dash | 5-tab dashboard, filters, drill-downs |
| Email Report | Jinja2, HTML/CSS | Self-contained executive report |
| Churn Prediction | XGBoost | Binary classifier, AUC 1.000* |
| Revenue Forecast | Prophet | 6-month MRR forecast, 95% CI |
| Automation | APScheduler | Monthly pipeline, cron scheduling |

*Note: AUC reflects synthetic data structure. See [narrative](insights/narrative.md#8-data--methodology-notes) for details.

---


## 📈 Key Metrics (June 2024)

| KPI | Value |
|-----|-------|
| Current MRR | $3.82M |
| YTD MRR Growth | +11.3% |
| Overall Churn Rate | 23.4% |
| SMB Churn Rate | 30.2% 🔴 |
| Enterprise Churn Rate | 8.8% 🟢 |
| CSAT Score | 3.63 / 5.00 |
| Net Revenue Retention | 99.8% |
| Dec 2024 MRR Forecast | $4.43M |
| High-Risk Customers | ML-scored, see dashboard |

---

## 🤖 ML Models

### Churn Prediction (XGBoost)
- 19 behavioral + demographic features
- Train/test split: 80/20, stratified
- Key features: engagement score, days since last access, usage count, tenure
- Output: churn probability (0–100%) for every active customer
- Risk tiers: Critical (≥75%) / High (50–74%) / Medium (30–49%) / Low (<30%)

### Revenue Forecasting (Prophet)
- Trained on 42 months of historical MRR data
- 6-month forward forecast with 95% confidence intervals
- July 2024: $3.99M | December 2024: $4.43M
- Expected growth: +16.0% over current MRR

---
*Built with Python 3.13 | Plotly Dash | XGBoost | Prophet | SQLite*
