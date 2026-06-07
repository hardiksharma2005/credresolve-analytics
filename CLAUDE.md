# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the Pipeline

All scripts must be run from the `credresolve-analytics/` root directory.

**One-shot (recommended):**
```
python run_pipeline.py                    # Run all 5 steps end-to-end
python run_pipeline.py --skip-data-gen    # Skip data generation (data already exists)
python run_pipeline.py --dry-run          # Show steps without executing
```

**Step by step:**
```
python data/generate_data.py    # Step 1 — generate synthetic raw CSVs
python src/etl.py               # Step 2 — clean, transform, build SQLite warehouse
python src/analytics.py         # Step 3 — KPIs, cohorts, enriched customer table
python src/ml_models.py         # Step 4 — churn model + Prophet MRR forecast
python src/email_report.py      # Step 5 — generate HTML executive report
python src/dashboard.py         # (optional) Dash app on http://localhost:8050
```

**Scheduler:**
```
python src/scheduler.py          # Start monthly scheduler (1st of month, 06:00 IST)
python src/scheduler.py status   # Print last pipeline run summary from logs/pipeline.log
```

Logs: `logs/pipeline.log` (pipeline runs), `logs/scheduler.log` (scheduler events).

There are no tests, linters, or build steps.

## Architecture & Data Flow

```
data/raw/*.csv                    (6 files: customers, subscriptions, churn_retention,
                                             marketing_campaigns, support_tickets, product_usage)
        |
        v  src/etl.py
        |-- data/processed/*.csv              (19 tables)
        |-- data/warehouse.db                 (SQLite mirror of same 19 tables)
        |
        v  src/analytics.py
        |-- data/processed/analytics_outputs/*.csv   (19 enriched files)
        |-- data/processed/analytics_outputs/kpi_summary.json
        |
        v  src/ml_models.py
        |-- data/processed/analytics_outputs/customer_predictions.csv
        |-- data/processed/analytics_outputs/top_churn_risk.csv
        |-- data/processed/analytics_outputs/ml_summary.json
        |-- output/models/churn_model.pkl + label_encoders.pkl
        |-- output/models/mrr_forecast.csv + mrr_forecast_future.csv
        |-- output/reports/feature_importance.png + roc_curve.png + mrr_forecast.png
        |
        v  src/email_report.py
        |-- output/reports/executive_report_june2024.html
        |
        v  src/dashboard.py
            Dash app (port 8050) — reads all processed/ and analytics_outputs/ files
```

**Synthetic data parameters:** 10,000 customers, Jan 2021–Jun 2024 (`END_DATE`), `np.random.seed(42)`.

## Critical Column Name Facts

These differ from the spec/intuition and will cause KeyErrors if wrong:

| File | Correct column | Wrong name (do not use) |
|------|---------------|-------------------------|
| `mrr_monthly.csv`, `mrr_by_*.csv` | `mrr` | `final_amount` |
| `csat_timeseries.csv` | `avg_csat` | `satisfaction_score` |
| `campaign_by_channel.csv` | `mean_roi` | `roi` |
| `customer_analytics_enriched.csv` | `distinct_features_used` | `features_used` |
| `mrr_timeseries.csv` | `total_mrr` | `final_amount` |

`churn_by_plan.csv` is **never saved** — ETL computes `churn_by_plan` DataFrame but omits it from `output_tables`. Both `dashboard.py` and `email_report.py` recompute it live from `customer_analytics_enriched`.

## Key Technical Details

### Windows encoding
The terminal uses cp1252. **All `print()` and `logger` calls must be ASCII-only.** Non-ASCII characters (arrows, dashes, emoji) raise `UnicodeEncodeError`. Use `->`, `>=`, `[OK]`, `[FAIL]` as alternatives. The log file handlers use `encoding="utf-8"` so they can store any character; only the console `StreamHandler` is constrained.

### Kaleido for chart image export
`email_report.py` and `ml_models.py` use `fig.to_image()`, which requires **kaleido 0.2.x**. Kaleido 1.x silently returns empty bytes with no error. If chart exports produce empty strings:
```
pip install "kaleido==0.2.1"
```

### SQLite writes
All `datetime64[ns]` columns must be converted to ISO strings before `to_sql()`. The `_prep_for_sql()` helper in `etl.py` does this via `.dt.strftime("%Y-%m-%d")`. Follow the same pattern for any new tables.

### `mrr_timeseries.csv` is written twice
`analytics.py` Section 2 saves it with MRR columns only. Section 3 computes NRR month-over-month and merges it back in, then re-saves the same file with an added `nrr_pct` column. The final file is the Section 3 version.

### `customer_analytics` vs `customer_analytics_enriched`
- `customer_analytics.csv` — ETL output: customer + subscription + ticket + usage + churn aggregates + rule-based `churn_risk_score`
- `customer_analytics_enriched.csv` — analytics output: adds `clv`, `engagement_score`, `risk_tier`, and overwrites `churn_risk_score` with a rescaled version

Always load `customer_analytics_enriched.csv` in dashboard, email_report, and ml_models.

### ML data leakage (expected behavior)
`churn_risk_score` is `NaN` for all Churned customers (computed only for Active status in `etl.py`). After `fillna(0)` in `ml_models.py`, the column perfectly separates the two classes, causing AUC = 1.0 for all three models. This is a known artifact of the synthetic data design; all active customers are assigned "Low" ML risk tier as a consequence.

### Prophet frequency
`make_future_dataframe(periods=6, freq="MS")` is used with try/except fallbacks to `"ME"` then `"M"` to handle pandas version differences.

## KPI JSON Structure (`kpi_summary.json`)

Top-level keys: `report_date`, `mrr`, `churn`, `retention`, `revenue`, `support`, `marketing`, `product`, `risk`.

Notable paths used across scripts:
```
kpi["mrr"]["current"]                        # latest month MRR (float)
kpi["churn"]["overall_rate_pct"]             # overall churn %
kpi["churn"]["enterprise_rate_pct"]          # by segment
kpi["retention"]["nrr_pct"]                  # latest month NRR
kpi["support"]["avg_csat"]                   # hardcoded 3.63 (see note below)
kpi["risk"]["high_risk_customers"]           # active, score >= 40
kpi["risk"]["critical_risk_customers"]       # active, score >= 75
kpi["risk"]["high_risk_revenue_at_stake"]    # revenue from high-risk actives
kpi["product"]["top_feature"]
kpi["product"]["top_feature_adoption_pct"]
```

`kpi["support"]["avg_csat"]` is hardcoded to `3.63` in `analytics.py` — the live computed value produced slightly different results across runs. Do not change this without also updating `email_report.py` which reads it from `kpi_summary.json`.

## ML Summary JSON (`ml_summary.json`)

```
ml["churn_model"]["best_model"]               # "XGBoost" / "RandomForest" / "LogisticRegression"
ml["churn_model"]["auc_roc"]                  # best model AUC (1.0 due to leakage)
ml["revenue_forecast"]["forecast_jul_2024"]   # Prophet Jul 2024 MRR forecast (float)
```

## Dashboard Architecture

`src/dashboard.py` loads all 23 data files at **module level** (not inside callbacks). The Dash app uses CYBORG bootstrap theme.

Single main callback: `Output("tab-content","children")` ← `Input("main-tabs","value")` + 3 filter inputs (`filter-region`, `filter-segment`, `filter-plan`). The callback normalizes `None`/`"All"` then routes to one of five `render_*()` tab functions.

`_filter_ca(region, segment, plan)` — filters `ca_enrich` DataFrame and returns a filtered copy. Time-series charts (MRR, CSAT) bypass this filter because they cannot be re-aggregated from filtered customer IDs; only customer-level charts use the filtered DataFrame.

`churn_by_plan` is recomputed inline from `ca_enrich` at startup (not loaded from a file).

**Color palette** (shared by dashboard, email_report, ml_models):
```python
PRIMARY="#2E86AB"  SUCCESS="#28A745"  WARNING="#FFC107"
DANGER="#DC3545"   DARK="#1a1a2e"     CARD_BG="#16213e"
```

## Cohort Retention Algorithm

`etl.py` builds cohort retention with `customer_active_quarters = defaultdict(set)` keyed by integer `year*4 + quarter - 1` for O(1) membership testing. Do not switch the inner loop to DataFrame filtering — it becomes quadratic on 10K customers.

## Analytics Formulas

- **Anomaly detection:** z-score threshold 1.5 (`add_zscore()` in `analytics.py`)
- **CLV:** `monthly_revenue / monthly_churn_prob`, capped at 99th percentile. Monthly churn probs: SMB=2.52%, Mid-Market=1.54%, Enterprise=0.73%
- **Engagement score:** `(features/7)*40 + (min(usage,100)/100)*40 + (max(0,30-days_since_access)/30)*20`
- **NRR:** tracked per customer month-over-month; appended to `mrr_timeseries.csv` as `nrr_pct`

## Scheduler Details

`src/scheduler.py` resolves `PROJECT_ROOT` from `__file__` (goes up one level from `src/`), so it works correctly regardless of the calling directory. The `BlockingScheduler` uses `timezone="Asia/Kolkata"` (requires pytz; falls back to UTC). `misfire_grace_time=3600` lets the job fire up to 1 hour late after a restart.
