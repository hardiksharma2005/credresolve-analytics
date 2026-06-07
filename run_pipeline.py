"""
CredResolve Analytics Pipeline Runner
Orchestrates all pipeline steps end-to-end.
Run from credresolve-analytics/ root: python run_pipeline.py
"""
import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# =============================================================================
# SECTION 1 -- Logging Setup
# =============================================================================
Path("logs").mkdir(exist_ok=True)

_formatter = logging.Formatter(
    fmt="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

_file_handler    = logging.FileHandler("logs/pipeline.log", encoding="utf-8")
_console_handler = logging.StreamHandler(sys.stdout)

_file_handler.setFormatter(_formatter)
_console_handler.setFormatter(_formatter)

logger = logging.getLogger("pipeline")
logger.setLevel(logging.INFO)
logger.addHandler(_file_handler)
logger.addHandler(_console_handler)


# =============================================================================
# SECTION 2 -- Pipeline Step Runner
# =============================================================================
def run_step(step_name: str, script_path: str, description: str):
    """Run a single pipeline step. Returns (success: bool, elapsed: float)."""
    logger.info("=" * 50)
    logger.info(f"STARTING: {step_name}")
    logger.info(f"Description: {description}")
    logger.info(f"Script: {script_path}")

    start_time = time.time()

    result = subprocess.run(
        [sys.executable, script_path],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=os.getcwd(),
    )

    elapsed = time.time() - start_time

    if result.returncode == 0:
        logger.info(f"[DONE] COMPLETED: {step_name} in {elapsed:.1f}s")
        stdout_lines = result.stdout.strip().split("\n")
        for line in stdout_lines[-5:]:
            if line.strip():
                logger.info(f"   > {line.strip()}")
        return True, elapsed
    else:
        logger.error(f"[FAIL] FAILED: {step_name} after {elapsed:.1f}s")
        stderr_tail = result.stderr[-500:] if result.stderr else "(no stderr)"
        logger.error(f"STDERR: {stderr_tail}")
        return False, elapsed


# =============================================================================
# SECTION 3 -- Pipeline Definition
# =============================================================================
PIPELINE_STEPS = [
    (
        "Data Generation",
        "data/generate_data.py",
        "Generate synthetic SaaS datasets",
    ),
    (
        "ETL & Integration",
        "src/etl.py",
        "Clean, transform and load all datasets to warehouse",
    ),
    (
        "Analytics Engine",
        "src/analytics.py",
        "Compute KPIs, cohorts, and analytical summaries",
    ),
    (
        "ML Models",
        "src/ml_models.py",
        "Train churn model and generate revenue forecast",
    ),
    (
        "Email Report",
        "src/email_report.py",
        "Generate HTML executive email report",
    ),
]


# =============================================================================
# SECTION 4 -- Main Pipeline Runner
# =============================================================================
def run_pipeline(skip_data_gen: bool = False, steps_only: list = None) -> bool:
    """
    Run the full analytics pipeline.

    skip_data_gen: if True, skip step 1 (useful when data already exists)
    steps_only:    list of step names to run (None = run all)
    """
    pipeline_start = time.time()

    logger.info("CREDRESOLVE ANALYTICS PIPELINE STARTING")
    logger.info(f"Run timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Working directory: {os.getcwd()}")
    logger.info(f"Python: {sys.version.split()[0]}")

    results = []

    for step_name, script_path, description in PIPELINE_STEPS:

        # Skip data generation if flag is set
        if skip_data_gen and step_name == "Data Generation":
            logger.info(f"[SKIP] SKIPPED: {step_name} (--skip-data-gen flag)")
            results.append((step_name, "SKIPPED", 0.0))
            continue

        # Skip if steps_only filter is active
        if steps_only and step_name not in steps_only:
            logger.info(f"[SKIP] SKIPPED: {step_name} (not in selected steps)")
            results.append((step_name, "SKIPPED", 0.0))
            continue

        success, elapsed = run_step(step_name, script_path, description)

        if success:
            results.append((step_name, "SUCCESS", elapsed))
        else:
            results.append((step_name, "FAILED", elapsed))
            logger.error(f"Pipeline halted at step: {step_name}")
            logger.error("Fix the error above and re-run.")
            break

    # ── Summary ───────────────────────────────────────────────────────────────
    total_time = time.time() - pipeline_start

    logger.info("")
    logger.info("=" * 60)
    logger.info("PIPELINE EXECUTION SUMMARY")
    logger.info("=" * 60)

    all_success = True
    for step_name, status, elapsed in results:
        icon = "[OK]  " if status == "SUCCESS" else "[SKIP]" if status == "SKIPPED" else "[FAIL]"
        logger.info(f"{icon} {step_name:<28} {status:<10} {elapsed:.1f}s")
        if status == "FAILED":
            all_success = False

    logger.info("-" * 60)
    logger.info(f"Total pipeline time: {total_time:.1f}s ({total_time/60:.1f} minutes)")

    if all_success:
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info(f"Report saved to: output/reports/executive_report_june2024.html")

        # Load and print key KPIs from saved outputs
        try:
            with open("data/processed/analytics_outputs/kpi_summary.json",
                      encoding="utf-8") as fh:
                kpi = json.load(fh)
            with open("data/processed/analytics_outputs/ml_summary.json",
                      encoding="utf-8") as fh:
                ml = json.load(fh)

            logger.info("")
            logger.info("KEY METRICS THIS RUN:")
            logger.info(f"   MRR:                   ${kpi['mrr']['current']:>14,.0f}")
            logger.info(f"   Churn Rate:             {kpi['churn']['overall_rate_pct']:>13.1f}%")
            logger.info(f"   CSAT:                   {kpi['support']['avg_csat']:>13.2f}/5.00")
            logger.info(f"   High-Risk Customers:    {kpi['risk']['high_risk_customers']:>13,}")
            logger.info(f"   ML Model AUC:           {ml['churn_model']['auc_roc']:>13.3f}")
            logger.info(
                f"   Jul 2024 MRR Forecast:  "
                f"${ml['revenue_forecast']['forecast_jul_2024']:>13,.0f}"
            )
        except Exception as exc:
            logger.warning(f"Could not load KPI summary: {exc}")

        return True

    else:
        logger.error("PIPELINE FAILED -- check logs above")
        return False


# =============================================================================
# SECTION 5 -- CLI Argument Parser
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CredResolve Analytics Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py                          Run full pipeline
  python run_pipeline.py --skip-data-gen          Skip data generation (data exists)
  python run_pipeline.py --steps ETL Analytics    Run specific steps only
  python run_pipeline.py --dry-run                Show steps without running
        """,
    )
    parser.add_argument(
        "--skip-data-gen",
        action="store_true",
        help="Skip data generation step (use existing raw data)",
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=[
            "Data Generation",
            "ETL & Integration",
            "Analytics Engine",
            "ML Models",
            "Email Report",
        ],
        metavar="STEP",
        help="Run only specific steps (space-separated step names)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print pipeline steps without executing",
    )

    args = parser.parse_args()

    if args.dry_run:
        print("\nPIPELINE STEPS (dry run):")
        for i, (name, script, desc) in enumerate(PIPELINE_STEPS, 1):
            skip_tag = "  [SKIP --skip-data-gen]" if args.skip_data_gen and name == "Data Generation" else ""
            print(f"  {i}. {name:<28} -> {script}{skip_tag}")
            print(f"     {desc}")
        print()
        sys.exit(0)

    success = run_pipeline(
        skip_data_gen=args.skip_data_gen,
        steps_only=args.steps,
    )

    sys.exit(0 if success else 1)
