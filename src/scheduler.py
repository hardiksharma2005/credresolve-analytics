"""
CredResolve Analytics Scheduler
Runs the analytics pipeline on a monthly schedule using APScheduler 3.x.
Run from credresolve-analytics/ root: python src/scheduler.py
Check last run status:             python src/scheduler.py status
"""
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# APScheduler 3.x imports
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

# =============================================================================
# SECTION 1 -- Logging Setup
# =============================================================================
# Project root is one level up from src/
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR      = os.path.join(PROJECT_ROOT, "logs")

Path(LOG_DIR).mkdir(parents=True, exist_ok=True)

_formatter = logging.Formatter(
    fmt="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

_file_handler    = logging.FileHandler(
    os.path.join(LOG_DIR, "scheduler.log"), encoding="utf-8"
)
_console_handler = logging.StreamHandler(sys.stdout)

_file_handler.setFormatter(_formatter)
_console_handler.setFormatter(_formatter)

logger = logging.getLogger("scheduler")
logger.setLevel(logging.INFO)
logger.addHandler(_file_handler)
logger.addHandler(_console_handler)


# =============================================================================
# SECTION 2 -- Monthly Pipeline Job
# =============================================================================
def run_monthly_pipeline() -> None:
    """
    Job function called by APScheduler on the 1st of each month at 06:00 IST.
    Runs run_pipeline.py --skip-data-gen in the project root directory.
    """
    run_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info("=" * 60)
    logger.info(f"MONTHLY PIPELINE JOB STARTED: {run_ts}")
    logger.info("Invoking: run_pipeline.py --skip-data-gen")

    pipeline_script = os.path.join(PROJECT_ROOT, "run_pipeline.py")

    try:
        result = subprocess.run(
            [sys.executable, pipeline_script, "--skip-data-gen"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=PROJECT_ROOT,
        )

        if result.returncode == 0:
            logger.info("[OK] Monthly pipeline completed successfully")
            # Log last few lines of stdout for quick visibility
            if result.stdout:
                for line in result.stdout.strip().split("\n")[-8:]:
                    if line.strip():
                        logger.info(f"   > {line.strip()}")
        else:
            logger.error("[FAIL] Monthly pipeline FAILED (exit code %d)", result.returncode)
            if result.stderr:
                logger.error("STDERR (last 500 chars):")
                logger.error(result.stderr[-500:])

    except Exception as exc:
        logger.error(f"[ERROR] Exception during monthly pipeline: {exc}")

    logger.info("=" * 60)


# =============================================================================
# SECTION 3 -- Pipeline Status Checker
# =============================================================================
def check_pipeline_status() -> None:
    """
    Read logs/pipeline.log and print a summary of the last pipeline run.
    Called when: python src/scheduler.py status
    """
    log_path = os.path.join(LOG_DIR, "pipeline.log")

    if not os.path.exists(log_path):
        print("No pipeline log found. The pipeline has not been run yet.")
        print(f"Expected at: {log_path}")
        return

    try:
        with open(log_path, encoding="utf-8") as fh:
            lines = fh.readlines()
    except Exception as exc:
        print(f"Error reading pipeline log: {exc}")
        return

    if not lines:
        print("Pipeline log is empty.")
        return

    # Find the last occurrence of "PIPELINE EXECUTION SUMMARY"
    summary_start = None
    for i in range(len(lines) - 1, -1, -1):
        if "PIPELINE EXECUTION SUMMARY" in lines[i]:
            summary_start = i
            break

    if summary_start is None:
        print("No completed pipeline run found in the log.")
        # Print last 10 lines anyway
        print("\nLast 10 log lines:")
        for line in lines[-10:]:
            print(line.rstrip())
        return

    # Print from the summary header to the end of that run block
    print("\n" + "=" * 60)
    print("LAST PIPELINE RUN SUMMARY")
    print("=" * 60)
    # Print summary block (up to 20 lines after the header)
    for line in lines[summary_start : summary_start + 20]:
        print(line.rstrip())

    # Also find and print the timestamp of that run
    for i in range(summary_start, -1, -1):
        if "PIPELINE STARTING" in lines[i]:
            print(f"\nRun timestamp: {lines[i].split('|')[0].strip()}")
            break


# =============================================================================
# SECTION 4 -- Scheduler Entry Point
# =============================================================================
def start_scheduler() -> None:
    """
    Configure and start the blocking APScheduler.
    Runs run_monthly_pipeline() on the 1st of every month at 06:00 IST.
    """
    logger.info("=" * 60)
    logger.info("CREDRESOLVE ANALYTICS SCHEDULER STARTING")
    logger.info(f"Project root: {PROJECT_ROOT}")
    logger.info(f"Log directory: {LOG_DIR}")

    # Timezone: Asia/Kolkata (IST, UTC+5:30)
    # APScheduler 3.x accepts IANA timezone strings directly via pytz.
    try:
        scheduler = BlockingScheduler(timezone="Asia/Kolkata")
    except Exception:
        # Fallback: let APScheduler use UTC if pytz/zoneinfo isn't available
        logger.warning("Could not set Asia/Kolkata timezone, falling back to UTC")
        scheduler = BlockingScheduler(timezone="UTC")

    # Cron: 1st of every month at 06:00
    trigger = CronTrigger(day=1, hour=6, minute=0)

    scheduler.add_job(
        run_monthly_pipeline,
        trigger=trigger,
        id="monthly_analytics_pipeline",
        name="Monthly CredResolve Analytics Pipeline",
        replace_existing=True,
        misfire_grace_time=3600,    # Allow job to be up to 1 hour late (e.g., server restart)
        max_instances=1,            # Prevent overlapping runs
    )

    # Log the next 3 scheduled run times
    logger.info("Scheduled job: Monthly CredResolve Analytics Pipeline")
    logger.info("Schedule: 1st of every month at 06:00 IST")
    try:
        job = scheduler.get_job("monthly_analytics_pipeline")
        if job and job.next_run_time:
            logger.info(f"Next run: {job.next_run_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    except Exception:
        pass

    logger.info("Scheduler is running. Press Ctrl+C to stop.")
    logger.info("=" * 60)

    try:
        scheduler.start()
    except KeyboardInterrupt:
        logger.info("Scheduler stopped by user (KeyboardInterrupt).")
        scheduler.shutdown(wait=False)
        logger.info("Scheduler shut down cleanly.")


# =============================================================================
# SECTION 5 -- CLI
# =============================================================================
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1].lower() == "status":
        check_pipeline_status()
    else:
        start_scheduler()
