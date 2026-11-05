# main.py
# يشغل المشروع كاملاً بأمر واحد: python main.py
# الخيارات:
#   --skip-eda      يتخطى رسوم التحليل الاستكشافي (اليومان 5 و6)
#   --from-scratch  يحذف data/processed قبل البدء
# كل خطوة تعمل في عملية مستقلة، وأي فشل يوقف الخط فوراً.

import shutil
import subprocess
import sys
import time

from src import config

STEPS = [
    ("Audit raw tables and ERD", "run_day1.py", False),
    ("Clean tables", "run_day2.py", False),
    ("Build one-row-per-order table", "run_day3.py", False),
    ("Target, eligibility and features", "run_day4.py", False),
    ("EDA questions 1-3", "run_day5.py", True),
    ("EDA questions 4-6", "run_day6.py", True),
    ("Split, baseline and two models", "run_day7.py", False),
    ("Validation: threshold and model selection", "run_day8_validation.py", False),
    ("Final test evaluation and error analysis", "run_day8_test.py", False),
]


def run_step(index, total, title, script, extra_args):
    print(f"\n[{index}/{total}] {title}  ({script})")
    start = time.perf_counter()
    cmd = [sys.executable, script] + extra_args
    result = subprocess.run(cmd, cwd=config.PROJECT_ROOT,
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    seconds = time.perf_counter() - start
    if result.returncode != 0:
        print(f"    FAILED after {seconds:.0f}s")
        print(result.stderr[-2000:])
        raise SystemExit(f"Pipeline stopped at: {script}")
    print(f"    OK in {seconds:.0f}s")
    return seconds


def main():
    skip_eda = "--skip-eda" in sys.argv
    if "--from-scratch" in sys.argv and config.PROCESSED_DIR.exists():
        shutil.rmtree(config.PROCESSED_DIR)
        print("Removed data/processed (rebuilding everything from the raw files).")

    # تجميد جديد للقرارات: إعادة الإنتاج الكاملة تعيد المرحلتين معاً
    for stale in (config.DECISIONS_FILE, config.TEST_METRICS_FILE):
        if stale.exists():
            stale.unlink()

    steps = [s for s in STEPS if not (skip_eda and s[2])]
    total = len(steps)
    print(f"Running {total} steps. Raw data in {config.RAW_DIR}")
    elapsed = sum(run_step(i, total, title, script, [])
                  for i, (title, script, _eda) in enumerate(steps, start=1))
    print(f"\nPipeline finished in {elapsed / 60:.1f} minutes.")
    print(f"Tables: {config.TABLES_DIR}")
    print(f"Figures: {config.FIGURES_DIR}")
    print(f"Final test metrics: {config.TEST_METRICS_FILE}")


if __name__ == "__main__":
    main()