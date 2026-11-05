# collect_outputs.py
# يقرأ جداول النتائج الجاهزة ويجمعها في ملف نصي واحد (للقراءة فقط، لا يعدل شيئاً).
# التشغيل: python collect_outputs.py

import json
import platform
import subprocess
import sys
from importlib.metadata import version

import pandas as pd

from src import config

TABLES = [
    # اليوم 3
    "join_log", "naive_join_demo", "order_table_validation", "seller_count_distribution",
    "orders_without_items_by_status", "payment_reconciliation", "shipping_limit_checks",
    # اليوم 4
    "exclusion_summary", "target_distribution", "delay_description",
    "numpy_feature_summary", "log_transform_effect", "feature_missing_eligible",
    # اليوم 7
    "split_summary", "experiment_log", "imputation_statistics_check",
    # اليوم 8: التحقق
    "val_threshold_choice", "val_model_selection", "val_metrics", "val_topk", "val_calibration",
    # اليوم 8: الاختبار النهائي
    "test_metrics", "test_calibration", "test_topk", "test_error_by_month",
    "test_error_by_route", "test_error_by_planned_window", "test_error_by_seller_structure",
    "test_error_by_customer_state", "test_outcome_profile",
    "test_worst_false_positives", "test_worst_false_negatives", "test_logistic_coefficients",
]


def run(cmd):
    try:
        r = subprocess.run(cmd, cwd=config.PROJECT_ROOT, capture_output=True, text=True, timeout=900)
        return (r.stdout + r.stderr).strip()
    except Exception as e:  # noqa: BLE001
        return f"(failed: {e})"


def main():
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 50)
    pd.set_option("display.max_colwidth", 70)
    lines = []

    def section(title):
        lines.append("\n" + "=" * 100)
        lines.append(title)
        lines.append("=" * 100)

    section("ENVIRONMENT")
    lines.append(f"Python {platform.python_version()} on {platform.system()} {platform.release()}")
    for pkg in ["numpy", "pandas", "matplotlib", "seaborn", "scikit-learn", "joblib", "pytest"]:
        try:
            lines.append(f"{pkg}=={version(pkg)}")
        except Exception:  # noqa: BLE001
            lines.append(f"{pkg}: not found")

    section("FROZEN DECISIONS (json)")
    if config.DECISIONS_FILE.exists():
        lines.append(json.dumps(json.loads(config.DECISIONS_FILE.read_text(encoding="utf-8")), indent=2))
    else:
        lines.append("missing: run run_day8_validation.py")

    for name in TABLES:
        section(f"TABLE: {name}")
        path = config.TABLES_DIR / f"{name}.csv"
        if path.exists():
            lines.append(pd.read_csv(path).round(4).to_string(index=False))
        else:
            lines.append("MISSING FILE")

    section("FIGURES FOUND")
    lines.append("\n".join(sorted(p.name for p in config.FIGURES_DIR.glob("*.png"))))

    section("PROCESSED FILES (name, size in MB)")
    for p in sorted(config.PROCESSED_DIR.rglob("*")):
        if p.is_file():
            lines.append(f"{p.relative_to(config.PROJECT_ROOT)}  {p.stat().st_size / 1e6:.2f}")

    section("GIT LOG (oneline)")
    lines.append(run(["git", "log", "--oneline", "--no-decorate"]))
    section("GIT TAGS")
    lines.append(run(["git", "tag"]))
    section("GIT COMMITS PER AUTHOR")
    lines.append(run(["git", "shortlog", "-sn", "--all"]))

    section("PYTEST (summary)")
    lines.append(run([sys.executable, "-m", "pytest", "-q", "--tb=short"])[-1500:])

    out = config.PROJECT_ROOT / "outputs" / "collected_outputs.txt"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved: {out}  ({len(lines)} lines)")


if __name__ == "__main__":
    main()