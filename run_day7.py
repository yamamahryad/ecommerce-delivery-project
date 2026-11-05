# run_day7.py
# التقسيم الزمني + خط المعالجة + المرجع + النموذجان: python run_day7.py
# يتطلب تشغيل run_day4.py أولاً. الاختبار النهائي لا يستخدم في هذا الملف.

import pandas as pd

from src import config
from src.features import load_modeling_table
from src.modeling import (BASELINE, HGB, LOGISTIC, learned_medians, run_experiments,
                          save_models, save_val_predictions)
from src.splitting import split_by_month, split_summary, temporal_split


def show(title, df):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)
    print(df.to_string(index=False))


def main():
    pd.set_option("display.width", 220)
    pd.set_option("display.max_colwidth", 60)
    config.ensure_output_dirs()

    modeling = load_modeling_table()
    splits = temporal_split(modeling)
    summary = split_summary(splits)
    by_month = split_by_month(splits)

    show("1) Temporal split by order_approved_at (test = newest period, untouched today)", summary)
    show("2) Months inside each split (distribution shift)", by_month)

    print("\nTraining and comparing candidates (HistGradientBoosting takes a few minutes)...")
    log, selected = run_experiments(splits["train"], splits["val"])

    show("3) Experiment log: all candidates, scored on VALIDATION only",
         log[["model", "settings", "val_average_precision", "val_roc_auc", "ap_lift",
              "train_average_precision", "fit_seconds", "selected"]])
    show("4) Selected setting per model", log[log["selected"]]
         [["model", "settings", "val_average_precision", "ap_lift"]])

    # إثبات أن خط المعالجة ضُبط على التدريب فقط
    medians = learned_medians(selected[LOGISTIC][0])
    check = pd.DataFrame([{
        "feature": f,
        "median_learned_from_train": round(float(medians[f]), 3),
        "median_of_all_eligible_orders": round(float(modeling[f].median()), 3),
    } for f in ["planned_days", "items_price_sum", "purchase_to_approval_hours", "log_weight_g"]])
    show("5) Imputation statistics come from TRAIN only", check)

    val_rate = summary.loc[summary["split"] == "val", "late_rate_pct"].iloc[0]
    print(f"\nReference: the baseline AP on validation equals the validation late rate ({val_rate}%).")

    out = config.TABLES_DIR
    summary.to_csv(out / "split_summary.csv", index=False, encoding="utf-8-sig")
    by_month.to_csv(out / "split_by_month.csv", index=False, encoding="utf-8-sig")
    log.to_csv(out / "experiment_log.csv", index=False, encoding="utf-8-sig")
    check.to_csv(out / "imputation_statistics_check.csv", index=False, encoding="utf-8-sig")

    save_models(selected)
    save_val_predictions(splits["val"], selected)
    assert set(selected) == {BASELINE, LOGISTIC, HGB}
    print("\nDone. Models saved in data/processed/models, validation predictions in "
          "data/processed/val_predictions.csv.")
    print("The final test set was NOT used.")


if __name__ == "__main__":
    main()