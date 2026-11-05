# run_day4.py
# يبني الهدف والخصائص وجدول النمذجة: python run_day4.py
# يتطلب تشغيل run_day2.py ثم run_day3.py أولاً.

import pandas as pd

from src import config
from src.feature_checks import (delay_description, feature_missing_report,
                                log_transform_effect, numpy_summary,
                                status_by_exclusion, target_distribution)
from src.features import (MODEL_FEATURES, build_modeling_tables, feature_dictionary,
                          load_modeling_table, save_feature_tables)
from src.merging import load_order_table
from src.target import TARGET


def show(title, df):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)
    print(df.to_string(index=False))


def main():
    pd.set_option("display.width", 220)
    pd.set_option("display.max_colwidth", 60)
    config.ensure_output_dirs()

    order_table = load_order_table()
    full, modeling, exclusions = build_modeling_tables(order_table)
    eligible = full[full["eligible"]]

    dictionary = feature_dictionary()
    status_x = status_by_exclusion(full)
    target_dist = target_distribution(modeling)
    missing = feature_missing_report(eligible)
    summary_cols = ["planned_days", "purchase_to_approval_hours", "items_price_sum",
                    "items_freight_sum", "freight_share", "log_weight_g", "log_volume_cm3"]
    np_summary = numpy_summary(eligible, summary_cols)
    log_effect = log_transform_effect(eligible)
    delay = delay_description(eligible)

    show("1) Exclusions: one reason per order, in priority order", exclusions)
    show("2) Order status x exclusion reason", status_x)
    show("3) Target distribution among eligible orders", target_dist)
    show("4) Descriptive: delay in days (analysis only, NOT a model input)", delay)
    show("5) Feature dictionary (short view; full text in the CSV)",
         dictionary[["feature", "derived", "type", "model_input", "available_at"]])
    show("6) Missing values in model features (eligible orders only)", missing)
    show("7) NumPy summary (nanpercentile / nanmean)", np_summary)
    show("8) Effect of log1p on skewness (why we transform)", log_effect)

    print("\n" + "=" * 100)
    print("9) Sample of the modeling table (3 rows, transposed)")
    print("=" * 100)
    print(modeling.head(3).T.to_string())

    # حفظ جداول التوثيق
    out = config.TABLES_DIR
    exclusions.to_csv(out / "exclusion_summary.csv", index=False, encoding="utf-8-sig")
    status_x.to_csv(out / "status_by_exclusion.csv", index=False, encoding="utf-8-sig")
    target_dist.to_csv(out / "target_distribution.csv", index=False, encoding="utf-8-sig")
    delay.to_csv(out / "delay_description.csv", index=False, encoding="utf-8-sig")
    dictionary.to_csv(out / "feature_dictionary.csv", index=False, encoding="utf-8-sig")
    missing.to_csv(out / "feature_missing_eligible.csv", index=False, encoding="utf-8-sig")
    np_summary.to_csv(out / "numpy_feature_summary.csv", index=False, encoding="utf-8-sig")
    log_effect.to_csv(out / "log_transform_effect.csv", index=False, encoding="utf-8-sig")

    # حفظ الجداول ثم إعادة قراءتها للتأكد من سلامتها
    save_feature_tables(full, modeling)
    reloaded = load_modeling_table()
    assert reloaded.shape == modeling.shape, "Shape changed after reload"
    assert reloaded["order_id"].is_unique, "order_id not unique in modeling table"
    assert set(reloaded[TARGET].unique()) <= {0, 1}, "target must be 0/1"
    assert len(MODEL_FEATURES) == len(set(MODEL_FEATURES)), "duplicate feature names"
    print("\nRound-trip check passed.")
    print(f"Full table: {full.shape}  Modeling table: {reloaded.shape}")


if __name__ == "__main__":
    main()