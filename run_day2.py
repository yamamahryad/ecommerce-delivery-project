# run_day2.py
# يشغل تنظيف اليوم الثاني كاملاً: python run_day2.py

import pandas as pd

from src import config
from src.audit import build_column_audit
from src.cleaning import clean_all_tables, load_clean_tables, save_clean_tables
from src.data_loading import load_all_tables
from src.quality_checks import (check_date_order, compare_before_after, date_ranges,
                                list_orphan_categories, status_vs_delivery)


def show(title, df):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)
    print(df.to_string(index=False))


def main():
    pd.set_option("display.width", 220)
    config.ensure_output_dirs()

    raw = load_all_tables()
    clean, log = clean_all_tables(raw)
    decisions = log.to_frame()

    show("1) Cleaning decision log (short view; full text in the CSV)",
         decisions[["table", "issue", "records_affected", "action"]])
    show("2) Event order check (violations are counted, NOT deleted)",
         check_date_order(clean["orders"]))
    show("3) Order status vs presence of customer delivery date",
         status_vs_delivery(clean["orders"]))
    show("4) Date ranges", date_ranges(clean))
    show("5) Order status counts",
         clean["orders"]["order_status"].value_counts().rename_axis("order_status")
         .reset_index(name="n_orders"))
    show("6) Categories without translation (raw data)",
         list_orphan_categories(raw["products"], raw["category_translation"]))
    comparison = compare_before_after(raw, clean)
    show("7) Before vs after cleaning", comparison)

    column_audit_after = build_column_audit(clean)
    show("8) Columns still missing values after cleaning",
         column_audit_after[column_audit_after["n_missing"] > 0])

    # حفظ الجداول التوثيقية (utf-8-sig ليفتحها Excel بشكل سليم)
    decisions.to_csv(config.TABLES_DIR / "cleaning_decision_log.csv",
                     index=False, encoding="utf-8-sig")
    check_date_order(clean["orders"]).to_csv(config.TABLES_DIR / "date_order_checks.csv",
                                             index=False, encoding="utf-8-sig")
    status_vs_delivery(clean["orders"]).to_csv(config.TABLES_DIR / "status_vs_delivery.csv",
                                               index=False, encoding="utf-8-sig")
    comparison.to_csv(config.TABLES_DIR / "audit_before_vs_after.csv",
                      index=False, encoding="utf-8-sig")
    column_audit_after.to_csv(config.TABLES_DIR / "audit_after_columns.csv",
                              index=False, encoding="utf-8-sig")

    # .حفظ الجداول النظيفة ثم إعادة قراءتها للتأكد من سلامة الحفظ
    save_clean_tables(clean)
    reloaded = load_clean_tables()
    for name in clean:
        assert clean[name].shape == reloaded[name].shape, f"Shape changed after reload: {name}"
    print("\nRound-trip check passed: processed files reload with the same shapes.")
    print("Done. Clean tables are in data/processed, documentation tables in outputs/tables.")


if __name__ == "__main__":
    main()
