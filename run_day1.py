# run_day1.py
# يشغل تدقيق اليوم الأول كاملاً: python run_day1.py

import pandas as pd

from src import config
from src.audit import build_column_audit, build_table_summary
from src.data_loading import load_all_tables
from src.erd import build_erd
from src.relationships import check_foreign_keys, check_primary_keys, describe_identifiers


def show(title, df):
    # يطبع جدولاً بعنوان واضح
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)
    print(df.to_string(index=False))


def main():
    pd.set_option("display.width", 220)
    config.ensure_output_dirs()

    tables = load_all_tables()

    summary = build_table_summary(tables)
    column_audit = build_column_audit(tables)
    primary_keys = check_primary_keys(tables)
    foreign_keys = check_foreign_keys(tables)
    identifiers = describe_identifiers(tables)

    show("1) Table summary before cleaning", summary)
    show("2) Column audit (types and missing values)", column_audit)
    show("3) Columns with missing values only",
         column_audit[column_audit["n_missing"] > 0])
    show("4) Primary key checks", primary_keys)
    show("5) Foreign key (relationship integrity) checks", foreign_keys)
    show("6) Identifier facts and multiple rows per order", identifiers)

    # utf-8-sig لتفتح الملفات في Excel بشكل سليم
    summary.to_csv(config.TABLES_DIR / "audit_before_table_summary.csv",
                   index=False, encoding="utf-8-sig")
    column_audit.to_csv(config.TABLES_DIR / "audit_before_columns.csv",
                        index=False, encoding="utf-8-sig")
    primary_keys.to_csv(config.TABLES_DIR / "key_checks_primary.csv",
                        index=False, encoding="utf-8-sig")
    foreign_keys.to_csv(config.TABLES_DIR / "key_checks_foreign.csv",
                        index=False, encoding="utf-8-sig")
    identifiers.to_csv(config.TABLES_DIR / "identifier_facts.csv",
                       index=False, encoding="utf-8-sig")

    build_erd(config.FIGURES_DIR / "erd_tables.png")
    print("\nDone. Tables are in outputs/tables and the diagram is in outputs/figures/erd_tables.png")


if __name__ == "__main__":
    main()