# run_day3.py
# يبني جدول الطلب الواحد ويتحقق منه: python run_day3.py
# يتطلب تشغيل run_day2.py أولاً (لأنه يقرأ الجداول النظيفة من data/processed).

import numpy as np  # noqa: F401  (تستخدمه دوال التحليل، وللتذكير بالمكتبة المطلوبة)
import pandas as pd

from src import config
from src.audit import build_column_audit
from src.cleaning import load_clean_tables
from src.merge_checks import (naive_join_demo, orders_without_items_by_status,
                              orders_without_payments, reconcile_payments_with_items,
                              seller_count_distribution, seller_group_pivot,
                              shipping_limit_checks, validate_order_table)
from src.merging import build_order_table, load_order_table, save_order_table


def show(title, df):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)
    print(df.to_string(index=False))


def main():
    pd.set_option("display.width", 220)
    config.ensure_output_dirs()

    clean = load_clean_tables()
    order_table, join_log = build_order_table(clean)

    naive = naive_join_demo(clean)
    validation = validate_order_table(order_table, clean)
    seller_dist = seller_count_distribution(order_table)
    no_items = orders_without_items_by_status(order_table)
    no_payments = orders_without_payments(order_table)
    reconcile = reconcile_payments_with_items(order_table)
    pivot = seller_group_pivot(order_table)
    shipping = shipping_limit_checks(clean["order_items"], clean["orders"])
    columns_audit = build_column_audit({"order_base": order_table})

    show("1) Join log: effect of every join on the number of rows", join_log)
    show("2) Why we aggregate first: naive join of items and payments", naive)
    show("3) Validation: one row per order and correct totals", validation)
    show("4) Number of sellers per order (orders with items)", seller_dist)
    show("5) Orders without items, by status", no_items)
    show("6) Orders without payment records", no_payments)
    show("7) Payment total vs items price + freight", reconcile)
    show("8) Pivot: order status x seller group (orders with items)", pivot)
    show("9) shipping_limit_date check (not used as a model input)", shipping)
    show("10) Columns of the order table", columns_audit)

    print("\n" + "=" * 100)
    print("11) Sample of the order table (3 rows, transposed)")
    print("=" * 100)
    print(order_table.head(3).T.to_string())

    if not validation["passed"].all():
        raise SystemExit("VALIDATION FAILED: see section 3 above.")

    # حفظ جداول التوثيق
    out = config.TABLES_DIR
    join_log.to_csv(out / "join_log.csv", index=False, encoding="utf-8-sig")
    naive.to_csv(out / "naive_join_demo.csv", index=False, encoding="utf-8-sig")
    validation.to_csv(out / "order_table_validation.csv", index=False, encoding="utf-8-sig")
    seller_dist.to_csv(out / "seller_count_distribution.csv", index=False, encoding="utf-8-sig")
    no_items.to_csv(out / "orders_without_items_by_status.csv", index=False, encoding="utf-8-sig")
    reconcile.to_csv(out / "payment_reconciliation.csv", index=False, encoding="utf-8-sig")
    pivot.to_csv(out / "status_by_seller_group.csv", index=False, encoding="utf-8-sig")
    shipping.to_csv(out / "shipping_limit_checks.csv", index=False, encoding="utf-8-sig")
    columns_audit.to_csv(out / "audit_order_base_columns.csv", index=False, encoding="utf-8-sig")

    # حفظ الجدول ثم إعادة قراءته للتأكد من سلامته
    save_order_table(order_table)
    reloaded = load_order_table()
    assert reloaded.shape == order_table.shape, "Shape changed after reload"
    assert reloaded["order_id"].is_unique, "order_id not unique after reload"
    print("\nAll validation checks passed. Round-trip check passed.")
    print(f"Order table saved: {config.ORDER_TABLE_FILE}  shape={reloaded.shape}")


if __name__ == "__main__":
    main()