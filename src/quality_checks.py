# src/quality_checks.py
# فحوص الجودة: ترتيب الأحداث الزمنية، تطابق الحالة مع التواريخ،
# الفئات اليتيمة، ومقارنة قبل وبعد المعالجة.

import pandas as pd

from src import config
from src.audit import build_table_summary

# (الحدث الأسبق، الحدث الأحدث): يجب أن يكون الثاني غير أقدم من الأول
DATE_PAIRS = [
    ("order_purchase_timestamp", "order_approved_at"),
    ("order_approved_at", "order_delivered_carrier_date"),
    ("order_delivered_carrier_date", "order_delivered_customer_date"),
    ("order_purchase_timestamp", "order_delivered_customer_date"),
    ("order_purchase_timestamp", "order_estimated_delivery_date"),
]


def check_date_order(orders):
    # لكل زوج نقارن فقط الصفوف التي فيها التاريخان معاً (غير مفقودين)
    rows = []
    for first, second in DATE_PAIRS:
        both_known = orders[first].notna() & orders[second].notna()
        violated = both_known & (orders[second] < orders[first])
        n_compared = int(both_known.sum())
        n_bad = int(violated.sum())
        rows.append({
            "earlier_event": first,
            "later_event": second,
            "rows_compared": n_compared,
            "violations": n_bad,
            "pct_violations": round(n_bad / n_compared * 100, 3) if n_compared else 0.0,
        })
    return pd.DataFrame(rows)


def status_vs_delivery(orders):
    # جدول تقاطعي: حالة الطلب مقابل وجود تاريخ تسليم للعميل
    table = pd.crosstab(orders["order_status"],
                        orders["order_delivered_customer_date"].notna())
    table = table.reindex(columns=[False, True], fill_value=0)
    table.columns = ["no_customer_delivery_date", "has_customer_delivery_date"]
    return table.reset_index()


def date_ranges(clean):
    # المدى الزمني لكل عمود تاريخ، وهل كل قيمه عند منتصف الليل
    rows = []
    for table, cols in config.DATE_COLUMNS.items():
        for col in cols:
            series = clean[table][col].dropna()
            rows.append({
                "table": table,
                "column": col,
                "min": series.min(),
                "max": series.max(),
                "all_at_midnight": bool((series == series.dt.normalize()).all()),
            })
    return pd.DataFrame(rows)


def list_orphan_categories(products_raw, translation_raw):
    # الفئات الموجودة في المنتجات وليس لها ترجمة (ليست المفقودة)
    known = set(translation_raw["product_category_name"])
    categories = products_raw["product_category_name"].dropna()
    orphans = categories[~categories.isin(known)]
    return (orphans.value_counts()
            .rename_axis("product_category_name")
            .reset_index(name="n_products"))


def compare_before_after(raw, clean):
    # جدول مقارنة: الصفوف والأعمدة والخلايا المفقودة قبل المعالجة وبعدها
    cols = ["table", "rows", "columns", "missing_cells"]
    before = build_table_summary(raw)[cols]
    after = build_table_summary(clean)[cols]
    merged = before.merge(after, on="table", suffixes=("_before", "_after"))
    merged["rows_removed"] = merged["rows_before"] - merged["rows_after"]
    merged["missing_change"] = merged["missing_cells_after"] - merged["missing_cells_before"]
    return merged