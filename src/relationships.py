# src/relationships.py
# فحص المفاتيح والعلاقات، وتوضيح معنى كل معرّف.

import pandas as pd

from src import config


def check_primary_keys(tables):
    # هل المفتاح الأساسي لكل جدول فريد وخال من القيم المفقودة؟
    rows = []
    for name, key_cols in config.PRIMARY_KEYS.items():
        df = tables[name]
        n_rows = len(df)
        distinct_keys = len(df[key_cols].drop_duplicates())
        has_nulls = bool(df[key_cols].isna().any().any())
        rows.append({
            "table": name,
            "key": " + ".join(key_cols),
            "rows": n_rows,
            "distinct_keys": distinct_keys,
            "duplicate_key_rows": n_rows - distinct_keys,
            "key_has_nulls": has_nulls,
            "is_valid_key": (n_rows == distinct_keys) and (not has_nulls),
        })
    return pd.DataFrame(rows)


def check_foreign_keys(tables):
    # هل كل قيمة في الجدول الابن موجودة في الجدول الأب؟ (سلامة العلاقة)
    rows = []
    for child, child_col, parent, parent_col in config.RELATIONSHIPS:
        child_values = tables[child][child_col]
        parent_values = tables[parent][parent_col].dropna().unique()

        n_null = int(child_values.isna().sum())
        non_null_values = child_values.dropna()

        # الصفوف التي لا يوجد لقيمتها أب (يتيمة)
        orphan_mask = ~non_null_values.isin(parent_values)

        rows.append({
            "child": f"{child}.{child_col}",
            "parent": f"{parent}.{parent_col}",
            "child_rows": len(child_values),
            "null_in_child": n_null,
            "orphan_rows": int(orphan_mask.sum()),
            "orphan_distinct_values": int(non_null_values[orphan_mask].nunique()),
            "is_valid": int(orphan_mask.sum()) == 0,
        })
    return pd.DataFrame(rows)


def describe_identifiers(tables):
    # حقائق رقمية توضح الفرق بين المعرفات وتكشف تعدد الصفوف لكل طلب
    orders = tables["orders"]
    customers = tables["customers"]
    items = tables["order_items"]
    payments = tables["payments"]

    items_per_order = items.groupby("order_id").size()
    payments_per_order = payments.groupby("order_id").size()
    sellers_per_order = items.groupby("order_id")["seller_id"].nunique()
    ids_per_person = customers.groupby("customer_unique_id")["customer_id"].nunique()

    facts = [
        ("orders: distinct order_id (number of orders)", orders["order_id"].nunique()),
        ("customers: distinct customer_id", customers["customer_id"].nunique()),
        ("customers: distinct customer_unique_id (real people)", customers["customer_unique_id"].nunique()),
        ("people with more than one customer_id (repeat buyers)", int((ids_per_person > 1).sum())),
        ("orders with no row in order_items", int((~orders["order_id"].isin(items["order_id"])).sum())),
        ("orders with no payment record", int((~orders["order_id"].isin(payments["order_id"])).sum())),
        ("orders with more than one item", int((items_per_order > 1).sum())),
        ("max items in a single order", int(items_per_order.max())),
        ("orders with more than one payment record", int((payments_per_order > 1).sum())),
        ("max payment records in a single order", int(payments_per_order.max())),
        ("multi-seller orders (more than one seller_id)", int((sellers_per_order > 1).sum())),
    ]
    return pd.DataFrame(facts, columns=["fact", "value"])