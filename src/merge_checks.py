# src/merge_checks.py
# فحوص الربط: عرض خطأ الربط المباشر، والتحقق من صف واحد لكل طلب وصحة المجاميع،
# وتحليل الطلبات متعددة البائعين، وفحص شذوذ shipping_limit_date.

import numpy as np
import pandas as pd


def naive_join_demo(clean):
    # يوضح ما يحدث لو ربطنا العناصر والدفع مباشرة دون تجميع.
    items = clean["order_items"][["order_id", "order_item_id", "price", "freight_value"]]
    payments = clean["payments"][["order_id", "payment_sequential", "payment_value"]]

    naive = items.merge(payments, on="order_id", how="inner")

    rows = [
        {"measure": "number of rows", "correct": len(items), "naive_join": len(naive)},
        {"measure": "sum of price", "correct": round(float(items["price"].sum()), 2),
         "naive_join": round(float(naive["price"].sum()), 2)},
        {"measure": "sum of freight_value", "correct": round(float(items["freight_value"].sum()), 2),
         "naive_join": round(float(naive["freight_value"].sum()), 2)},
    ]
    out = pd.DataFrame(rows)
    out["naive_minus_correct"] = (out["naive_join"] - out["correct"]).round(2)
    return out


def validate_order_table(order_table, clean):
    # كل فحص يقارن ما في جدول الطلب بحساب مستقل من الجداول النظيفة.
    orders = clean["orders"]
    items = clean["order_items"]
    payments = clean["payments"]

    rows = []

    def add(name, expected, actual, tol=0.0):
        passed = abs(float(expected) - float(actual)) <= tol
        if tol > 0:
            expected, actual = round(float(expected), 2), round(float(actual), 2)
        rows.append({"check": name, "expected": expected, "actual": actual,
                     "passed": bool(passed)})

    add("rows equal number of orders", len(orders), len(order_table))
    add("distinct order_id equals number of orders", len(orders), order_table["order_id"].nunique())
    add("no missing order_id", 0, int(order_table["order_id"].isna().sum()))
    add("order_id set identical to orders table (1 = yes)", 1,
        int(set(order_table["order_id"]) == set(orders["order_id"])))
    add("sum of n_items equals item rows", len(items), int(order_table["n_items"].sum()))
    add("sum of items_price_sum equals sum of price", items["price"].sum(),
        order_table["items_price_sum"].sum(), tol=0.05)
    add("sum of items_freight_sum equals sum of freight_value", items["freight_value"].sum(),
        order_table["items_freight_sum"].sum(), tol=0.05)
    add("sum of n_payment_records equals payment rows", len(payments),
        int(order_table["n_payment_records"].sum()))
    add("sum of payment_value_sum equals sum of payment_value", payments["payment_value"].sum(),
        order_table["payment_value_sum"].sum(), tol=0.05)
    add("orders without items", int((~orders["order_id"].isin(items["order_id"])).sum()),
        int((~order_table["has_items"]).sum()))
    add("orders without payments", int((~orders["order_id"].isin(payments["order_id"])).sum()),
        int((~order_table["has_payments"]).sum()))
    add("multi-seller orders",
        int((items.groupby("order_id")["seller_id"].nunique() > 1).sum()),
        int((order_table["n_sellers"] > 1).sum()))

    return pd.DataFrame(rows)


def seller_count_distribution(order_table):
    # توزيع عدد البائعين في الطلب (للطلبات التي لها عناصر)
    with_items = order_table[order_table["has_items"]]
    dist = (with_items["n_sellers"].astype(int).value_counts().sort_index()
            .rename_axis("n_sellers").reset_index(name="n_orders"))
    dist["pct_of_orders_with_items"] = (dist["n_orders"] / dist["n_orders"].sum() * 100).round(3)
    return dist


def seller_group_pivot(order_table):
    # جدول محوري: حالة الطلب مقابل (بائع واحد / عدة بائعين)
    with_items = order_table[order_table["has_items"]].copy()
    with_items["seller_group"] = np.where(with_items["n_sellers"] > 1,
                                          "multi_seller", "single_seller")
    pivot = pd.pivot_table(with_items, index="order_status", columns="seller_group",
                           values="order_id", aggfunc="count", fill_value=0)
    pivot.columns.name = None
    return pivot.reset_index()


def orders_without_items_by_status(order_table):
    subset = order_table[~order_table["has_items"]]
    return (subset["order_status"].value_counts()
            .rename_axis("order_status").reset_index(name="n_orders_without_items"))


def orders_without_payments(order_table):
    subset = order_table[~order_table["has_payments"]]
    return subset[["order_id", "order_status", "has_items"]]


def reconcile_payments_with_items(order_table):
    # هل مجموع الدفع يساوي السعر + الشحن؟ (قد يختلف بسبب الفوائد والقسائم)
    both = order_table[order_table["has_items"] & order_table["has_payments"]]
    expected = both["items_price_sum"] + both["items_freight_sum"]
    diff = (both["payment_value_sum"] - expected).abs()
    rows = [
        ("orders compared (have items and payments)", len(both)),
        ("difference <= 0.01", int((diff <= 0.01).sum())),
        ("difference between 0.01 and 1.00", int(((diff > 0.01) & (diff <= 1.0)).sum())),
        ("difference > 1.00", int((diff > 1.0).sum())),
        ("max absolute difference", round(float(diff.max()), 2)),
    ]
    return pd.DataFrame(rows, columns=["measure", "value"])


def shipping_limit_checks(items, orders):
    # فحص شذوذ shipping_limit_date بالنسبة لوقت الشراء (الأصل: تاريخ ظهر حتى 2020)
    purchase = orders[["order_id", "order_purchase_timestamp"]]
    df = items[["order_id", "shipping_limit_date"]].merge(
        purchase, on="order_id", how="left", validate="many_to_one")
    gap_days = (df["shipping_limit_date"] - df["order_purchase_timestamp"]).dt.total_seconds() / 86400

    rows = [
        ("item rows checked", len(df)),
        ("limit date earlier than purchase time", int((gap_days < 0).sum())),
        ("limit date more than 90 days after purchase", int((gap_days > 90).sum())),
        ("min gap in days", round(float(gap_days.min()), 2)),
        ("max gap in days", round(float(gap_days.max()), 2)),
        ("latest shipping_limit_date", str(df["shipping_limit_date"].max())),
    ]
    return pd.DataFrame(rows, columns=["measure", "value"])