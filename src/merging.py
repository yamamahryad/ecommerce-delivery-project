# src/merging.py
# تجميع العناصر والدفع إلى مستوى الطلب، ثم بناء جدول بصف واحد لكل طلب.
# كل عملية ربط تسجل أثرها على عدد الصفوف في سجل الربط.

import pandas as pd

from src import config


def merge_with_log(left, right, key, validate, step, log):
    # left join مع ثلاث حمايات:
    # 1) validate: pandas يرفع MergeError إن كانت علاقة المفاتيح غير ما نتوقع
    # 2) نتحقق أن عدد الصفوف لم يتغير
    # 3) نسجل في log عدد الصفوف قبل وبعد وعدد الصفوف التي وجدت مقابلاً
    rows_left = len(left)
    matched = int(left[key].isin(right[key]).sum())

    merged = left.merge(right, on=key, how="left", validate=validate)

    if len(merged) != rows_left:
        raise ValueError(
            f"Join '{step}' changed the number of rows: {rows_left} -> {len(merged)}"
        )

    log.append({
        "step": step,
        "key": key,
        "validate": validate,
        "left_rows": rows_left,
        "right_rows": len(right),
        "result_rows": len(merged),
        "rows_added": len(merged) - rows_left,
        "left_rows_with_match": matched,
        "left_rows_without_match": rows_left - matched,
    })
    return merged


def aggregate_items(items, products, sellers, log):
    # 1) نضيف خصائص المنتج والبائع إلى كل عنصر (ربط من نوع كثير-إلى-واحد)
    item_cols = ["order_id", "order_item_id", "product_id", "seller_id",
                 "price", "freight_value"]
    product_cols = ["product_id", "product_category_english", "product_weight_g",
                    "product_length_cm", "product_height_cm", "product_width_cm"]

    df = items[item_cols].copy()
    df = merge_with_log(df, products[product_cols], "product_id",
                        "many_to_one", "items + products (item level)", log)
    df = merge_with_log(df, sellers[["seller_id", "seller_state"]], "seller_id",
                        "many_to_one", "items + sellers (item level)", log)

    # 2) أعمدة على مستوى العنصر بعمليات متجهة (بدون حلقات)
    df["product_volume_cm3"] = (df["product_length_cm"] * df["product_height_cm"]
                                * df["product_width_cm"])
    # 1 إذا كان وزن العنصر أو حجمه مجهولاً
    df["item_size_missing"] = (df[["product_weight_g", "product_volume_cm3"]]
                               .isna().any(axis=1).astype(int))

    # 3) التجميع إلى مستوى الطلب
    grouped = df.groupby("order_id")
    agg = grouped.agg(
        n_items=("order_item_id", "size"),
        n_distinct_products=("product_id", "nunique"),
        n_sellers=("seller_id", "nunique"),
        n_seller_states=("seller_state", "nunique"),
        n_categories=("product_category_english", "nunique"),
        items_price_sum=("price", "sum"),
        items_freight_sum=("freight_value", "sum"),
        max_item_price=("price", "max"),
        n_items_missing_size=("item_size_missing", "sum"),
    )
    # sum(min_count=1): إذا كانت كل القيم مفقودة تبقى النتيجة مفقودة وليست صفراً
    agg["total_weight_g"] = grouped["product_weight_g"].sum(min_count=1)
    agg["total_volume_cm3"] = grouped["product_volume_cm3"].sum(min_count=1)

    # 4) العنصر الرئيسي = الأعلى سعراً، وعند التعادل أصغر order_item_id
    ranked = df.sort_values(["order_id", "price", "order_item_id"],
                            ascending=[True, False, True])
    primary = ranked.drop_duplicates(subset="order_id", keep="first")
    primary = (primary[["order_id", "seller_id", "seller_state", "product_category_english"]]
               .rename(columns={"seller_id": "primary_seller_id",
                                "seller_state": "primary_seller_state",
                                "product_category_english": "primary_category"})
               .set_index("order_id"))

    agg = agg.join(primary, how="left")
    return agg.reset_index()


def aggregate_payments(payments):
    grouped = payments.groupby("order_id")
    agg = grouped.agg(
        n_payment_records=("payment_sequential", "size"),
        n_payment_types=("payment_type", "nunique"),
        payment_value_sum=("payment_value", "sum"),
        max_installments=("payment_installments", "max"),
    )

    # وسيلة الدفع الرئيسية = صاحبة أكبر مبلغ، وعند التعادل أصغر payment_sequential
    ranked = payments.sort_values(["order_id", "payment_value", "payment_sequential"],
                                  ascending=[True, False, True])
    main = ranked.drop_duplicates(subset="order_id", keep="first")
    main_type = main.set_index("order_id")["payment_type"].rename("main_payment_type")

    return agg.join(main_type, how="left").reset_index()


def build_order_table(clean):
    # يعيد (جدول الطلب الواحد، سجل الربط)
    log = []

    items_agg = aggregate_items(clean["order_items"], clean["products"],
                                clean["sellers"], log)
    payments_agg = aggregate_payments(clean["payments"])
    customers = clean["customers"][["customer_id", "customer_unique_id",
                                    "customer_zip_code_prefix", "customer_city",
                                    "customer_state"]]

    table = clean["orders"].copy()
    table = merge_with_log(table, items_agg, "order_id", "one_to_one",
                           "orders + item aggregates", log)
    table = merge_with_log(table, payments_agg, "order_id", "one_to_one",
                           "orders + payment aggregates", log)
    table = merge_with_log(table, customers, "customer_id", "many_to_one",
                           "orders + customers", log)

    # علامات تفرق بين "لا عناصر" و"عناصر قيمتها مفقودة"
    table["has_items"] = table["n_items"].notna()
    table["has_payments"] = table["n_payment_records"].notna()

    return table, pd.DataFrame(log)


def save_order_table(table):
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(config.ORDER_TABLE_FILE, index=False)


def load_order_table():
    if not config.ORDER_TABLE_FILE.exists():
        raise FileNotFoundError(f"{config.ORDER_TABLE_FILE} not found. Run: python run_day3.py")
    return pd.read_csv(
        config.ORDER_TABLE_FILE,
        dtype={"customer_zip_code_prefix": str},
        parse_dates=config.DATE_COLUMNS["orders"],
    )