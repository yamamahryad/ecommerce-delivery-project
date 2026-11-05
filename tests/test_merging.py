# tests/test_merging.py
# اختبارات الربط: صف واحد لكل طلب، عدم تضاعف الصفوف، صحة المجاميع، متعدد البائعين.
# التشغيل: python -m pytest -v

import pandas as pd
import pytest

from src.merge_checks import naive_join_demo, validate_order_table
from src.merging import build_order_table


def make_tiny_tables():
    # ثلاثة طلبات نعرف نتيجتها مسبقاً:
    # o1: عنصران من بائعين مختلفين ودفعتان (بطاقة 120 + قسيمة 45)
    # o2: عنصر واحد ودفعة واحدة
    # o3: بلا عناصر ولا دفع
    return {
        "orders": pd.DataFrame({
            "order_id": ["o1", "o2", "o3"],
            "customer_id": ["c1", "c2", "c3"],
            "order_status": ["delivered", "delivered", "canceled"],
        }),
        "customers": pd.DataFrame({
            "customer_id": ["c1", "c2", "c3"],
            "customer_unique_id": ["u1", "u2", "u3"],
            "customer_zip_code_prefix": ["01046", "20000", "30000"],
            "customer_city": ["a", "b", "c"],
            "customer_state": ["SP", "RJ", "MG"],
        }),
        "sellers": pd.DataFrame({
            "seller_id": ["s1", "s2"],
            "seller_zip_code_prefix": ["11111", "22222"],
            "seller_city": ["x", "y"],
            "seller_state": ["SP", "RJ"],
        }),
        "products": pd.DataFrame({
            "product_id": ["p1", "p2"],
            "product_category_name": ["cat_a", "cat_b"],
            "product_category_english": ["cat_a_en", "cat_b_en"],
            "product_weight_g": [500.0, 300.0],
            "product_length_cm": [10.0, 5.0],
            "product_height_cm": [10.0, 5.0],
            "product_width_cm": [10.0, 4.0],
        }),
        "order_items": pd.DataFrame({
            "order_id": ["o1", "o1", "o2"],
            "order_item_id": [1, 2, 1],
            "product_id": ["p1", "p2", "p1"],
            "seller_id": ["s1", "s2", "s1"],
            "price": [100.0, 50.0, 20.0],
            "freight_value": [10.0, 5.0, 5.0],
        }),
        "payments": pd.DataFrame({
            "order_id": ["o1", "o1", "o2"],
            "payment_sequential": [1, 2, 1],
            "payment_type": ["credit_card", "voucher", "boleto"],
            "payment_installments": [3, 1, 1],
            "payment_value": [120.0, 45.0, 25.0],
        }),
    }


# ---------------- اختبارات على أمثلة صغيرة نعرف جوابها ----------------

def test_tiny_one_row_per_order():
    table, _log = build_order_table(make_tiny_tables())
    assert len(table) == 3
    assert table["order_id"].is_unique


def test_tiny_totals_are_not_inflated():
    table, _log = build_order_table(make_tiny_tables())
    o1 = table[table["order_id"] == "o1"].iloc[0]
    assert o1["n_items"] == 2
    assert o1["items_price_sum"] == 150.0          # وليس 300 كما في الربط المباشر
    assert o1["items_freight_sum"] == 15.0
    assert o1["n_payment_records"] == 2
    assert o1["payment_value_sum"] == 165.0        # = 150 + 15
    assert o1["total_weight_g"] == 800.0
    assert o1["total_volume_cm3"] == 1100.0        # 10*10*10 + 5*5*4


def test_tiny_multi_seller_is_handled_explicitly():
    table, _log = build_order_table(make_tiny_tables())
    o1 = table[table["order_id"] == "o1"].iloc[0]
    assert o1["n_sellers"] == 2
    assert o1["n_seller_states"] == 2
    assert o1["primary_seller_id"] == "s1"         # صاحب العنصر الأعلى سعراً
    assert o1["primary_seller_state"] == "SP"
    assert o1["primary_category"] == "cat_a_en"
    assert o1["main_payment_type"] == "credit_card"
    assert o1["max_installments"] == 3


def test_tiny_order_without_items_and_payments_stays_with_flags():
    table, _log = build_order_table(make_tiny_tables())
    o3 = table[table["order_id"] == "o3"].iloc[0]
    assert not o3["has_items"]
    assert not o3["has_payments"]
    assert pd.isna(o3["n_items"])
    assert pd.isna(o3["payment_value_sum"])


def test_tiny_missing_weight_stays_missing_not_zero():
    tables = make_tiny_tables()
    tables["products"].loc[tables["products"]["product_id"] == "p1", "product_weight_g"] = float("nan")
    table, _log = build_order_table(tables)
    o1 = table[table["order_id"] == "o1"].iloc[0]
    o2 = table[table["order_id"] == "o2"].iloc[0]
    assert o1["total_weight_g"] == 300.0           # وزن p2 فقط
    assert o1["n_items_missing_size"] == 1
    assert pd.isna(o2["total_weight_g"])           # لا صفر مضلل
    assert o2["n_items_missing_size"] == 1


def test_tiny_join_log_shows_no_row_growth():
    _table, log = build_order_table(make_tiny_tables())
    assert (log["rows_added"] == 0).all()
    order_level = log[log["step"].str.startswith("orders +")]
    assert (order_level["result_rows"] == 3).all()


def test_duplicate_key_in_lookup_table_is_caught():
    tables = make_tiny_tables()
    tables["customers"] = pd.concat([tables["customers"], tables["customers"].iloc[[0]]],
                                    ignore_index=True)
    with pytest.raises(pd.errors.MergeError):
        build_order_table(tables)


def test_naive_join_inflates_but_order_table_does_not(clean_tables):
    demo = naive_join_demo(clean_tables)
    rows = demo[demo["measure"] == "number of rows"].iloc[0]
    assert rows["naive_join"] > rows["correct"]


# ---------------- اختبارات على البيانات الحقيقية ----------------

def test_one_row_per_order_real(order_table_and_log, clean_tables):
    table, _log = order_table_and_log
    orders = clean_tables["orders"]
    assert len(table) == len(orders)
    assert table["order_id"].is_unique
    assert table["order_id"].equals(orders["order_id"])   # نفس الترتيب أيضاً


def test_no_row_growth_after_each_join_real(order_table_and_log, clean_tables):
    _table, log = order_table_and_log
    assert (log["rows_added"] == 0).all()
    order_level = log[log["step"].str.startswith("orders +")]
    assert (order_level["result_rows"] == len(clean_tables["orders"])).all()


def test_totals_match_source_tables(order_table_and_log, clean_tables):
    table, _log = order_table_and_log
    items = clean_tables["order_items"]
    payments = clean_tables["payments"]
    assert int(table["n_items"].sum()) == len(items)
    assert int(table["n_payment_records"].sum()) == len(payments)
    assert table["items_price_sum"].sum() == pytest.approx(items["price"].sum(), abs=0.05)
    assert table["items_freight_sum"].sum() == pytest.approx(items["freight_value"].sum(), abs=0.05)
    assert table["payment_value_sum"].sum() == pytest.approx(payments["payment_value"].sum(), abs=0.05)


def test_orders_without_items_or_payments_are_flagged(order_table_and_log, clean_tables):
    table, _log = order_table_and_log
    orders = clean_tables["orders"]
    no_items = int((~orders["order_id"].isin(clean_tables["order_items"]["order_id"])).sum())
    no_pay = int((~orders["order_id"].isin(clean_tables["payments"]["order_id"])).sum())
    assert int((~table["has_items"]).sum()) == no_items
    assert int((~table["has_payments"]).sum()) == no_pay
    assert table.loc[~table["has_items"], "n_items"].isna().all()


def test_multi_seller_orders_counted_correctly(order_table_and_log, clean_tables):
    table, _log = order_table_and_log
    sellers_per_order = clean_tables["order_items"].groupby("order_id")["seller_id"].nunique()
    assert int((table["n_sellers"] > 1).sum()) == int((sellers_per_order > 1).sum())
    # كل طلب له عناصر يجب أن يكون له بائع رئيسي
    with_items = table[table["has_items"]]
    assert with_items["primary_seller_id"].notna().all()
    assert (with_items["n_sellers"] >= 1).all()


def test_shipping_limit_date_not_carried_into_order_table(order_table_and_log):
    table, _log = order_table_and_log
    assert "shipping_limit_date" not in table.columns


def test_validation_report_all_passed(order_table_and_log, clean_tables):
    table, _log = order_table_and_log
    report = validate_order_table(table, clean_tables)
    assert report["passed"].all(), report[~report["passed"]]