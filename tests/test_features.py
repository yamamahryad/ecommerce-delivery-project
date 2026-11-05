# tests/test_features.py
# اختبارات الهدف والأهلية والخصائص والحقول المحظورة.
# التشغيل: python -m pytest -v

import numpy as np
import pandas as pd
import pytest

from src.features import (FEATURE_DICTIONARY, FORBIDDEN_COLUMNS, MODEL_FEATURES,
                          add_features, build_modeling_tables, safe_divide)
from src.target import TARGET, compute_is_late

DATE_COLS = ["order_purchase_timestamp", "order_approved_at",
             "order_delivered_carrier_date", "order_delivered_customer_date",
             "order_estimated_delivery_date"]

BASE_ROW = dict(
    order_id="o", customer_id="c", customer_unique_id="u", order_status="delivered",
    order_purchase_timestamp="2018-01-01 09:00:00",
    order_approved_at="2018-01-01 10:00:00",
    order_delivered_carrier_date="2018-01-03 10:00:00",
    order_delivered_customer_date="2018-01-08 15:00:00",
    order_estimated_delivery_date="2018-01-10 00:00:00",
    n_items=2, n_distinct_products=2, n_sellers=1, n_seller_states=1, n_categories=1,
    items_price_sum=100.0, items_freight_sum=20.0, max_item_price=60.0,
    n_items_missing_size=0, total_weight_g=1000.0, total_volume_cm3=2000.0,
    primary_seller_id="s", primary_seller_state="SP", primary_category="toys",
    n_payment_records=1, n_payment_types=1, payment_value_sum=120.0,
    max_installments=3, main_payment_type="credit_card",
    customer_state="RJ", has_items=True, has_payments=True,
)


def make_orders(overrides):
    # كل عنصر في overrides يعدل الصف الأساسي. 2018-01-01 يوم اثنين.
    rows = [{**BASE_ROW, **o} for o in overrides]
    df = pd.DataFrame(rows)
    for col in DATE_COLS:
        df[col] = pd.to_datetime(df[col])
    return df


def row_of(df, order_id):
    return df[df["order_id"] == order_id].iloc[0]


# ---------------- اختبارات الهدف (الحالات الثلاث التي يطلبها الملف) ----------------

def test_target_delivery_on_estimated_day_is_not_late():
    delivered = pd.to_datetime(pd.Series(["2018-01-10 23:59:00"]))   # نفس اليوم، آخر الليل
    estimated = pd.to_datetime(pd.Series(["2018-01-10 00:00:00"]))
    assert compute_is_late(delivered, estimated).iloc[0] == 0


def test_target_delivery_before_estimated_day_is_not_late():
    delivered = pd.to_datetime(pd.Series(["2018-01-05 12:00:00"]))
    estimated = pd.to_datetime(pd.Series(["2018-01-10 00:00:00"]))
    assert compute_is_late(delivered, estimated).iloc[0] == 0


def test_target_delivery_after_estimated_day_is_late():
    delivered = pd.to_datetime(pd.Series(["2018-01-11 00:01:00"]))   # اليوم التالي
    estimated = pd.to_datetime(pd.Series(["2018-01-10 00:00:00"]))
    assert compute_is_late(delivered, estimated).iloc[0] == 1


def test_target_missing_date_gives_missing_target_not_zero():
    delivered = pd.to_datetime(pd.Series([None, "2018-01-11 10:00:00"]))
    estimated = pd.to_datetime(pd.Series(["2018-01-10 00:00:00", None]))
    result = compute_is_late(delivered, estimated)
    assert result.isna().all()


# ---------------- الأهلية والاستبعاد على مثال صغير ----------------

def tiny_bundle():
    orders = make_orders([
        dict(order_id="on_time_same_day", order_delivered_customer_date="2018-01-10 23:59:00"),
        dict(order_id="early"),
        dict(order_id="late_next_day", order_delivered_customer_date="2018-01-11 00:01:00"),
        dict(order_id="missing_delivery_date", order_delivered_customer_date=None),
        dict(order_id="canceled_with_date", order_status="canceled"),
        dict(order_id="missing_approval", order_approved_at=None),
        dict(order_id="approval_after_delivery", order_approved_at="2018-01-09 10:00:00"),
        dict(order_id="no_items", has_items=False),
    ])
    return build_modeling_tables(orders)


def test_tiny_eligibility_and_reasons():
    full, modeling, _ex = tiny_bundle()
    expected = {
        "on_time_same_day": "eligible",
        "early": "eligible",
        "late_next_day": "eligible",
        "missing_delivery_date": "delivered_without_customer_date",
        "canceled_with_date": "not_delivered_status",
        "missing_approval": "missing_approval_date",
        "approval_after_delivery": "approval_after_delivery",
        "no_items": "no_items",
    }
    for order_id, reason in expected.items():
        assert row_of(full, order_id)["exclusion_reason"] == reason, order_id
    assert sorted(modeling["order_id"]) == ["early", "late_next_day", "on_time_same_day"]


def test_tiny_target_values_and_excluded_have_no_target():
    full, modeling, _ex = tiny_bundle()
    assert row_of(full, "on_time_same_day")[TARGET] == 0
    assert row_of(full, "early")[TARGET] == 0
    assert row_of(full, "late_next_day")[TARGET] == 1
    for order_id in ("missing_delivery_date", "canceled_with_date", "missing_approval"):
        assert pd.isna(row_of(full, order_id)[TARGET]), order_id
    assert set(modeling[TARGET]) <= {0, 1}


def test_tiny_exclusion_summary_adds_up():
    full, modeling, exclusions = tiny_bundle()
    assert exclusions["orders_excluded"].sum() == len(full) - len(modeling)
    assert exclusions["orders_remaining"].iloc[-1] == len(modeling)


# ---------------- الخصائص بقيم نعرف جوابها ----------------

def test_feature_values_on_known_example():
    df = add_features(make_orders([dict(order_id="x")]))
    r = df.iloc[0]
    assert r["approval_dow"] == 0                      # الاثنين
    assert r["approval_hour"] == 10
    assert r["approval_month"] == 1
    assert r["purchase_to_approval_hours"] == pytest.approx(1.0)
    assert r["planned_days"] == 9                      # من 1 يناير إلى 10 يناير
    assert r["avg_item_price"] == pytest.approx(50.0)
    assert r["freight_share"] == pytest.approx(20 / 120)
    assert r["log_weight_g"] == pytest.approx(np.log1p(1000.0))
    assert r["is_multi_seller"] == 0
    assert r["same_state"] == 0                        # SP مقابل RJ
    assert r["state_route"] == "SP>RJ"


def test_safe_divide_never_returns_inf():
    result = safe_divide(pd.Series([10.0, 5.0, 3.0]), pd.Series([2.0, 0.0, np.nan]))
    assert result.iloc[0] == 5.0
    assert result.iloc[1:].isna().all()
    assert not np.isinf(result.to_numpy()).any()


# ---------------- استبعاد الحقول المحظورة ----------------

def test_forbidden_columns_are_not_model_features():
    assert set(FORBIDDEN_COLUMNS).isdisjoint(MODEL_FEATURES)
    assert TARGET not in MODEL_FEATURES


def test_features_do_not_depend_on_forbidden_columns():
    # نحذف الحقول المحظورة كلها قبل حساب الخصائص: يجب أن تبقى النتيجة نفسها
    orders = make_orders([dict(order_id="a"), dict(order_id="b", n_sellers=3)])
    with_all = add_features(orders)[MODEL_FEATURES]
    without_forbidden = add_features(
        orders.drop(columns=[c for c in FORBIDDEN_COLUMNS if c in orders.columns])
    )[MODEL_FEATURES]
    pd.testing.assert_frame_equal(with_all, without_forbidden)


def test_feature_dictionary_is_complete():
    documented = {e["feature"] for e in FEATURE_DICTIONARY}
    assert set(MODEL_FEATURES) <= documented
    assert sum(1 for e in FEATURE_DICTIONARY if e["derived"]) >= 8
    for e in FEATURE_DICTIONARY:
        for field in ("formula", "source", "why_relevant", "available_at"):
            assert e[field], f"{e['feature']} missing {field}"


# ---------------- اختبارات على البيانات الحقيقية ----------------

def test_real_full_table_has_one_row_per_order(modeling_bundle, order_table_and_log):
    full, _modeling, _ex = modeling_bundle
    table, _log = order_table_and_log
    assert len(full) == len(table)
    assert full["order_id"].is_unique


def test_real_modeling_table_is_valid(modeling_bundle):
    full, modeling, _ex = modeling_bundle
    assert modeling["order_id"].is_unique
    assert len(modeling) == int(full["eligible"].sum())
    assert modeling[TARGET].isin([0, 1]).all()
    assert (modeling["planned_days"] >= 0).all()
    assert set(FORBIDDEN_COLUMNS).isdisjoint(modeling.columns)


def test_real_eligible_orders_are_delivered_with_valid_dates(modeling_bundle):
    full, _modeling, _ex = modeling_bundle
    eligible = full[full["eligible"]]
    assert (eligible["order_status"] == "delivered").all()
    assert eligible["order_delivered_customer_date"].notna().all()
    assert eligible["order_approved_at"].notna().all()
    assert (eligible["order_approved_at"] <= eligible["order_delivered_customer_date"]).all()


def test_real_exclusions_add_up(modeling_bundle):
    full, modeling, exclusions = modeling_bundle
    assert exclusions["orders_excluded"].sum() == len(full) - len(modeling)
    assert exclusions["orders_remaining"].iloc[-1] == len(modeling)


def test_real_target_matches_independent_calculation(modeling_bundle):
    full, _modeling, _ex = modeling_bundle
    eligible = full[full["eligible"]]
    independent = (eligible["order_delivered_customer_date"].dt.normalize()
                   > eligible["order_estimated_delivery_date"].dt.normalize()).astype(int)
    assert (eligible[TARGET].astype(int) == independent).all()
    assert full.loc[~full["eligible"], TARGET].isna().all()