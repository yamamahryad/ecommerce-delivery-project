# tests/test_eda.py
# اختبارات التحليل الاستكشافي: فاصل Wilson، المقامات، الخلايا الصغيرة، والرسوم.
# التشغيل: python -m pytest -v

import numpy as np
import pandas as pd
import pytest

from src.eda import (MIN_N, make_all_figures, monthly_summary, prepare, rate_table,
                     route_matrix, wilson_interval)
from src.target import TARGET


def test_wilson_known_values():
    low, high = wilson_interval(0, 10)
    assert float(low) == pytest.approx(0.0, abs=1e-3)
    assert float(high) == pytest.approx(0.2775, abs=1e-3)
    low, high = wilson_interval(50, 100)
    assert float(low) == pytest.approx(0.4038, abs=1e-3)
    assert float(high) == pytest.approx(0.5962, abs=1e-3)


def test_wilson_stays_inside_zero_one_and_handles_empty_group():
    low, high = wilson_interval(np.array([0, 5, 10]), np.array([10, 10, 10]))
    assert (low >= 0).all() and (high <= 1).all()
    low, high = wilson_interval(0, 0)
    assert np.isnan(float(low)) and np.isnan(float(high))


def test_rate_table_on_small_example():
    df = pd.DataFrame({"g": ["A", "A", "A", "B", "B"], TARGET: [1, 0, 0, 1, 1]})
    t = rate_table(df, "g", min_n=3).set_index("g")
    assert t.loc["A", "n"] == 3 and t.loc["A", "late"] == 1
    assert t.loc["A", "rate"] == pytest.approx(1 / 3)
    assert not t.loc["A", "small_sample"]
    assert t.loc["B", "small_sample"]               # n=2 أقل من 3
    assert t.loc["B", "rate"] == 1.0


def test_monthly_summary_counts_and_missing_rate():
    full = pd.DataFrame({
        "order_purchase_timestamp": pd.to_datetime(["2018-01-05", "2018-01-20", "2018-02-03"]),
        "eligible": [True, True, False],
        TARGET: [1.0, 0.0, np.nan],
    })
    m = monthly_summary(full).set_index("month")
    jan, feb = m.iloc[0], m.iloc[1]
    assert (jan["orders_all"], jan["orders_eligible"], jan["late_orders"]) == (2, 2, 1)
    assert jan["late_rate"] == 0.5
    assert (feb["orders_all"], feb["orders_eligible"]) == (1, 0)
    assert np.isnan(feb["late_rate"])               # لا نخترع نسبة لشهر بلا طلبات مؤهلة


def test_route_matrix_blanks_small_cells():
    rows = ([{"primary_seller_state": "SP", "customer_state": "SP", TARGET: int(i < 10)}
             for i in range(40)]
            + [{"primary_seller_state": "SP", "customer_state": "RJ", TARGET: 1} for _ in range(5)])
    rate, n = route_matrix(pd.DataFrame(rows), top_sellers=2, top_customers=2, min_cell_n=30)
    assert n.loc["SP", "SP"] == 40 and n.loc["SP", "RJ"] == 5
    assert rate.loc["SP", "SP"] == pytest.approx(0.25)
    assert pd.isna(rate.loc["SP", "RJ"])            # 5 طلبات فقط: الخلية فارغة


# ---------------- على البيانات الحقيقية ----------------

def test_real_denominators_add_up(modeling_bundle):
    full, modeling, _ex = modeling_bundle
    el, tables = prepare(full)
    assert len(el) == len(modeling)
    m = tables["monthly_summary"]
    assert m["orders_all"].sum() == len(full)
    assert m["orders_eligible"].sum() == len(el)
    assert m["late_orders"].sum() == el[TARGET].sum()
    for name in ("late_by_state", "late_by_seller_group", "late_by_same_state",
                 "late_by_category", "late_by_n_items", "late_by_price_group"):
        assert tables[name]["n"].sum() == len(el), name
        assert tables[name]["rate"].between(0, 1).all(), name


def test_real_figures_are_created(modeling_bundle, tmp_path):
    full, _modeling, _ex = modeling_bundle
    el, tables = prepare(full)
    paths = make_all_figures(el, tables, tmp_path)
    assert len(paths) == 6
    for p in paths:
        assert p.exists() and p.stat().st_size > 5_000