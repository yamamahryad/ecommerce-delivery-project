# tests/test_eda_part2.py
# اختبارات اليوم 6: الشرائح، توازن الهدف، مقارنة الفواصل، الارتباط، المفقود، الرسوم.
# التشغيل: python -m pytest -v

import numpy as np
import pandas as pd
import pytest

from src.eda import prepare
from src.eda_part2 import (CORR_COLUMNS, PEAK_LABEL, _compare, add_planned_columns,
                           build_part2_tables, correlation_matrix, excluded_by_month,
                           make_part2_figures, missingness_vs_target, planned_day_table,
                           representation_by_state, target_balance)
from src.target import TARGET


@pytest.fixture(scope="module")
def part2(modeling_bundle):
    full, _modeling, _ex = modeling_bundle
    el, tables = prepare(full)
    el = add_planned_columns(el)
    t2 = build_part2_tables(full, el, tables)
    return full, el, tables, t2


# ---------------- أمثلة صغيرة نعرف جوابها ----------------

def test_planned_group_boundaries_and_period():
    df = pd.DataFrame({
        "planned_days": [0, 10, 11, 15, 16, 40, 41, 120],
        "order_purchase_timestamp": pd.to_datetime(["2017-11-05"] * 4 + ["2017-06-01"] * 4),
        TARGET: [0] * 8,
    })
    out = add_planned_columns(df)
    assert out["planned_group"].astype(str).tolist() == [
        "0-10", "0-10", "11-15", "11-15", "16-20", "31-40", "41+", "41+"]
    assert (out["period"].iloc[:4] == PEAK_LABEL).all()
    assert (out["period"].iloc[4:] == "other months").all()


def test_target_balance_tiny():
    el = pd.DataFrame({TARGET: [0] * 8 + [1] * 2})
    t = dict(target_balance(el).values)
    assert t["late share (%)"] == 20.0
    assert t["imbalance ratio (on time : late)"] == 4.0
    assert t["accuracy of always predicting 'on time' (%)"] == 80.0


def test_compare_detects_overlap_and_separation():
    separated = pd.DataFrame({"g": ["A", "B"], "n": [1000, 1000], "rate": [0.10, 0.02],
                              "ci_low": [0.08, 0.01], "ci_high": [0.12, 0.03]})
    row = _compare("p", separated, "g", "A", "B")
    assert row["intervals_overlap"] is False or row["intervals_overlap"] == False  # noqa: E712
    assert row["difference_pp"] == 8.0

    overlapping = pd.DataFrame({"g": ["A", "B"], "n": [100, 100], "rate": [0.05, 0.06],
                                "ci_low": [0.02, 0.03], "ci_high": [0.09, 0.10]})
    assert _compare("p", overlapping, "g", "A", "B")["intervals_overlap"]


def test_missingness_vs_target_tiny():
    df = pd.DataFrame({"a": [1, np.nan, 3, np.nan],
                       "primary_category": ["x", "unknown", "x", "x"],
                       TARGET: [0, 1, 0, 1]})
    out = missingness_vs_target(df, columns=["a"]).set_index("feature")
    assert out.loc["a", "n_missing"] == 2
    assert out.loc["a", "late_rate_when_missing_pct"] == 100.0
    assert out.loc["a", "late_rate_when_observed_pct"] == 0.0
    assert out.loc["primary_category == 'unknown'", "n_missing"] == 1


# ---------------- على البيانات الحقيقية ----------------

def test_real_planned_groups_cover_all_orders(part2):
    _full, el, _tables, t2 = part2
    assert el["planned_group"].notna().all()
    assert t2["late_by_planned_group"]["n"].sum() == len(el)
    assert t2["late_by_planned_x_route"]["n"].sum() == len(el)
    assert t2["late_by_period_x_route"]["n"].sum() == len(el)


def test_real_planned_day_table_respects_min_cell(part2):
    _full, el, _tables, _t2 = part2
    t = planned_day_table(el, min_cell_n=30)
    assert (t["n"] >= 30).all()
    assert t["rate"].between(0, 1).all()


def test_real_correlation_matrix_is_valid(part2):
    _full, el, _tables, _t2 = part2
    corr = correlation_matrix(el)
    assert corr.shape == (len(CORR_COLUMNS) + 1, len(CORR_COLUMNS) + 1)
    assert np.allclose(corr.to_numpy(), corr.to_numpy().T, equal_nan=True)
    assert np.allclose(np.diag(corr.to_numpy()), 1.0)
    assert (corr.abs().to_numpy() <= 1.000001).all()


def test_real_exclusion_tables_add_up(part2):
    full, el, _tables, t2 = part2
    assert t2["exclusion_reasons"]["n_orders"].sum() == len(full) - len(el)
    assert excluded_by_month(full)["orders_excluded"].sum() == len(full) - len(el)
    states = representation_by_state(full)
    assert states["n_all"].sum() == len(full)
    assert states["n_eligible"].sum() == len(el)
    assert states["share_all_pct"].sum() == pytest.approx(100, abs=0.01)
    assert states["share_eligible_pct"].sum() == pytest.approx(100, abs=0.01)


def test_real_missingness_counts_match(part2):
    _full, el, _tables, t2 = part2
    for _, row in t2["missingness_vs_target"].iterrows():
        if row["feature"] in el.columns:
            assert row["n_missing"] == int(el[row["feature"]].isna().sum())


def test_real_pattern_summary(part2):
    _full, _el, _tables, t2 = part2
    p = t2["pattern_summary"]
    assert len(p) == 8
    route = p[p["pattern"] == "Route type"].iloc[0]
    assert route["rate_a_pct"] > route["rate_b_pct"]      # ولاية مختلفة أعلى من نفس الولاية
    assert not route["intervals_overlap"]


def test_real_part2_figures_created(part2, tmp_path):
    full, el, tables, t2 = part2
    paths = make_part2_figures(full, el, tables, t2, tmp_path)
    assert len(paths) == 6
    for p in paths:
        assert p.exists() and p.stat().st_size > 5_000