# tests/test_modeling.py
# اختبارات التقسيم الزمني وخط المعالجة ومنع التسرب.
# التشغيل: python -m pytest -q

import inspect

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score

from src import config
from src.features import (CATEGORICAL_FEATURES, FORBIDDEN_COLUMNS, MODEL_FEATURES,
                          NUMERIC_FEATURES)
from src.modeling import (build_preprocessor, make_baseline, make_hgb_pipeline,
                          make_logistic_pipeline, run_experiments)
from src.splitting import (SPLIT_NAMES, TIME_COL, get_xy, split_summary,
                           temporal_split)
from src.target import TARGET


@pytest.fixture(scope="module")
def splits(modeling_bundle):
    _full, modeling, _ex = modeling_bundle
    return temporal_split(modeling)


def tiny_frame(n=300, seed=0):
    # جدول صغير بكل أعمدة النمذجة، نصنعه بأنفسنا
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({c: rng.normal(size=n) for c in NUMERIC_FEATURES})
    states = np.array(["SP", "RJ", "MG"])
    df["customer_state"] = rng.choice(states, n)
    df["primary_seller_state"] = rng.choice(states, n)
    df["primary_category"] = rng.choice(["a", "b"], n)
    df["main_payment_type"] = rng.choice(["credit_card", "boleto"], n)
    df["order_id"] = [f"o{i}" for i in range(n)]
    df[TIME_COL] = pd.date_range("2018-01-01", periods=n, freq="h")
    y = (rng.random(n) < 0.2).astype(int)
    y[:2] = [0, 1]                       # نضمن وجود الفئتين
    df[TARGET] = y
    return df


# ---------------- التقسيم الزمني ----------------

def test_split_partitions_all_orders_without_overlap(splits, modeling_bundle):
    _full, modeling, _ex = modeling_bundle
    ids = [set(splits[name]["order_id"]) for name in SPLIT_NAMES]
    assert sum(len(s) for s in ids) == len(modeling)
    assert ids[0].isdisjoint(ids[1]) and ids[0].isdisjoint(ids[2]) and ids[1].isdisjoint(ids[2])
    assert set().union(*ids) == set(modeling["order_id"])


def test_split_is_strictly_chronological_and_test_is_newest(splits):
    assert splits["train"][TIME_COL].max() < splits["val"][TIME_COL].min()
    assert splits["val"][TIME_COL].max() < splits["test"][TIME_COL].min()


def test_split_fractions_are_close_to_target(splits):
    total = sum(len(d) for d in splits.values())
    for name, frac in zip(SPLIT_NAMES, config.SPLIT_FRACTIONS):
        assert len(splits[name]) / total == pytest.approx(frac, abs=0.01)


def test_split_with_tied_timestamps_never_shares_a_timestamp():
    times = pd.to_datetime(["2018-01-01"] * 4 + ["2018-01-02"] * 4 + ["2018-01-03"] * 4)
    df = pd.DataFrame({"order_id": list("abcdefghijkl"), TIME_COL: times,
                       TARGET: [0, 1] * 6})
    parts = temporal_split(df)
    assert sum(len(p) for p in parts.values()) == len(df)
    assert parts["train"][TIME_COL].max() < parts["val"][TIME_COL].min() \
        if len(parts["val"]) else True
    seen = [set(p[TIME_COL]) for p in parts.values()]
    assert seen[0].isdisjoint(seen[1]) and seen[0].isdisjoint(seen[2]) and seen[1].isdisjoint(seen[2])


def test_split_summary_adds_up(splits):
    s = split_summary(splits)
    assert s["orders"].sum() == sum(len(d) for d in splits.values())
    assert s["late_orders"].sum() == sum(int(d[TARGET].sum()) for d in splits.values())
    assert s["share_pct"].sum() == pytest.approx(100, abs=0.05)


# ---------------- المدخلات ومنع التسرب ----------------

def test_xy_uses_only_the_whitelist(splits):
    X, y = get_xy(splits["train"])
    assert list(X.columns) == MODEL_FEATURES
    for col in FORBIDDEN_COLUMNS + ["order_id", "customer_unique_id", TIME_COL,
                                    "order_purchase_timestamp", TARGET]:
        assert col not in X.columns
    assert set(y.unique()) <= {0, 1}


def test_preprocessor_learns_statistics_from_train_only():
    
    train, other = tiny_frame(200, seed=1), tiny_frame(600, seed=2)
    first = NUMERIC_FEATURES[0]
    other[first] = other[first] + 1000          # قيم متطرفة خارج التدريب
    X_tr, _ = get_xy(train)
    X_other, _ = get_xy(other)

    pre = build_preprocessor(scale=True).fit(X_tr)
    learned = pre.named_transformers_["num"].named_steps["impute"].statistics_[0]
    pooled = pd.concat([X_tr[first], X_other[first]]).median()
    assert learned == pytest.approx(X_tr[first].median())
    assert abs(learned - pooled) > 1          # لو تسرب شيء لاقتربا من بعضهما


def test_unseen_category_does_not_crash():
    train, test = tiny_frame(200, seed=3), tiny_frame(50, seed=4)
    test["customer_state"] = "ZZ"              # ولاية لم تظهر في التدريب
    X_tr, y_tr = get_xy(train)
    X_te, _ = get_xy(test)
    proba = make_logistic_pipeline().fit(X_tr, y_tr).predict_proba(X_te)[:, 1]
    assert np.isfinite(proba).all()


def test_pipeline_handles_missing_values():
    train = tiny_frame(200, seed=5)
    train.loc[:20, NUMERIC_FEATURES[0]] = np.nan
    train.loc[:20, "primary_category"] = np.nan
    X, y = get_xy(train)
    proba = make_logistic_pipeline().fit(X, y).predict_proba(X)[:, 1]
    assert np.isfinite(proba).all() and ((proba >= 0) & (proba <= 1)).all()


# ---------------- النماذج ----------------

def test_baseline_predicts_majority_with_train_prevalence():
    train = tiny_frame(300, seed=6)
    X, y = get_xy(train)
    pipe = make_baseline().fit(X, y)
    assert (pipe.predict(X) == 0).all()
    assert np.allclose(pipe.predict_proba(X)[:, 1], y.mean())


def test_hgb_pipeline_runs_on_small_data():
    train = tiny_frame(300, seed=7)
    X, y = get_xy(train)
    proba = make_hgb_pipeline().fit(X, y).predict_proba(X)
    assert proba.shape == (300, 2)


def test_run_experiments_cannot_receive_the_test_set():
    assert list(inspect.signature(run_experiments).parameters) == ["train", "val"]


def test_real_smoke_logistic_on_a_subset(splits):
    X_tr, y_tr = get_xy(splits["train"].head(20000))
    X_va, y_va = get_xy(splits["val"])
    proba = make_logistic_pipeline().fit(X_tr, y_tr).predict_proba(X_va)[:, 1]
    assert np.isfinite(proba).all()
    assert 0 <= average_precision_score(y_va, proba) <= 1
    # القاعدة الثابتة: AP تساوي نسبة التأخر
    constant = np.full(len(y_va), 0.3)
    assert average_precision_score(y_va, constant) == pytest.approx(y_va.mean(), abs=1e-6)