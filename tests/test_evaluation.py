# tests/test_evaluation.py
# اختبارات المقاييس والعتبة والتجميد وتحليل الأخطاء.
# التشغيل: python -m pytest -q

import numpy as np
import pandas as pd
import pytest

from src import config
from src.evaluation import (choose_threshold_f1, classification_metrics, error_by_group,
                            load_decisions, save_decisions, topk_table)
from src.modeling import make_logistic_pipeline
from src.splitting import get_xy, temporal_split
from src.target import TARGET


def test_threshold_on_perfectly_separable_scores():
    y = [0, 0, 0, 1, 1]
    p = [0.1, 0.2, 0.3, 0.8, 0.9]
    thr, f1 = choose_threshold_f1(y, p)
    assert f1 == pytest.approx(1.0)
    assert thr == pytest.approx(0.8)            # أول عتبة تفصل الفئتين تماماً


def test_metrics_on_a_known_example():
    y = [1, 1] + [0] * 8
    p = [0.9, 0.4, 0.8] + [0.1] * 7             # عند 0.5: TP=1, FN=1, FP=1, TN=7
    m = classification_metrics(y, p, 0.5)
    assert (m["tp"], m["fn"], m["fp"], m["tn"]) == (1, 1, 1, 7)
    assert m["accuracy"] == pytest.approx(0.8)
    assert m["precision"] == pytest.approx(0.5)
    assert m["recall"] == pytest.approx(0.5)
    assert m["f1"] == pytest.approx(0.5)
    assert m["flagged_pct"] == pytest.approx(20.0)


def test_metrics_when_nothing_is_predicted_late_do_not_crash():
    y = [1, 0, 0, 0]
    m = classification_metrics(y, np.full(4, 0.05), 0.5)   # مثل النموذج المرجعي
    assert m["precision"] == 0 and m["recall"] == 0 and m["f1"] == 0
    assert m["accuracy"] == pytest.approx(0.75)             # دقة عالية بلا أي فائدة
    assert m["average_precision"] == pytest.approx(0.25)    # AP لقاعدة ثابتة = نسبة التأخر


def test_topk_with_perfect_ranking():
    y = [1, 1] + [0] * 8
    p = np.linspace(1.0, 0.1, 10)
    t = topk_table(y, p, fractions=(0.2,)).iloc[0]
    assert t["orders_flagged"] == 2 and t["late_caught"] == 2
    assert t["precision_pct"] == 100.0 and t["recall_pct"] == 100.0
    assert t["lift"] == pytest.approx(5.0)                  # (1.0) / (0.2)


def test_error_by_group_counts():
    df = pd.DataFrame({"g": ["A"] * 4, TARGET: [1, 1, 0, 0], "pred": [1, 0, 1, 0]})
    row = error_by_group(df, "g").iloc[0]
    assert (row["n"], row["tp"], row["fp"], row["fn"]) == (4, 1, 1, 1)
    assert row["precision"] == pytest.approx(0.5)
    assert row["recall"] == pytest.approx(0.5)
    assert row["late_rate_pct"] == pytest.approx(50.0)
    assert row["small_sample"]                              # n=4 أقل من 200


def test_frozen_decisions_roundtrip_is_exact(tmp_path):
    decisions = {"selected_model": "m", "thresholds": {"m": 0.1234567890123456},
                 "model_sha256": {"m": "abc"}}
    path = tmp_path / "decisions.json"
    save_decisions(decisions, path)
    assert load_decisions(path) == decisions                # العتبة تعود بالدقة نفسها


def test_validation_script_never_touches_the_test_split():
    text = (config.PROJECT_ROOT / "run_day8_validation.py").read_text(encoding="utf-8")
    assert 'splits["test"]' not in text


def test_real_metrics_are_consistent(modeling_bundle):
    _full, modeling, _ex = modeling_bundle
    splits = temporal_split(modeling)
    X_tr, y_tr = get_xy(splits["train"].head(20000))
    X_va, y_va = get_xy(splits["val"])
    proba = make_logistic_pipeline().fit(X_tr, y_tr).predict_proba(X_va)[:, 1]
    thr, f1 = choose_threshold_f1(y_va, proba)
    m = classification_metrics(y_va, proba, thr)
    assert m["tp"] + m["fp"] + m["fn"] + m["tn"] == len(y_va)
    assert m["f1"] == pytest.approx(f1, abs=1e-9)
    assert 0 <= m["precision"] <= 1 and 0 <= m["recall"] <= 1