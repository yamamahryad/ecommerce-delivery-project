# run_day8_test.py
# المرحلة 2: تقييم نهائي واحد على الاختبار بالقرارات المجمدة، ثم تحليل الأخطاء.
# التشغيل: python run_day8_test.py
# إعادة التشغيل مرفوضة دون --force (والأرقام تبقى نفسها لأن كل شيء حتمي).

import sys

import numpy as np
import pandas as pd

from src import config
from src.eda_part2 import PLANNED_BINS, PLANNED_LABELS
from src.evaluation import (calibration_table, error_by_group, evaluate_models, file_sha256,
                            load_decisions, logistic_coefficients, outcome_profile,
                            topk_table, worst_errors)
from src.evaluation_plots import (fig_confusion, fig_error_by_month, fig_pr_curves,
                                  fig_score_distribution)
from src.features import load_modeling_table
from src.modeling import BASELINE, HGB, LOGISTIC, load_model
from src.splitting import get_xy, temporal_split
from src.target import TARGET


def show(title, df):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)
    print(df.round(4).to_string(index=False))


def main():
    pd.set_option("display.width", 220)
    config.ensure_output_dirs()

    if config.TEST_METRICS_FILE.exists() and "--force" not in sys.argv:
        raise SystemExit("The final test evaluation was already done. Evaluating again to tune "
                         "anything is not allowed. Use --force only to reproduce the same numbers.")

    decisions = load_decisions(config.DECISIONS_FILE)
    thresholds = decisions["thresholds"]
    selected = decisions["selected_model"]
    names = [BASELINE, LOGISTIC, HGB]

    # النماذج لم تتغير منذ تجميد العتبات؟
    for name in names:
        current = file_sha256(config.MODELS_DIR / f"{name}.joblib")
        if current != decisions["model_sha256"][name]:
            raise SystemExit(f"Model file '{name}' changed after the thresholds were frozen. "
                             "Run python run_day8_validation.py again BEFORE the test evaluation.")

    splits = temporal_split(load_modeling_table())
    test = splits["test"].copy()
    X_test, y_test = get_xy(test)
    models = {name: load_model(name) for name in names}
    probas = {name: models[name].predict_proba(X_test)[:, 1] for name in names}

    metrics = evaluate_models(y_test, probas, thresholds, "test")
    calibration = calibration_table(y_test, probas, "test")
    topk = pd.concat([topk_table(y_test, probas[n]).assign(model=n) for n in (LOGISTIC, HGB)],
                     ignore_index=True)

    # ---- تحليل الأخطاء للنموذج المختار ----
    t = test.copy()
    t["proba"] = probas[selected]
    t["pred"] = (t["proba"] >= thresholds[selected]).astype(int)
    t["approval_month"] = t["order_approved_at"].dt.to_period("M").astype(str)
    t["route_group"] = np.where(t["same_state"] == 1, "same state", "different state")
    t["seller_group"] = np.where(t["is_multi_seller"] == 1, "multi-seller", "single seller")
    t["planned_group"] = pd.cut(t["planned_days"], bins=PLANNED_BINS, labels=PLANNED_LABELS)

    err_tables = {
        "error_by_month": error_by_group(t, "approval_month"),
        "error_by_route": error_by_group(t, "route_group"),
        "error_by_planned_window": error_by_group(t, "planned_group"),
        "error_by_seller_structure": error_by_group(t, "seller_group"),
        "error_by_customer_state": error_by_group(t, "customer_state")
                                   .sort_values("n", ascending=False).head(10),
    }
    profile = outcome_profile(t)
    fp_cases, fn_cases = worst_errors(t)

    print(f"Selected model (frozen before the test): {selected}  "
          f"threshold={thresholds[selected]:.4f}")
    print(f"Test orders: {len(test):,}   late rate: {y_test.mean() * 100:.2f}%")
    show("1) FINAL test metrics (frozen thresholds; last row = reference rule)", metrics[
        ["model", "threshold", "accuracy", "precision", "recall", "f1", "average_precision",
         "roc_auc", "tn", "fp", "fn", "tp", "flagged_pct", "precision_lift"]])
    show("2) Calibration: mean predicted vs actual late rate (distribution shift)", calibration)
    show("3) Test: reviewing the top k% highest-risk orders", topk)
    show("4) Error analysis by approval month", err_tables["error_by_month"])
    show("5) Error analysis by route type", err_tables["error_by_route"])
    show("6) Error analysis by planned delivery window", err_tables["error_by_planned_window"])
    show("7) Error analysis by seller structure", err_tables["error_by_seller_structure"])
    show("8) Error analysis by customer state (10 largest)", err_tables["error_by_customer_state"])
    show("9) Profile of TP / FP / FN / TN orders", profile)
    show("10) False alarms with the highest scores (FP)", fp_cases)
    show("11) Late orders with the lowest scores (FN, missed)", fn_cases)

    out = config.TABLES_DIR
    coefs = None
    if selected == LOGISTIC:
        coefs = logistic_coefficients(models[LOGISTIC])
        show("12) Logistic regression: largest standardized coefficients", coefs)
        coefs.to_csv(out / "test_logistic_coefficients.csv", index=False, encoding="utf-8-sig")

    for name, table in err_tables.items():
        table.to_csv(out / f"test_{name}.csv", index=False, encoding="utf-8-sig")
    calibration.to_csv(out / "test_calibration.csv", index=False, encoding="utf-8-sig")
    topk.to_csv(out / "test_topk.csv", index=False, encoding="utf-8-sig")
    profile.to_csv(out / "test_outcome_profile.csv", index=False, encoding="utf-8-sig")
    fp_cases.to_csv(out / "test_worst_false_positives.csv", index=False, encoding="utf-8-sig")
    fn_cases.to_csv(out / "test_worst_false_negatives.csv", index=False, encoding="utf-8-sig")

    preds = test[["order_id", TARGET]].copy()
    for name in names:
        preds[f"proba_{name}"] = probas[name]
    preds["pred_selected"] = t["pred"]
    preds.to_csv(config.TEST_PREDICTIONS_FILE, index=False)

    fig_names = [LOGISTIC, HGB]
    fig_pr_curves(y_test, probas, thresholds, fig_names, config.FIGURES_DIR / "fig14_pr_curves_test.png")
    fig_confusion(y_test, probas, thresholds, names, config.FIGURES_DIR / "fig15_confusion_matrices_test.png")
    fig_score_distribution(y_test, probas[selected], thresholds[selected], selected,
                           config.FIGURES_DIR / "fig16_score_distribution_test.png")
    fig_error_by_month(err_tables["error_by_month"], config.FIGURES_DIR / "fig17_error_by_month_test.png")

    # يُكتب آخراً: وجوده يعني أن التقييم النهائي اكتمل
    metrics.to_csv(config.TEST_METRICS_FILE, index=False, encoding="utf-8-sig")
    print("\nFinal evaluation done. Do not change the model, features or thresholds after this.")


if __name__ == "__main__":
    main()