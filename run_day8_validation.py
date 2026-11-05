# run_day8_validation.py
# المرحلة 1: اختيار العتبة والنموذج النهائي على التحقق، ثم تجميد القرارات.
# هذا الملف لا يقرأ مجموعة الاختبار إطلاقاً.
# التشغيل: python run_day8_validation.py   (بعد run_day7.py)

import sys

import pandas as pd

from src import config
from src.evaluation import (BASELINE_THRESHOLD, calibration_table, choose_threshold_f1,
                            evaluate_models, file_sha256, save_decisions,
                            select_final_model, topk_table)
from src.evaluation_plots import fig_threshold_selection
from src.features import load_modeling_table
from src.modeling import BASELINE, HGB, LOGISTIC, load_model
from src.splitting import get_xy, temporal_split

CANDIDATES = [LOGISTIC, HGB]


def show(title, df):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)
    print(df.round(4).to_string(index=False))


def main():
    pd.set_option("display.width", 220)
    config.ensure_output_dirs()

    if config.TEST_METRICS_FILE.exists() and "--force" not in sys.argv:
        raise SystemExit("The final evaluation already exists. Changing thresholds or the "
                         "selected model after seeing it is not allowed.")

    splits = temporal_split(load_modeling_table())
    X_val, y_val = get_xy(splits["val"])
    names = [BASELINE] + CANDIDATES
    models = {name: load_model(name) for name in names}
    val_probas = {name: models[name].predict_proba(X_val)[:, 1] for name in names}

    # 1) العتبة لكل نموذج: تعظيم F1 على التحقق
    thresholds = {BASELINE: BASELINE_THRESHOLD}
    rows = []
    for name in CANDIDATES:
        thr, f1 = choose_threshold_f1(y_val, val_probas[name])
        thresholds[name] = thr
        rows.append({"model": name, "chosen_threshold": thr, "validation_f1_at_threshold": f1})
    threshold_table = pd.DataFrame(rows)

    # 2) النموذج النهائي: أعلى AP على التحقق (معيار معلن قبل رؤية الاختبار)
    selected, selection_table = select_final_model(y_val, val_probas, CANDIDATES)

    val_metrics = evaluate_models(y_val, val_probas, thresholds, "validation")
    val_topk = pd.concat([topk_table(y_val, val_probas[n]).assign(model=n) for n in CANDIDATES],
                         ignore_index=True)
    val_calibration = calibration_table(y_val, val_probas, "validation")

    show("1) Chosen thresholds (maximize F1 on VALIDATION)", threshold_table)
    show("2) Model selection (highest validation average precision)", selection_table)
    show("3) Validation metrics at the chosen thresholds", val_metrics[
        ["model", "threshold", "accuracy", "precision", "recall", "f1", "average_precision",
         "tn", "fp", "fn", "tp", "flagged_pct", "precision_lift"]])
    show("4) Validation: reviewing the top k% highest-risk orders", val_topk)
    show("5) Validation calibration: mean predicted vs actual late rate", val_calibration)
    print(f"\nSELECTED MODEL (decided before the test set is touched): {selected}")

    # 3) تجميد القرارات مع بصمة ملفات النماذج
    decisions = {
        "threshold_criterion": "maximize F1 of the late class on the validation set",
        "model_criterion": "highest average precision on the validation set",
        "selected_model": selected,
        "thresholds": thresholds,
        "model_sha256": {n: file_sha256(config.MODELS_DIR / f"{n}.joblib") for n in names},
    }
    save_decisions(decisions, config.DECISIONS_FILE)

    out = config.TABLES_DIR
    threshold_table.to_csv(out / "val_threshold_choice.csv", index=False, encoding="utf-8-sig")
    selection_table.to_csv(out / "val_model_selection.csv", index=False, encoding="utf-8-sig")
    val_metrics.to_csv(out / "val_metrics.csv", index=False, encoding="utf-8-sig")
    val_topk.to_csv(out / "val_topk.csv", index=False, encoding="utf-8-sig")
    val_calibration.to_csv(out / "val_calibration.csv", index=False, encoding="utf-8-sig")
    fig_threshold_selection(y_val, val_probas, thresholds, CANDIDATES,
                            config.FIGURES_DIR / "fig13_threshold_selection_validation.png")

    print(f"\nDecisions frozen in {config.DECISIONS_FILE.name}. Next: python run_day8_test.py")


if __name__ == "__main__":
    main()