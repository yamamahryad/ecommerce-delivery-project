# src/evaluation.py
# مقاييس التصنيف، اختيار العتبة على التحقق، تجميد القرارات، وجداول تحليل الأخطاء.
# لا يحتوي هذا الملف أي اتصال بمجموعة الاختبار: يستقبل مصفوفات y والاحتمالات فقط.

import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score)

from src.target import TARGET

BASELINE_THRESHOLD = 0.5   # المرجع يتنبأ دائماً بـ "في الموعد"


# ------------------------------------------------------------------ العتبة والمقاييس

def choose_threshold_f1(y_true, proba):
    # يجرب كل عتبة ممكنة ويعيد العتبة التي تعطي أعلى F1 للفئة المتأخرة.
    # التنبؤ: متأخر إذا كان الاحتمال >= العتبة (نفس قاعدة precision_recall_curve).
    y = np.asarray(y_true)
    precision, recall, thresholds = precision_recall_curve(y, np.asarray(proba))
    p, r = precision[:-1], recall[:-1]            # آخر عنصر لا يقابل عتبة
    denom = p + r
    f1 = np.divide(2 * p * r, denom, out=np.zeros_like(denom), where=denom > 0)
    best = int(np.argmax(f1))
    return float(thresholds[best]), float(f1[best])


def classification_metrics(y_true, proba, threshold):
    # كل المقاييس عند عتبة واحدة. zero_division=0: إن لم يتنبأ النموذج بأي "متأخر" يكون الإحكام 0.
    y = np.asarray(y_true)
    proba = np.asarray(proba, dtype=float)
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    prevalence = float(y.mean())
    precision = precision_score(y, pred, zero_division=0)
    return {
        "threshold": float(threshold),
        "accuracy": accuracy_score(y, pred),
        "precision": precision,
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "average_precision": average_precision_score(y, proba),   # يعتمد على الاحتمالات لا العتبة
        "roc_auc": roc_auc_score(y, proba),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "flagged_pct": float(pred.mean() * 100),
        "precision_lift": float(precision / prevalence) if prevalence > 0 else float("nan"),
        "late_rate_pct": prevalence * 100,
    }


def evaluate_models(y_true, probas, thresholds, split, include_reference=True):
    # جدول مقارنة: صف لكل نموذج، وسطر مرجعي لقاعدة "كل الطلبات متأخرة"
    rows = []
    for name, proba in probas.items():
        rows.append({"model": name, "split": split,
                     **classification_metrics(y_true, proba, thresholds[name])})
    if include_reference:
        ones = np.ones(len(y_true))
        rows.append({"model": "reference_always_late", "split": split,
                     **classification_metrics(y_true, ones, 0.5)})
    return pd.DataFrame(rows)


def select_final_model(y_val, val_probas, candidates):
    # المعيار المعلن: أعلى Average Precision على التحقق بين النموذجين (المرجع غير مرشح)
    rows = [{"model": n, "val_average_precision": average_precision_score(y_val, val_probas[n])}
            for n in candidates]
    table = pd.DataFrame(rows).sort_values("val_average_precision", ascending=False)
    table = table.reset_index(drop=True)
    return str(table.loc[0, "model"]), table


def topk_table(y_true, proba, fractions=(0.02, 0.05, 0.10, 0.20, 0.30)):
    # لو راجعنا أعلى k% من الطلبات خطراً: كم متأخراً نلتقط؟ (تفسير تشغيلي بلا عتبة)
    y = np.asarray(y_true)
    p = np.asarray(proba, dtype=float)
    order = np.argsort(-p, kind="mergesort")      # ترتيب مستقر عند التعادل
    prevalence = y.mean()
    rows = []
    for f in fractions:
        k = max(1, int(round(len(y) * f)))
        caught = int(y[order[:k]].sum())
        rows.append({
            "flagged_share_pct": round(f * 100, 1),
            "orders_flagged": k,
            "late_caught": caught,
            "precision_pct": round(caught / k * 100, 2),
            "recall_pct": round(caught / y.sum() * 100, 2),
            "lift": round((caught / k) / prevalence, 2),
        })
    return pd.DataFrame(rows)


def calibration_table(y_true, probas, split):
    # متوسط الاحتمال المتوقع مقابل نسبة التأخر الفعلية (يكشف تضخم الاحتمالات بسبب الانزياح)
    y = np.asarray(y_true)
    rows = [{"model": name, "split": split,
             "mean_predicted_pct": round(float(np.mean(p)) * 100, 2),
             "actual_late_pct": round(float(y.mean()) * 100, 2)}
            for name, p in probas.items()]
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ تجميد القرارات

def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save_decisions(decisions, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(decisions, f, indent=2)


def load_decisions(path):
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: python run_day8_validation.py")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ------------------------------------------------------------------ تحليل الأخطاء

def error_by_group(df, group_col, y_col=TARGET, pred_col="pred", min_n=200):
    # لكل مجموعة: الحجم، نسبة التأخر الفعلية، نسبة ما صُنّف متأخراً، الإحكام، الاستدعاء، FP، FN
    d = df.copy()
    d["_tp"] = ((d[y_col] == 1) & (d[pred_col] == 1)).astype(int)
    d["_fp"] = ((d[y_col] == 0) & (d[pred_col] == 1)).astype(int)
    d["_fn"] = ((d[y_col] == 1) & (d[pred_col] == 0)).astype(int)
    t = d.groupby(group_col, observed=True).agg(
        n=(y_col, "size"), late=(y_col, "sum"), flagged=(pred_col, "sum"),
        tp=("_tp", "sum"), fp=("_fp", "sum"), fn=("_fn", "sum"))
    t["late_rate_pct"] = t["late"] / t["n"] * 100
    t["flagged_pct"] = t["flagged"] / t["n"] * 100
    t["precision"] = t["tp"] / t["flagged"].where(t["flagged"] > 0)   # NaN إن لم يُصنَّف شيء
    t["recall"] = t["tp"] / t["late"].where(t["late"] > 0)
    t["small_sample"] = t["n"] < min_n
    return t.round(4).reset_index()


def outcome_profile(df):
    # خصائص الطلبات في الخانات الأربع: TP و FP و FN و TN
    d = df.copy()
    d["outcome"] = np.select(
        [(d[TARGET] == 1) & (d["pred"] == 1),
         (d[TARGET] == 0) & (d["pred"] == 1),
         (d[TARGET] == 1) & (d["pred"] == 0)],
        ["TP", "FP", "FN"], default="TN")
    d["same_state_pct"] = d["same_state"] * 100
    d["multi_seller_pct"] = d["is_multi_seller"] * 100
    g = d.groupby("outcome").agg(
        n=("order_id", "size"), mean_proba=("proba", "mean"),
        median_planned_days=("planned_days", "median"),
        same_state_pct=("same_state_pct", "mean"),
        multi_seller_pct=("multi_seller_pct", "mean"),
        median_items_price=("items_price_sum", "median"),
        mean_freight=("items_freight_sum", "mean"))
    return g.reindex(["TP", "FP", "FN", "TN"]).round(3).reset_index()


def worst_errors(df, k=8):
    # أسوأ الأخطاء: متأخرة بأقل درجة (فاتتنا)، وفي الموعد بأعلى درجة (إنذار كاذب)
    cols = ["order_id", "proba", "planned_days", "same_state", "customer_state",
            "primary_seller_state", "approval_month", "items_price_sum"]
    fp = df[(df[TARGET] == 0) & (df["pred"] == 1)].nlargest(k, "proba")[cols]
    fn = df[(df[TARGET] == 1) & (df["pred"] == 0)].nsmallest(k, "proba")[cols]
    return fp.round(3).reset_index(drop=True), fn.round(3).reset_index(drop=True)


def logistic_coefficients(pipe, top=15):
    # أكبر المعاملات المطلقة. الأرقام مُحجَّمة (StandardScaler) فتقارن بين الأرقام.
    names = pipe.named_steps["prep"].get_feature_names_out()
    coef = pipe.named_steps["model"].coef_[0]
    t = pd.DataFrame({"feature": names, "coefficient": coef})
    t["abs_coefficient"] = t["coefficient"].abs()
    return t.sort_values("abs_coefficient", ascending=False).head(top).round(4).reset_index(drop=True)