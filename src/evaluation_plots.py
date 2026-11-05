# src/evaluation_plots.py
# رسوم التقييم. النصوص بالإنجليزية لأن Matplotlib لا يربط الحروف العربية.

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve

from src.eda import BLUE, GREEN, GREY, POP, RED, _finish

PRETTY = {
    "baseline_prior": "Baseline (always on time)",
    "logistic_regression": "Logistic regression",
    "hist_gradient_boosting": "Hist. gradient boosting",
}
COLORS = {"logistic_regression": BLUE, "hist_gradient_boosting": GREEN, "baseline_prior": GREY}


def fig_threshold_selection(y_val, probas, thresholds, names, path):
    # رسم 13 (منحنيات): الإحكام والاستدعاء وF1 مقابل العتبة على التحقق
    y = np.asarray(y_val)
    fig, axes = plt.subplots(1, len(names), figsize=(6 * len(names), 5), sharey=True)
    for ax, name in zip(np.atleast_1d(axes), names):
        p, r, t = precision_recall_curve(y, probas[name])
        p, r = p[:-1], r[:-1]
        f1 = np.divide(2 * p * r, p + r, out=np.zeros_like(p), where=(p + r) > 0)
        ax.plot(t, p, color=BLUE, label="precision")
        ax.plot(t, r, color=GREEN, label="recall")
        ax.plot(t, f1, color=RED, lw=2, label="F1")
        ax.axvline(thresholds[name], ls="--", color="#444444",
                   label=f"chosen threshold {thresholds[name]:.3f}")
        ax.set_xlim(0, min(1.0, float(t.max()) * 1.02))
        ax.set_xlabel("Decision threshold (predict late if probability >= threshold)")
        ax.set_title(PRETTY[name])
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    np.atleast_1d(axes)[0].set_ylabel("Score")
    fig.suptitle("Threshold selection on the VALIDATION set (late class)", fontsize=12)
    return _finish(fig, path, "Population: validation orders only. The test set is not used here. "
                   "The threshold maximizes F1 on validation.")


def fig_pr_curves(y_test, probas, thresholds, names, path):
    # رسم 14 (منحنيات): منحنى الإحكام والاستدعاء على الاختبار مع نقطة العتبة المجمدة
    y = np.asarray(y_test)
    fig, ax = plt.subplots(figsize=(8, 6))
    for name in names:
        p, r, _ = precision_recall_curve(y, probas[name])
        ap = average_precision_score(y, probas[name])
        ax.plot(r, p, color=COLORS[name], label=f"{PRETTY[name]} (AP = {ap:.3f})")
        pred = probas[name] >= thresholds[name]
        tp = int(((y == 1) & pred).sum())
        precision = tp / pred.sum() if pred.sum() else 0.0
        recall = tp / (y == 1).sum()
        ax.scatter([recall], [precision], color=COLORS[name], s=80, edgecolors="black",
                   zorder=5)
    ax.axhline(y.mean(), ls="--", color=RED, label=f"no skill = late rate {y.mean() * 100:.1f}%")
    ax.set_xlabel("Recall (late class)")
    ax.set_ylabel("Precision (late class)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, min(1.0, max(0.3, float(y.mean()) * 8)))
    ax.set_title("Precision-recall curves on the TEST set")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return _finish(fig, path, POP + " (test period). Black-edged dots = frozen validation threshold.")


def fig_confusion(y_test, probas, thresholds, names, path):
    # رسم 15 (خريطة حرارية): مصفوفات الالتباس للنماذج الثلاثة
    y = np.asarray(y_test)
    fig, axes = plt.subplots(1, len(names), figsize=(5.2 * len(names), 4.8))
    for ax, name in zip(np.atleast_1d(axes), names):
        pred = (probas[name] >= thresholds[name]).astype(int)
        cm = confusion_matrix(y, pred, labels=[0, 1])
        rates = cm / cm.sum(axis=1, keepdims=True)
        labels = np.array([[f"{cm[i, j]:,}\n({rates[i, j] * 100:.1f}% of row)"
                            for j in range(2)] for i in range(2)])
        sns.heatmap(rates, annot=labels, fmt="", cmap="Blues", vmin=0, vmax=1, cbar=False,
                    xticklabels=["on time", "late"], yticklabels=["on time", "late"], ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"{PRETTY[name]}\nthreshold {thresholds[name]:.3f}", fontsize=10)
    fig.suptitle("Confusion matrices on the TEST set", fontsize=12)
    return _finish(fig, path, POP + " (test period). Colors show the share of each ACTUAL class, "
                   "not raw counts.")


def fig_score_distribution(y_test, proba, threshold, name, path):
    # رسم 16 (مدرج): توزيع الاحتمالات للطلبات المتأخرة وغير المتأخرة
    y = np.asarray(y_test)
    proba = np.asarray(proba)
    upper = float(np.quantile(proba, 0.995))
    bins = np.linspace(0, upper, 41)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.hist(proba[y == 0], bins=bins, density=True, alpha=0.55, color=BLUE,
            label=f"on time (n={int((y == 0).sum()):,})")
    ax.hist(proba[y == 1], bins=bins, density=True, alpha=0.55, color=RED,
            label=f"late (n={int((y == 1).sum()):,})")
    ax.axvline(threshold, ls="--", color="#444444", label=f"threshold {threshold:.3f}")
    ax.set_xlabel("Predicted probability of being late")
    ax.set_ylabel("Density (each class sums to 1)")
    ax.set_title(f"Score distribution by actual outcome: {PRETTY[name]} (TEST)")
    ax.legend()
    ax.grid(alpha=0.3)
    return _finish(fig, path, POP + " (test period). The two classes overlap heavily: "
                   "that overlap is why precision stays low. Top 0.5% of scores not shown.")


def fig_error_by_month(table, path):
    # رسم 17 (أعمدة): الأداء حسب شهر الاعتماد داخل فترة الاختبار
    t = table.reset_index(drop=True)
    x = np.arange(len(t))
    w = 0.2
    series = [("actual late rate", t["late_rate_pct"], GREY),
              ("share flagged as late", t["flagged_pct"], BLUE),
              ("recall", t["recall"] * 100, GREEN),
              ("precision", t["precision"] * 100, RED)]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for i, (label, values, color) in enumerate(series):
        ax.bar(x + (i - 1.5) * w, values.fillna(0), w, label=label, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{m}\n(n={n:,})" for m, n in zip(t["approval_month"], t["n"])])
    ax.set_ylabel("Percent")
    ax.set_title("Error analysis by approval month inside the TEST period")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    return _finish(fig, path, POP + " (test period). The late rate differs by month, so the same "
                   "threshold flags very different shares of orders.")