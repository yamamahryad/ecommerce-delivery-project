# src/feature_checks.py
# ملخصات NumPy الإحصائية، وأثر log، والمفقود في الخصائص، وتوزيع الهدف.

import numpy as np
import pandas as pd

from src.features import MODEL_FEATURES
from src.target import TARGET


def numpy_skewness(values):
    # معامل الالتواء = متوسط (z^3). موجب كبير = ذيل طويل من القيم الكبيرة.
    x = np.asarray(values, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 3 or x.std() == 0:
        return float("nan")
    z = (x - x.mean()) / x.std()
    return float(np.mean(z ** 3))


def numpy_summary(df, columns):
    # إحصاءات وصفية بـ NumPy. دوال nan* تتجاهل القيم المفقودة بدل أن تفسد النتيجة.
    rows = []
    for col in columns:
        x = df[col].to_numpy(dtype=float)
        p = np.nanpercentile(x, [0, 25, 50, 75, 95, 100])
        rows.append({
            "feature": col, "n_missing": int(np.isnan(x).sum()),
            "min": round(p[0], 3), "p25": round(p[1], 3), "median": round(p[2], 3),
            "p75": round(p[3], 3), "p95": round(p[4], 3), "max": round(p[5], 3),
            "mean": round(float(np.nanmean(x)), 3),
        })
    return pd.DataFrame(rows)


def log_transform_effect(df):
    # يقارن الالتواء قبل log1p وبعده: الدليل على سبب استخدام التحويل
    pairs = [("total_weight_g", "log_weight_g"),
             ("total_volume_cm3", "log_volume_cm3"),
             ("items_price_sum", None)]
    rows = []
    for raw_col, log_col in pairs:
        raw = df[raw_col].to_numpy(dtype=float)
        logged = np.log1p(raw) if log_col is None else df[log_col].to_numpy(dtype=float)
        rows.append({"column": raw_col,
                     "skewness_raw": round(numpy_skewness(raw), 3),
                     "skewness_after_log1p": round(numpy_skewness(logged), 3)})
    return pd.DataFrame(rows)


def feature_missing_report(eligible):
    # المفقود في الخصائص بين الطلبات المؤهلة (يعالج لاحقاً داخل Pipeline على التدريب فقط)
    miss = eligible[MODEL_FEATURES].isna().sum()
    out = miss[miss > 0].rename_axis("feature").reset_index(name="n_missing")
    out["pct_missing"] = (out["n_missing"] / len(eligible) * 100).round(3)
    return out


def target_distribution(modeling):
    counts = modeling[TARGET].value_counts().sort_index()
    out = counts.rename_axis(TARGET).reset_index(name="n_orders")
    out["pct"] = (out["n_orders"] / out["n_orders"].sum() * 100).round(3)
    return out


def status_by_exclusion(full):
    # حالة الطلب مقابل سبب الاستبعاد (جدول تقاطعي)
    table = pd.crosstab(full["order_status"], full["exclusion_reason"])
    return table.reset_index()


def delay_description(eligible):
    # وصف التأخر الفعلي (تحليل وصفي فقط، ليس مدخلاً للنموذج)
    d = eligible["delay_days"].to_numpy(dtype=float)
    late = d[d > 0]
    rows = [
        ("eligible orders", len(d)),
        ("late orders (delay_days > 0)", int((d > 0).sum())),
        ("delivered earlier than estimated (delay_days < 0)", int((d < 0).sum())),
        ("delivered on the estimated day (delay_days == 0)", int((d == 0).sum())),
        ("median delay among late orders (days)", float(np.median(late)) if len(late) else float("nan")),
        ("max delay (days)", float(np.max(d))),
        ("median days early among early orders",
         float(np.median(-d[d < 0])) if (d < 0).any() else float("nan")),
    ]
    return pd.DataFrame(rows, columns=["measure", "value"])