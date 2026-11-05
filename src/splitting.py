# src/splitting.py
# التقسيم الزمني (تدريب / تحقق / اختبار) حسب وقت اعتماد الطلب، وتجهيز X و y.

import pandas as pd

from src import config
from src.features import CATEGORICAL_FEATURES, MODEL_FEATURES
from src.target import TARGET

TIME_COL = "order_approved_at"           # لحظة التنبؤ المعتمدة في الملف
SPLIT_NAMES = ("train", "val", "test")


def temporal_split(modeling, fractions=config.SPLIT_FRACTIONS):
    # الأقدم للتدريب، ثم التحقق، والأحدث للاختبار النهائي.
    if abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError("fractions must sum to 1")

    # نرتب بالزمن ثم بـ order_id ليكون الترتيب حتمياً عند تساوي الطوابع
    df = modeling.sort_values([TIME_COL, "order_id"]).reset_index(drop=True)
    n = len(df)
    n_train = int(n * fractions[0])
    n_val = int(n * fractions[1])

    # الحدود تؤخذ من الطابع الزمني، فلا يقع طابع واحد في مجموعتين
    cut_train = df[TIME_COL].iloc[n_train - 1]
    cut_val = df[TIME_COL].iloc[n_train + n_val - 1]
    t = df[TIME_COL]

    return {
        "train": df[t <= cut_train].reset_index(drop=True),
        "val": df[(t > cut_train) & (t <= cut_val)].reset_index(drop=True),
        "test": df[t > cut_val].reset_index(drop=True),
    }


def split_summary(splits):
    # الحدود الزمنية والأعداد ونسبة التأخر في كل مجموعة
    total = sum(len(d) for d in splits.values())
    rows = []
    for name in SPLIT_NAMES:
        d = splits[name]
        late = int(d[TARGET].sum())
        rows.append({
            "split": name,
            "orders": len(d),
            "share_pct": round(len(d) / total * 100, 2),
            "approved_from": d[TIME_COL].min(),
            "approved_to": d[TIME_COL].max(),
            "late_orders": late,
            "late_rate_pct": round(late / len(d) * 100, 3),
        })
    return pd.DataFrame(rows)


def split_by_month(splits):
    # أي شهور وقعت في أي مجموعة وبأي نسبة تأخر (يوضح انزياح التوزيع)
    parts = []
    for name in SPLIT_NAMES:
        d = splits[name]
        month = d[TIME_COL].dt.to_period("M").astype(str)
        g = d.groupby(month)[TARGET].agg(orders="size", late="sum")
        g["late_rate_pct"] = (g["late"] / g["orders"] * 100).round(2)
        g.insert(0, "split", name)
        parts.append(g.rename_axis("approval_month").reset_index())
    return pd.concat(parts, ignore_index=True)


def get_xy(df):
    # المدخلات = القائمة البيضاء MODEL_FEATURES فقط. المعرفات والتواريخ لا تدخل.
    X = df[MODEL_FEATURES].copy()
    for col in CATEGORICAL_FEATURES:
        X[col] = X[col].astype(object)   # يتوافق مع SimpleImputer للنصوص
    y = df[TARGET].astype(int)
    return X, y