# src/target.py
# تعريف الهدف is_late وتحديد الطلبات المؤهلة مع سبب استبعاد واحد لكل طلب.

import numpy as np
import pandas as pd

TARGET = "is_late"

# ترتيب الأولوية: الطلب المستبعد يأخذ أول سبب ينطبق عليه
REASON_ORDER = [
    "not_delivered_status",            # الحالة ليست delivered (ملغى، شحن، ...)
    "delivered_without_customer_date", # مسلّم لكن بلا تاريخ تسليم للعميل
    "missing_approval_date",           # لا يوجد تاريخ اعتماد (لحظة التنبؤ مجهولة)
    "missing_estimated_date",          # لا يوجد موعد متوقع
    "approval_after_delivery",         # الاعتماد بعد التسليم (تناقض زمني)
    "estimated_before_approval",       # الموعد المتوقع قبل يوم الاعتماد
    "no_items",                        # لا توجد عناصر لبناء الخصائص
    "no_payment_record",               # لا يوجد سجل دفع
]

REASON_RULES = {
    "not_delivered_status": "order_status != 'delivered'",
    "delivered_without_customer_date": "order_delivered_customer_date is missing",
    "missing_approval_date": "order_approved_at is missing",
    "missing_estimated_date": "order_estimated_delivery_date is missing",
    "approval_after_delivery": "order_approved_at > order_delivered_customer_date",
    "estimated_before_approval": "estimated delivery day < approval day",
    "no_items": "order has no rows in order_items",
    "no_payment_record": "order has no payment record",
}


def compute_is_late(delivered, estimated):
    # الهدف: 1 إذا كان يوم التسليم الفعلي بعد اليوم المتوقع، 0 إذا كان فيه أو قبله،
    # و NaN إذا كان أي من التاريخين مفقوداً (لا نخمن النتيجة).
    delivered_day = delivered.dt.normalize()   # نحذف الساعة: المقارنة بالأيام
    estimated_day = estimated.dt.normalize()

    known = (delivered_day.notna() & estimated_day.notna()).to_numpy()
    late = (delivered_day > estimated_day).to_numpy()

    # NumPy: np.where يختار بين قيمتين لكل صف دفعة واحدة (عملية متجهة بلا حلقة)
    values = np.where(known, late.astype(float), np.nan)
    return pd.Series(values, index=delivered.index, name=TARGET)


def add_target_and_eligibility(table):
    # يضيف: exclusion_reason و eligible و is_late وعمودي التحليل الوصفي فقط
    df = table.copy()
    approved_day = df["order_approved_at"].dt.normalize()
    estimated_day = df["order_estimated_delivery_date"].dt.normalize()

    # كل شرط مصفوفة منطقية (قناع). المقارنة مع NaT تعطي False فلا تسبب أخطاء.
    conditions = [
        (df["order_status"] != "delivered").to_numpy(),
        df["order_delivered_customer_date"].isna().to_numpy(),
        df["order_approved_at"].isna().to_numpy(),
        df["order_estimated_delivery_date"].isna().to_numpy(),
        (df["order_approved_at"] > df["order_delivered_customer_date"]).to_numpy(),
        (estimated_day < approved_day).to_numpy(),
        (~df["has_items"].astype(bool)).to_numpy(),
        (~df["has_payments"].astype(bool)).to_numpy(),
    ]
    # NumPy: np.select يطبق الشروط بالترتيب ويأخذ أول شرط صحيح لكل صف
    df["exclusion_reason"] = np.select(conditions, REASON_ORDER, default="eligible")
    df["eligible"] = df["exclusion_reason"] == "eligible"

    # الهدف يُحسب للجميع ثم يُمحى لغير المؤهل (NaN وليس 0)
    df[TARGET] = compute_is_late(df["order_delivered_customer_date"],
                                 df["order_estimated_delivery_date"]).where(df["eligible"])

    # للتحليل الوصفي فقط: ممنوعة كمدخلات للنموذج
    delivered_day = df["order_delivered_customer_date"].dt.normalize()
    df["delay_days"] = (delivered_day - estimated_day).dt.days
    df["actual_delivery_days"] = ((df["order_delivered_customer_date"]
                                   - df["order_purchase_timestamp"]).dt.total_seconds() / 86400)
    return df


def exclusion_summary(full):
    # جدول الاستبعادات: كل سبب وعدده والمتبقي بعده
    counts = full["exclusion_reason"].value_counts()
    remaining = len(full)
    rows = [{"step": 0, "reason": "all orders", "rule": "-",
             "orders_excluded": 0, "orders_remaining": remaining}]
    for i, reason in enumerate(REASON_ORDER, start=1):
        n = int(counts.get(reason, 0))
        remaining -= n
        rows.append({"step": i, "reason": reason, "rule": REASON_RULES[reason],
                     "orders_excluded": n, "orders_remaining": remaining})
    return pd.DataFrame(rows)