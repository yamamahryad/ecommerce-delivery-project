# src/features.py
# الخصائص المشتقة، قائمة المدخلات البيضاء، الحقول المحظورة، وقاموس الخصائص.
# القاعدة: كل خاصية تُحسب من معلومات متاحة لحظة اعتماد الطلب فقط.

import numpy as np
import pandas as pd

from src import config
from src.target import TARGET, add_target_and_eligibility, exclusion_summary

# معرفات للربط والتتبع فقط (لا تدخل النموذج)
ID_COLUMNS = ["order_id", "customer_unique_id"]
# أعمدة وقت اعتماد الطلب: للتقسيم الزمني لاحقاً، وليست مدخلات
TIME_COLUMNS = ["order_purchase_timestamp", "order_approved_at"]

# الحقول المحظورة كمدخلات (حسب الملف) + نتائج لاحقة مشتقة منها
FORBIDDEN_COLUMNS = [
    "order_status",
    "order_delivered_customer_date",
    "order_delivered_carrier_date",
    "actual_delivery_days",
    "delay_days",
]

NUMERIC_FEATURES = [
    "approval_dow", "approval_hour", "approval_month",
    "purchase_to_approval_hours", "planned_days",
    "n_items", "n_distinct_products", "n_sellers", "n_seller_states", "n_categories",
    "items_price_sum", "items_freight_sum", "max_item_price", "avg_item_price",
    "freight_share", "log_weight_g", "log_volume_cm3", "n_items_missing_size",
    "is_multi_seller", "same_state",
    "n_payment_records", "n_payment_types", "max_installments",
]
CATEGORICAL_FEATURES = ["customer_state", "primary_seller_state",
                        "primary_category", "main_payment_type"]
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES   # القائمة البيضاء


def safe_divide(numerator, denominator):
    # قسمة آمنة: إن كان المقام صفراً أو مفقوداً تكون النتيجة NaN (وليست inf)
    num = numerator.to_numpy(dtype=float)
    den = denominator.to_numpy(dtype=float)
    out = np.full(num.shape, np.nan)
    np.divide(num, den, out=out, where=(den > 0))   # تقسم فقط حيث المقام > 0
    return pd.Series(out, index=numerator.index)


def add_features(table):
    # لا تستخدم هذه الدالة أي عمود محظور (والاختبار يثبت ذلك بحذفها كلها)
    df = table.copy()
    approved = df["order_approved_at"]
    purchase = df["order_purchase_timestamp"]
    estimated_day = df["order_estimated_delivery_date"].dt.normalize()

    # ---- زمنية ----
    df["approval_dow"] = approved.dt.dayofweek            # 0 = الاثنين
    df["approval_hour"] = approved.dt.hour
    df["approval_month"] = approved.dt.month
    df["purchase_to_approval_hours"] = (approved - purchase).dt.total_seconds() / 3600
    df["planned_days"] = (estimated_day - approved.dt.normalize()).dt.days

    # ---- عددية ----
    df["avg_item_price"] = safe_divide(df["items_price_sum"], df["n_items"])
    df["freight_share"] = safe_divide(df["items_freight_sum"],
                                      df["items_price_sum"] + df["items_freight_sum"])
    # log1p يضغط القيم الكبيرة جداً (الأوزان والأحجام ملتوية) ويحافظ على الصفر
    df["log_weight_g"] = np.log1p(df["total_weight_g"])
    df["log_volume_cm3"] = np.log1p(df["total_volume_cm3"])

    # ---- مشتقة من العلاقات ----
    df["is_multi_seller"] = np.where(df["n_sellers"].isna(), np.nan,
                                     (df["n_sellers"] > 1).astype(float))
    df["same_state"] = np.where(df["primary_seller_state"].isna(), np.nan,
                                (df["primary_seller_state"] == df["customer_state"]).astype(float))
    # للتحليل (السؤال 2) وليس للنموذج: مسار الولاية
    df["state_route"] = df["primary_seller_state"] + ">" + df["customer_state"]
    return df


def build_modeling_tables(order_table):
    # يعيد: (الجدول الكامل، جدول النمذجة، جدول الاستبعادات)
    full = add_features(order_table)
    full = add_target_and_eligibility(full)
    exclusions = exclusion_summary(full)

    modeling = full.loc[full["eligible"], ID_COLUMNS + TIME_COLUMNS + MODEL_FEATURES + [TARGET]]
    modeling = modeling.copy()
    modeling[TARGET] = modeling[TARGET].astype(int)
    return full, modeling.reset_index(drop=True), exclusions


def _entry(name, derived, kind, formula, source, why, available_at, model_input=True):
    return {"feature": name, "derived": derived, "type": kind, "formula": formula,
            "source": source, "why_relevant": why, "available_at": available_at,
            "model_input": model_input}


APPROVAL = "at approval time"
ASSUMED = "at approval time (assumption of the brief)"

FEATURE_DICTIONARY = [
    _entry("approval_dow", True, "temporal", "day of week of order_approved_at (0=Mon)",
           "orders", "weekend approvals may wait for the next working day", APPROVAL),
    _entry("approval_hour", True, "temporal", "hour of order_approved_at",
           "orders", "late-day approvals may miss the same-day dispatch", APPROVAL),
    _entry("approval_month", True, "temporal", "month of order_approved_at",
           "orders", "captures seasonality such as holiday peaks", APPROVAL),
    _entry("purchase_to_approval_hours", True, "temporal",
           "(order_approved_at - order_purchase_timestamp) in hours",
           "orders", "slow approval may signal payment or fraud checks", APPROVAL),
    _entry("planned_days", True, "temporal",
           "estimated delivery day - approval day (days)",
           "orders", "length of the promised delivery window", ASSUMED),
    _entry("avg_item_price", True, "numeric", "items_price_sum / n_items (NaN if no items)",
           "order_items", "average value per item", ASSUMED),
    _entry("freight_share", True, "numeric", "freight / (price + freight)",
           "order_items", "high freight share can signal heavy or distant shipments", ASSUMED),
    _entry("log_weight_g", True, "numeric", "log1p(sum of product weights)",
           "order_items + products", "heavy orders may ship differently; log reduces skew", ASSUMED),
    _entry("log_volume_cm3", True, "numeric", "log1p(sum of length*height*width)",
           "order_items + products", "bulky orders may ship differently; log reduces skew", ASSUMED),
    _entry("is_multi_seller", True, "relationship-derived", "1 if n_sellers > 1 else 0",
           "order_items", "several sellers need several dispatches", ASSUMED),
    _entry("same_state", True, "relationship-derived",
           "1 if primary seller state == customer state else 0",
           "sellers + customers", "same-state routes are shorter", ASSUMED),
    _entry("state_route", True, "categorical (analysis only)",
           "primary_seller_state + '>' + customer_state",
           "sellers + customers", "route-level analysis (question 2); too many levels for the model",
           ASSUMED, model_input=False),
    # ---- أعمدة مجمّعة في اليوم 3 وتمر كما هي ----
    _entry("n_items", False, "numeric", "count of item rows per order", "order_items",
           "order size", ASSUMED),
    _entry("n_distinct_products", False, "numeric", "distinct product_id per order",
           "order_items", "order variety", ASSUMED),
    _entry("n_sellers", False, "numeric", "distinct seller_id per order", "order_items",
           "multi-seller orders are handled explicitly", ASSUMED),
    _entry("n_seller_states", False, "numeric", "distinct seller states per order",
           "order_items + sellers", "geographic spread of sellers", ASSUMED),
    _entry("n_categories", False, "numeric", "distinct categories per order",
           "order_items + products", "order variety", ASSUMED),
    _entry("items_price_sum", False, "numeric", "sum of price", "order_items",
           "order value", ASSUMED),
    _entry("items_freight_sum", False, "numeric", "sum of freight_value", "order_items",
           "shipping cost", ASSUMED),
    _entry("max_item_price", False, "numeric", "max price in order", "order_items",
           "most valuable item", ASSUMED),
    _entry("n_items_missing_size", False, "numeric", "items with unknown weight or volume",
           "products", "warns that weight/volume totals may be incomplete", ASSUMED),
    _entry("n_payment_records", False, "numeric", "count of payment rows", "payments",
           "split payments", ASSUMED),
    _entry("n_payment_types", False, "numeric", "distinct payment types", "payments",
           "mixed payment methods", ASSUMED),
    _entry("max_installments", False, "numeric", "max payment_installments", "payments",
           "financing choice", ASSUMED),
    _entry("customer_state", False, "categorical", "customer state code", "customers",
           "destination region", APPROVAL),
    _entry("primary_seller_state", False, "categorical",
           "state of the seller of the highest-priced item", "order_items + sellers",
           "origin region (multi-seller rule from day 3)", ASSUMED),
    _entry("primary_category", False, "categorical",
           "category of the highest-priced item", "order_items + products",
           "product type", ASSUMED),
    _entry("main_payment_type", False, "categorical",
           "payment type with the largest value", "payments", "payment method", ASSUMED),
]


def feature_dictionary():
    return pd.DataFrame(FEATURE_DICTIONARY)


def save_feature_tables(full, modeling):
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    full.to_csv(config.FEATURE_TABLE_FILE, index=False)
    modeling.to_csv(config.MODELING_TABLE_FILE, index=False)


def load_modeling_table():
    if not config.MODELING_TABLE_FILE.exists():
        raise FileNotFoundError(f"{config.MODELING_TABLE_FILE} not found. Run: python run_day4.py")
    return pd.read_csv(config.MODELING_TABLE_FILE, parse_dates=TIME_COLUMNS)