# src/cleaning.py
# تنظيف كل جدول على حدة مع تسجيل كل قرار في سجل واضح.
# قاعدة: لا نعدل الخام. كل دالة تأخذ نسخة (copy) وتعيد جدولاً جديداً.

import numpy as np
import pandas as pd

from src import config


class DecisionLog:
    # يجمع قرارات التنظيف: الجدول، المشكلة، عدد السجلات، الإجراء، السبب
    def __init__(self):
        self.rows = []

    def add(self, table, issue, count, action, reason):
        self.rows.append({
            "table": table,
            "issue": issue,
            "records_affected": int(count),
            "action": action,
            "reason": reason,
        })

    def to_frame(self):
        return pd.DataFrame(self.rows)


# ---------------------------------------------------------------- أدوات مساعدة

def drop_full_duplicates(df, table, log):
    # يحذف الصفوف المكررة بكل أعمدتها فقط (تكرار order_id وحده ليس خطأ)
    n_dup = int(df.duplicated().sum())
    if n_dup > 0:
        df = df.drop_duplicates()
    action = "dropped" if n_dup > 0 else "none found, nothing removed"
    log.add(table, "fully duplicated rows", n_dup, action,
            "Only rows identical in ALL columns are treated as duplicates")
    return df.copy()


def parse_date_columns(df, columns, table, log):
    # يحول أعمدة نصية إلى datetime. errors="coerce" يحول الفاشل إلى NaT
    # بدل أن يوقف البرنامج، لذلك نعد القيم المفقودة قبل التحويل وبعده
    # لنعرف إن كان التحويل أفسد شيئاً.
    df = df.copy()
    for col in columns:
        missing_before = int(df[col].isna().sum())
        df[col] = pd.to_datetime(df[col], errors="coerce")
        missing_after = int(df[col].isna().sum())
        new_missing = missing_after - missing_before
        converted = len(df) - missing_after

        log.add(table, f"{col}: dates stored as text", converted,
                "converted to datetime with pd.to_datetime(errors='coerce')",
                f"Needed for time arithmetic; values that failed to parse: {new_missing}")

        if missing_before > 0:
            log.add(table, f"{col}: missing values", missing_before,
                    "kept as missing (NaT), no imputation",
                    "A missing date means the event did not happen or was not recorded; "
                    "eligibility is decided when the target is defined")
    return df


def normalize_text_column(df, col, case, table, log):
    # يزيل المسافات الزائدة ويوحد حالة الأحرف، ويعد الصفوف التي تغيرت فعلاً
    old = df[col]
    new = old.str.strip()
    if case == "lower":
        new = new.str.lower()
    elif case == "upper":
        new = new.str.upper()
    n_changed = int((new != old).sum())
    df[col] = new
    log.add(table, f"{col}: inconsistent spaces/case", n_changed,
            f"stripped spaces and converted to {case}case" if n_changed else "none needed",
            "Same city/state must have one spelling to group correctly")
    return df


def pad_zip_codes(df, col, table, log):
    # الرمز البريدي نص من 5 خانات. إن كان أقصر نضيف أصفاراً في بدايته.
    old = df[col].str.strip()
    n_short = int((old.str.len() < 5).sum())
    df[col] = old.str.zfill(5)
    log.add(table, f"{col}: zip prefix shorter than 5 digits", n_short,
            "left-padded with zeros to 5 characters (kept as text)",
            "Leading zeros can be lost in the source; text keeps the code intact")
    return df


# ---------------------------------------------------------------- تنظيف كل جدول

def clean_category_translation(translation, log):
    df = drop_full_duplicates(translation, "category_translation", log)
    for col in ("product_category_name", "product_category_name_english"):
        df[col] = df[col].str.strip()
    return df


def clean_orders(orders, log):
    df = drop_full_duplicates(orders, "orders", log)
    df = normalize_text_column(df, "order_status", "lower", "orders", log)
    df = parse_date_columns(df, config.DATE_COLUMNS["orders"], "orders", log)
    return df


def clean_customers(customers, log):
    df = drop_full_duplicates(customers, "customers", log)
    df = normalize_text_column(df, "customer_city", "lower", "customers", log)
    df = normalize_text_column(df, "customer_state", "upper", "customers", log)
    df = pad_zip_codes(df, "customer_zip_code_prefix", "customers", log)
    return df


def clean_order_items(items, log):
    df = drop_full_duplicates(items, "order_items", log)
    df = parse_date_columns(df, config.DATE_COLUMNS["order_items"], "order_items", log)

    # قيم غير منطقية محتملة: نعدها ونوثقها ولا نحذفها
    n_bad_price = int((df["price"] <= 0).sum())
    n_bad_freight = int((df["freight_value"] < 0).sum())
    n_free_ship = int((df["freight_value"] == 0).sum())
    log.add("order_items", "price <= 0", n_bad_price, "kept",
            "Counted only; none expected. Any case would be reviewed, not silently removed")
    log.add("order_items", "freight_value < 0", n_bad_freight, "kept",
            "Counted only; none expected")
    log.add("order_items", "freight_value == 0", n_free_ship, "kept",
            "Free shipping is plausible, so it is not an error")
    return df


def clean_payments(payments, log):
    df = drop_full_duplicates(payments, "payments", log)

    n_neg = int((df["payment_value"] < 0).sum())
    n_zero_value = int((df["payment_value"] == 0).sum())
    n_zero_inst = int((df["payment_installments"] == 0).sum())
    n_undefined = int((df["payment_type"] == "not_defined").sum())

    log.add("payments", "payment_value < 0", n_neg, "kept", "Counted only; none expected")
    log.add("payments", "payment_value == 0", n_zero_value, "kept",
            "Rare rows; they add 0 to the order total, so sums are unaffected")
    log.add("payments", "payment_installments == 0", n_zero_inst, "kept",
            "Illogical value (at least 1 installment expected) but rare; payment_installments "
            "may be handled when features are built")
    log.add("payments", "payment_type == 'not_defined'", n_undefined, "kept",
            "Treated as its own payment type category")
    return df


def clean_products(products, translation, log):
    df = drop_full_duplicates(products, "products", log)
    cat = "product_category_name"

    # 1) فئة مفقودة -> "unknown" (قيمة ثابتة لا تسرب معلومات)
    descriptive = ["product_category_name", "product_name_lenght",
                   "product_description_lenght", "product_photos_qty"]
    n_missing_cat = int(df[cat].isna().sum())
    n_all_four = int(df[descriptive].isna().all(axis=1).sum())
    df[cat] = df[cat].fillna("unknown")
    log.add("products", "product_category_name missing", n_missing_cat,
            "filled with constant label 'unknown'",
            f"{n_all_four} products miss all four descriptive columns together. "
            "The numeric descriptive columns stay missing; they are imputed later "
            "on training data only")

    # 2) أوزان وأبعاد غير منطقية (صفر أو سالب) -> NaN
    measures = ["product_weight_g", "product_length_cm",
                "product_height_cm", "product_width_cm"]
    for col in measures:
        bad_mask = df[col] <= 0          # NaN <= 0 تعطي False فلا تتأثر المفقودة
        n_bad = int(bad_mask.sum())
        df.loc[bad_mask, col] = np.nan
        log.add("products", f"{col} <= 0 (illogical)", n_bad,
                "set to missing (NaN)" if n_bad else "none found",
                "A physical product cannot have zero/negative size; "
                "imputation is done later on training data only")
    n_missing_measures = int(df[measures].isna().any(axis=1).sum())
    log.add("products", "weight/dimensions missing (after fix)", n_missing_measures,
            "kept as missing", "To be imputed with training statistics in the model pipeline")

    # 3) عمود الفئة بالإنجليزية. الفئة اليتيمة تحتفظ باسمها البرتغالي.
    mapping = dict(zip(translation["product_category_name"],
                       translation["product_category_name_english"]))
    english = df[cat].map(mapping)
    unmatched = english.isna() & (df[cat] != "unknown")
    orphan_values = sorted(df.loc[unmatched, cat].unique())
    english = english.where(~unmatched, df[cat])
    english = english.fillna("unknown")
    df["product_category_english"] = english
    log.add("products", "category without English translation", int(unmatched.sum()),
            "new column product_category_english keeps the Portuguese name for these",
            f"Distinct untranslated categories: {orphan_values}. "
            "The raw translation table is not modified")
    return df


def clean_sellers(sellers, log):
    df = drop_full_duplicates(sellers, "sellers", log)
    df = normalize_text_column(df, "seller_city", "lower", "sellers", log)
    df = normalize_text_column(df, "seller_state", "upper", "sellers", log)
    df = pad_zip_codes(df, "seller_zip_code_prefix", "sellers", log)
    return df


# ---------------------------------------------------------------- التشغيل الكامل

def clean_all_tables(raw):
    # يأخذ قاموس الجداول الخام ويعيد (الجداول النظيفة، سجل القرارات)
    log = DecisionLog()
    clean = {}
    clean["category_translation"] = clean_category_translation(raw["category_translation"], log)
    clean["orders"] = clean_orders(raw["orders"], log)
    clean["customers"] = clean_customers(raw["customers"], log)
    clean["order_items"] = clean_order_items(raw["order_items"], log)
    clean["payments"] = clean_payments(raw["payments"], log)
    clean["products"] = clean_products(raw["products"], clean["category_translation"], log)
    clean["sellers"] = clean_sellers(raw["sellers"], log)

    # نعيد الترتيب نفسه المعتمد في config
    ordered = {name: clean[name] for name in config.TABLE_FILES}
    return ordered, log


def save_clean_tables(clean):
    # الناتج يحفظ في data/processed (مجلد لا يرفع إلى Git)
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for name, df in clean.items():
        df.to_csv(config.PROCESSED_DIR / f"{name}_clean.csv", index=False)


def load_clean_tables():
    # يقرأ الجداول المعالجة مع الأنواع الصحيحة (نص للرموز، datetime للتواريخ)
    tables = {}
    for name in config.TABLE_FILES:
        path = config.PROCESSED_DIR / f"{name}_clean.csv"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Run: python run_day2.py")
        tables[name] = pd.read_csv(
            path,
            dtype=config.TEXT_COLUMNS.get(name),
            parse_dates=config.DATE_COLUMNS.get(name, []),
        )
    return tables