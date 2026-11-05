# src/config.py
# كل الإعدادات الثابتة في مكان واحد، فإذا تغير اسم ملف نعدله هنا فقط.

from pathlib import Path

# جذر المشروع: هذا الملف داخل src، فنصعد مستويين (src ثم الجذر).
PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"

# اسمنا المختصر لكل جدول، واسم الملف الفعلي داخل data/raw
TABLE_FILES = {
    "orders": "olist_orders_dataset.csv",
    "customers": "olist_customers_dataset.csv",
    "order_items": "olist_order_items_dataset.csv",
    "payments": "olist_order_payments_dataset.csv",
    "products": "olist_products_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "category_translation": "product_category_name_translation.csv",
}

# أعمدة الرموز البريدية تقرأ كنصوص حتى لا تضيع الأصفار من بدايتها
# (مثال: 01046 تصبح 1046 لو قرئت كرقم).
TEXT_COLUMNS = {
    "customers": {"customer_zip_code_prefix": str},
    "sellers": {"seller_zip_code_prefix": str},
}

# المفتاح الأساسي لكل جدول (قد يتكون من عمود أو أكثر)
PRIMARY_KEYS = {
    "orders": ["order_id"],
    "customers": ["customer_id"],
    "order_items": ["order_id", "order_item_id"],
    "payments": ["order_id", "payment_sequential"],
    "products": ["product_id"],
    "sellers": ["seller_id"],
    "category_translation": ["product_category_name"],
}

# العلاقات: (الجدول الابن، عموده، الجدول الأب، عموده)
RELATIONSHIPS = [
    ("orders", "customer_id", "customers", "customer_id"),
    ("order_items", "order_id", "orders", "order_id"),
    ("order_items", "product_id", "products", "product_id"),
    ("order_items", "seller_id", "sellers", "seller_id"),
    ("payments", "order_id", "orders", "order_id"),
    ("products", "product_category_name", "category_translation", "product_category_name"),
]


def ensure_output_dirs():
    # ينشئ المجلدات إن لم تكن موجودة (exist_ok يمنع الخطأ إن وجدت)
    for folder in (RAW_DIR, PROCESSED_DIR, FIGURES_DIR, TABLES_DIR):
        folder.mkdir(parents=True, exist_ok=True)

# الأعمدة التي تحتوي تواريخ في كل جدول (تقرأ كنصوص في الخام وتحول عند التنظيف)
DATE_COLUMNS = {
    "orders": [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ],
    "order_items": ["shipping_limit_date"],
}
# جدول الطلب الواحد (قاعدة جدول النمذجة) الناتج عن اليوم 3
ORDER_TABLE_FILE = PROCESSED_DIR / "order_base.csv"
# مخرجات اليوم 4
FEATURE_TABLE_FILE = PROCESSED_DIR / "order_features.csv"    # كل الطلبات مع الهدف وسبب الاستبعاد
MODELING_TABLE_FILE = PROCESSED_DIR / "modeling_table.csv"   # الطلبات المؤهلة فقط
# مخرجات اليوم 7
SPLIT_FRACTIONS = (0.70, 0.15, 0.15)   # تدريب / تحقق / اختبار نهائي
RANDOM_SEED = 42                        # بذرة ثابتة لإعادة إنتاج النتائج
MODELS_DIR = PROCESSED_DIR / "models"
VAL_PREDICTIONS_FILE = PROCESSED_DIR / "val_predictions.csv"
# مخرجات اليوم 8
DECISIONS_FILE = TABLES_DIR / "frozen_decisions.json"     # العتبات والنموذج المختار (تُجمَّد قبل الاختبار)
TEST_METRICS_FILE = TABLES_DIR / "test_metrics.csv"       # وجوده = تم التقييم النهائي
TEST_PREDICTIONS_FILE = PROCESSED_DIR / "test_predictions.csv"