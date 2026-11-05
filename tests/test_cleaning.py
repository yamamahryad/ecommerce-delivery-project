# tests/test_cleaning.py
# اختبارات آلية لتنظيف اليوم الثاني. التشغيل: python -m pytest -v

import pandas as pd
import pytest

from src import config
from src.cleaning import (DecisionLog, clean_all_tables, clean_customers,
                          clean_products, parse_date_columns)
from src.data_loading import load_all_tables


@pytest.fixture(scope="module")
def raw():
    return load_all_tables()


@pytest.fixture(scope="module")
def cleaned(raw):
    clean, _log = clean_all_tables(raw)
    return clean


# ---------------- اختبارات على البيانات الحقيقية ----------------

def test_no_rows_dropped(raw, cleaned):
    for name in raw:
        assert len(cleaned[name]) == len(raw[name]), f"rows changed in {name}"


def test_primary_keys_still_unique(cleaned):
    for name, key_cols in config.PRIMARY_KEYS.items():
        df = cleaned[name]
        assert not df[key_cols].isna().any().any(), f"null key in {name}"
        assert not df.duplicated(subset=key_cols).any(), f"duplicate key in {name}"


def test_order_ids_unchanged(raw, cleaned):
    assert cleaned["orders"]["order_id"].equals(raw["orders"]["order_id"])


def test_date_columns_are_datetime(cleaned):
    for table, cols in config.DATE_COLUMNS.items():
        for col in cols:
            assert pd.api.types.is_datetime64_any_dtype(cleaned[table][col]), f"{table}.{col}"


def test_parsing_creates_no_new_missing_dates(raw, cleaned):
    for table, cols in config.DATE_COLUMNS.items():
        for col in cols:
            assert cleaned[table][col].isna().sum() == raw[table][col].isna().sum(), f"{table}.{col}"


def test_zip_codes_are_text_with_five_digits(cleaned):
    for table, col in (("customers", "customer_zip_code_prefix"),
                       ("sellers", "seller_zip_code_prefix")):
        series = cleaned[table][col]
        assert (series.str.len() == 5).all()
        assert series.str.isdigit().all()


def test_no_full_duplicate_rows(cleaned):
    for name, df in cleaned.items():
        assert not df.duplicated().any(), f"full duplicates in {name}"


def test_product_categories_have_no_missing(cleaned):
    products = cleaned["products"]
    assert products["product_category_name"].notna().all()
    assert products["product_category_english"].notna().all()


def test_product_measures_are_positive_or_missing(cleaned):
    products = cleaned["products"]
    for col in ("product_weight_g", "product_length_cm",
                "product_height_cm", "product_width_cm"):
        assert ((products[col] > 0) | products[col].isna()).all(), col


# ---------------- اختبارات على أمثلة صغيرة نصنعها بأنفسنا ----------------

def test_unparseable_date_becomes_missing():
    df = pd.DataFrame({"d": ["2017-10-02 10:56:33", "not a date", None]})
    out = parse_date_columns(df, ["d"], "t", DecisionLog())
    assert out["d"].notna().sum() == 1
    assert out["d"].isna().sum() == 2


def test_customer_cleaning_pads_zip_and_normalizes_text():
    df = pd.DataFrame({
        "customer_id": ["a"],
        "customer_unique_id": ["u"],
        "customer_zip_code_prefix": ["1046"],
        "customer_city": ["  Sao Paulo "],
        "customer_state": ["sp"],
    })
    out = clean_customers(df, DecisionLog())
    assert out.loc[0, "customer_zip_code_prefix"] == "01046"
    assert out.loc[0, "customer_city"] == "sao paulo"
    assert out.loc[0, "customer_state"] == "SP"


def test_products_cleaning_logic_on_small_example():
    products = pd.DataFrame({
        "product_id": ["p1", "p2", "p3"],
        "product_category_name": ["beleza_saude", None, "pc_gamer"],
        "product_name_lenght": [40.0, None, 30.0],
        "product_description_lenght": [200.0, None, 150.0],
        "product_photos_qty": [1.0, None, 2.0],
        "product_weight_g": [0.0, 500.0, 200.0],
        "product_length_cm": [10.0, 10.0, 10.0],
        "product_height_cm": [5.0, 5.0, 5.0],
        "product_width_cm": [8.0, 8.0, 8.0],
    })
    translation = pd.DataFrame({
        "product_category_name": ["beleza_saude"],
        "product_category_name_english": ["health_beauty"],
    })
    out = clean_products(products, translation, DecisionLog())
    assert out["product_category_name"].tolist() == ["beleza_saude", "unknown", "pc_gamer"]
    assert out["product_category_english"].tolist() == ["health_beauty", "unknown", "pc_gamer"]
    assert pd.isna(out.loc[0, "product_weight_g"])   # الوزن صفر تحول إلى مفقود