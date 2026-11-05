# tests/conftest.py
# مثبتات (fixtures) مشتركة بين ملفات الاختبار، تحسب مرة واحدة لكل جلسة.

import pytest

from src.cleaning import clean_all_tables
from src.data_loading import load_all_tables
from src.merging import build_order_table


@pytest.fixture(scope="session")
def clean_tables():
    clean, _log = clean_all_tables(load_all_tables())
    return clean

@pytest.fixture(scope="session")
def order_table_and_log(clean_tables):
    return build_order_table(clean_tables)

from src.features import build_modeling_tables


@pytest.fixture(scope="session")
def modeling_bundle(order_table_and_log):
    table, _log = order_table_and_log
    return build_modeling_tables(table)