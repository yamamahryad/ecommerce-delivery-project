# src/data_loading.py
# مسؤول فقط عن قراءة الجداول الخام كما هي، دون أي تنظيف.

import pandas as pd

from src import config


def list_raw_files():
    # يعيد أسماء الملفات الموجودة في data/raw (لرسالة الخطأ المفيدة)
    return sorted(p.name for p in config.RAW_DIR.glob("*") if p.is_file())


def load_table(name):
    # يقرأ جدولاً واحداً حسب اسمه المختصر
    if name not in config.TABLE_FILES:
        raise KeyError(f"اسم جدول غير معروف: {name}")

    path = config.RAW_DIR / config.TABLE_FILES[name]
    if not path.exists():
        raise FileNotFoundError(
            f"الملف غير موجود: {path}\n"
            f"الملفات الموجودة في data/raw: {list_raw_files()}\n"
            f"عدّل TABLE_FILES في src/config.py ليطابق الأسماء الفعلية."
        )

    # dtype يجبر أعمدة الرموز البريدية على النص، ويتجاهل الأعمدة غير الموجودة
    dtype = config.TEXT_COLUMNS.get(name)
    return pd.read_csv(path, dtype=dtype)


def load_all_tables():
    # يعيد قاموساً: اسم الجدول -> DataFrame
    return {name: load_table(name) for name in config.TABLE_FILES}