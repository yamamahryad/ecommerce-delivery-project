# src/audit.py
# تدقيق الجداول قبل المعالجة: الحجم والأنواع والقيم المفقودة والتكرار.

import pandas as pd


def build_table_summary(tables):
    # صف واحد لكل جدول
    rows = []
    for name, df in tables.items():
        missing_cells = int(df.isna().sum().sum())
        rows.append({
            "table": name,
            "rows": len(df),
            "columns": df.shape[1],
            "missing_cells": missing_cells,
            "pct_missing_cells": round(missing_cells / df.size * 100, 3),
            "fully_duplicated_rows": int(df.duplicated().sum()),
            "memory_mb": round(df.memory_usage(deep=True).sum() / 1024 ** 2, 2),
        })
    return pd.DataFrame(rows)


def build_column_audit(tables):
    # صف واحد لكل عمود في كل جدول
    rows = []
    for name, df in tables.items():
        for col in df.columns:
            series = df[col]
            non_null = series.dropna()
            example = str(non_null.iloc[0]) if len(non_null) > 0 else ""
            rows.append({
                "table": name,
                "column": col,
                "dtype": str(series.dtype),
                "n_missing": int(series.isna().sum()),
                "pct_missing": round(series.isna().mean() * 100, 3),
                "n_unique": int(series.nunique(dropna=True)),
                "example": example,
            })
    return pd.DataFrame(rows)