# run_day5.py
# يبني جداول ورسوم الأسئلة 1 إلى 3: python run_day5.py
# يتطلب تشغيل run_day4.py أولاً.

import pandas as pd

from src import config
from src.eda import load_full_table, make_all_figures, prepare
from src.target import TARGET


def show(title, df):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)
    print(df.round(4).to_string(index=False))


def main():
    pd.set_option("display.width", 220)
    config.ensure_output_dirs()

    full = load_full_table()
    el, tables = prepare(full)
    overall = el[TARGET].mean()

    print(f"All orders: {len(full):,}   Eligible orders (analysis population): {len(el):,}")
    print(f"Overall late rate among eligible orders: {overall * 100:.2f}%")

    show("1) Monthly summary (Q1). small_sample = fewer than 200 eligible orders",
         tables["monthly_summary"])
    show("2) Late rate by customer state (Q2)", tables["late_by_state"])
    show("3) Route matrix: late rate (blank = fewer than 30 orders)", tables["route_late_rate"])
    show("4) Route matrix: number of orders", tables["route_n"])
    show("5) Single vs multi-seller (Q2)", tables["late_by_seller_group"])
    show("6) Same state vs different state (Q2)", tables["late_by_same_state"])
    show("7) Late rate by primary category, largest 20 (Q3)",
         tables["late_by_category"].head(20))
    show("8) Late rate by number of items (Q3)", tables["late_by_n_items"])
    show("9) Late rate by items value quintile (Q3)", tables["late_by_price_group"])
    show("10) Price quintile edges", tables["price_group_edges"])

    for name, df in tables.items():
        df.to_csv(config.TABLES_DIR / f"eda_{name}.csv", index=False, encoding="utf-8-sig")

    paths = make_all_figures(el, tables, config.FIGURES_DIR)
    print("\nFigures saved:")
    for p in paths:
        print("  ", p.name)


if __name__ == "__main__":
    main()