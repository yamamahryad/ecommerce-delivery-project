# run_day6.py

import pandas as pd

from src import config
from src.eda import load_full_table, prepare
from src.eda_part2 import add_planned_columns, build_part2_tables, make_part2_figures
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
    el = add_planned_columns(el)
    t2 = build_part2_tables(full, el, tables)

    print(f"All orders: {len(full):,}   Eligible orders: {len(el):,}   "
          f"Overall late rate: {el[TARGET].mean() * 100:.2f}%")

    show("1) Target balance", t2["target_balance"])
    show("2) Q4: late rate by planned delivery window", t2["late_by_planned_group"])
    show("3) Q4: planned days by outcome (0 = on time, 1 = late)", t2["planned_days_by_class"])
    show("4) Q4: late rate by planned window x route type", t2["late_by_planned_x_route"])
    show("5) Correlation with is_late (Spearman), sorted by strength",
         t2["correlation_with_target"])
    show("6) Q5: exclusion reasons", t2["exclusion_reasons"])
    show("7) Q5: excluded share by purchase month", t2["excluded_by_month"])
    show("8) Q5: states with the highest excluded share (top 10)",
         t2["representation_by_state"].head(10))
    show("9) Q5: all orders vs eligible orders (profile)", t2["representation_summary"])
    show("10) Q5: is missingness informative? (eligible orders)", t2["missingness_vs_target"])
    show("11) Q6: pattern summary (intervals_overlap = True means do not claim a difference)",
         t2["pattern_summary"])
    show("12) Q6 check: route effect inside peak vs other months", t2["late_by_period_x_route"])
    show("13) Q6 check: planned days by seller structure", t2["planned_days_by_seller_group"])
    show("14) Q6 check: planned days by route type", t2["planned_days_by_route_group"])

    for name, df in t2.items():
        df.to_csv(config.TABLES_DIR / f"eda2_{name}.csv", index=False, encoding="utf-8-sig")

    paths = make_part2_figures(full, el, tables, t2, config.FIGURES_DIR)
    print("\nFigures saved:")
    for p in paths:
        print("  ", p.name)


if __name__ == "__main__":
    main()