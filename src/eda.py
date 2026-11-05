# src/eda.py
# دوال التحليل الاستكشافي (الأسئلة 1 إلى 3): جداول النسب والرسوم.
# نصوص الرسوم بالإنجليزية لأن Matplotlib لا يربط الحروف العربية دون مكتبات إضافية.

import matplotlib

matplotlib.use("Agg")  # للحفظ في ملفات دون نافذة
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src import config
from src.target import TARGET

MIN_N = 200        # مجموعة أصغر من هذا تعد صغيرة (رمادية ومنبّه عليها)
MIN_CELL_N = 30    # خلية الخريطة الحرارية الأصغر من هذا تترك فارغة
BLUE = "#2F4B7C"
GREY = "#B0B0B0"
RED = "#B5452F"
GREEN = "#6BA368"


# ------------------------------------------------------------------ تحميل وتحضير

def load_full_table():
    # الجدول الكامل (كل الطلبات مع الهدف وسبب الاستبعاد) الناتج عن اليوم 4
    path = config.FEATURE_TABLE_FILE
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: python run_day4.py")
    return pd.read_csv(
        path,
        dtype={"customer_zip_code_prefix": str},
        parse_dates=config.DATE_COLUMNS["orders"],
    )


def get_eligible(full):
    # المجتمع المحلَّل في تحليل التأخر: الطلبات المؤهلة فقط
    eligible = full[full["eligible"]].copy()
    eligible[TARGET] = eligible[TARGET].astype(int)
    return eligible


def add_group_columns(eligible):
    # أعمدة تجميعية للتحليل فقط (ليست مدخلات للنموذج)
    df = eligible.copy()
    df["seller_group"] = np.where(df["is_multi_seller"] == 1, "multi-seller", "single seller")
    df["route_group"] = np.where(df["same_state"] == 1, "same state", "different state")
    # حجم الطلب بعدد العناصر
    df["n_items_group"] = pd.cut(df["n_items"], bins=[0, 1, 2, 3, np.inf],
                                 labels=["1 item", "2 items", "3 items", "4+ items"])
    # حجم الطلب بالقيمة: 5 شرائح متساوية العدد
    df["price_group"] = pd.qcut(df["items_price_sum"], q=5,
                                labels=["Q1 lowest", "Q2", "Q3", "Q4", "Q5 highest"])
    return df


# ------------------------------------------------------------------ نسب التأخر

def wilson_interval(k, n, z=1.96):
    # فاصل ثقة Wilson لنسبة k/n (أصلح من الفاصل العادي للنسب الصغيرة)
    k = np.asarray(k, dtype=float)
    n = np.asarray(n, dtype=float)
    n = np.where(n > 0, n, np.nan)          # n=0 يعطي NaN بدل خطأ قسمة
    p = k / n
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return np.clip(centre - half, 0, 1), np.clip(centre + half, 0, 1)


def rate_table(df, group_col, min_n=MIN_N):
    # لكل مجموعة: الحجم n وعدد المتأخر ونسبة التأخر وفاصل الثقة وعلامة العينة الصغيرة
    g = df.groupby(group_col, observed=True)[TARGET].agg(n="size", late="sum")
    g["rate"] = g["late"] / g["n"]
    low, high = wilson_interval(g["late"].to_numpy(), g["n"].to_numpy())
    g["ci_low"] = low
    g["ci_high"] = high
    g["small_sample"] = g["n"] < min_n
    return g.reset_index()


def monthly_summary(full, min_n=MIN_N):
    # أعداد الطلبات (كلها) ونسبة التأخر (المؤهلة) لكل شهر شراء
    df = full.copy()
    df["month"] = df["order_purchase_timestamp"].dt.to_period("M").dt.to_timestamp()
    all_n = df.groupby("month").size().rename("orders_all")

    eligible = get_eligible(df)
    g = eligible.groupby("month")[TARGET].agg(orders_eligible="size", late_orders="sum")

    out = all_n.to_frame().join(g, how="left").fillna(0)
    out["orders_eligible"] = out["orders_eligible"].astype(int)
    out["late_orders"] = out["late_orders"].astype(int)
    out["late_rate"] = out["late_orders"] / out["orders_eligible"].replace(0, np.nan)
    low, high = wilson_interval(out["late_orders"].to_numpy(), out["orders_eligible"].to_numpy())
    out["ci_low"] = low
    out["ci_high"] = high
    out["small_sample"] = out["orders_eligible"] < min_n
    return out.reset_index()


def route_matrix(eligible, top_sellers=8, top_customers=10, min_cell_n=MIN_CELL_N):
    # جدول محوري (pivot_table): ولاية البائع الرئيسي × ولاية العميل.
    s_top = eligible["primary_seller_state"].value_counts().head(top_sellers).index
    c_top = eligible["customer_state"].value_counts().head(top_customers).index
    sub = eligible[eligible["primary_seller_state"].isin(s_top)
                   & eligible["customer_state"].isin(c_top)]

    n = pd.pivot_table(sub, index="primary_seller_state", columns="customer_state",
                       values=TARGET, aggfunc="count", fill_value=0)
    rate = pd.pivot_table(sub, index="primary_seller_state", columns="customer_state",
                          values=TARGET, aggfunc="mean")
    n = n.reindex(index=s_top, columns=c_top, fill_value=0)
    rate = rate.reindex(index=s_top, columns=c_top)
    rate = rate.where(n >= min_cell_n)          # الخلايا الصغيرة تصبح فارغة
    return rate, n


def prepare(full):
    # يعيد (الطلبات المؤهلة مع أعمدة التجميع، قاموس الجداول)
    el = add_group_columns(get_eligible(full))
    rate, n = route_matrix(el)
    tables = {
        "monthly_summary": monthly_summary(full),
        "late_by_state": rate_table(el, "customer_state").sort_values("rate", ascending=False),
        "route_late_rate": rate.rename_axis(columns=None).reset_index(),
        "route_n": n.rename_axis(columns=None).reset_index(),
        "late_by_seller_group": rate_table(el, "seller_group"),
        "late_by_same_state": rate_table(el, "route_group"),
        "late_by_category": rate_table(el, "primary_category").sort_values("n", ascending=False),
        "late_by_n_items": rate_table(el, "n_items_group"),
        "late_by_price_group": rate_table(el, "price_group"),
        "price_group_edges": el.groupby("price_group", observed=True)["items_price_sum"]
                               .agg(n="size", min_price="min", max_price="max").reset_index(),
    }
    return el, tables


# ------------------------------------------------------------------ أدوات الرسم

def _finish(fig, path, note=None):
    if note:
        fig.text(0.01, -0.01, note, fontsize=8, color="#444444", ha="left", va="top")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _bars_h(ax, table, label_col, overall, xlabel, title):
    # أعمدة أفقية مرتبة بالنسبة: فاصل الثقة + n بجانب كل عمود + الرمادي للعينة الصغيرة
    t = table.sort_values("rate").reset_index(drop=True)
    y = np.arange(len(t))
    colors = [GREY if s else BLUE for s in t["small_sample"]]
    xerr = [(t["rate"] - t["ci_low"]) * 100, (t["ci_high"] - t["rate"]) * 100]
    ax.barh(y, t["rate"] * 100, color=colors, xerr=xerr,
            error_kw=dict(ecolor="#444444", capsize=2, lw=1))
    ax.set_yticks(y)
    ax.set_yticklabels(t[label_col].astype(str))
    xmax = float((t["ci_high"] * 100).max())
    for yi, n in zip(y, t["n"]):
        ax.text(xmax * 1.02, yi, f"n={n:,}", va="center", fontsize=8)
    ax.axvline(overall * 100, ls="--", color=RED, lw=1.2,
               label=f"overall rate {overall * 100:.1f}%")
    ax.set_xlim(0, xmax * 1.2)
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontsize=11)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="x", alpha=0.3)


def _columns_v(ax, table, label_col, overall, ylabel, title):
    # أعمدة رأسية بترتيب الجدول (ترتيب طبيعي) مع فاصل الثقة و n فوق كل عمود
    t = table.reset_index(drop=True)
    x = np.arange(len(t))
    colors = [GREY if s else BLUE for s in t["small_sample"]]
    yerr = [(t["rate"] - t["ci_low"]) * 100, (t["ci_high"] - t["rate"]) * 100]
    ax.bar(x, t["rate"] * 100, color=colors, yerr=yerr,
           error_kw=dict(ecolor="#444444", capsize=3, lw=1))
    ax.set_xticks(x)
    ax.set_xticklabels(t[label_col].astype(str))
    ymax = float((t["ci_high"] * 100).max())
    for xi, n, hi in zip(x, t["n"], t["ci_high"]):
        ax.text(xi, hi * 100 + ymax * 0.03, f"n={n:,}", ha="center", fontsize=8)
    ax.axhline(overall * 100, ls="--", color=RED, lw=1.2,
               label=f"overall rate {overall * 100:.1f}%")
    ax.set_ylim(0, ymax * 1.2)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)


POP = "Population: eligible orders (delivered, valid dates)"
CI_NOTE = "Error bars: 95% Wilson interval. Grey = fewer than 200 orders (unreliable, not a ranking)."


# ------------------------------------------------------------------ الرسوم

def fig_monthly(monthly, path):
    # رسم 1 (منحنى زمني): الأعداد ونسبة التأخر لكل شهر شراء
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True)
    x = monthly["month"]
    ax1.plot(x, monthly["orders_all"], marker="o", color=BLUE, label="all orders")
    ax1.plot(x, monthly["orders_eligible"], marker="s", color=GREEN,
             label="eligible orders (delivered, valid dates)")
    ax1.set_ylabel("Number of orders")
    ax1.set_title("Q1. Orders per purchase month")
    ax1.legend()
    ax1.grid(alpha=0.3)

    big = monthly.loc[~monthly["small_sample"]]
    small = monthly.loc[monthly["small_sample"]]
    overall = monthly["late_orders"].sum() / monthly["orders_eligible"].sum()
    ax2.errorbar(big["month"], big["late_rate"] * 100, fmt="-o", color=RED, capsize=3,
                 yerr=[(big["late_rate"] - big["ci_low"]) * 100,
                       (big["ci_high"] - big["late_rate"]) * 100],
                 label="late rate (95% interval)")
    ax2.scatter(small["month"], small["late_rate"] * 100, facecolors="none",
                edgecolors="grey", s=50, label=f"month with n < {MIN_N} (unreliable)")
    ax2.axhline(overall * 100, ls="--", color=BLUE, lw=1, label=f"overall {overall * 100:.1f}%")
    ax2.set_ylabel("Late orders (%)")
    ax2.set_xlabel("Purchase month")
    ax2.set_title("Late-delivery rate per purchase month")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)
    fig.autofmt_xdate()
    return _finish(fig, path, POP + ". Top panel also shows all 99,441 orders. Late rate uses eligible orders only.")


def fig_state(table, overall, path):
    # رسم 2 (أعمدة): نسبة التأخر حسب ولاية العميل
    fig, ax = plt.subplots(figsize=(9, 9))
    _bars_h(ax, table, "customer_state", overall, "Late orders (%)",
            "Q2. Late rate by customer state")
    return _finish(fig, path, POP + ". " + CI_NOTE)


def fig_route_heatmap(rate, n, path):
    # رسم 3 (خريطة حرارية): ولاية البائع الرئيسي × ولاية العميل
        fig, ax = plt.subplots(figsize=(11, 6.5))
        labels = rate.mul(100).round(1).astype(str) + "\n(n=" + n.fillna(0).astype(int).astype(str) + ")"
    
        sns.heatmap(rate * 100, annot=labels, fmt="", cmap="YlOrRd", linewidths=0.5,
                linecolor="white", mask=rate.isna(), ax=ax,
                annot_kws={"fontsize": 8},
                cbar_kws={"label": "Late orders (%)"})

        ax.set_xlabel("Customer state (destination)")
        ax.set_ylabel("Primary seller state (origin)")
        ax.set_title("Q2. Late rate by route: seller state to customer state")
        return _finish(fig, path, POP + f". Blank cells have fewer than {MIN_CELL_N} orders. "
                   "Top 8 seller states x top 10 customer states.")


def fig_seller_and_route(seller_table, route_table, overall, path):
    # رسم 4 (أعمدة): تعدد البائعين، ثم نفس الولاية أو ولاية مختلفة
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
    _columns_v(ax1, seller_table, "seller_group", overall, "Late orders (%)",
               "Single vs multiple sellers")
    _columns_v(ax2, route_table, "route_group", overall, "Late orders (%)",
               "Seller and customer in the same state?")
    fig.suptitle("Q2. Late rate by seller structure and route type", fontsize=12)
    return _finish(fig, path, POP + ". " + CI_NOTE)


def fig_category(table, overall, path, top=20):
    # رسم 5 (أعمدة): أكبر 20 فئة منتج رئيسية (بشرط n >= MIN_N)
    t = table[~table["small_sample"]].sort_values("n", ascending=False).head(top)
    omitted = len(table) - len(t)
    fig, ax = plt.subplots(figsize=(9, 8))
    _bars_h(ax, t, "primary_category", overall, "Late orders (%)",
            f"Q3. Late rate by primary product category (top {len(t)} by size)")
    return _finish(fig, path, POP + f". {omitted} smaller categories are not shown. "
                   "Primary category = category of the highest-priced item. "
                   "Error bars: 95% Wilson interval.")


def fig_order_size(items_table, price_table, overall, path):
    # رسم 6 (أعمدة): حجم الطلب بالعناصر وبالقيمة
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    _columns_v(ax1, items_table, "n_items_group", overall, "Late orders (%)",
               "By number of items")
    _columns_v(ax2, price_table, "price_group", overall, "Late orders (%)",
               "By items value (equal-size quintiles)")
    ax1.set_xlabel("Items in the order")
    ax2.set_xlabel("Quintile of total items price")
    fig.suptitle("Q3. Late rate by order size", fontsize=12)
    return _finish(fig, path, POP + ". " + CI_NOTE)


def make_all_figures(el, tables, folder):
    # يرسم الأشكال الستة ويعيد مساراتها
    folder.mkdir(parents=True, exist_ok=True)
    overall = float(el[TARGET].mean())
    rate, n = route_matrix(el)
    return [
        fig_monthly(tables["monthly_summary"], folder / "fig01_monthly_orders_and_late_rate.png"),
        fig_state(tables["late_by_state"], overall, folder / "fig02_late_rate_by_customer_state.png"),
        fig_route_heatmap(rate, n, folder / "fig03_route_heatmap.png"),
        fig_seller_and_route(tables["late_by_seller_group"], tables["late_by_same_state"],
                             overall, folder / "fig04_seller_structure_and_route.png"),
        fig_category(tables["late_by_category"], overall, folder / "fig05_late_rate_by_category.png"),
        fig_order_size(tables["late_by_n_items"], tables["late_by_price_group"], overall,
                       folder / "fig06_late_rate_by_order_size.png"),
    ]