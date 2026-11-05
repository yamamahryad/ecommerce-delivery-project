# src/eda_part2.py

import matplotlib

matplotlib.use("Agg")  # للحفظ في ملفات دون نافذة
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.eda import (BLUE, CI_NOTE, GREEN, GREY, MIN_N, POP, RED, _finish,
                     monthly_summary, rate_table)
from src.features import MODEL_FEATURES
from src.target import TARGET

# شرائح المدة المخططة بالأيام (الحد الأعلى مشمول): (-1,10] تعني 0..10
PLANNED_BINS = [-1, 10, 15, 20, 25, 30, 40, np.inf]
PLANNED_LABELS = ["0-10", "11-15", "16-20", "21-25", "26-30", "31-40", "41+"]

# شهور الذروة التي لاحظناها في اليوم 5 (نسبة تأخر مرتفعة بوضوح)
PEAK_MONTHS = ["2017-11", "2018-02", "2018-03"]
PEAK_LABEL = "peak months (2017-11, 2018-02, 2018-03)"
OTHER_LABEL = "other months"

# أعمدة رقمية مختارة لتحليل الارتباط (كلها متاحة لحظة الاعتماد)
CORR_COLUMNS = [
    "planned_days", "purchase_to_approval_hours", "approval_month", "approval_dow",
    "approval_hour", "n_items", "n_sellers", "items_price_sum", "items_freight_sum",
    "freight_share", "log_weight_g", "log_volume_cm3", "is_multi_seller", "same_state",
    "max_installments", "n_payment_records",
]


# ------------------------------------------------------------------ تحضير

def add_planned_columns(el):
    # يضيف شريحة المدة المخططة وفترة الذروة (للتحليل فقط)
    df = el.copy()
    df["planned_group"] = pd.cut(df["planned_days"], bins=PLANNED_BINS, labels=PLANNED_LABELS)
    month = df["order_purchase_timestamp"].dt.to_period("M").astype(str)
    df["period"] = np.where(month.isin(PEAK_MONTHS), PEAK_LABEL, OTHER_LABEL)
    return df


# ------------------------------------------------------------------ توزيع الهدف

def target_balance(el):
    # حقائق عدم التوازن، ودقة قاعدة "كل الطلبات في الموعد"
    n = len(el)
    late = int(el[TARGET].sum())
    on_time = n - late
    rows = [
        ("eligible orders", n),
        ("on time (is_late = 0)", on_time),
        ("late (is_late = 1)", late),
        ("late share (%)", round(late / n * 100, 3)),
        ("imbalance ratio (on time : late)", round(on_time / late, 2)),
        ("accuracy of always predicting 'on time' (%)", round(on_time / n * 100, 3)),
        ("recall for the late class of that rule (%)", 0.0),
    ]
    return pd.DataFrame(rows, columns=["measure", "value"])


# ------------------------------------------------------------------ السؤال 4: المدة المخططة

def planned_day_table(el, min_cell_n=30):
    # نسبة التأخر لكل قيمة صحيحة من planned_days (نحذف الأيام التي n فيها صغير جداً)
    t = rate_table(el, "planned_days")
    return t[t["n"] >= min_cell_n].reset_index(drop=True)


def describe_by(df, group_col, value_col="planned_days"):
    # وصف إحصائي لعمود حسب مجموعة (pandas.describe)
    out = df.groupby(group_col, observed=True)[value_col].describe(percentiles=[0.25, 0.5, 0.75])
    return out.round(2).reset_index()


# ------------------------------------------------------------------ الارتباط

def correlation_matrix(el):
    # ارتباط Spearman (رتبي) بين الخصائص المختارة والهدف
    cols = CORR_COLUMNS + [TARGET]
    return el[cols].astype(float).corr(method="spearman")


def target_correlations(corr):
    s = corr[TARGET].drop(TARGET)
    out = s.reindex(s.abs().sort_values(ascending=False).index)
    return out.rename_axis("feature").reset_index(name="spearman_with_is_late").round(4)


# ------------------------------------------------------------------ السؤال 5: المفقود والاستبعاد

def exclusion_reason_table(full):
    counts = full.loc[~full["eligible"], "exclusion_reason"].value_counts()
    out = counts.rename_axis("reason").reset_index(name="n_orders")
    out["pct_of_all_orders"] = (out["n_orders"] / len(full) * 100).round(3)
    return out


def excluded_by_month(full):
    m = monthly_summary(full)
    out = m[["month", "orders_all"]].copy()
    out["orders_excluded"] = m["orders_all"] - m["orders_eligible"]
    out["pct_excluded"] = (out["orders_excluded"] / out["orders_all"] * 100).round(3)
    out["small_sample"] = out["orders_all"] < MIN_N
    return out


def representation_by_state(full):
    # هل خرجت ولايات بنسب أكبر من غيرها؟ (تمثيل الولايات قبل وبعد الاستبعاد)
    a = full.groupby("customer_state").size().rename("n_all")
    e = full[full["eligible"]].groupby("customer_state").size().rename("n_eligible")
    t = pd.concat([a, e], axis=1).fillna(0)
    t["n_eligible"] = t["n_eligible"].astype(int)
    t["pct_excluded"] = (1 - t["n_eligible"] / t["n_all"]) * 100
    t["share_all_pct"] = t["n_all"] / t["n_all"].sum() * 100
    t["share_eligible_pct"] = t["n_eligible"] / t["n_eligible"].sum() * 100
    t["share_diff_pp"] = t["share_eligible_pct"] - t["share_all_pct"]
    return t.round(3).sort_values("pct_excluded", ascending=False).reset_index()


def _profile(df):
    items = df[df["has_items"].astype(bool)]
    return {
        "orders": len(df),
        "share of customers in SP (%)": (df["customer_state"] == "SP").mean() * 100,
        "multi-seller share among orders with items (%)": items["is_multi_seller"].mean() * 100,
        "same-state share among orders with items (%)": items["same_state"].mean() * 100,
        "median items price sum": items["items_price_sum"].median(),
        "median planned days": df["planned_days"].median(),
        "mean number of items": items["n_items"].mean(),
    }


def representation_summary(full, el):
    # مقارنة خصائص كل الطلبات بالطلبات المؤهلة: هل تغيّر "شكل" البيانات بعد الاستبعاد؟
    a, b = _profile(full), _profile(el)
    rows = [{"measure": k, "all_orders": round(a[k], 3), "eligible_orders": round(b[k], 3),
             "difference": round(b[k] - a[k], 3)} for k in a]
    return pd.DataFrame(rows)


def missingness_vs_target(el, columns=MODEL_FEATURES):
    # هل المفقود "معلومة"؟ نقارن نسبة التأخر عند المفقود وعند المعروف
    masks = {col: el[col].isna() for col in columns if col in el.columns}
    if "primary_category" in el.columns:
        masks["primary_category == 'unknown'"] = el["primary_category"] == "unknown"
    rows = []
    for name, miss in masks.items():
        if miss.sum() == 0:
            continue
        rows.append({
            "feature": name,
            "n_missing": int(miss.sum()),
            "pct_missing": round(miss.mean() * 100, 3),
            "late_rate_when_missing_pct": round(el.loc[miss, TARGET].mean() * 100, 2),
            "late_rate_when_observed_pct": round(el.loc[~miss, TARGET].mean() * 100, 2),
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ السؤال 6: ملخص الأنماط

def _row(pattern, a_label, a, b_label, b):
    # فاصلا الثقة لا يتداخلان => الفرق غالباً ليس ضجيج عينة (لا يعني أنه سبب)
    overlap = not (a["ci_low"] > b["ci_high"] or b["ci_low"] > a["ci_high"])
    return {
        "pattern": pattern,
        "group_a": str(a_label), "n_a": int(a["n"]), "rate_a_pct": round(a["rate"] * 100, 2),
        "group_b": str(b_label), "n_b": int(b["n"]), "rate_b_pct": round(b["rate"] * 100, 2),
        "difference_pp": round((a["rate"] - b["rate"]) * 100, 2),
        "intervals_overlap": overlap,
    }


def _compare(pattern, table, label_col, a_label, b_label):
    t = table.set_index(label_col)
    return _row(pattern, a_label, t.loc[a_label], b_label, t.loc[b_label])


def _extremes(pattern, table, label_col):
    # أعلى وأدنى نسبة بين المجموعات غير الصغيرة فقط
    t = table[~table["small_sample"]].sort_values("rate")
    lo, hi = t.iloc[0], t.iloc[-1]
    return _row(pattern, hi[label_col], hi, lo[label_col], lo)


def pattern_summary(tables, planned_table):
    monthly = (tables["monthly_summary"].dropna(subset=["late_rate"])
               .rename(columns={"late_rate": "rate", "orders_eligible": "n"}).copy())
    monthly["month"] = monthly["month"].dt.strftime("%Y-%m")
    rows = [
        _extremes("Purchase month (highest vs lowest)", monthly, "month"),
        _extremes("Customer state (highest vs lowest, n >= 200)", tables["late_by_state"], "customer_state"),
        _compare("Route type", tables["late_by_same_state"], "route_group",
                 "different state", "same state"),
        _extremes("Planned delivery window (highest vs lowest group)", planned_table, "planned_group"),
        _compare("Seller structure", tables["late_by_seller_group"], "seller_group",
                 "multi-seller", "single seller"),
        _compare("Items value (quintiles)", tables["late_by_price_group"], "price_group",
                 "Q5 highest", "Q1 lowest"),
        _extremes("Product category (highest vs lowest, top 20 by size)",
                  tables["late_by_category"].head(20), "primary_category"),
        _compare("Number of items", tables["late_by_n_items"], "n_items_group",
                 "1 item", "4+ items"),
    ]
    return pd.DataFrame(rows)


def build_part2_tables(full, el, tables):
    # el يجب أن يحتوي planned_group و period (من add_planned_columns)
    t = {}
    t["target_balance"] = target_balance(el)
    t["late_by_planned_group"] = rate_table(el, "planned_group")
    t["planned_days_by_class"] = describe_by(el, TARGET)
    t["planned_days_by_seller_group"] = describe_by(el, "seller_group")
    t["planned_days_by_route_group"] = describe_by(el, "route_group")
    t["late_by_planned_x_route"] = rate_table(el, ["planned_group", "route_group"])
    t["late_by_period_x_route"] = rate_table(el, ["period", "route_group"])
    corr = correlation_matrix(el)
    t["correlation_matrix"] = corr.rename_axis("feature").reset_index().round(4)
    t["correlation_with_target"] = target_correlations(corr)
    t["exclusion_reasons"] = exclusion_reason_table(full)
    t["excluded_by_month"] = excluded_by_month(full)
    t["representation_by_state"] = representation_by_state(full)
    t["representation_summary"] = representation_summary(full, el)
    t["missingness_vs_target"] = missingness_vs_target(el)
    t["pattern_summary"] = pattern_summary(tables, t["late_by_planned_group"])
    return t


# ------------------------------------------------------------------ الرسوم

def fig_target(el, path):
    # رسم 7 (أعمدة): توزيع الهدف + لماذا الدقة وحدها مضللة
    n = len(el)
    late = int(el[TARGET].sum())
    on = n - late
    acc = on / n * 100
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

    ax1.bar(["on time (0)", "late (1)"], [on, late], color=[BLUE, RED])
    for x, v in enumerate([on, late]):
        ax1.text(x, v, f"{v:,}\n({v / n * 100:.1f}%)", ha="center", va="bottom", fontsize=9)
    ax1.set_ylim(0, on * 1.2)
    ax1.set_ylabel("Number of orders")
    ax1.set_title("Target distribution (is_late)")
    ax1.grid(axis="y", alpha=0.3)

    values = [acc, 0.0, 0.0]
    ax2.bar(["accuracy", "recall (late)", "F1 (late)"], values, color=[GREY, RED, RED])
    for x, v in enumerate(values):
        ax2.text(x, v + 1.5, f"{v:.1f}%", ha="center", fontsize=9)
    ax2.set_ylim(0, 112)
    ax2.set_ylabel("Score (%)")
    ax2.set_title("A rule that always predicts 'on time'")
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle("Class imbalance: a high accuracy can hide a useless model", fontsize=12)
    return _finish(fig, path, POP + ". Precision for the late class is undefined (the rule never predicts late).")


def fig_planned_hist(el, path, clip=60):
    # رسم 8 (مدرج تكراري): توزيع المدة المخططة لكل فئة (كل فئة مطبّعة على حدة)
    d = el[["planned_days", TARGET]].copy()
    n_clipped = int((d["planned_days"] > clip).sum())
    d = d[d["planned_days"] <= clip]
    d["class"] = np.where(d[TARGET] == 1, "late", "on time")

    fig, ax = plt.subplots(figsize=(10, 5.5))
    sns.histplot(data=d, x="planned_days", hue="class", bins=np.arange(0, clip + 2, 2),
                 stat="density", common_norm=False, element="step",
                 palette={"on time": BLUE, "late": RED}, ax=ax)
    ax.set_xlabel("Planned delivery window (days from approval to estimated date)")
    ax.set_ylabel("Density (each class sums to 1)")
    ax.set_title("Q4. Planned delivery window of on-time vs late orders")
    ax.grid(alpha=0.3)
    return _finish(fig, path, POP + f". Each class normalized separately because only about 7% of orders are late. "
                   f"{n_clipped:,} orders with more than {clip} planned days are not shown.")


def fig_boxes(el, path):
    # رسم 9 (صناديق): المدة المخططة وزمن الاعتماد حسب الفئة
    specs = [("planned_days", "Planned days (approval to estimated date)"),
             ("purchase_to_approval_hours", "Hours from purchase to approval")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
    for ax, (col, label) in zip(axes, specs):
        on = el.loc[el[TARGET] == 0, col].dropna()
        late = el.loc[el[TARGET] == 1, col].dropna()
        bp = ax.boxplot([on, late], patch_artist=True, showfliers=False, widths=0.5)
        for patch, color in zip(bp["boxes"], [BLUE, RED]):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        for i, series in enumerate([on, late], start=1):
            ax.text(i + 0.3, float(series.median()), f"median {series.median():.1f}",
                    fontsize=8, va="center")
        ax.set_xticks([1, 2])
        ax.set_xticklabels([f"on time\n(n={len(on):,})", f"late\n(n={len(late):,})"])
        ax.set_ylabel(label)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Q4. Distribution by delivery outcome (box = middle 50%, whiskers = 1.5 x IQR)", fontsize=12)
    return _finish(fig, path, POP + ". Outliers are hidden for readability; they are still in the data.")


def fig_planned_scatter(day_table, overall, path):
    # رسم 10 (تشتت): نسبة التأخر لكل قيمة من المدة المخططة، حجم النقطة = n
    t = day_table
    fig, ax = plt.subplots(figsize=(10, 5.5))
    colors = [GREY if s else BLUE for s in t["small_sample"]]
    sizes = 12 + np.sqrt(t["n"].to_numpy()) * 1.2
    ax.scatter(t["planned_days"], t["rate"] * 100, s=sizes, c=colors, alpha=0.8, edgecolors="white")
    ax.axhline(overall * 100, ls="--", color=RED, lw=1.2, label=f"overall rate {overall * 100:.1f}%")
    ax.set_xlabel("Planned delivery window (days)")
    ax.set_ylabel("Late orders (%)")
    ax.set_title("Q4. Late rate by planned delivery window (one point per day)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return _finish(fig, path, POP + ". Point size grows with the number of orders; grey = fewer than 200 orders; "
                   "days with fewer than 30 orders are not shown.")


def fig_corr(corr, path):
    # رسم 11 (خريطة حرارية): ارتباط Spearman
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", annot_kws={"fontsize": 7},
                cmap="coolwarm", center=0, vmin=-1, vmax=1, linewidths=0.4,
                cbar_kws={"label": "Spearman correlation"}, ax=ax)
    ax.set_title("Spearman correlations between order features and is_late")
    return _finish(fig, path, POP + ". Correlation measures association only; it does not show cause. "
                   "approval_month is cyclic, so its rank correlation is only a rough guide.")


def fig_exclusion_impact(reasons, by_month, path):
    # رسم 12 (أعمدة): أسباب الاستبعاد ونسبة المستبعد لكل شهر
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), gridspec_kw={"width_ratios": [1, 1.2]})
    r = reasons.sort_values("n_orders")
    ax1.barh(r["reason"], r["n_orders"], color=BLUE)
    ax1.set_xscale("log")
    top = float(r["n_orders"].max())
    ax1.set_xlim(right=top * 6)
    for y, v in enumerate(r["n_orders"]):
        ax1.text(v * 1.1, y, f"{v:,}", va="center", fontsize=8)
    ax1.set_xlabel("Orders excluded (log scale)")
    ax1.set_title("Why orders are excluded")
    ax1.grid(axis="x", alpha=0.3)

    big = by_month[~by_month["small_sample"]]
    ax2.bar(big["month"], big["pct_excluded"], width=20, color=BLUE)
    ax2.set_ylabel("Orders excluded (%)")
    ax2.set_xlabel("Purchase month")
    ax2.set_title("Share of orders excluded, by purchase month")
    ax2.tick_params(axis="x", rotation=45)
    ax2.grid(axis="y", alpha=0.3)
    fig.suptitle("Q5. Effect of exclusions on the data", fontsize=12)
    return _finish(fig, path, "Population: all 99,441 orders. Months with fewer than 200 orders are not shown (their shares are unstable).")


def make_part2_figures(full, el, tables, t2, folder):
    folder.mkdir(parents=True, exist_ok=True)
    overall = float(el[TARGET].mean())
    corr = t2["correlation_matrix"].set_index("feature")
    return [
        fig_target(el, folder / "fig07_target_distribution.png"),
        fig_planned_hist(el, folder / "fig08_planned_days_histogram.png"),
        fig_boxes(el, folder / "fig09_boxplots_by_outcome.png"),
        fig_planned_scatter(planned_day_table(el), overall, folder / "fig10_planned_days_scatter.png"),
        fig_corr(corr, folder / "fig11_correlation_heatmap.png"),
        fig_exclusion_impact(t2["exclusion_reasons"], t2["excluded_by_month"],
                             folder / "fig12_exclusion_impact.png"),
    ]