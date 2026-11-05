# src/erd.py
# يرسم مخطط العلاقات بين الجداول بواسطة Matplotlib.
# ملاحظة: إن اختلفت أسماء الأعمدة في حزمتكم فعدلوها في TABLES أدناه.

import matplotlib

matplotlib.use("Agg")  # للحفظ في ملف دون نافذة رسومية
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

BOX_WIDTH = 26.0
LINE_STEP = 1.5
HEADER_H = 3.2

# اسم الجدول: (مركز x، مركز y، قائمة (الوسم، اسم العمود))
TABLES = {
    "customers": (14, 58, [
        ("PK", "customer_id"),
        ("", "customer_unique_id"),
        ("", "customer_zip_code_prefix"),
        ("", "customer_city"),
        ("", "customer_state"),
    ]),
    "orders": (50, 58, [
        ("PK", "order_id"),
        ("FK", "customer_id"),
        ("", "order_status"),
        ("", "order_purchase_timestamp"),
        ("", "order_approved_at"),
        ("", "order_delivered_carrier_date"),
        ("", "order_delivered_customer_date"),
        ("", "order_estimated_delivery_date"),
    ]),
    "payments": (86, 58, [
        ("PK,FK", "order_id"),
        ("PK", "payment_sequential"),
        ("", "payment_type"),
        ("", "payment_installments"),
        ("", "payment_value"),
    ]),
    "products": (14, 33, [
        ("PK", "product_id"),
        ("FK", "product_category_name"),
        ("", "product_name_lenght"),
        ("", "product_description_lenght"),
        ("", "product_photos_qty"),
        ("", "product_weight_g"),
        ("", "product_length_cm"),
        ("", "product_height_cm"),
        ("", "product_width_cm"),
    ]),
    "order_items": (50, 33, [
        ("PK,FK", "order_id"),
        ("PK", "order_item_id"),
        ("FK", "product_id"),
        ("FK", "seller_id"),
        ("", "shipping_limit_date"),
        ("", "price"),
        ("", "freight_value"),
    ]),
    "sellers": (86, 33, [
        ("PK", "seller_id"),
        ("", "seller_zip_code_prefix"),
        ("", "seller_city"),
        ("", "seller_state"),
    ]),
    "category_translation": (14, 6, [
        ("PK", "product_category_name"),
        ("", "product_category_name_english"),
    ]),
}

# (الابن، جهة الابن، الأب، جهة الأب، نص الرابط)
RELATIONS = [
    ("orders", "left", "customers", "right", "customer_id\nN : 1"),
    ("payments", "left", "orders", "right", "order_id\nN : 1"),
    ("order_items", "top", "orders", "bottom", "order_id  N : 1"),
    ("order_items", "left", "products", "right", "product_id\nN : 1"),
    ("order_items", "right", "sellers", "left", "seller_id\nN : 1"),
    ("products", "bottom", "category_translation", "top", "product_category_name  N : 1"),
]


def draw_table(ax, title, columns, cx, cy):
    # يرسم صندوق جدول واحد ويعيد حدوده
    height = HEADER_H + LINE_STEP * len(columns) + 1.0
    x0 = cx - BOX_WIDTH / 2
    y0 = cy - height / 2

    ax.add_patch(Rectangle((x0, y0), BOX_WIDTH, height,
                           facecolor="#F7F9FC", edgecolor="#2F4B7C", linewidth=1.5))
    ax.add_patch(Rectangle((x0, y0 + height - HEADER_H), BOX_WIDTH, HEADER_H,
                           facecolor="#2F4B7C", edgecolor="#2F4B7C"))
    ax.text(cx, y0 + height - HEADER_H / 2, title, ha="center", va="center",
            color="white", fontsize=10, fontweight="bold")

    for i, (tag, col) in enumerate(columns):
        y = y0 + height - HEADER_H - 0.5 - LINE_STEP * (i + 0.5)
        ax.text(x0 + 1.0, y, tag, ha="left", va="center",
                fontsize=8, fontweight="bold", color="#B5452F")
        ax.text(x0 + 5.5, y, col, ha="left", va="center",
                fontsize=8, color="#1B1B1B")

    return {"x0": x0, "y0": y0, "x1": x0 + BOX_WIDTH, "y1": y0 + height}


def anchor(box, side):
    # نقطة منتصف ضلع الصندوق المطلوب
    cx = (box["x0"] + box["x1"]) / 2
    cy = (box["y0"] + box["y1"]) / 2
    points = {
        "left": (box["x0"], cy),
        "right": (box["x1"], cy),
        "top": (cx, box["y1"]),
        "bottom": (cx, box["y0"]),
    }
    return points[side]


def draw_relation(ax, child_box, child_side, parent_box, parent_side, label):
    # سهم من الابن (N) إلى الأب (1) مع نص يوضح العلاقة
    start = anchor(child_box, child_side)
    end = anchor(parent_box, parent_side)
    ax.annotate("", xy=end, xytext=start,
                arrowprops=dict(arrowstyle="->", color="#444444", lw=1.5))

    mid_x = (start[0] + end[0]) / 2
    mid_y = (start[1] + end[1]) / 2
    if child_side in ("top", "bottom"):
        ax.text(mid_x + 1.0, mid_y, label, ha="left", va="center", fontsize=8,
                bbox=dict(facecolor="white", edgecolor="none", pad=1.0))
    else:
        ax.text(mid_x, mid_y + 0.8, label, ha="center", va="bottom", fontsize=8,
                bbox=dict(facecolor="white", edgecolor="none", pad=1.0))


def build_erd(output_path):
    fig, ax = plt.subplots(figsize=(16, 11.2))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 70)
    ax.axis("off")

    boxes = {}
    for name, (cx, cy, columns) in TABLES.items():
        boxes[name] = draw_table(ax, name, columns, cx, cy)

    for child, child_side, parent, parent_side, label in RELATIONS:
        draw_relation(ax, boxes[child], child_side, boxes[parent], parent_side, label)

    ax.set_title("Entity-Relationship Diagram of the Olist tables", fontsize=14, pad=12)
    ax.text(99, 1.0, "PK = primary key, FK = foreign key. Arrow: child (N) to parent (1).",
            ha="right", va="bottom", fontsize=8, color="#555555")

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)