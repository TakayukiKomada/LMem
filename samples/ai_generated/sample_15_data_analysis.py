"""
データ分析パイプライン
Python基礎: pandas操作、集計、フィルタリング
データサイエンス語録: DataFrame, groupby, describe, plot
"""
import json
from pathlib import Path
from typing import Optional


def load_sales_data(path: Path) -> list[dict]:
    """売上データを読み込む"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def aggregate_by_category(records: list[dict]) -> dict[str, dict]:
    """カテゴリ別に集計する"""
    result: dict[str, dict] = {}
    for rec in records:
        cat = rec.get("category_name", "未分類")
        if cat not in result:
            result[cat] = {
                "total_revenue": 0.0,
                "total_quantity": 0,
                "order_count": 0,
                "average_price": 0.0,
            }
        result[cat]["total_revenue"] += rec.get("total_amount", 0)
        result[cat]["total_quantity"] += rec.get("quantity", 0)
        result[cat]["order_count"] += 1

    for cat, stats in result.items():
        if stats["total_quantity"] > 0:
            stats["average_price"] = round(stats["total_revenue"] / stats["total_quantity"], 2)
        stats["total_revenue"] = round(stats["total_revenue"], 2)

    return result


def calculate_summary_stats(values: list[float]) -> dict[str, float]:
    """基本統計量を計算する"""
    if not values:
        return {"count": 0, "mean": 0, "min": 0, "max": 0, "std": 0}

    n = len(values)
    mean = sum(values) / n
    sorted_vals = sorted(values)
    variance = sum((v - mean) ** 2 for v in values) / n
    std = variance ** 0.5

    return {
        "count": n,
        "mean": round(mean, 2),
        "min": sorted_vals[0],
        "max": sorted_vals[-1],
        "median": sorted_vals[n // 2],
        "std": round(std, 2),
        "total": round(sum(values), 2),
    }


def filter_by_date_range(
    records: list[dict],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> list[dict]:
    """日付範囲でフィルタする"""
    filtered = records
    if start_date:
        filtered = [r for r in filtered if r.get("order_date", "") >= start_date]
    if end_date:
        filtered = [r for r in filtered if r.get("order_date", "") <= end_date]
    return filtered


def rank_products(
    records: list[dict],
    sort_by: str = "total_revenue",
    limit: int = 10,
) -> list[dict]:
    """商品をランキングする"""
    product_stats: dict[str, dict] = {}
    for rec in records:
        pid = str(rec.get("product_id", ""))
        if pid not in product_stats:
            product_stats[pid] = {
                "product_id": rec.get("product_id"),
                "product_name": rec.get("product_name", ""),
                "total_revenue": 0.0,
                "total_quantity": 0,
                "order_count": 0,
            }
        product_stats[pid]["total_revenue"] += rec.get("total_amount", 0)
        product_stats[pid]["total_quantity"] += rec.get("quantity", 0)
        product_stats[pid]["order_count"] += 1

    ranked = sorted(product_stats.values(), key=lambda x: -x[sort_by])
    return ranked[:limit]


def generate_report(records: list[dict]) -> str:
    """分析レポートを生成する"""
    revenues = [r.get("total_amount", 0) for r in records]
    stats = calculate_summary_stats(revenues)
    categories = aggregate_by_category(records)
    top_products = rank_products(records, limit=5)

    lines = [
        "=== 売上分析レポート ===",
        f"総注文数: {stats['count']}",
        f"総売上: {stats['total']:,.2f}",
        f"平均単価: {stats['mean']:,.2f}",
        f"最小: {stats['min']:,.2f} / 最大: {stats['max']:,.2f}",
        f"標準偏差: {stats['std']:,.2f}",
        "",
        "--- カテゴリ別売上 ---",
    ]
    for cat, s in sorted(categories.items(), key=lambda x: -x[1]["total_revenue"]):
        lines.append(f"  {cat}: {s['total_revenue']:,.2f} ({s['order_count']}件)")

    lines.append("")
    lines.append("--- 商品ランキング Top 5 ---")
    for i, p in enumerate(top_products, 1):
        lines.append(f"  {i}. {p['product_name']}: {p['total_revenue']:,.2f}")

    return "\n".join(lines)


if __name__ == "__main__":
    sample = [
        {"product_id": 1, "product_name": "商品A", "category_name": "家電", "total_amount": 15000, "quantity": 1},
        {"product_id": 2, "product_name": "商品B", "category_name": "食品", "total_amount": 3000, "quantity": 5},
        {"product_id": 1, "product_name": "商品A", "category_name": "家電", "total_amount": 15000, "quantity": 1},
    ]
    report = generate_report(sample)
    print(report)
