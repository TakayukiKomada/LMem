"""
評価メトリクス計算
Python基礎: 統計計算、Counter、辞書内包表記
AI語録: accuracy, precision, recall, f1_score
"""
from collections import Counter
from typing import Optional


def accuracy(predictions: list[int], labels: list[int]) -> float:
    """正答率を計算する"""
    if not predictions:
        return 0.0
    correct = sum(1 for p, l in zip(predictions, labels) if p == l)
    return correct / len(labels)


def precision_recall_f1(
    predictions: list[int],
    labels: list[int],
    target_class: int,
) -> dict[str, float]:
    """特定クラスの適合率・再現率・F1を計算する"""
    tp = sum(1 for p, l in zip(predictions, labels) if p == target_class and l == target_class)
    fp = sum(1 for p, l in zip(predictions, labels) if p == target_class and l != target_class)
    fn = sum(1 for p, l in zip(predictions, labels) if p != target_class and l == target_class)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
    }


def confusion_matrix(
    predictions: list[int],
    labels: list[int],
    num_classes: Optional[int] = None,
) -> list[list[int]]:
    """混同行列を計算する"""
    if num_classes is None:
        num_classes = max(max(predictions), max(labels)) + 1

    matrix = [[0] * num_classes for _ in range(num_classes)]
    for pred, label in zip(predictions, labels):
        matrix[label][pred] += 1

    return matrix


def char_level_accuracy(predicted: str, expected: str) -> float:
    """文字レベルの一致率を計算する（LMem復元評価用）"""
    pred = predicted.rstrip()
    exp = expected.rstrip()
    if pred == exp:
        return 100.0
    matches = sum(1 for a, b in zip(pred, exp) if a == b)
    max_len = max(len(pred), len(exp))
    if max_len == 0:
        return 100.0
    return matches / max_len * 100


def token_reduction_rate(original_tokens: int, compressed_tokens: int) -> float:
    """トークン削減率を計算する"""
    if original_tokens == 0:
        return 0.0
    return (1 - compressed_tokens / original_tokens) * 100


def print_classification_report(
    predictions: list[int],
    labels: list[int],
) -> None:
    """分類レポートを表示する"""
    classes = sorted(set(labels))
    print(f"{'クラス':>8} {'適合率':>8} {'再現率':>8} {'F1':>8}")
    print("-" * 40)

    for cls in classes:
        metrics = precision_recall_f1(predictions, labels, cls)
        print(f"{cls:>8} {metrics['precision']:>8.4f} {metrics['recall']:>8.4f} {metrics['f1_score']:>8.4f}")

    acc = accuracy(predictions, labels)
    print("-" * 40)
    print(f"{'正答率':>8} {acc:>24.4f}")


if __name__ == "__main__":
    preds = [0, 1, 2, 0, 1, 2, 0, 0, 1, 2]
    labs = [0, 1, 2, 0, 2, 2, 1, 0, 1, 2]
    print_classification_report(preds, labs)
