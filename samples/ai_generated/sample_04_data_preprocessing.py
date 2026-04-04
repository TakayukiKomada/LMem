"""
データ前処理パイプライン
Python基礎: ファイルI/O、例外処理、ジェネレータ
AI語録: Dataset, DataLoader, batch_size, shuffle
"""
import json
import random
from pathlib import Path
from typing import Optional


def load_jsonl(path: Path) -> list[dict]:
    """JSONLファイルを読み込む"""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def save_jsonl(data: list[dict], path: Path) -> None:
    """JSONLファイルに保存する"""
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"保存: {path} ({len(data)} 件)")


def split_dataset(
    data: list[dict],
    train_ratio: float = 0.8,
    eval_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    """データセットをtrain/eval/testに分割する"""
    random.seed(seed)
    shuffled = data.copy()
    random.shuffle(shuffled)

    n = len(shuffled)
    train_end = int(n * train_ratio)
    eval_end = train_end + int(n * eval_ratio)

    train_data = shuffled[:train_end]
    eval_data = shuffled[train_end:eval_end]
    test_data = shuffled[eval_end:]

    return train_data, eval_data, test_data


def create_instruction_pair(
    code: str,
    compressed: str,
) -> list[dict]:
    """圧縮・復元のインストラクションペアを生成する"""
    pairs = [
        {
            "instruction": "PythonコードをLMemに圧縮してください。",
            "input": code,
            "output": compressed,
        },
        {
            "instruction": "LMemコードをPythonに復元してください。",
            "input": compressed,
            "output": code,
        },
    ]
    return pairs


def filter_by_length(
    data: list[dict],
    min_length: int = 10,
    max_length: int = 500,
    key: str = "input",
) -> list[dict]:
    """長さでフィルタリングする"""
    return [
        item for item in data
        if min_length <= len(item.get(key, "")) <= max_length
    ]


if __name__ == "__main__":
    sample = [{"text": f"サンプル{i}", "label": i % 3} for i in range(100)]
    train, val, test = split_dataset(sample)
    print(f"train: {len(train)}, eval: {len(val)}, test: {len(test)}")
