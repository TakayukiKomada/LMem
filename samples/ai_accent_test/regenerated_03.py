"""
トークナイザの使い方
"""
import json
from pathlib import Path

import tiktoken


def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    """テキストのトークン数を計算する"""
    enc = tiktoken.get_encoding(encoding_name)
    tokens = enc.encode(text)
    return len(tokens)


def analyze_token_distribution(texts: list[str]) -> dict:
    """テキスト群のトークン数分布を分析する"""
    enc = tiktoken.get_encoding("cl100k_base")
    counts = [len(enc.encode(t)) for t in texts]

    return {
        "total_texts": len(texts),
        "total_tokens": sum(counts),
        "mean_tokens": sum(counts) / max(len(counts), 1),
        "min_tokens": min(counts) if counts else 0,
        "max_tokens": max(counts) if counts else 0,
    }


def build_chat_messages(system_prompt: str, user_message: str) -> list[dict]:
    """チャットメッセージ形式を構築する"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]
    return messages


def save_tokenized_data(
    texts: list[str],
    output_path: Path,
    encoding_name: str = "cl100k_base",
) -> None:
    """テキストをトークン化して保存する"""
    enc = tiktoken.get_encoding(encoding_name)
    results = []

    for text in texts:
        tokens = enc.encode(text)
        results.append({
            "text": text,
            "tokens": tokens,
            "num_tokens": len(tokens),
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"保存完了: {output_path} ({len(results)} 件)")


if __name__ == "__main__":
    sample_texts = [
        "import torch",
        "def forward(self, x):",
        "model = AutoModelForCausalLM.from_pretrained(model_name)",
    ]
    stats = analyze_token_distribution(sample_texts)
    for key, value in stats.items():
        print(f"  {key}: {value}")
