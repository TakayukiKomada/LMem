"""
LMem圧縮ユーティリティ
Python基礎: 関数合成、辞書操作、統計処理
AI語録: tokens, compression, encoding, dictionary
"""
import json
from pathlib import Path
from typing import Optional

import tiktoken


def load_dictionary(path: Path) -> list[tuple[str, str]]:
    """固定辞書をロードする"""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return [(entry[0], entry[1]) for entry in raw]


def compress_code(
    code: str,
    dict_entries: list[tuple[str, str]],
) -> tuple[str, list[tuple[str, str]]]:
    """コードを辞書ベースで圧縮する"""
    code_chars = set(code)
    current = code
    used = []

    for pattern, char in dict_entries:
        if char in code_chars:
            continue
        if pattern in current:
            current = current.replace(pattern, char)
            used.append((pattern, char))

    return current, used


def decompress_code(
    compressed: str,
    used_entries: list[tuple[str, str]],
) -> str:
    """圧縮済みコードを復元する"""
    result = compressed
    for pattern, char in reversed(used_entries):
        result = result.replace(char, pattern)
    return result


def verify_roundtrip(code: str, dict_entries: list[tuple[str, str]]) -> bool:
    """往復テストを検証する"""
    compressed, used = compress_code(code, dict_entries)
    restored = decompress_code(compressed, used)
    return restored == code


def compute_stats(
    code: str,
    compressed: str,
    enc: Optional[tiktoken.Encoding] = None,
) -> dict:
    """圧縮統計を計算する"""
    if enc is None:
        enc = tiktoken.get_encoding("cl100k_base")

    original_tokens = len(enc.encode(code))
    compressed_tokens = len(enc.encode(compressed))
    reduction = (1 - compressed_tokens / original_tokens) * 100 if original_tokens > 0 else 0.0

    return {
        "original_chars": len(code),
        "compressed_chars": len(compressed),
        "original_tokens": original_tokens,
        "compressed_tokens": compressed_tokens,
        "reduction_pct": round(reduction, 2),
        "char_ratio": round(len(compressed) / max(len(code), 1), 4),
    }


def batch_compress(
    files: list[Path],
    dict_entries: list[tuple[str, str]],
) -> dict:
    """複数ファイルを一括圧縮して統計を返す"""
    enc = tiktoken.get_encoding("cl100k_base")
    total_orig = 0
    total_comp = 0
    passed = 0
    failed = 0

    for file_path in files:
        try:
            code = file_path.read_text(encoding="utf-8")
        except Exception:
            failed += 1
            continue

        compressed, used = compress_code(code, dict_entries)
        restored = decompress_code(compressed, used)

        if restored != code:
            failed += 1
            continue

        stats = compute_stats(code, compressed, enc)
        total_orig += stats["original_tokens"]
        total_comp += stats["compressed_tokens"]
        passed += 1

    overall_reduction = (1 - total_comp / total_orig) * 100 if total_orig > 0 else 0.0

    return {
        "total_files": len(files),
        "passed": passed,
        "failed": failed,
        "total_original_tokens": total_orig,
        "total_compressed_tokens": total_comp,
        "overall_reduction_pct": round(overall_reduction, 2),
    }


if __name__ == "__main__":
    sample = "def forward(self, x):\n    return self.linear(x)\n"
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = len(enc.encode(sample))
    print(f"サンプルコード: {tokens} トークン")
