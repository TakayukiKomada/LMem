"""
LMem 極限圧縮器 v3

辞書コスト = 0 前提（LLMが辞書を既知）
- 出現1回でも多トークン列なら置換
- イテレーティブ圧縮（圧縮後テキストを再スキャンして更に圧縮）
- 1,148個の1トークン文字を最大限活用
"""

import json
from pathlib import Path

import tiktoken


def load_single_token_chars() -> list[str]:
    """可視1トークン文字を読み込み"""
    path = Path(__file__).parent / "visible_single_tokens.json"
    with open(path, "r", encoding="utf-8") as f:
        entries = json.load(f)

    EXCLUDE_CATS = {"Cc", "Cf", "Mn", "Mc", "Me", "Zs", "Zl", "Zp", "Co"}
    chars = []
    for e in entries:
        cp = e["cp"]
        if cp <= 0x7F or 0xFF00 <= cp <= 0xFFEF:
            continue
        if e["category"] in EXCLUDE_CATS:
            continue
        chars.append(e["char"])
    return chars


def find_all_substrings(text: str, min_len: int = 2, max_len: int = 200) -> list[tuple[str, int]]:
    """全部分文字列と出現回数を返す（出現1回含む）"""
    max_len = min(max_len, len(text))
    counts = {}
    text_len = len(text)

    for length in range(min_len, max_len + 1):
        for i in range(text_len - length + 1):
            sub = text[i:i + length]
            counts[sub] = counts.get(sub, 0) + 1

    return list(counts.items())


def find_best_replacement(
    text: str, enc: tiktoken.Encoding, replacement_char: str,
    min_len: int = 2, max_len: int = 200
) -> tuple[str, int] | None:
    """BPE実測で最もトークン削減効果の高い部分文字列を返す"""
    original_tokens = len(enc.encode(text))
    max_len = min(max_len, len(text))

    # フェーズ1: 理論スコアで候補を絞り込み
    candidates = {}
    text_len = len(text)
    for length in range(min_len, max_len + 1):
        for i in range(text_len - length + 1):
            sub = text[i:i + length]
            if sub in candidates or replacement_char in sub:
                continue
            count = text.count(sub)
            tok_count = len(enc.encode(sub))
            # 理論削減 = (トークン数 - 1) × 出現回数
            theory = (tok_count - 1) * count
            if theory > 0:
                candidates[sub] = (count, theory)

    if not candidates:
        return None

    # 理論スコア上位100件をBPE実測
    sorted_cands = sorted(candidates.items(), key=lambda x: -x[1][1])[:100]

    best = None
    best_saving = 0
    for sub, (count, theory) in sorted_cands:
        replaced = text.replace(sub, replacement_char)
        saving = original_tokens - len(enc.encode(replaced))
        if saving > best_saving:
            best_saving = saving
            best = sub

    if best is None or best_saving <= 0:
        return None

    return best, best_saving


def compress(code: str, verbose: bool = True) -> tuple[str, list[tuple[str, str]]]:
    """LMem極限圧縮（イテレーティブ）

    戻り値: (圧縮テキスト, [(パターン, 置換文字), ...] の置換履歴)
    """
    enc = tiktoken.get_encoding("cl100k_base")
    all_chars = load_single_token_chars()

    # ソースコード中の文字を除外
    used_chars = set(code)
    available = [c for c in all_chars if c not in used_chars]

    original_tokens = len(enc.encode(code))
    if verbose:
        print(f"利用可能文字: {len(available)}")
        print(f"元トークン数: {original_tokens}\n")

    current = code
    history = []  # 置換履歴（復元用）
    char_idx = 0
    total_saved = 0

    while char_idx < len(available):
        replacement = available[char_idx]

        # 使用済み文字がテキスト中に出現していないか確認
        if replacement in current:
            char_idx += 1
            continue

        result = find_best_replacement(current, enc, replacement)

        if result is None:
            break

        pattern, saving = result
        current = current.replace(pattern, replacement)
        history.append((pattern, replacement))
        total_saved += saving
        char_idx += 1

        # 使用済みの文字をスキップリストに追加
        used_chars.add(replacement)

        current_tokens = len(enc.encode(current))
        pct = (1 - current_tokens / original_tokens) * 100

        if verbose:
            step = len(history)
            tok_in_pat = len(enc.encode(pattern))
            count = len(code)  # 概算
            display = repr(pattern) if len(pattern) <= 50 else repr(pattern[:47] + "...")
            print(
                f"  [{step:>3}] {replacement} ← {display:<55} "
                f"BPE:{saving:>+3}  累計:{total_saved:>+4}  "
                f"残:{current_tokens:>4} ({pct:.1f}%)"
            )

        # 残りトークンが十分小さくなったら終了
        if current_tokens <= original_tokens * 0.05:
            break

    if verbose:
        print(f"\n圧縮完了: {len(history)}回置換")

    return current, history


def decompress(compressed: str, history: list[tuple[str, str]]) -> str:
    """逆順復元"""
    result = compressed
    for pattern, char in reversed(history):
        result = result.replace(char, pattern)
    return result


SAMPLE_CODE = '''\
from pathlib import Path
from typing import Optional, List, Dict, Any
from collections import defaultdict
from dataclasses import dataclass
import os
import sys
import json
import logging


@dataclass
class Config:
    """設定データクラス"""
    base_dir: str = "."
    recursive: bool = True
    extensions: list[str] = None


class FileProcessor:
    """ファイル処理クラス"""

    def __init__(self, config: Config):
        self.config = config
        self.base_dir = Path(config.base_dir)
        self._files: list[str] = []
        self._processed: dict[str, Any] = {}
        self._logger = logging.getLogger(__name__)

    def __repr__(self) -> str:
        return f"FileProcessor(base_dir={self.base_dir})"

    def __len__(self) -> int:
        return len(self._files)

    def __iter__(self):
        return iter(self._files)

    def __contains__(self, item) -> bool:
        return item in self._files

    def __getitem__(self, index: int) -> str:
        return self._files[index]

    @property
    def file_count(self) -> int:
        return len(self._files)

    @staticmethod
    def is_valid_extension(path: Path, extensions: set) -> bool:
        return path.suffix in extensions

    @classmethod
    def from_dict(cls, data: dict) -> "FileProcessor":
        config = Config(**data)
        return cls(config)

    def scan(self, extensions: Optional[set] = None) -> list[str]:
        if not self.base_dir.exists():
            raise ValueError(f"Directory not found: {self.base_dir}")

        if extensions is None:
            extensions = self.config.extensions

        pattern = "**/*" if self.config.recursive else "*"
        for path in self.base_dir.glob(pattern):
            if path.is_file():
                if self.is_valid_extension(path, extensions):
                    self._files.append(str(path))
                    self._logger.debug(f"Found: {path}")

        self._files = sorted(self._files)
        self._logger.info(f"Scanned {len(self._files)} files")
        return self._files

    def process(self, callback=None) -> dict[str, Any]:
        for i in range(len(self._files)):
            file_path = self._files[i]
            try:
                with open(file_path, "r") as f:
                    content = f.read()
                if callback is not None:
                    result = callback(content)
                else:
                    result = content
                self._processed[file_path] = result
            except FileNotFoundError:
                self._logger.error(f"File not found: {file_path}")
                self._processed[file_path] = None
            except PermissionError:
                self._logger.error(f"Permission denied: {file_path}")
                self._processed[file_path] = None
            except Exception as e:
                self._logger.error(f"Error: {file_path}: {e}")
                self._processed[file_path] = None

        return self._processed

    def get_results(self) -> dict[str, Any]:
        return dict(self._processed)

    def summary(self) -> None:
        total = len(self._processed)
        success = sum(1 for v in self._processed.values() if v is not None)
        failed = total - success
        self._logger.info(f"Processed: {success}/{total}")
        print(f"Total: {total}")
        print(f"Success: {success}")
        print(f"Failed: {failed}")

    def to_json(self, output_path: str) -> None:
        data = {
            "base_dir": str(self.base_dir),
            "file_count": len(self._files),
            "results": {
                k: v for k, v in self._processed.items()
                if v is not None
            },
        }
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)


def count_lines(text: str) -> int:
    return len(text.split("\\n"))


def word_frequency(text: str) -> dict[str, int]:
    words = text.lower().split()
    freq: dict[str, int] = {}
    for word in words:
        freq[word] = freq.get(word, 0) + 1
    return dict(sorted(freq.items(), key=lambda x: -x[1]))


async def async_process(processor: FileProcessor) -> None:
    results = processor.process(callback=count_lines)
    for path, count in results.items():
        if count is not None:
            print(f"{path}: {count} lines")


def main():
    logging.basicConfig(level=logging.INFO)
    config = Config(
        base_dir="/tmp/data",
        recursive=True,
        extensions=[".py", ".txt", ".md"],
    )
    processor = FileProcessor(config)
    files = processor.scan()
    print(f"Found {len(files)} files")

    results = processor.process(callback=count_lines)
    processor.summary()
    processor.to_json("/tmp/output.json")

    if len(processor) > 0:
        print(f"First file: {processor[0]}")

    for path in processor:
        if path.endswith(".py"):
            print(f"Python file: {path}")

    return None


if __name__ == "__main__":
    main()
'''


def main():
    enc = tiktoken.get_encoding("cl100k_base")

    print("=" * 70)
    print("LMem 極限圧縮器 v3")
    print("=" * 70)

    original_tokens = len(enc.encode(SAMPLE_CODE))
    original_chars = len(SAMPLE_CODE)

    compressed, history = compress(SAMPLE_CODE, verbose=True)
    restored = decompress(compressed, history)

    compressed_tokens = len(enc.encode(compressed))

    print(f"\n{'='*70}")
    print(f"  元:     {original_tokens:>6} トークン / {original_chars:>6} 文字")
    print(f"  圧縮後: {compressed_tokens:>6} トークン / {len(compressed):>6} 文字")
    print(f"  削減:   {original_tokens - compressed_tokens:>6} トークン ({(1-compressed_tokens/original_tokens)*100:.1f}%)")
    print(f"  置換数: {len(history):>6}")
    print(f"  復元:   {'PASS' if restored == SAMPLE_CODE else 'FAIL'}")
    print(f"{'='*70}")

    print(f"\n--- 圧縮後 ---")
    print(compressed[:800])

    # 保存
    output = {
        "version": "3.0",
        "original_tokens": original_tokens,
        "compressed_tokens": compressed_tokens,
        "reduction_pct": round((1 - compressed_tokens / original_tokens) * 100, 1),
        "history": [(p, c) for p, c in history],
    }
    out_path = Path(__file__).parent / "lmem_compressed.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
