"""
LMem 理論証明パイプライン
=========================
Top 20 パターンに絞り、学習 → テスト → 100点 を証明する。

Step 1: Top 20 辞書構築
Step 2: 学習データ生成（stdlib から）
Step 3: テストセット分離（学習に未使用のコード）
Step 4: 全データ保存

使い方:
    python prove_theory.py
"""

import json
import random
import sys
from pathlib import Path
from collections import Counter

import tiktoken

from lmem_deterministic import load_fixed_dict

# ============================
#  設定
# ============================
TOP_N       = 20           # 使うパターン数
TEST_COUNT  = 20           # テスト用スニペット数
MAX_FILES   = 800          # stdlib ファイル上限
MIN_LINES   = 8
MAX_LINES   = 50
MIN_TOKENS  = 50
MAX_TOKENS  = 300
SEED        = 42
OUT_DIR     = Path("training_data/prove")


# ============================
#  圧縮・復元（グローバル辞書版）
# ============================

def compress(code: str, dictionary: list[tuple[str, str]]) -> str:
    result = code
    for pattern, char in dictionary:
        result = result.replace(pattern, char)
    return result


def decompress(compressed: str, dictionary: list[tuple[str, str]]) -> str:
    result = compressed
    for pattern, char in reversed(dictionary):
        result = result.replace(char, pattern)
    return result


# ============================
#  コーパス収集
# ============================

def collect_files() -> list[Path]:
    base = Path(sys.prefix) / "Lib"
    files: list[Path] = []
    for f in base.rglob("*.py"):
        try:
            size = f.stat().st_size
            if 300 < size < 15000:
                files.append(f)
        except (OSError, PermissionError):
            continue
        if len(files) >= MAX_FILES:
            break
    return files


def extract_chunks(path: Path, enc: tiktoken.Encoding) -> list[str]:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []
    lines = text.splitlines()
    if len(lines) < MIN_LINES:
        return []
    chunks: list[str] = []
    # ランダム断片 x3
    for _ in range(3):
        size  = random.randint(MIN_LINES, min(MAX_LINES, len(lines)))
        start = random.randint(0, max(0, len(lines) - size))
        chunk = "\n".join(lines[start : start + size])
        n = len(enc.encode(chunk))
        if MIN_TOKENS <= n <= MAX_TOKENS and chunk.strip():
            chunks.append(chunk)
    return chunks


# ============================
#  メイン
# ============================

def main() -> None:
    random.seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    enc = tiktoken.get_encoding("cl100k_base")

    # ---- Step 1: 全辞書ロード ----
    print("=" * 60)
    print("Step 1: 辞書ロード + コーパス収集")
    print("=" * 60)
    raw_dict = load_fixed_dict()
    print(f"  固定辞書: {len(raw_dict)} エントリ")

    files = collect_files()
    random.shuffle(files)
    print(f"  stdlib ファイル: {len(files)}")

    all_chunks: list[str] = []
    for f in files:
        all_chunks.extend(extract_chunks(f, enc))
    random.shuffle(all_chunks)
    all_chunks = all_chunks[:3000]
    print(f"  チャンク: {len(all_chunks)}")

    # ---- Step 2: 安全辞書 → Top 20 抽出 ----
    print("\n" + "=" * 60)
    print("Step 2: Top 20 パターン抽出")
    print("=" * 60)

    # コーパスに出現する文字を集めて安全辞書を作る
    global_chars: set[str] = set()
    for chunk in all_chunks:
        global_chars.update(chunk)
    safe_dict = [(pat, char) for pat, char in raw_dict if char not in global_chars]
    print(f"  安全辞書: {len(safe_dict)} エントリ")

    # 全チャンクを安全辞書で圧縮してパターン使用頻度を集計
    symbol_count: Counter = Counter()
    symbol_to_pattern = {char: pat for pat, char in safe_dict}
    for chunk in all_chunks:
        compressed = compress(chunk, safe_dict)
        for ch in compressed:
            if ch in symbol_to_pattern:
                symbol_count[ch] += 1

    # Top N を選出
    top_symbols = [sym for sym, _ in symbol_count.most_common(TOP_N)]
    top_dict = [(pat, char) for pat, char in safe_dict if char in set(top_symbols)]
    # 元の辞書順を維持（長いパターンが先）
    top_dict_ordered = [e for e in safe_dict if e[1] in set(top_symbols)]

    print(f"\n  Top {TOP_N} パターン:")
    for rank, (pat, char) in enumerate(top_dict_ordered, 1):
        cnt = symbol_count[char]
        print(f"    {rank:2d}. {repr(pat):<35} ({cnt:,} 回)")

    # ---- Step 3: Top20 辞書で学習データ生成 ----
    print("\n" + "=" * 60)
    print("Step 3: 学習データ + テストセット生成")
    print("=" * 60)

    good_pairs: list[tuple[str, str]] = []  # (original, compressed)
    for chunk in all_chunks:
        compressed = compress(chunk, top_dict_ordered)
        if compressed == chunk:
            continue
        restored = decompress(compressed, top_dict_ordered)
        if restored != chunk:
            continue
        good_pairs.append((chunk, compressed))

    random.shuffle(good_pairs)
    print(f"  ロスレスペア: {len(good_pairs)}")

    # テストセットを先に分離
    test_pairs  = good_pairs[:TEST_COUNT]
    train_pairs = good_pairs[TEST_COUNT:]
    print(f"  テスト: {len(test_pairs)} 件（学習に含まない）")
    print(f"  学習:   {len(train_pairs)} 件")

    # 学習データ作成（.strip() で先頭/末尾空白を正規化）
    train_examples: list[dict] = []
    for code, compressed in train_pairs:
        code_s = code.strip()
        comp_s = compressed.strip()
        train_examples.append({
            "instruction": "PythonコードをLMemに圧縮してください。スペース・インデント・改行を1文字も変えずに正確に変換すること。",
            "input":  code_s,
            "output": comp_s,
        })
        train_examples.append({
            "instruction": "LMemコードをPythonに復元してください。スペース・インデント・改行を1文字も変えずに正確に復元すること。",
            "input":  comp_s,
            "output": code_s,
        })
    random.shuffle(train_examples)

    # eval 分割（学習データの10%）
    split = int(len(train_examples) * 0.9)
    train_data = train_examples[:split]
    eval_data  = train_examples[split:]

    # テストデータ（復元方向のみ＝理論証明に必要なのはこれ）
    test_data: list[dict] = []
    for code, compressed in test_pairs:
        test_data.append({
            "instruction": "LMemコードをPythonに復元してください。スペース・インデント・改行を1文字も変えずに正確に復元すること。",
            "input":  compressed.strip(),
            "expected_output": code.strip(),
        })

    # ---- Step 4: 保存 ----
    print("\n" + "=" * 60)
    print("Step 4: データ保存")
    print("=" * 60)

    # 辞書
    dict_path = OUT_DIR / "top20_dict.json"
    with open(dict_path, "w", encoding="utf-8") as f:
        json.dump(top_dict_ordered, f, ensure_ascii=False, indent=2)
    print(f"  辞書:     {dict_path}  ({len(top_dict_ordered)} エントリ)")

    # 学習データ
    for fname, data in [("train.jsonl", train_data), ("eval.jsonl", eval_data)]:
        p = OUT_DIR / fname
        with open(p, "w", encoding="utf-8") as f:
            for ex in data:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"  {fname}: {p}  ({len(data)} 件)")

    # テストデータ
    test_path = OUT_DIR / "test.jsonl"
    with open(test_path, "w", encoding="utf-8") as f:
        for ex in test_data:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"  テスト:   {test_path}  ({len(test_data)} 件)")

    # ---- 統計 ----
    print("\n" + "=" * 60)
    print("統計サマリー")
    print("=" * 60)

    reductions = []
    for code, compressed in good_pairs:
        n_orig = len(enc.encode(code))
        n_comp = len(enc.encode(compressed))
        reductions.append((1 - n_comp / n_orig) * 100)

    avg_red = sum(reductions) / len(reductions)
    print(f"  パターン数:     {TOP_N}")
    print(f"  平均トークン削減: {avg_red:.1f}%")
    print(f"  学習データ:     {len(train_data)} 件  (compress + decompress)")
    print(f"  評価データ:     {len(eval_data)} 件")
    print(f"  テストデータ:   {len(test_data)} 件  (decompress のみ)")
    print(f"  出力先:         {OUT_DIR}")
    print()
    print("次のステップ:")
    print("  1. python train_lmem_prove.py   (学習)")
    print("  2. python test_lmem_prove.py    (テスト → 100点?)")


if __name__ == "__main__":
    main()
