"""
LMem Deterministic Compressor
==============================
fixed_dict.json を使って完全に決定論的に圧縮・復元する。
- Opusや外部LLMに依存しない
- 同じ入力 → 常に同じ出力（再現性100%）
- 学習データ生成・評価・デバッグに使用

使い方:
    python lmem_deterministic.py compress  <input.py>          # 圧縮
    python lmem_deterministic.py decompress <compressed.txt>   # 復元
    python lmem_deterministic.py roundtrip <input.py>          # 往復テスト
    python lmem_deterministic.py stats     <input.py>          # 統計表示
    python lmem_deterministic.py batch     <dir> [--out <dir>] # ディレクトリ一括処理
"""

import json
import sys
from pathlib import Path
from typing import Optional

import tiktoken

# fixed_dict.json のデフォルトパス（このスクリプトと同じ training_data/ 以下）
_DEFAULT_DICT_PATH = Path(__file__).parent / "training_data" / "fixed_dict.json"


# ===========================
#  辞書ロード
# ===========================

def load_fixed_dict(path: Optional[Path] = None) -> list[tuple[str, str]]:
    """
    fixed_dict.json を読み込み、[(pattern, char), ...] のリストを返す。
    順序はファイルに記録された順（最長一致が先頭）を保証する。
    """
    dict_path = path or _DEFAULT_DICT_PATH
    with open(dict_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return [(entry[0], entry[1]) for entry in raw]


# ===========================
#  圧縮・復元
# ===========================

def compress(code: str, dict_entries: list[tuple[str, str]]) -> tuple[str, list[tuple[str, str]]]:
    """
    コードを固定辞書で圧縮する。

    Returns:
        compressed  : 圧縮後の文字列
        used_entries: 実際に適用したエントリ（復元に必要）
    """
    # ソースコードに既に含まれている置換文字はスキップ（衝突防止）
    code_chars = set(code)
    current = code
    used: list[tuple[str, str]] = []

    for pattern, char in dict_entries:
        if char in code_chars:
            # この文字はソース中に既存 → 使用不可
            continue
        if pattern in current:
            current = current.replace(pattern, char)
            used.append((pattern, char))

    return current, used


def decompress(compressed: str, used_entries: list[tuple[str, str]]) -> str:
    """
    圧縮済みコードを元のPythonコードに復元する。
    適用順と逆順で置換することで完全ロスレス復元。
    """
    result = compressed
    for pattern, char in reversed(used_entries):
        result = result.replace(char, pattern)
    return result


# ===========================
#  統計
# ===========================

def calc_stats(
    original: str,
    compressed: str,
    used_entries: list[tuple[str, str]],
    enc: tiktoken.Encoding,
) -> dict:
    """圧縮統計を計算して辞書で返す。"""
    orig_tokens  = len(enc.encode(original))
    comp_tokens  = len(enc.encode(compressed))
    reduction    = (1 - comp_tokens / orig_tokens) * 100 if orig_tokens > 0 else 0.0
    return {
        "original_chars":    len(original),
        "compressed_chars":  len(compressed),
        "original_tokens":   orig_tokens,
        "compressed_tokens": comp_tokens,
        "reduction_pct":     round(reduction, 2),
        "used_entries":      len(used_entries),
    }


def print_stats(stats: dict) -> None:
    """統計をターミナルに見やすく表示する。"""
    print(f"  原文字数:    {stats['original_chars']:>8,} chars")
    print(f"  圧縮文字数:  {stats['compressed_chars']:>8,} chars")
    print(f"  原トークン:  {stats['original_tokens']:>8,} tokens")
    print(f"  圧縮トークン:{stats['compressed_tokens']:>8,} tokens")
    print(f"  削減率:      {stats['reduction_pct']:>7.1f}%")
    print(f"  適用パターン:{stats['used_entries']:>8,} / {stats['used_entries']} entries")


# ===========================
#  CLIコマンド実装
# ===========================

def cmd_compress(input_path: Path, dict_path: Optional[Path] = None) -> None:
    """指定ファイルを圧縮して <name>.lmem として保存する。"""
    code = input_path.read_text(encoding="utf-8")
    entries = load_fixed_dict(dict_path)
    compressed, used = compress(code, entries)

    out_path = input_path.with_suffix(".lmem")
    # 復元に必要な used_entries もサイドカーとして保存
    meta_path = input_path.with_suffix(".lmem.meta.json")

    out_path.write_text(compressed, encoding="utf-8")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(used, f, ensure_ascii=False, indent=2)

    enc = tiktoken.get_encoding("cl100k_base")
    stats = calc_stats(code, compressed, used, enc)
    print(f"[compress] {input_path.name} → {out_path.name}")
    print_stats(stats)
    print(f"  メタデータ: {meta_path.name}")


def cmd_decompress(compressed_path: Path) -> None:
    """<name>.lmem を復元して <name>.restored.py として保存する。"""
    # サイドカーの meta を探す
    meta_path = Path(str(compressed_path) + ".meta.json")
    if not meta_path.exists():
        print(f"[ERROR] メタファイルが見つかりません: {meta_path}", file=sys.stderr)
        sys.exit(1)

    compressed = compressed_path.read_text(encoding="utf-8")
    with open(meta_path, "r", encoding="utf-8") as f:
        used_entries: list[list[str]] = json.load(f)

    restored = decompress(compressed, [(p, c) for p, c in used_entries])

    # 出力ファイル名: foo.lmem → foo.restored.py
    stem = compressed_path.stem  # .lmem を除いた部分
    out_path = compressed_path.parent / (stem + ".restored.py")
    out_path.write_text(restored, encoding="utf-8")
    print(f"[decompress] {compressed_path.name} → {out_path.name}  ({len(restored)} chars)")


def cmd_roundtrip(input_path: Path, dict_path: Optional[Path] = None) -> None:
    """圧縮 → 復元 → 一致確認の往復テスト。"""
    code = input_path.read_text(encoding="utf-8")
    entries = load_fixed_dict(dict_path)

    compressed, used = compress(code, entries)
    restored = decompress(compressed, used)

    enc = tiktoken.get_encoding("cl100k_base")
    stats = calc_stats(code, compressed, used, enc)
    print(f"[roundtrip] {input_path.name}")
    print_stats(stats)

    if restored == code:
        print("  [PASS] 完全一致（ロスレス）")
    else:
        # 差分の詳細を表示
        orig_lines  = code.splitlines()
        rest_lines  = restored.splitlines()
        diffs = [
            (i + 1, a, b)
            for i, (a, b) in enumerate(zip(orig_lines, rest_lines))
            if a != b
        ]
        print(f"  [FAIL] {len(diffs)} 行の差異")
        for lineno, orig, rest in diffs[:5]:
            print(f"    line {lineno}:")
            print(f"      原文  : {repr(orig)}")
            print(f"      復元  : {repr(rest)}")
        if len(diffs) > 5:
            print(f"    ... 他 {len(diffs) - 5} 行")
        sys.exit(1)


def cmd_stats(input_path: Path, dict_path: Optional[Path] = None) -> None:
    """圧縮統計のみ表示（ファイル保存なし）。"""
    code = input_path.read_text(encoding="utf-8")
    entries = load_fixed_dict(dict_path)
    compressed, used = compress(code, entries)
    enc = tiktoken.get_encoding("cl100k_base")
    stats = calc_stats(code, compressed, used, enc)

    print(f"[stats] {input_path.name}")
    print_stats(stats)

    # 適用されたパターン一覧（置換文字はU+XXXX形式で表示）
    if used:
        print("\n  適用パターン (先頭10件):")
        for pat, char in used[:10]:
            cp = f"U+{ord(char):04X}"
            # repr() で安全にエスケープ、末尾の ' を除去して整形
            pat_repr = repr(pat)
            print(f"    {pat_repr:<42} -> {cp}")
        if len(used) > 10:
            print(f"    ... 他 {len(used) - 10} 件")


def cmd_batch(
    input_dir: Path,
    output_dir: Optional[Path] = None,
    dict_path: Optional[Path] = None,
) -> None:
    """
    ディレクトリ内の全 .py ファイルを一括圧縮する。
    往復テストに失敗したファイルはスキップ。
    """
    out_dir = output_dir or (input_dir / "lmem_output")
    out_dir.mkdir(parents=True, exist_ok=True)

    entries = load_fixed_dict(dict_path)
    enc     = tiktoken.get_encoding("cl100k_base")

    py_files = sorted(input_dir.rglob("*.py"))
    print(f"[batch] {len(py_files)} ファイルを処理: {input_dir}")

    total_orig  = 0
    total_comp  = 0
    passed      = 0
    failed      = 0

    for py_file in py_files:
        try:
            code = py_file.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            failed += 1
            continue

        compressed, used = compress(code, entries)
        restored = decompress(compressed, used)

        if restored != code:
            # ロスレス失敗はスキップ
            failed += 1
            continue

        stats = calc_stats(code, compressed, used, enc)
        total_orig  += stats["original_tokens"]
        total_comp  += stats["compressed_tokens"]

        # 出力: フラット構造（元パスのフォルダ区切りを _ に変換）
        rel   = py_file.relative_to(input_dir)
        stem  = str(rel).replace("/", "_").replace("\\", "_").replace(".py", "")
        out_py   = out_dir / (stem + ".lmem")
        out_meta = out_dir / (stem + ".lmem.meta.json")

        out_py.write_text(compressed, encoding="utf-8")
        with open(out_meta, "w", encoding="utf-8") as f:
            json.dump(used, f, ensure_ascii=False)

        passed += 1

    overall = (1 - total_comp / total_orig) * 100 if total_orig > 0 else 0
    print(f"\n  成功: {passed} / {len(py_files)} ファイル (失敗: {failed})")
    print(f"  合計削減: {total_orig:,} → {total_comp:,} tokens  ({overall:.1f}%)")
    print(f"  出力先: {out_dir}")


# ===========================
#  エントリポイント
# ===========================

def _usage() -> None:
    print(__doc__)
    sys.exit(0)


def main() -> None:
    args = sys.argv[1:]
    if not args:
        _usage()

    cmd = args[0]

    # --dict オプションのパース
    dict_path: Optional[Path] = None
    if "--dict" in args:
        idx = args.index("--dict")
        dict_path = Path(args[idx + 1])
        args = args[:idx] + args[idx + 2:]

    if cmd == "compress":
        if len(args) < 2:
            print("使い方: lmem_deterministic.py compress <input.py>", file=sys.stderr)
            sys.exit(1)
        cmd_compress(Path(args[1]), dict_path)

    elif cmd == "decompress":
        if len(args) < 2:
            print("使い方: lmem_deterministic.py decompress <compressed.lmem>", file=sys.stderr)
            sys.exit(1)
        cmd_decompress(Path(args[1]))

    elif cmd == "roundtrip":
        if len(args) < 2:
            print("使い方: lmem_deterministic.py roundtrip <input.py>", file=sys.stderr)
            sys.exit(1)
        cmd_roundtrip(Path(args[1]), dict_path)

    elif cmd == "stats":
        if len(args) < 2:
            print("使い方: lmem_deterministic.py stats <input.py>", file=sys.stderr)
            sys.exit(1)
        cmd_stats(Path(args[1]), dict_path)

    elif cmd == "batch":
        if len(args) < 2:
            print("使い方: lmem_deterministic.py batch <dir> [--out <outdir>]", file=sys.stderr)
            sys.exit(1)
        out_dir: Optional[Path] = None
        if "--out" in args:
            idx = args.index("--out")
            out_dir = Path(args[idx + 1])
        cmd_batch(Path(args[1]), out_dir, dict_path)

    else:
        print(f"不明なコマンド: {cmd}", file=sys.stderr)
        _usage()


if __name__ == "__main__":
    main()
