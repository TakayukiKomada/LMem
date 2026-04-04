"""
LMem — Python極限圧縮ツール

任意のPythonファイルを入力し、1トークンUnicode文字で極限圧縮する。
辞書はコードごとに動的生成。復元も完全保証。

使い方:
    python lmem.py compress input.py              # 圧縮 → .lmem + .lmem.json
    python lmem.py decompress input.py.lmem       # 復元 → .restored.py
    python lmem.py test input.py                   # 圧縮→復元→一致確認
    python lmem.py demo                            # 内蔵デモ
"""

import json
import sys
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


def find_best_replacement(
    text: str, enc: tiktoken.Encoding, replacement_char: str,
    min_len: int = 2, max_len: int = 200
) -> tuple[str, int] | None:
    """BPE実測で最もトークン削減効果の高い部分文字列を返す"""
    original_tokens = len(enc.encode(text))
    max_len = min(max_len, len(text))

    # 理論スコアで候補を絞り込み
    candidates = {}
    text_len = len(text)
    for length in range(min_len, max_len + 1):
        for i in range(text_len - length + 1):
            sub = text[i:i + length]
            if sub in candidates or replacement_char in sub:
                continue
            count = text.count(sub)
            tok_count = len(enc.encode(sub))
            theory = (tok_count - 1) * count
            if theory > 0:
                candidates[sub] = theory

    if not candidates:
        return None

    # 理論スコア上位100件をBPE実測
    sorted_cands = sorted(candidates.items(), key=lambda x: -x[1])[:100]

    best = None
    best_saving = 0
    for sub, theory in sorted_cands:
        replaced = text.replace(sub, replacement_char)
        saving = original_tokens - len(enc.encode(replaced))
        if saving > best_saving:
            best_saving = saving
            best = sub

    if best is None or best_saving <= 0:
        return None

    return best, best_saving


def compress(code: str, verbose: bool = False) -> tuple[str, list[tuple[str, str]]]:
    """任意のテキストをLMem極限圧縮

    戻り値: (圧縮テキスト, [(パターン, 置換文字), ...])
    """
    enc = tiktoken.get_encoding("cl100k_base")
    available = load_single_token_chars()

    # ソースコード中の文字を除外
    code_chars = set(code)
    available = [c for c in available if c not in code_chars]

    original_tokens = len(enc.encode(code))
    if verbose:
        print(f"利用可能文字: {len(available)} / 元トークン: {original_tokens}")

    current = code
    history = []
    char_idx = 0
    total_saved = 0

    while char_idx < len(available):
        replacement = available[char_idx]

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

        current_tokens = len(enc.encode(current))
        pct = (1 - current_tokens / original_tokens) * 100

        if verbose:
            step = len(history)
            display = repr(pattern) if len(pattern) <= 50 else repr(pattern[:47] + "...")
            print(f"  [{step:>3}] {replacement} ← {display:<55} BPE:{saving:>+3}  残:{current_tokens:>5} ({pct:.1f}%)")

        if current_tokens <= max(original_tokens * 0.03, 5):
            break

    return current, history


def decompress(compressed: str, history: list[tuple[str, str]]) -> str:
    """逆順復元"""
    result = compressed
    for pattern, char in reversed(history):
        result = result.replace(char, pattern)
    return result


def save_compressed(
    input_path: Path, compressed: str, history: list[tuple[str, str]],
    original_tokens: int, compressed_tokens: int
):
    """圧縮結果をファイルに保存"""
    # 圧縮テキスト
    lmem_path = input_path.with_suffix(input_path.suffix + ".lmem")
    lmem_path.write_text(compressed, encoding="utf-8")

    # 辞書（復元用）
    dict_path = input_path.with_suffix(input_path.suffix + ".lmem.json")
    data = {
        "version": "3.0",
        "source": input_path.name,
        "original_tokens": original_tokens,
        "compressed_tokens": compressed_tokens,
        "reduction_pct": round((1 - compressed_tokens / original_tokens) * 100, 1),
        "steps": len(history),
        "history": [(p, c) for p, c in history],
    }
    dict_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    return lmem_path, dict_path


def load_compressed(lmem_path: Path) -> tuple[str, list[tuple[str, str]]]:
    """圧縮ファイルと辞書を読み込み"""
    compressed = lmem_path.read_text(encoding="utf-8")
    dict_path = Path(str(lmem_path) + ".json")
    data = json.loads(dict_path.read_text(encoding="utf-8"))
    history = [(p, c) for p, c in data["history"]]
    return compressed, history


def cmd_compress(input_path: str):
    """圧縮コマンド"""
    enc = tiktoken.get_encoding("cl100k_base")
    path = Path(input_path)
    code = path.read_text(encoding="utf-8")
    original_tokens = len(enc.encode(code))

    print(f"圧縮中: {path.name} ({original_tokens} tokens, {len(code)} chars)")
    compressed, history = compress(code, verbose=True)

    compressed_tokens = len(enc.encode(compressed))
    pct = (1 - compressed_tokens / original_tokens) * 100

    lmem_path, dict_path = save_compressed(
        path, compressed, history, original_tokens, compressed_tokens
    )

    print(f"\n結果: {original_tokens} → {compressed_tokens} tokens ({pct:.1f}% 削減)")
    print(f"  圧縮: {lmem_path}")
    print(f"  辞書: {dict_path}")


def cmd_decompress(lmem_path_str: str):
    """復元コマンド"""
    lmem_path = Path(lmem_path_str)
    compressed, history = load_compressed(lmem_path)
    restored = decompress(compressed, history)

    # .py.lmem → .restored.py
    out_name = lmem_path.stem  # "foo.py"
    out_path = lmem_path.parent / (Path(out_name).stem + ".restored" + Path(out_name).suffix)
    out_path.write_text(restored, encoding="utf-8")
    print(f"復元: {out_path}")


def cmd_test(input_path: str):
    """圧縮→復元→一致確認"""
    enc = tiktoken.get_encoding("cl100k_base")
    path = Path(input_path)
    code = path.read_text(encoding="utf-8")
    original_tokens = len(enc.encode(code))

    print(f"テスト: {path.name} ({original_tokens} tokens)")
    compressed, history = compress(code, verbose=True)

    compressed_tokens = len(enc.encode(compressed))
    restored = decompress(compressed, history)
    ok = restored == code
    pct = (1 - compressed_tokens / original_tokens) * 100

    print(f"\n  {original_tokens} → {compressed_tokens} tokens ({pct:.1f}% 削減)")
    print(f"  置換数: {len(history)}")
    print(f"  往復: {'PASS' if ok else 'FAIL'}")

    if not ok:
        for i, (a, b) in enumerate(zip(code, restored)):
            if a != b:
                print(f"  差異位置 {i}: {repr(code[max(0,i-20):i+20])}")
                break

    return ok


def cmd_demo():
    """内蔵デモ"""
    from lmem_compressor import SAMPLE_CODE
    enc = tiktoken.get_encoding("cl100k_base")
    original_tokens = len(enc.encode(SAMPLE_CODE))

    print(f"デモコード: {original_tokens} tokens / {len(SAMPLE_CODE)} chars\n")
    compressed, history = compress(SAMPLE_CODE, verbose=True)

    compressed_tokens = len(enc.encode(compressed))
    restored = decompress(compressed, history)
    pct = (1 - compressed_tokens / original_tokens) * 100

    print(f"\n{'='*60}")
    print(f"  {original_tokens} → {compressed_tokens} tokens ({pct:.1f}% 削減)")
    print(f"  往復: {'PASS' if restored == SAMPLE_CODE else 'FAIL'}")
    print(f"{'='*60}")
    print(f"\n--- 圧縮後 ---")
    print(compressed)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return

    cmd = sys.argv[1]

    if cmd == "compress" and len(sys.argv) >= 3:
        cmd_compress(sys.argv[2])
    elif cmd == "decompress" and len(sys.argv) >= 3:
        cmd_decompress(sys.argv[2])
    elif cmd == "test" and len(sys.argv) >= 3:
        cmd_test(sys.argv[2])
    elif cmd == "demo":
        cmd_demo()
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
