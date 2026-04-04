"""
Unicode全域スキャン（BMP + SMP）→ 1トークン文字リスト作成

tiktoken cl100k_base を使用して、Unicode BMP (U+0000〜U+FFFF) と
SMP (U+10000〜U+1FFFF) の全域をスキャンし、1トークンで表現される文字を抽出する。
"""

import json
import unicodedata
from pathlib import Path

import tiktoken


def scan_unicode_range(enc: tiktoken.Encoding, start: int, end: int) -> list[dict]:
    """指定範囲のUnicode文字をスキャンし、1トークン文字を返す"""
    results = []
    for cp in range(start, end):
        cat = unicodedata.category(chr(cp))
        # 未割当(Cn)、サロゲート(Cs)はスキップ
        if cat in ("Cn", "Cs"):
            continue
        c = chr(cp)
        try:
            tokens = enc.encode(c)
            if len(tokens) == 1:
                name = unicodedata.name(c, "")
                results.append({
                    "cp": cp,
                    "hex": f"U+{cp:04X}",
                    "char": c,
                    "category": cat,
                    "name": name,
                    "token_id": tokens[0],
                    "utf8_bytes": len(c.encode("utf-8")),
                })
        except Exception:
            pass
    return results


def classify_visibility(entry: dict) -> str:
    """文字の視認性を分類する"""
    cat = entry["category"]
    # 制御文字・結合文字・書式文字は不可視
    if cat in ("Cc", "Cf", "Mn", "Mc", "Me", "Zs", "Zl", "Zp"):
        return "invisible"
    # 私用域は環境依存
    if cat == "Co":
        return "unreliable"
    # それ以外は可視
    return "visible"


def main():
    enc = tiktoken.get_encoding("cl100k_base")
    output_dir = Path(__file__).parent

    # === BMP スキャン (U+0000 〜 U+FFFF) ===
    print("BMP (U+0000 〜 U+FFFF) をスキャン中...")
    bmp_results = scan_unicode_range(enc, 0x0000, 0x10000)
    print(f"  1トークン文字数: {len(bmp_results)}")

    # === SMP スキャン (U+10000 〜 U+1FFFF) ===
    print("SMP (U+10000 〜 U+1FFFF) をスキャン中...")
    smp_results = scan_unicode_range(enc, 0x10000, 0x20000)
    print(f"  1トークン文字数: {len(smp_results)}")

    all_results = bmp_results + smp_results

    # 視認性分類を追加
    for entry in all_results:
        entry["visibility"] = classify_visibility(entry)

    visible = [e for e in all_results if e["visibility"] == "visible"]
    invisible = [e for e in all_results if e["visibility"] == "invisible"]

    print(f"\n=== スキャン結果サマリー ===")
    print(f"総1トークン文字数: {len(all_results)}")
    print(f"  可視文字: {len(visible)}")
    print(f"  不可視文字: {len(invisible)}")

    # カテゴリ別集計
    cats = {}
    for e in visible:
        cats[e["category"]] = cats.get(e["category"], 0) + 1
    print(f"\n可視文字のカテゴリ別内訳:")
    for cat, count in sorted(cats.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}")

    # JSON保存
    output_path = output_dir / "scan_results_tiktoken.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "encoding": "cl100k_base",
            "bmp_count": len(bmp_results),
            "smp_count": len(smp_results),
            "total_count": len(all_results),
            "visible_count": len(visible),
            "characters": all_results,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n結果保存先: {output_path}")

    # 可視文字のみのリストも保存
    visible_path = output_dir / "visible_single_tokens.json"
    with open(visible_path, "w", encoding="utf-8") as f:
        json.dump(visible, f, ensure_ascii=False, indent=2)
    print(f"可視文字リスト: {visible_path}")

    return all_results


if __name__ == "__main__":
    main()
