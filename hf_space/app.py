"""LMem Middleware — Lossless Code Compression (Gradio UI)"""

import json
import zlib
import base64
from pathlib import Path
from collections import defaultdict

import gradio as gr
import tiktoken

# === 辞書読み込み ===
_dict_path = Path(__file__).parent / "dict.json.zlib.b64"
if _dict_path.exists():
    # base64 + zlib 圧縮形式
    with open(_dict_path, "r") as f:
        b64 = f.read().strip()
    DICT_ENTRIES = json.loads(zlib.decompress(base64.b64decode(b64)).decode("utf-8"))
else:
    # 生JSON形式（フォールバック）
    raw_path = Path(__file__).parent / "final_dict.json"
    with open(raw_path, "r", encoding="utf-8") as f:
        DICT_ENTRIES = json.load(f)

_enc = tiktoken.get_encoding("cl100k_base")
_bpe_cache = {}


def _bpe_cost(s):
    if s not in _bpe_cache:
        _bpe_cache[s] = len(_enc.encode(s))
    return _bpe_cache[s]


def token_count(text):
    """BPEトークン数を返す"""
    return len(_enc.encode(text))


def compress(code):
    """DP最適圧縮"""
    N = len(code)
    if N == 0:
        return code, []
    safe = [e for e in DICT_ENTRIES if e["symbol"] not in code]
    starts = defaultdict(list)
    for entry in safe:
        pat = entry["pattern"]
        idx = 0
        while True:
            pos = code.find(pat, idx)
            if pos == -1:
                break
            starts[pos].append((pos + len(pat), entry))
            idx = pos + 1
    if not starts:
        return code, []
    INF = float("inf")
    dp = [INF] * (N + 1)
    choice = [None] * (N + 1)
    dp[N] = 0
    for i in range(N - 1, -1, -1):
        for end, entry in starts.get(i, []):
            sc = _bpe_cost(entry["symbol"])
            if dp[end] + sc < dp[i]:
                dp[i] = dp[end] + sc
                choice[i] = (end, entry)
        cc = _bpe_cost(code[i])
        if dp[i + 1] + cc < dp[i]:
            dp[i] = dp[i + 1] + cc
            choice[i] = (i + 1, None)
        for cl in range(2, min(11, N - i + 1)):
            ch = code[i : i + cl]
            chc = _bpe_cost(ch)
            r = dp[i + cl] if i + cl <= N else 0
            if r + chc < dp[i]:
                dp[i] = r + chc
                choice[i] = (i + cl, None)
    used_set = set()
    segments = []
    pos = 0
    while pos < N:
        np_, entry = choice[pos]
        if entry:
            segments.append(entry["symbol"])
            used_set.add(id(entry))
        else:
            segments.append(code[pos:np_])
        pos = np_
    compressed = "".join(segments)
    used = [e for e in safe if id(e) in used_set]
    if decompress(compressed, used) != code:
        # Greedy フォールバック
        current = code
        used = []
        for entry in sorted(DICT_ENTRIES, key=lambda x: -len(x["pattern"])):
            pat, sym = entry["pattern"], entry["symbol"]
            if sym in code or sym in current:
                continue
            if pat in current:
                current = current.replace(pat, sym)
                used.append(entry)
        return current, used
    return compressed, used


def decompress(compressed, used_entries):
    """復元（完全ロスレス）"""
    sym_map = {e["symbol"]: e["pattern"] for e in used_entries}
    if not sym_map:
        return compressed
    sorted_syms = sorted(sym_map.keys(), key=len, reverse=True)
    result = []
    i = 0
    while i < len(compressed):
        matched = False
        for sym in sorted_syms:
            if compressed[i : i + len(sym)] == sym:
                result.append(sym_map[sym])
                i += len(sym)
                matched = True
                break
        if not matched:
            result.append(compressed[i])
            i += 1
    return "".join(result)


# === Gradio UI ===

# 最後の圧縮結果を保持（復元タブ用）
_last_result = {"compressed": "", "used": []}


def do_compress(code):
    """圧縮タブのハンドラ"""
    global _last_result
    if not code.strip():
        return "", "", "コードを入力してください"
    orig = token_count(code)
    compressed, used = compress(code)
    comp = token_count(compressed)
    pct = (1 - comp / orig) * 100 if orig > 0 else 0
    restored = decompress(compressed, used)
    ok = restored == code

    _last_result = {"compressed": compressed, "used": used}

    # 保存用JSON
    save_json = json.dumps(
        {"compressed": compressed, "used": used}, ensure_ascii=False, indent=2
    )

    stats = (
        f"Before:   {orig:,} tokens  ({len(code):,} chars)\n"
        f"After:    {comp:,} tokens  ({len(compressed):,} chars)\n"
        f"Saved:    {orig - comp:,} tokens  ({pct:.1f}%)\n"
        f"Lossless: {'PASS' if ok else 'FAIL'}"
    )
    return compressed, save_json, stats


def do_restore(restore_json):
    """復元タブのハンドラ"""
    if not restore_json.strip():
        return "", "JSONを入力してください"
    try:
        data = json.loads(restore_json)
        compressed = data["compressed"]
        used = data["used"]
    except (json.JSONDecodeError, KeyError) as e:
        return "", f"JSONの解析エラー: {e}"

    restored = decompress(compressed, used)
    tok = token_count(restored)
    stats = f"Restored: {tok:,} tokens  ({len(restored):,} chars)"
    return restored, stats


# デモコード
DEMO_CODE = '''\
from pathlib import Path
from typing import Optional, List, Dict, Any
import json
import logging


class DataProcessor:
    def __init__(self, config: dict):
        self.config = config
        self.base_dir = Path(config.get("base_dir", "."))
        self._results: dict[str, Any] = {}
        self._logger = logging.getLogger(__name__)

    def process(self, file_path: str) -> Optional[dict]:
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
            self._results[file_path] = data
            self._logger.info(f"Processed: {file_path}")
            return data
        except Exception as e:
            self._logger.error(f"Error: {file_path}: {e}")
            return None

    def summary(self) -> None:
        total = len(self._results)
        print(f"Total processed: {total}")


def main():
    logging.basicConfig(level=logging.INFO)
    processor = DataProcessor({"base_dir": "/tmp/data"})
    processor.process("input.json")
    processor.summary()


if __name__ == "__main__":
    main()
'''

with gr.Blocks(title="LMem Middleware", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# LMem Middleware — Lossless Code Compression\n"
        "Paste Python code → compress → save JSON → restore anytime.\n"
        "9,076-entry dictionary (syntax + AI accent). Always lossless."
    )

    with gr.Tabs():
        # === 圧縮タブ ===
        with gr.TabItem("Compress"):
            with gr.Row():
                with gr.Column():
                    input_code = gr.Textbox(
                        label="Input (Python code)",
                        lines=20,
                        value=DEMO_CODE,
                        placeholder="Paste your Python code here...",
                    )
                    compress_btn = gr.Button("Compress", variant="primary", size="lg")
                with gr.Column():
                    output_compressed = gr.Textbox(
                        label="Compressed text", lines=10, interactive=False
                    )
                    output_json = gr.Textbox(
                        label="Save this JSON (for restoration)",
                        lines=8,
                        interactive=True,
                        info="Copy this JSON. Paste it in the Restore tab to get back the original.",
                    )
                    output_stats = gr.Textbox(
                        label="Stats", lines=4, interactive=False
                    )

            compress_btn.click(
                fn=do_compress,
                inputs=[input_code],
                outputs=[output_compressed, output_json, output_stats],
            )

        # === 復元タブ ===
        with gr.TabItem("Restore"):
            with gr.Row():
                with gr.Column():
                    restore_input = gr.Textbox(
                        label='Paste the JSON from "Compress" tab',
                        lines=15,
                        placeholder='{"compressed": "...", "used": [...]}',
                    )
                    restore_btn = gr.Button("Restore", variant="primary", size="lg")
                with gr.Column():
                    restore_output = gr.Textbox(
                        label="Restored code", lines=20, interactive=False
                    )
                    restore_stats = gr.Textbox(
                        label="Stats", lines=2, interactive=False
                    )

            restore_btn.click(
                fn=do_restore,
                inputs=[restore_input],
                outputs=[restore_output, restore_stats],
            )

if __name__ == "__main__":
    demo.launch()
