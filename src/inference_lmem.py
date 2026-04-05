"""
LMem学習済みモデル推論テスト

学習済みLoRAアダプターを読み込み、圧縮/復元の精度を検証する。
"""

import json
import torch
from pathlib import Path

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from generate_training_data_v3 import compress_fixed, decompress, build_fixed_dict
import tiktoken


MODEL_NAME = "Qwen/Qwen3.5-0.8B"
ADAPTER_DIR = Path(__file__).parent / "lmem_model"


def load_model():
    """学習済みモデルを読み込む"""
    print(f"モデル読み込み: {MODEL_NAME} + LoRA")

    tokenizer = AutoTokenizer.from_pretrained(
        str(ADAPTER_DIR),
        trust_remote_code=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    model = PeftModel.from_pretrained(model, str(ADAPTER_DIR))

    return model, tokenizer


def inference(model, tokenizer, instruction: str, input_text: str, max_new_tokens: int = 4096) -> str:
    """推論実行"""
    messages = [
        {"role": "system", "content": "あなたはLMem圧縮システムです。Pythonコードの圧縮・復元を正確に行います。"},
        {"role": "user", "content": f"{instruction}\n\n{input_text}"},
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            do_sample=True,
            top_p=0.95,
        )

    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response


def test_decompress(model, tokenizer, code: str) -> bool:
    """圧縮→モデルで復元→一致確認"""
    print(f"\n--- テスト: decompress ---")
    print(f"元コード ({len(code)} chars):")
    print(code[:200] + "..." if len(code) > 200 else code)

    # 固定辞書で圧縮
    enc = tiktoken.get_encoding("cl100k_base")
    dict_entries = build_fixed_dict(enc)
    compressed, history = compress_fixed(code, dict_entries)

    # モデルで復元（辞書なし！）
    restored = inference(
        model, tokenizer,
        "LMemコードをPythonに復元してください。",
        compressed,
    )

    # 正解と比較
    ok = restored.strip() == code.strip()
    print(f"\nモデル出力 ({len(restored)} chars):")
    print(restored[:200] + "..." if len(restored) > 200 else restored)
    print(f"\n一致: {'PASS' if ok else 'FAIL'}")

    if not ok:
        # プログラム的復元と比較
        correct = decompress(compressed, history)
        print(f"正解復元との一致: {correct == code}")

    return ok


def test_compress(model, tokenizer, code: str) -> bool:
    """元コード→モデルで圧縮→プログラム復元→一致確認"""
    print(f"\n--- テスト: compress ---")

    # まずプログラムで辞書を生成（正解参照用）
    enc = tiktoken.get_encoding("cl100k_base")
    dict_entries = build_fixed_dict(enc)
    compressed_ref, history = compress_fixed(code, dict_entries)

    # モデルで圧縮（辞書なし！）
    compressed = inference(
        model, tokenizer,
        "PythonコードをLMemに圧縮してください。",
        code,
    )

    # プログラムで復元
    try:
        restored = decompress(compressed.strip(), history)
        ok = restored.strip() == code.strip()
    except Exception as e:
        print(f"復元エラー: {e}")
        ok = False

    print(f"モデル圧縮 → 復元一致: {'PASS' if ok else 'FAIL'}")
    print(f"参照圧縮との一致: {compressed.strip() == compressed_ref.strip()}")

    return ok


def main():
    model, tokenizer = load_model()

    # テスト用コード
    test_codes = [
        # 小さいコード
        '''def fibonacci(n: int) -> list[int]:
    result = [0, 1]
    for i in range(2, n):
        result.append(result[-1] + result[-2])
    return result

if __name__ == "__main__":
    print(fibonacci(10))
''',
        # 中サイズ
        '''from pathlib import Path
import json

class Config:
    def __init__(self, path: str):
        self.path = Path(path)
        self.data = {}

    def load(self) -> dict:
        if not self.path.exists():
            raise FileNotFoundError(f"Config not found: {self.path}")
        with open(self.path, "r") as f:
            self.data = json.load(f)
        return self.data

    def save(self) -> None:
        with open(self.path, "w") as f:
            json.dump(self.data, f, indent=2)

    def get(self, key: str, default=None):
        return self.data.get(key, default)

    def set(self, key: str, value) -> None:
        self.data[key] = value
        self.save()
''',
    ]

    results = []
    for code in test_codes:
        r1 = test_decompress(model, tokenizer, code)
        r2 = test_compress(model, tokenizer, code)
        results.extend([r1, r2])

    print(f"\n{'='*60}")
    print(f"結果: {sum(results)}/{len(results)} PASS")


if __name__ == "__main__":
    main()
