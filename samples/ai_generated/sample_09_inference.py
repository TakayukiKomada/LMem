"""
推論パイプライン
Python基礎: コンテキストマネージャ、ジェネレータ、例外処理
AI語録: generate, inference, model.eval(), torch.no_grad()
"""
import json
import sys
from pathlib import Path
from typing import Optional


def load_model_and_tokenizer(model_path: str, device: str = "auto"):
    """モデルとトークナイザをロードする"""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        model.eval()
        return model, tokenizer

    except Exception as e:
        print(f"モデルロード失敗: {e}", file=sys.stderr)
        raise


def generate_text(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    do_sample: bool = True,
    top_p: float = 0.9,
) -> str:
    """テキストを生成する"""
    import torch

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated = output_ids[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(generated, skip_special_tokens=True)
    return text


def batch_inference(
    model,
    tokenizer,
    prompts: list[str],
    batch_size: int = 8,
    **kwargs,
) -> list[str]:
    """バッチ推論を実行する"""
    results = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        for prompt in batch:
            result = generate_text(model, tokenizer, prompt, **kwargs)
            results.append(result)
        print(f"  進捗: {min(i + batch_size, len(prompts))}/{len(prompts)}")
    return results


def save_results(
    results: list[dict],
    output_path: Path,
) -> None:
    """推論結果を保存する"""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"結果保存: {output_path} ({len(results)} 件)")


if __name__ == "__main__":
    print("推論パイプライン - モデルが必要です")
