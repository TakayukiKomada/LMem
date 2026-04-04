"""
LoRAファインチューニング
Python基礎: 引数パース、パス操作、ログ出力
AI語録: LoRA, PeftModel, from_pretrained, TrainingArguments
"""
import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """コマンドライン引数をパースする"""
    parser = argparse.ArgumentParser(description="LoRAファインチューニング")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3.5-0.8B",
                        help="ベースモデル名")
    parser.add_argument("--data-path", type=str, required=True,
                        help="学習データのパス")
    parser.add_argument("--output-dir", type=str, default="lora_output",
                        help="出力ディレクトリ")
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=16,
                        help="LoRAのランク")
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_training_data(path: Path) -> list[dict]:
    """学習データをJSONLから読み込む"""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    print(f"学習データ読み込み: {len(data)} 件")
    return data


def format_prompt(instruction: str, input_text: str) -> str:
    """プロンプトをフォーマットする"""
    messages = [
        {"role": "system", "content": "あなたはLMem圧縮システムです。"},
        {"role": "user", "content": f"{instruction}\n\n{input_text}"},
    ]
    # メッセージをテンプレート形式にする
    formatted = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        formatted += f"<|{role}|>\n{content}\n"
    return formatted


def save_training_log(log: list[dict], output_dir: Path) -> None:
    """学習ログを保存する"""
    log_path = output_dir / "training_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)
    print(f"ログ保存: {log_path}")


if __name__ == "__main__":
    args = parse_args()
    print(f"モデル: {args.model_name}")
    print(f"LoRA r={args.lora_r}, alpha={args.lora_alpha}")
    print(f"学習率: {args.learning_rate}")
    print(f"エポック数: {args.num_epochs}")
