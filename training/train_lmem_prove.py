"""
LMem 理論証明 — 学習スクリプト
================================
Top 20 辞書データで学習。100点を目指す設定。

変更点（train_lmem.py から）:
  - データパス: training_data/prove/
  - エポック: 2 → 5（完全記憶させる）
  - 出力: lmem_model_prove/
"""

import torch
from pathlib import Path

from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig


# === 設定 ===
MODEL_NAME  = "Qwen/Qwen3.5-0.8B"
OUTPUT_DIR  = Path(__file__).parent / "lmem_model_prove"
DATA_DIR    = Path(__file__).parent / "training_data" / "prove"
NUM_EPOCHS  = 12


def format_chat(example: dict, tokenizer) -> dict:
    """チャット形式に変換する。"""
    messages = [
        {"role": "system", "content": "あなたはLMem圧縮システムです。Pythonコードの圧縮・復元を1文字の狂いもなく正確に行います。スペース数・インデント幅・改行位置を絶対に変えないでください。"},
        {"role": "user", "content": f"{example['instruction']}\n\n{example['input']}"},
        {"role": "assistant", "content": example["output"]},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    return {"text": text}


def main():
    print("=" * 60)
    print("LMem 理論証明 - LoRA 学習")
    print(f"  モデル:   {MODEL_NAME}")
    print(f"  データ:   {DATA_DIR}")
    print(f"  エポック: {NUM_EPOCHS}")
    print(f"  出力:     {OUTPUT_DIR}")
    print("=" * 60)

    # データ読み込み
    print("\nデータ読み込み...")
    dataset = load_dataset("json", data_files={
        "train": str(DATA_DIR / "train.jsonl"),
        "eval":  str(DATA_DIR / "eval.jsonl"),
    })
    print(f"  train: {len(dataset['train'])} 件")
    print(f"  eval:  {len(dataset['eval'])} 件")

    # トークナイザー
    print(f"\nトークナイザー: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # モデル
    print(f"モデル読み込み: {MODEL_NAME}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    # LoRA
    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.0,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    trainable, total = model.get_nb_trainable_parameters()
    print(f"  全パラメータ:   {total:,}")
    print(f"  学習パラメータ: {trainable:,} ({trainable/total*100:.2f}%)")

    # データ変換
    train_dataset = dataset["train"].map(lambda ex: format_chat(ex, tokenizer))
    eval_dataset  = dataset["eval"].map(lambda ex: format_chat(ex, tokenizer))

    # 学習設定（100点狙い）
    training_args = SFTConfig(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=3e-4,
        weight_decay=0.005,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        num_train_epochs=NUM_EPOCHS,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        bf16=True,
        gradient_checkpointing=True,
        max_length=512,
        dataset_text_field="text",
        optim="adamw_torch",
        report_to="none",
    )

    # 学習
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    print(f"\n学習開始（{NUM_EPOCHS} エポック）...")
    trainer.train()

    # 保存
    print(f"\nモデル保存: {OUTPUT_DIR}")
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    print("\n=== 学習完了 ===")


if __name__ == "__main__":
    main()
