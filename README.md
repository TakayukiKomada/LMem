# LMem — Extreme Python Code Compression for LLMs

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TakayukiKomada/LMem/blob/main/LMem_Middleware.ipynb)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/Takayu/lmem-middleware)

LMem is an experimental compression system that maps Python code patterns to single-token Unicode characters, dramatically reducing the token cost of LLM communication.

> **Try it now:** [Hugging Face Space](https://huggingface.co/spaces/Takayu/lmem-middleware) (no install needed) | [Google Colab](https://colab.research.google.com/github/TakayukiKomada/LMem/blob/main/LMem_Middleware.ipynb)

## Getting Started

Three ways to use LMem — pick whichever suits you:

### Option 1: Hugging Face Space (easiest — no install, no login)

1. Open [https://huggingface.co/spaces/Takayu/lmem-middleware](https://huggingface.co/spaces/Takayu/lmem-middleware)
2. **Compress tab**: Paste your Python code → click **Compress** → see result
3. **Copy the JSON** in the "Save this JSON" box (this is your restore key)
4. **Restore tab**: Paste the JSON → click **Restore** → get back the original code

### Option 2: Google Colab (no install, full Python access)

1. Click the **"Open in Colab"** badge above (or [this link](https://colab.research.google.com/github/TakayukiKomada/LMem/blob/main/LMem_Middleware.ipynb))
2. **Runtime → Run all**
3. Scroll to **"Try it"** — paste your code in `MY_CODE` and run the cell
4. To restore: `decompress(result["compressed"], result["used"])`

### Option 3: CLI (local)

```bash
pip install tiktoken
python lmem.py compress your_script.py
python lmem.py decompress compressed_output.txt
```

## Concept

When LLMs communicate with each other (agent-to-agent), they pay token costs per message. LMem compresses Python code by replacing repeated patterns with single-token Unicode characters — achieving up to **97%+ token reduction** with lossless round-trip.

```
Original:  1,140 tokens
Compressed:   31 tokens  (97.3% reduction)
Dictionary:   24 entries (can be pre-loaded into system prompt = cost 0)
```

The core insight: **dictionary cost is zero** when pre-loaded into the system prompt once, or when baked into LoRA weights so the model "knows" the encoding natively.

## How It Works

### Dynamic Compression (`lmem_compressor.py`)
Scans all substrings of the input code, greedily replaces the most frequent patterns with single-token Unicode characters. Achieves 97%+ compression but requires sending the dictionary alongside the compressed code.

### Fixed-Dictionary Compression (`generate_training_data_v3.py`)
Uses a pre-built dictionary of 453 Python patterns. Achieves ~13% reduction. The dictionary is fixed, so it can be pre-loaded (cost 0) or baked into a fine-tuned model.

### LoRA Fine-tuning (`train_lmem.py`)
Fine-tunes a small LLM (Qwen3.5-0.8B) to understand the compressed format without needing the dictionary in the prompt — the mapping is baked into model weights.

## Architecture

```
LMem_Middleware.ipynb       → Colab middleware (compress/decompress, ready to use)
harness_evaluator.py        → DP-optimal compression engine + evaluation
harness_orchestrator.py     → Automated dictionary building pipeline
unicode_scanner.py          → Scan BMP+SMP for single-token Unicode chars
python_elements_v2.py       → 453 Python patterns for the fixed dictionary
generate_training_data_v3.py → Generate compress/decompress training pairs
train_lmem.py               → QLoRA fine-tuning (RTX 4070 Ti compatible)
inference_lmem.py           → Test trained model on compress/decompress
lmem_compressor.py          → Dynamic extreme compression (97%+)
lmem.py                     → CLI tool (compress/decompress/test/demo)
experiment_understanding.py  → Test whether Opus can understand compressed code
```

## Experiment Results

### Compression
| Mode | Tokens | Reduction |
|------|--------|-----------|
| Original code | 1,140 | baseline |
| Fixed-dict (453 entries) | ~985 | 13.7% |
| AI accent dict (9,076 entries) | ~137 | ~50% (generalizes to unseen code) |
| Dynamic compressed | 31 | 97.3% (file-specific, no generalization) |

### Understanding (Claude Opus)
- **With dictionary**: 167/168 lines restored correctly (one `\n` escape diff)
- **Without dictionary**: Correctly identified the language and purpose from the 3% remaining plaintext
- The 3% residual plaintext = the most unique parts of the code (unrepeated fragments)

### LoRA Training (Qwen3.5-0.8B)
- Training data: 2,100 compress/decompress pairs
- eval_loss: 0.70
- Token accuracy: 84.3%
- Training time: ~67 minutes (RTX 4070 Ti)

## Middleware Details

### Python API (Colab / local)

```python
# Compress
result = process(your_code)

# Restore
restored = decompress(result["compressed"], result["used"])
assert restored == your_code  # always passes
```

### Save / Load (keep compressed data for later)

```python
import json

# Save
with open("compressed.json", "w") as f:
    json.dump({"compressed": result["compressed"], "used": result["used"]}, f)

# Load and restore
with open("compressed.json", "r") as f:
    loaded = json.load(f)
restored = decompress(loaded["compressed"], loaded["used"])
```

### LLM API Integration

```python
# Compress before sending to LLM API
compressed, used = compress_for_api(code)

# Generate a system prompt so the LLM understands the compressed code
system_prompt = make_dict_prompt(used)

# Restore the LLM response
restored = decompress_from_api(compressed, used)
```

### Performance

| Metric | Value |
|--------|-------|
| Dictionary | 9,076 entries (syntax 2,796 + ai_accent 6,280) |
| Compression | DP-optimal (greedy fallback for safety) |
| Lossless | Always guaranteed |
| Typical reduction | ~50% on unseen Python code |

## Token Counting

Uses `tiktoken` with `cl100k_base` encoding (GPT-4 / Claude compatible).

```python
from generate_training_data_v3 import build_fixed_dict, compress_fixed, decompress
import tiktoken

enc = tiktoken.get_encoding("cl100k_base")
dict_entries = build_fixed_dict(enc)

compressed, used = compress_fixed(your_code, dict_entries)
restored = decompress(compressed, used)
assert restored == your_code  # lossless

print(f"Original:   {len(enc.encode(your_code))} tokens")
print(f"Compressed: {len(enc.encode(compressed))} tokens")
```

## Unicode Mapping

1,278 single-token Unicode characters identified in `cl100k_base`. Used characters are filtered to exclude:
- ASCII (< 0x7F) — already in Python source
- Halfwidth/Fullwidth Forms (0xFF00–0xFFEF)
- Combining marks, control chars, private-use area

## Motivation

This is a pseudo-experiment for AI-to-AI communication compression. The ideal scenario: two LLM agents agree on a shared compression dictionary (or both fine-tuned with the same LoRA), then communicate in compressed form. The human here played the role of the "negotiating AI" to demonstrate feasibility.

## Requirements

- Python 3.10+
- `tiktoken` (required)
- `anthropic` (for Opus experiments)
- `torch`, `transformers`, `peft`, `trl` (for LoRA training, GPU recommended)

## License

MIT
