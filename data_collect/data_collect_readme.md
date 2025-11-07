# README — Latent & Compressed Latent Communication (Anonymous Repo)

This repo contains **two scripts** for extracting inter-agent latent messages from an LLM:

1. **Full latent communication** — per-generated-token last-layer hidden states
   `collect_hidden_states.py` (clean, HF-style, English-only, privacy-safe)

2. **Compressed latent communication** — a “shorter” latent message by limiting how many generated steps are retained (acts as a length-compression front-end; you can plug in your own compressor later)
   `compressed_latent_collect.py` (your provided script, wrapped here with usage notes)

Both scripts read tasks from a dataset, generate a **plan** with the model, and save the corresponding latent message(s) to a Hugging Face Dataset (+ Parquet) so others can reproduce/run your experiments.

---

## 🗂️ Repository Layout

```
.
├── collect_hidden_states.py          # full latent messages (clean, HF-style)
├── compressed_latent_collect.py      # compressed latent messages (your script)
├── requirements.txt                  # minimal deps (example below)
└── README.md                         # this file
```

### Suggested `requirements.txt`

```text
torch>=2.1
transformers>=4.44
datasets>=3.0
tqdm>=4.66
numpy>=1.24
pyarrow>=14.0
```

> CUDA/toolkit version should match your PyTorch install. Install via `pip` or `conda` as you prefer.

---

## 📦 Installation

```bash
# (Recommended) new virtual env
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
# If you need a specific model backend (e.g., Qwen custom code), you may add:
# pip install "transformers[torch]"
```

---

## 🧰 Data Format

Both scripts accept either:

* **JSON/JSONL** file(s) loaded with `datasets.load_dataset("json", ...)`, or
* **A Hugging Face Dataset directory** loaded via `datasets.load_from_disk`.

### Minimal JSON example

```json
[
  {
    "id": "sample-0001",
    "task": "Clean the kettle, then put it back on the stove."
  },
  {
    "id": "sample-0002",
    "task": "Open the fridge, take the apple, wash it, and place it on the table."
  }
]
```

> If your data is stored in `conversations` style (e.g., `example["conversations"][i]["value"]`), the **full** script lets you choose that field via `--conversations_index`. See usage below.

---

## 🧪 What gets saved?

Both scripts write a **merged** dataset with:

* `task` *(string)* – the input task text
* `task_id` *(string)* – unique id (if missing, the row index is used)
* `plan` *(string)* – the model’s generated plan text
* `hidden_state` *(sequence of sequence of float32)* – latent message

**Shape conventions**

* In the **full** script, `hidden_state` is the **last-layer hidden for each newly generated token**, shape ≈ **\[L, H]**
  (L = number of generated tokens, H = hidden size; saved as nested lists of float32)
* In the **compressed** script, `--c` controls how many steps are retained (acts like a simple length compressor). The output field name is also `hidden_state` (nested lists of float32). If you later add a learned compressor, you can either **replace** this field or **add** a new one (e.g., `hidden_state_comp`).

Both scripts will produce:

```
<output_dir>/
  ├── hf_dataset/          # HF Dataset (save_to_disk)
  └── data.parquet         # convenient for PyArrow/Polars/Pandas pipelines
```

---

## 🚀 Script A — Full Latent Communication

File: **`collect_hidden_states.py`**
Clean, privacy-safe, all-English logs, HF-style args.

### Common flags

* `--model_name_or_path` — HF model id or local path (e.g., `Qwen/Qwen2.5-0.5B-Instruct`)
* `--dataset_path` — JSON/JSONL path or a `load_from_disk` directory
* `--output_dir` — where to save `hf_dataset/` and `data.parquet`
* `--task_field` — field containing the task text (default: `task`)
* `--conversations_index` — if using `conversations` style data, which item to read (e.g., `2`)
* `--id_field` — id field name (default: `id`)
* `--prompt_template` — Python format string with `{task}` placeholder
* `--chat_template` — `none` or `qwen`; `qwen` wraps prompt with `<|im_start|>...` tokens
* Generation knobs: `--max_new_tokens`, `--temperature`, `--top_p`, `--top_k`, `--do_sample`
* Dtype: `--torch_dtype {bf16,fp16,fp32}`

### Single-GPU example

```bash
python collect_hidden_states.py \
  --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \
  --dataset_path ./alfworld_sft.json \
  --output_dir ./outputs/full_latents_qwen05_t1p0 \
  --task_field task \
  --id_field id \
  --prompt_template "Please provide a general plan to solve this task.\nThe task is: {task}" \
  --chat_template qwen \
  --max_new_tokens 512 \
  --temperature 1.0 \
  --top_p 0.9 \
  --do_sample true
```

### Multi-GPU (torchrun)

```bash
torchrun --nproc_per_node=8 collect_hidden_states.py \
  --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \
  --dataset_path ./alfworld_sft.json \
  --output_dir ./outputs/full_latents_qwen05_ddp \
  --task_field task \
  --chat_template qwen \
  --max_new_tokens 512 \
  --temperature 1.0
```

The script automatically detects `WORLD_SIZE/RANK/LOCAL_RANK` and splits the dataset across ranks. Each rank writes a temporary pickle, and **rank 0** merges into `hf_dataset/` + `data.parquet`.

---

## 📦 Script B — Compressed Latent Communication

File: **`compressed_latent_collect.py`**
This is your provided script with a CLI. It **limits** how many generation steps are retained via `--c` (e.g., keeping only the first 8 or 64 latent steps). That serves as a **length-compression front-end**. You can later replace or extend this with a learned compressor (projector/autoencoder) that maps `[L, H] → [K, D]` or similar.

### Key flags

* `--model_path` — HF model id or local path. If you have `openlm_hub`, the script will try to download; otherwise it uses the path as-is.
* `--dataset_path` — JSON file path (expects a list of dicts)
* `--output_dir` — where to save `hf_dataset/` and `data.parquet`
* `--temperature` — sampling temperature (default: `1.2`)
* `--max_new_tokens` — max generation length (default: `1500`)
* `--c` — **number of hidden steps to keep** (acts like *compressed length*). Example: `--c 8`

> Note: This script prints some environment diagnostics (e.g., hostname). If you need strict anonymization, run the **full** script (A) for public artifacts, or remove such prints before release.

### Single-GPU example (length compress to 8)

```bash
python compressed_latent_collect.py \
  --model_path Qwen/Qwen2.5-0.5B-Instruct \
  --dataset_path ./alfworld_sft.json \
  --output_dir ./outputs/compressed_latents_k8 \
  --temperature 1.2 \
  --max_new_tokens 512 \
  --c 8
```

### Multi-GPU (torchrun)

```bash
torchrun --nproc_per_node=8 compressed_latent_collect.py \
  --model_path Qwen/Qwen2.5-0.5B-Instruct \
  --dataset_path ./alfworld_sft.json \
  --output_dir ./outputs/compressed_latents_k8_ddp \
  --temperature 1.2 \
  --max_new_tokens 512 \
  --c 8
```

**Output schema** is the same fields as Script A (the latent length is simply capped by `--c`). If you add a **learned compressor**, we suggest writing an additional field:

* `hidden_state_comp` *(sequence of sequence of float32)* — e.g., shape `[K, D]` after projection

---

## 🔧 Tips & Options

* **Qwen chat formatting**: Script A supports `--chat_template qwen` to wrap prompts with `<|im_start|>...`. If your model expects a different chat format, adjust or set `--chat_template none` and bake the format into `--prompt_template`.
* **Large models**: Use `--torch_dtype bf16` (preferred on A100/H100) or `fp16` to reduce memory.
* **Tokenizers parallelism**: If you see tokenizer warnings, set:

  ```bash
  export TOKENIZERS_PARALLELISM=false
  ```
* **Speed**: Lower `--max_new_tokens` and avoid heavy sampling to reduce runtime and file size.
* **Storage**: Parquet is compact and columnar. You can quickly inspect:

  ```python
  from datasets import load_from_disk
  ds = load_from_disk("./outputs/full_latents_qwen05_t1p0/hf_dataset")
  print(ds[0].keys(), len(ds))
  ```

---

## 🔁 Re-using the saved dataset

Once you have `hf_dataset/`:

```python
from datasets import load_from_disk
ds = load_from_disk("./outputs/full_latents_qwen05_t1p0/hf_dataset")

# Example: get one latent sequence
hs = ds[0]["hidden_state"]  # Python nested list (L x H)
plan = ds[0]["plan"]
task = ds[0]["task"]
```

If you later add a compressor, you can load this dataset, transform `hidden_state` → `hidden_state_comp`, and save a **new** dataset:

```python
from datasets import Dataset
import numpy as np

def my_projector(hs_list):  # hs_list: list[list[float]]
    X = np.asarray(hs_list, dtype=np.float32)   # [L, H]
    # ... your projection to [K, D]
    return X[:8]  # toy: keep first 8

ds2 = ds.map(lambda ex: {"hidden_state_comp": my_projector(ex["hidden_state"]).tolist()})
ds2.save_to_disk("./outputs/with_comp/hf_dataset")
```

---

## 🛡️ Anonymization & Privacy

* Script A is designed to be **privacy-safe** out of the box (no hostname printing, no company paths).
* Script B prints some environment info. For a fully anonymized submission, either:

  1. run Script A only; or
  2. remove such prints from Script B before releasing artifacts.

Do **not** include internal absolute paths or machine hostnames in the repo.

---

## ❓ FAQ

**Q: What if my dataset has the task inside `conversations[i]['value']`?**
A: In Script A, pass `--task_field ""` and `--conversations_index i`. Example:

```bash
python collect_hidden_states.py \
  --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \
  --dataset_path ./alfworld_sft.json \
  --output_dir ./outputs/full_latents_from_conversations \
  --task_field "" \
  --conversations_index 2 \
  --chat_template qwen
```

**Q: Shapes don’t match what I expect.**
A: Script A saves **per-generated-token** last-layer hidden states `[L, H]`. Script B limits the number of steps via `--c`. If you add a learned compressor, write a new field (e.g., `hidden_state_comp`) with its own `[K, D]` and document it.

**Q: Can I run CPU-only?**
A: Technically yes, but it’s slow. GPU with enough memory is highly recommended.

---

## 📄 Citation

If you use this code or dataset in academic work, please cite the corresponding paper (details omitted here to preserve anonymity).

---

## 🧪 Minimal end-to-end demo

1. **Full latents**

```bash
python collect_hidden_states.py \
  --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \
  --dataset_path ./alfworld_sft.json \
  --output_dir ./outputs/full_demo \
  --chat_template qwen \
  --max_new_tokens 128 \
  --temperature 1.0
```

2. **Compressed latents (keep K=8 steps)**

```bash
python compressed_latent_collect.py \
  --model_path Qwen/Qwen2.5-0.5B-Instruct \
  --dataset_path ./alfworld_sft.json \
  --output_dir ./outputs/comp_demo_k8 \
  --max_new_tokens 128 \
  --temperature 1.2 \
  --c 8
```