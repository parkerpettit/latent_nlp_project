# Latent Communication Training (Anonymous Repo)

> **TL;DR**: This repo trains a two-agent setup where a **student LLM** generates a K-step **latent message** that’s inserted into a **frozen teacher LLM** right after the first human turn. Losses include CE on the teacher’s outputs, an **uncertainty-weighted KL** (teacher-with-data-latent → teacher-with-student-latent), and a **cosine alignment** between latent directions. Everything is Hugging Face-native, paths are user-provided, and no third-party logging is enabled.

---

## Contents

* `train_latent_comm.py` — single script to train **Student-over-Teacher** latent communication with Hugging Face `Trainer`.

---

## Why this repo fits double-blind review

* **No secrets**: No API keys, no trackers; `report_to=[]` disables online logging.
* **No local/vendor paths**: All models/data/outputs are passed via CLI args.
* **Reproducible**: Fixed seeds; deterministic data split.
* **Anonymized**: No author names or institutional references in code or README.

---

## Requirements

* Python **3.9+**
* PyTorch **2.1+** with CUDA (recommended)
* Hugging Face: `transformers >= 4.41`, `datasets >= 2.18`, `accelerate >= 0.28` (optional), `tqdm`, `numpy`
* (Optional) `deepspeed` for large-scale training

Install (CPU/GPU agnostic):

```bash
pip install "torch>=2.1" "transformers>=4.41" "datasets>=2.18" "accelerate>=0.28" tqdm numpy
# optional:
pip install deepspeed
```

> If you use Qwen/other architectures that require custom ops, keep `trust_remote_code=True` (already handled by the script).

---

## Data formats

### 1) Supervised JSON (`--data_path`)

Each sample:

```json
{
  "id": "train-000001",
  "conversations": [
    {"from": "human", "value": "Task description or instruction..."},
    {"from": "gpt",   "value": "Assistant reply..."},
    {"from": "human", "value": "Follow-up..."},
    {"from": "gpt",   "value": "Next reply..."}
  ]
  /* Optional fields below are ignored here; they are filled from the hidden-state repo by id */
  /* "plan": "...", "hidden_state": [[...], ...] */
}
```

* Roles accepted: `human|user|system` (treated as **input**, not supervised) and `gpt|assistant` (supervised).
* The script **inserts** the latent block **right after the first human message**.

### 2) Hidden-state dataset (`--hf_hidden_repo`)

A Hugging Face dataset **or** a local directory created by `datasets.load_from_disk`.
Fields per record:

* `task_id` **or** `id` — must match the JSON `"id"` above.
* `hidden_state` — shape `[L, H]` (list-of-lists or numpy array convertible to `float32`).
* `plan` — optional string (not required for training but allowed).

Example record (conceptual):

```python
{
  "task_id": "train-000001",
  "hidden_state": [[0.01, -0.02, ...], [ ... ], ...],  # L x H
  "plan": "Step-by-step outline produced by a reasoning agent."
}
```

> If your hidden-state repo is local: pass its path to `--hf_hidden_repo` and it will be loaded via `load_from_disk`.

---

## Quick start

### 1) Single GPU (most users)

```bash
python train_latent_comm.py \
  --teacher_model_name_or_path Qwen/Qwen2.5-7B-Instruct \
  --student_model_name_or_path Qwen/Qwen2.5-0.5B \
  --data_path /PATH/alfworld_sft.json \
  --hf_hidden_repo your-username/alfworld_hidden_states_train \
  --output_dir ./latent_out \
  --K 128 --bf16 --model_max_length 4096
```

### 2) Multi-GPU with `torchrun`

```bash
torchrun --nproc_per_node 8 train_latent_comm.py \
  --teacher_model_name_or_path Qwen/Qwen2.5-7B-Instruct \
  --student_model_name_or_path Qwen/Qwen2.5-0.5B \
  --data_path /PATH/alfworld_sft.json \
  --hf_hidden_repo /PATH/hidden_states_dataset  \
  --output_dir ./latent_out \
  --per_device_train_batch_size 1 --per_device_eval_batch_size 1 \
  --K 64 --bf16 --gradient_checkpointing
```

### 3) DeepSpeed (optional)

Create `ds_z2.json`:

```json
{
  "train_micro_batch_size_per_gpu": 1,
  "gradient_accumulation_steps": 4,
  "zero_optimization": { "stage": 2 },
  "bf16": { "enabled": true }
}
```

Run:

```bash
deepspeed train_latent_comm.py \
  --teacher_model_name_or_path Qwen/Qwen2.5-7B-Instruct \
  --student_model_name_or_path Qwen/Qwen2.5-0.5B \
  --data_path /PATH/alfworld_sft.json \
  --hf_hidden_repo your-username/alfworld_hidden_states_train \
  --output_dir ./latent_out_ds \
  --K 128 --bf16 \
  --deepspeed ds_z2.json
```

---

## What the script does (high level)

* **Tokenizer**: adds `<bop>` / `<eop>` for marking the latent block.
* **Student LLM**: given a short planning prefix + the first human turn, autoregressively emits **K hidden states** via a small `h2e` bridge.
* **Teacher Head**: `HiddenStateHead` maps raw student latents to teacher-space latents (MHA + GLU stack).
* **Teacher LLM (frozen)**: receives the **\[<bop>, latent (K × H), <eop>]** block inserted after the first human message and predicts next tokens.
* **Loss**:

  * CE on teacher outputs (with latent from student).
  * Uncertainty-weighted **KL** (teacher with **data latent** → teacher with **student latent**) on a supervised window after insertion.
  * **Cosine alignment** between mean-pooled student latent and data latent.

---

## Important CLI flags

* `--teacher_model_name_or_path` (required) HF id or local path.
* `--student_model_name_or_path` (required) HF id or local path.
* `--data_path` (required) path to the supervised JSON described above.
* `--hf_hidden_repo` (required) HF dataset id **or** local `load_from_disk` directory with `id/task_id`, `hidden_state`, `plan`.
* `--K` latent length (default 128).
* `--model_max_length` tokenizer max length (default 4096).
* Precision: `--bf16` or `--fp16`.
* Throughput: `--per_device_train_batch_size`, `--gradient_checkpointing`, `--deepspeed`.
* Early stop: `--early_stopping_patience`, `--early_stopping_threshold`.
* Optional: `--hidden_head_ckpt` to load a pre-trained head state dict.

Run `python train_latent_comm.py --help` to see all options.

---

## Reproducibility

* Seeds: PyTorch / NumPy / Python set to `--seed` (default 42).
* Fixed **random split**: `eval_ratio` (default 0.05) with a fixed RNG.
* Determinism can still vary across CUDA/cuDNN environments; for strict determinism, pin library versions and disable nondeterministic kernels.

---

## Expected resources

* The teacher can be large (e.g., 7B). For limited GPUs:

  * Reduce `--K`, `--model_max_length`, batch sizes.
  * Enable `--gradient_checkpointing`.
  * Use DeepSpeed ZeRO-2/3.
  * Prefer BF16 on Ampere+ GPUs.

---

## Troubleshooting

* **`Missing required arguments`**: Provide all four required flags (teacher, student, data\_path, hf\_hidden\_repo).
* **`Missing <bop>/<eop> in tokenizer!`**: The script adds them automatically; if you swapped tokenizer after init, re-add special tokens.
* **OOM**: Lower `--K`, sequence length, batch size; enable gradient checkpointing / ZeRO; use `--fp16`/`--bf16`.
* **`Sample is missing 'hidden_state'`**: Ensure the hidden-state repo has a record whose `id`/`task_id` matches the JSON `id`.
* **Slow / high memory in attention**: The script tries `flash_attention_2` when CUDA is available. If issues arise, it will fall back automatically.

---

## Minimal toy example (sanity check)

Create a tiny JSON at `/tmp/data.json`:

```json
[
  {
    "id": "toy-1",
    "conversations": [
      {"from": "human", "value": "Sort these numbers: 3, 1, 2."},
      {"from": "gpt",   "value": "They can be sorted as 1, 2, 3."}
    ]
  }
]
```

Create a tiny hidden-state HF dataset with one record `task_id="toy-1"` and shape `[L, H]` where `H` equals the teacher hidden size. Save it with `datasets.DatasetDict.save_to_disk("/tmp/hidden_states")`.

Run:

```bash
python train_latent_comm.py \
  --teacher_model_name_or_path Qwen/Qwen2.5-0.5B \
  --student_model_name_or_path Qwen/Qwen2.5-0.5B \
  --data_path /tmp/data.json \
  --hf_hidden_repo /tmp/hidden_states \
  --output_dir /tmp/out --K 8 --num_train_epochs 1 --per_device_train_batch_size 1
```