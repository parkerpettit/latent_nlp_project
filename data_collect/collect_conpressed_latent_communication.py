# infer_student_latents.py
# Generate K×Hs latents using only the student model during inference,
# warming up on the first conversation turn (task field), without any special markers.
# Also logs per-task latency to CSV.

import os
import json
import glob
import argparse
import numpy as np
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import time
import csv
import statistics

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

import pyarrow as pa
import pyarrow.parquet as pq


# Optional: safetensors support
try:
    from safetensors.torch import load_file as st_load_file, safe_open
except ImportError:
    st_load_file = None
    safe_open = None


def save_latents_parquet_nested(
    latents: torch.Tensor,
    samples: List[dict],
    output_path: str,
    texts: Optional[List[str]] = None
):
    """
    Save latents in nested Parquet format.

    Args:
        latents: [B, K, Hs] tensor
        samples: List of original samples (for 'id' field)
        output_path: Path to save .parquet file
        texts: Optional list of input texts, aligned with batch
    """
    latents_np = latents.to(torch.float32).cpu().numpy()  # [B, K, Hs]
    B, K, Hs = latents_np.shape

    # Use sample ID or fallback to index
    ids = [str(sample.get("id", i)) for i, sample in enumerate(samples)]
    latents_array = pa.array(
        [latents_np[i].tolist() for i in range(B)],
        type=pa.list_(pa.list_(pa.float32()))
    )

    columns = {
        "id": pa.array(ids, type=pa.string()),
        "latents": latents_array,
    }
    if texts is not None:
        assert len(texts) == B, f"Texts length ({len(texts)}) must match batch size ({B})"
        columns["text"] = pa.array([str(t) for t in texts], type=pa.string())

    table = pa.table(columns)

    # Add metadata for K and Hs
    metadata = {b"K": str(K).encode(), b"Hs": str(Hs).encode()}
    table = table.replace_schema_metadata(metadata)

    pq.write_table(table, output_path, compression="zstd")
    print(f"Parquet saved (nested format) → {output_path}")


def load_state_dict_from_checkpoint(checkpoint_dir: str) -> Dict[str, torch.Tensor]:
    """
    Load state dict from a checkpoint directory, supporting:
      - model.safetensors
      - model-*.safetensors (sharded)
      - pytorch_model.bin / model.bin / *.pt

    Returns:
        state_dict: Dict[str, torch.Tensor]
    """
    # 1. Single safetensors file
    single_path = os.path.join(checkpoint_dir, "model.safetensors")
    if os.path.isfile(single_path):
        if st_load_file is None:
            raise ImportError("safetensors not installed. Run: pip install safetensors")
        return st_load_file(single_path, device="cpu")

    # 2. Sharded safetensors
    shard_paths = sorted(glob.glob(os.path.join(checkpoint_dir, "model-*.safetensors")))
    if shard_paths:
        if safe_open is None:
            raise ImportError("safetensors not installed. Run: pip install safetensors")
        state_dict = {}
        for shard_path in shard_paths:
            with safe_open(shard_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    state_dict[key] = f.get_tensor(key)
        return state_dict

    # 3. Legacy .bin or .pt files
    for filename in ["pytorch_model.bin", "model.bin", "model.pt", "pytorch_model.pt"]:
        path = os.path.join(checkpoint_dir, filename)
        if os.path.isfile(path):
            return torch.load(path, map_location="cpu", weights_only=False)

    raise FileNotFoundError(
        f"No valid checkpoint found in {checkpoint_dir}. "
        "Expected: model.safetensors, model-*.safetensors, pytorch_model.bin, etc."
    )


def load_json_data(json_path: str) -> List[dict]:
    """
    Load conversation data from JSON.
    Expected format:
        [{"id": "...", "task": "..."}, ...]
    Supports both list and single dict (wrapped into list).

    Args:
        json_path: Path to JSON file

    Returns:
        List of samples
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Input JSON not found: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else [data]


def tokenize_first_turn(
    samples: List[dict],
    tokenizer,
    max_length: int = 2048
) -> Dict[str, torch.Tensor]:
    """
    Tokenize the 'task' field from each sample (first turn only).
    Applies padding and truncation.

    Args:
        samples: List of dicts with 'task' field
        tokenizer: Transformers tokenizer
        max_length: Max token length

    Returns:
        Dict with 'input_ids', 'attention_mask', and 'texts'
    """
    texts = [str(sample["task"]) for sample in samples]
    batch = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
        add_special_tokens=True
    )
    return {
        "input_ids": batch["input_ids"],
        "attention_mask": batch["attention_mask"],
        "texts": texts
    }


class StudentLatentGenerator(nn.Module):
    """
    Latent generator using only the student language model.
    Generates K latents per sample autoregressively.
    Warm-up uses optional prepend text + first turn (task).
    """

    def __init__(self, tokenizer, student_model: nn.Module, K: int = 128):
        super().__init__()
        self.tokenizer = tokenizer
        self.student_model = student_model.eval()
        self.K = K

        hidden_size = student_model.config.hidden_size
        self.h2e = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size, bias=False)
        )
        self.latent_bos = nn.Parameter(torch.empty(hidden_size))
        nn.init.normal_(self.latent_bos, mean=0.0, std=0.02)

    @torch.inference_mode()
    def generate_latents(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        K: Optional[int] = None,
        prepend_text: str = "Please generate a plan to solve this task: ",
        return_timing: bool = True
    ) -> Tuple[torch.Tensor, List[Dict]]:
        """
        Generate latents for a batch.

        Args:
            input_ids: [B, T]
            attention_mask: [B, T]
            K: Number of latents to generate (default: self.K)
            prepend_text: Optional prefix prompt. Use "" to disable.
            return_timing: Whether to return timing stats

        Returns:
            latents: [B, K, Hs]
            timing: List of per-sample timing stats (if return_timing=True)
        """
        self.eval()
        self.student_model.eval()
        device = input_ids.device
        B, T = input_ids.shape
        K = K or self.K

        latents_list = []
        timing_stats = []
        emb_module = self.student_model.get_input_embeddings()
        Hs = emb_module.embedding_dim

        # Preprocess prepend text
        if prepend_text.strip():
            prep_tokens = self.tokenizer(
                prepend_text,
                add_special_tokens=True,
                return_tensors="pt"
            )
            prep_ids = prep_tokens.input_ids.to(device)
            prep_attn = torch.ones_like(prep_ids, dtype=attention_mask.dtype, device=device)
            prepend_len = prep_ids.shape[1]
        else:
            prep_ids = prep_attn = None
            prepend_len = 0

        for batch_idx in tqdm(range(B), desc="Generating latents"):
            # Build context: [prepend] + [first turn]
            if prep_ids is not None:
                ctx_ids = torch.cat([prep_ids, input_ids[batch_idx:batch_idx+1]], dim=1)
                ctx_attn = torch.cat([prep_attn, attention_mask[batch_idx:batch_idx+1]], dim=1)
            else:
                ctx_ids = input_ids[batch_idx:batch_idx+1]
                ctx_attn = attention_mask[batch_idx:batch_idx+1]

            t_start = time.perf_counter()
            t_warmup_start = time.perf_counter()

            # Warm-up forward pass
            warmup_out = self.student_model(
                input_ids=ctx_ids,
                attention_mask=ctx_attn,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True
            )
            past_key_values = warmup_out.past_key_values
            past_seq_len = ctx_ids.size(1)
            t_warmup_end = time.perf_counter()

            # Autoregressive generation of K latents
            step_inputs = self.latent_bos.to(device, dtype=emb_module.weight.dtype).view(1, 1, -1)
            step_latents = []
            step_times = []

            for _ in range(K):
                t_step_start = time.perf_counter()
                pos_ids = torch.tensor([[past_seq_len]], device=device, dtype=torch.long)

                step_out = self.student_model(
                    inputs_embeds=step_inputs,
                    attention_mask=torch.ones_like(pos_ids),
                    past_key_values=past_key_values,
                    position_ids=pos_ids,
                    use_cache=True,
                    output_hidden_states=True,
                    return_dict=True
                )
                z_k = step_out.hidden_states[-1][:, -1, :]  # [1, Hs]
                step_latents.append(z_k.squeeze(0))  # [Hs]

                past_key_values = step_out.past_key_values
                past_seq_len += 1
                step_inputs = self.h2e(z_k).view(1, 1, -1)
                t_step_end = time.perf_counter()
                step_times.append(t_step_end - t_step_start)

            z_batch = torch.stack(step_latents, dim=0)  # [K, Hs]
            latents_list.append(z_batch)
            t_end = time.perf_counter()

            if return_timing:
                warmup_ms = (t_warmup_end - t_warmup_start) * 1000
                total_ms = (t_end - t_start) * 1000
                step_ms = [t * 1000 for t in step_times]
                mean_ms = float(statistics.fmean(step_ms)) if step_ms else 0.0
                p50_ms = float(np.percentile(step_ms, 50)) if step_ms else 0.0
                p95_ms = float(np.percentile(step_ms, 95)) if step_ms else 0.0
                p99_ms = float(np.percentile(step_ms, 99)) if step_ms else 0.0

                first_valid_len = int(attention_mask[batch_idx].sum().item())
                ctx_token_len = prepend_len + first_valid_len

                timing_stats.append({
                    "index": batch_idx,
                    "id": None,  # filled later
                    "K": K,
                    "Hs": Hs,
                    "warmup_ms": round(warmup_ms, 3),
                    "step_mean_ms": round(mean_ms, 3),
                    "step_p50_ms": round(p50_ms, 3),
                    "step_p95_ms": round(p95_ms, 3),
                    "step_p99_ms": round(p99_ms, 3),
                    "total_ms": round(total_ms, 3),
                    "ctx_token_len": ctx_token_len,
                })

        latents_out = torch.stack(latents_list, dim=0)  # [B, K, Hs]
        return latents_out, timing_stats


def main():
    parser = argparse.ArgumentParser(
        description="Generate latents using student model only, warm-up on first task."
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Path to trained checkpoint (contains model.safetensors or pytorch_model.bin)"
    )
    parser.add_argument(
        "--student_model_path",
        type=str,
        required=True,
        help="Path or Hugging Face repo ID for the base student model (e.g., 'Qwen/Qwen2.5-7B')"
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=None,
        help="Optional tokenizer path; defaults to student_model_path"
    )
    parser.add_argument(
        "--input_json",
        type=str,
        required=True,
        help="Path to input JSON with 'task' field per sample"
    )
    parser.add_argument(
        "--output_npy",
        type=str,
        default="latents.npy",
        help="Output .npy file for latents"
    )
    parser.add_argument(
        "--parquet_out",
        type=str,
        default=None,
        help="Output Parquet path. If not provided, uses checkpoint_dir/short_hidden_state_seen.parquet"
    )
    parser.add_argument(
        "--time_csv",
        type=str,
        default="latency_by_task.csv",
        help="CSV file to log per-task latency"
    )
    parser.add_argument(
        "--K",
        type=int,
        default=128,
        help="Number of latents to generate per sample"
    )
    parser.add_argument(
        "--model_max_length",
        type=int,
        default=2048,
        help="Max token length for tokenizer"
    )
    parser.add_argument(
        "--prepend_text",
        type=str,
        default="Please generate a plan to solve this task: ",
        help="Optional prefix prompt. Use empty string to disable."
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu"],
        help="Device to use. Auto-detect if not specified."
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default=None,
        choices=["bfloat16", "float16", "float32"],
        help="Torch dtype. Auto-detect if not specified."
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Allow remote code in Hugging Face models"
    )

    args = parser.parse_args()

    # Device and dtype
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype_str = args.dtype or ("bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float32")
    torch_dtype = getattr(torch, dtype_str)

    # Tokenizer
    tokenizer_path = args.tokenizer_path or args.student_model_path
    print(f"Loading tokenizer from: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        use_fast=False,
        trust_remote_code=args.trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Add special tokens if needed
    new_tokens = ['<FIRST_HUMAN_END>', '<bop>', '<eop>']
    tokenizer.add_special_tokens({'additional_special_tokens': new_tokens})
    tokenizer.model_max_length = args.model_max_length

    # Student model
    print(f"Loading student model: {args.student_model_path}")
    config = AutoConfig.from_pretrained(
        args.student_model_path,
        trust_remote_code=args.trust_remote_code
    )
    config.use_cache = False

    attn_impl = "flash_attention_2" if device == "cuda" else None
    try:
        student_model = AutoModelForCausalLM.from_pretrained(
            args.student_model_path,
            config=config,
            trust_remote_code=args.trust_remote_code,
            torch_dtype=torch_dtype,
            attn_implementation=attn_impl,
            device_map="auto" if device == "cuda" else None
        )
    except Exception as e:
        print(f"Failed to load model with device_map. Falling back to manual device placement: {e}")
        student_model = AutoModelForCausalLM.from_pretrained(
            args.student_model_path,
            config=config,
            trust_remote_code=args.trust_remote_code,
            torch_dtype=torch_dtype,
            attn_implementation=attn_impl
        ).to(device)

    student_model.resize_token_embeddings(len(tokenizer))
    student_model.eval()

    # Latent generator
    model = StudentLatentGenerator(tokenizer, student_model, K=args.K).to(device).eval()

    # Load trained weights (only student-related)
    print(f"Loading trained weights from: {args.checkpoint_dir}")
    state_dict = load_state_dict_from_checkpoint(args.checkpoint_dir)

    def strip_module_prefix(key: str) -> str:
        return key[len("module."):] if key.startswith("module.") else key

    filtered_state = {
        strip_module_prefix(k): v
        for k, v in state_dict.items()
        if strip_module_prefix(k).startswith(("student_model.", "h2e.", "latent_bos"))
    }

    missing, unexpected = model.load_state_dict(filtered_state, strict=False)
    print(f"Weights loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}")

    # Prepare data
    samples = load_json_data(args.input_json)
    print(f"Loaded {len(samples)} samples")

    tokenized = tokenize_first_turn(samples, tokenizer, max_length=args.model_max_length)
    input_ids = tokenized["input_ids"].to(device)
    attention_mask = tokenized["attention_mask"].to(device)
    texts = tokenized["texts"]

    # Match dtypes with model
    target_dtype = next(student_model.parameters()).dtype
    model.h2e.to(device=device, dtype=target_dtype)
    model.latent_bos.data = model.latent_bos.data.to(device=device, dtype=target_dtype)

    # Generate latents
    print("Starting latent generation...")
    latents, timing = model.generate_latents(
        input_ids=input_ids,
        attention_mask=attention_mask,
        K=args.K,
        prepend_text=args.prepend_text,
        return_timing=True
    )
    print(f"Latents shape: {tuple(latents.shape)}")

    # Save .npy
    np.save(args.output_npy, latents.float().cpu().numpy())
    print(f"Latents saved to: {args.output_npy}")

    # Parquet output path
    if args.parquet_out is None:
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        args.parquet_out = os.path.join(args.checkpoint_dir, "short_hidden_state_seen.parquet")
    save_latents_parquet_nested(latents, samples, args.parquet_out, texts)

    # Save timing CSV
    os.makedirs(os.path.dirname(args.time_csv) or ".", exist_ok=True)
    for row in timing:
        idx = row["index"]
        row["id"] = str(samples[idx].get("id", idx))
        row["task_char_len"] = len(texts[idx]) if idx < len(texts) else None

    fieldnames = [
        "index", "id", "K", "Hs",
        "ctx_token_len", "task_char_len",
        "warmup_ms", "step_mean_ms", "step_p50_ms", "step_p95_ms", "step_p99_ms",
        "total_ms"
    ]
    with open(args.time_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in timing:
            writer.writerow({k: row.get(k) for k in fieldnames})
    print(f"Latency CSV saved to: {args.time_csv}")


if __name__ == "__main__":
    main()
