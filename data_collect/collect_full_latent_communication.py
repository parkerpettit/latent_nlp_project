#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
collect_hidden_states.py

Clean, privacy-safe, Hugging Face–style script that:
1) Loads a model/tokenizer via HF Transformers.
2) Reads a dataset via HF Datasets (JSON/JSONL or load_from_disk).
3) Generates a plan (text) for each task and collects the last-layer hidden state
   for each newly generated token (shape [L, H]).
4) Saves a merged Hugging Face Dataset (+ Parquet) with fields:
   { "task", "task_id", "plan", "hidden_state" }.

Multi-GPU is optional (env://). Each rank saves local results then rank 0 merges.
All configuration is via argparse. All comments/logs are in English.

Author: (anonymous)
"""

import os
import argparse
import json
import math
import pickle
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer

from datasets import load_dataset, load_from_disk, Features, Value, Sequence, Dataset as HFDataset
from tqdm import tqdm
import numpy as np


# --------------------------- Utilities ---------------------------

def str2bool(v: str) -> bool:
    return str(v).lower() in ("1", "true", "yes", "y", "on")


def set_seed(seed: int) -> None:
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def torch_dtype_from_str(name: str) -> torch.dtype:
    name = name.lower()
    if name in ("bf16", "bfloat16"):
        return torch.bfloat16
    if name in ("fp16", "float16"):
        return torch.float16
    if name in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unsupported torch_dtype: {name}")


# --------------------------- Distributed ---------------------------

def setup_distributed(args: argparse.Namespace) -> Tuple[int, int, int]:
    """
    Initialize torch.distributed if possible (env://). Safe to run on single GPU/CPU.
    Returns (rank, local_rank, world_size).
    """
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    use_dist = world_size > 1 and torch.cuda.is_available()
    if use_dist:
        backend = "nccl"
        torch.cuda.set_device(local_rank)
        if not dist.is_initialized():
            dist.init_process_group(backend=backend, init_method="env://")
        # refresh real values
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        # single process fallback
        rank, local_rank, world_size = 0, 0, 1

    return rank, local_rank, world_size


def dist_barrier_safe():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def dist_destroy_safe():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


# --------------------------- Model I/O ---------------------------

def load_model_and_tokenizer(args: argparse.Namespace, device: torch.device):
    """
    Load model/tokenizer from Hugging Face. No private/download hub used.
    """
    dtype = torch_dtype_from_str(args.torch_dtype)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=dtype,
        trust_remote_code=str2bool(args.trust_remote_code),
        low_cpu_mem_usage=True,
    ).to(device)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=str2bool(args.trust_remote_code),
        use_fast=True
    )

    if tokenizer.pad_token_id is None:
        # fall back to eos as pad for generation
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer


# --------------------------- Dataset I/O ---------------------------

def load_hf_dataset(args: argparse.Namespace):
    """
    Load dataset from JSON/JSONL via HF Datasets, or from `load_from_disk`.
    """
    if str2bool(args.load_from_disk):
        ds = load_from_disk(args.dataset_path)
        split = args.dataset_split if args.dataset_split in ds else None
        dataset = ds[split] if split else ds
    else:
        # When dataset_path is a local json/jsonl, use the 'json' loader
        # Else you can also pass a Hub dataset name to --dataset_path and set --dataset_name to use custom loader.
        if args.dataset_name:
            dataset = load_dataset(args.dataset_name, data_files=args.dataset_path, split=args.dataset_split)
        else:
            ext = os.path.splitext(args.dataset_path)[-1].lower()
            loader = "json" if ext in (".json", ".jsonl") else "json"
            dataset = load_dataset(loader, data_files=args.dataset_path, split=args.dataset_split)

    return dataset


def default_task_extractor(
    ex: Dict[str, Any], task_field: str, conversations_index: Optional[int]
) -> str:
    """
    Extract the task text from an example.
    - If `task_field` exists, use it.
    - Else, if `conversations` exists and `conversations_index` is provided, try to read `conversations[i].value`.
    """
    if task_field and task_field in ex:
        return ex[task_field]

    if "conversations" in ex and conversations_index is not None:
        conv = ex["conversations"]
        if isinstance(conv, list) and 0 <= conversations_index < len(conv):
            node = conv[conversations_index]
            # commonly {'from': 'user', 'value': '...'} or similar
            if isinstance(node, dict):
                # try several common keys
                for k in ("value", "text", "content"):
                    if k in node and isinstance(node[k], str):
                        return node[k]

    raise KeyError(
        f"Could not extract task text: missing field '{task_field}' and conversations[{conversations_index}]"
    )


def format_prompt(task_text: str, template: str, chat_template: str) -> str:
    """
    Format the input prompt. If chat_template='qwen', wrap with Qwen-style special tokens.
    Otherwise, use the provided `template` or plain task.
    """
    base = template.format(task=task_text) if template else task_text

    if chat_template == "qwen":
        # Minimal Qwen chat wrapper
        return "<|im_start|>user\n" + base + "\n<|im_end|>\n<|im_start|>assistant\n"
    else:
        return base


# --------------------------- Generation & Hidden States ---------------------------

@torch.no_grad()
def generate_with_hidden(
    model,
    tokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    do_sample: bool,
    eos_token_id: Optional[int],
    pad_token_id: Optional[int],
) -> Tuple[str, torch.Tensor]:
    """
    Run generation and collect the last-layer hidden state for each newly generated token.

    Returns:
        generated_text: str
        hidden_seq: torch.Tensor with shape [L, H] (L = #generated tokens, H = hidden size)
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask", None)
    input_len = input_ids.shape[1]

    gen_kwargs = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p if top_p > 0 else None,
        top_k=top_k if top_k > 0 else None,
        num_beams=1,  # pure sampling by default
        return_dict_in_generate=True,
        output_hidden_states=True,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
    )

    outputs = model.generate(**gen_kwargs)

    # Decode only the newly generated tokens
    gen_ids = outputs.sequences[0][input_len:]
    generated_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

    # Collect last-layer hidden of each new step: outputs.hidden_states is a list over steps
    step_hiddens = []
    for step_tuple in outputs.hidden_states:        # list length == #generation steps
        last_layer = step_tuple[-1]                 # [B, S_step, H]
        h_last = last_layer[:, -1, :]               # [B, H] -> just the new token position
        step_hiddens.append(h_last)

    hidden_seq = torch.stack(step_hiddens, dim=1)   # [B, L, H]
    if hidden_seq.size(0) == 1:
        hidden_seq = hidden_seq.squeeze(0)          # [L, H]

    return generated_text, hidden_seq  # [L, H]


# --------------------------- Saving & Merging ---------------------------

def save_rank_pickle(rank: int, data: List[Dict[str, Any]], temp_dir: str) -> str:
    os.makedirs(temp_dir, exist_ok=True)
    path = os.path.join(temp_dir, f"rank_{rank}.pkl")
    with open(path, "wb") as f:
        pickle.dump(data, f)
    return path


def convert_and_save_hf_dataset(data: List[Dict[str, Any]], output_dir: str) -> None:
    """
    Convert in-memory list to HF Dataset and save both HF Dataset and Parquet files.
    """
    os.makedirs(output_dir, exist_ok=True)

    tasks = [d["task"] for d in data]
    task_ids = [d["task_id"] for d in data]
    plans = [d["plan"] for d in data]
    hidden_states = [np.asarray(d["hidden_state"], dtype=np.float32) for d in data]  # list of [L, H]

    features = Features({
        "task": Value("string"),
        "task_id": Value("string"),
        "plan": Value("string"),
        "hidden_state": Sequence(Sequence(Value("float32"))),  # variable [L, H]
    })

    ds = HFDataset.from_dict({
        "task": tasks,
        "task_id": task_ids,
        "plan": plans,
        "hidden_state": [hs.tolist() for hs in hidden_states],
    }, features=features)

    # HF Dataset format
    ds_path = os.path.join(output_dir, "hf_dataset")
    ds.save_to_disk(ds_path)
    print(f"[Info] Saved HuggingFace Dataset to: {ds_path}")

    # Parquet
    pq_path = os.path.join(output_dir, "data.parquet")
    ds.to_parquet(pq_path)
    print(f"[Info] Saved Parquet to: {pq_path}")


def merge_and_save_all_ranks(rank: int, world_size: int, output_dir: str, temp_dir: str, local_buffer: List[Dict[str, Any]]) -> None:
    """
    Each rank writes its local buffer to temp_dir. Rank 0 merges and writes HF Dataset + Parquet.
    """
    path = save_rank_pickle(rank, local_buffer, temp_dir)
    print(f"[Rank {rank}] wrote {len(local_buffer)} samples -> {path}")

    dist_barrier_safe()

    if rank == 0:
        print("[Rank 0] Merging data from all ranks...")
        full: List[Dict[str, Any]] = []
        for i in range(world_size):
            p = os.path.join(temp_dir, f"rank_{i}.pkl")
            if not os.path.exists(p):
                print(f"[Warning] Missing file {p}, skipping.")
                continue
            with open(p, "rb") as f:
                part = pickle.load(f)
                full.extend(part)

        convert_and_save_hf_dataset(full, output_dir)

        # cleanup
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
        print(f"[Rank 0] Removed temp dir: {temp_dir}")


# --------------------------- Main Loop ---------------------------

def run(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    rank, local_rank, world_size = setup_distributed(args)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    if rank == 0:
        print(f"[Info] world_size={world_size}, device={device}")
        print(f"[Info] model={args.model_name_or_path}")

    model, tokenizer = load_model_and_tokenizer(args, device)

    dataset = load_hf_dataset(args)
    total = len(dataset)
    if rank == 0:
        print(f"[Info] Loaded dataset split='{args.dataset_split}' with {total} samples from {args.dataset_path}")

    # split indices across ranks
    if world_size > 1:
        per_rank = int(math.ceil(total / world_size))
        start = rank * per_rank
        end = min(start + per_rank, total)
    else:
        start, end = 0, total

    if rank == 0:
        print(f"[Info] Output dir: {args.output_dir}")

    local_results: List[Dict[str, Any]] = []

    iterator = range(start, end)
    pbar = tqdm(iterator, disable=(args.disable_tqdm or (world_size > 1 and rank != 0)), desc=f"Rank {rank}")

    for i in pbar:
        ex = dataset[i]

        # extract id
        if args.id_field not in ex:
            # fall back to string index if no id field
            task_id = str(i)
        else:
            val = ex[args.id_field]
            task_id = str(val)

        # extract task text
        task_text = default_task_extractor(
            ex, task_field=args.task_field, conversations_index=args.conversations_index
        )

        # format prompt
        prompt = format_prompt(task_text, template=args.prompt_template, chat_template=args.chat_template)

        # generate + collect hidden states
        plan, hidden_seq = generate_with_hidden(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            do_sample=str2bool(args.do_sample),
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

        # convert to numpy float32 for serialization
        hidden_np = hidden_seq.detach().cpu().numpy().astype(np.float32)  # [L, H]

        local_results.append({
            "task": task_text,
            "task_id": task_id,
            "plan": plan,
            "hidden_state": hidden_np,   # will be list-of-list in HF dataset
        })

        # optional: flush periodically
        if len(local_results) % args.flush_every == 0:
            print(f"[Rank {rank}] processed {len(local_results)} local samples so far...")

    # synchronize before IO
    dist_barrier_safe()

    # merge and save
    merge_and_save_all_ranks(
        rank=rank,
        world_size=world_size,
        output_dir=args.output_dir,
        temp_dir=args.temp_dir,
        local_buffer=local_results,
    )

    dist_destroy_safe()
    if rank == 0:
        print("[Done] All ranks finished.")


# --------------------------- CLI ---------------------------

def build_arg_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect per-token last-layer hidden states during generation.")

    # Model & Tokenizer
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2.5-0.5B-Instruct",
                        help="HF model name or local path.")
    parser.add_argument("--torch_dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"],
                        help="Computation dtype for model weights.")
    parser.add_argument("--trust_remote_code", type=str, default="false",
                        help="Set to true if the model repo requires custom code.")
    parser.add_argument("--chat_template", type=str, default="none", choices=["none", "qwen"],
                        help="Optional chat wrapping. 'qwen' uses <|im_start|>/<|im_end|> style.")

    # Dataset
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path to JSON/JSONL file or to a load_from_disk directory.")
    parser.add_argument("--dataset_split", type=str, default="train",
                        help="Split to load (train/validation/test). For JSON, this is ignored; for load_from_disk, must exist.")
    parser.add_argument("--dataset_name", type=str, default="",
                        help="Optional HF dataset script name if not using 'json' loader (rare).")
    parser.add_argument("--load_from_disk", type=str, default="false",
                        help="If true, load the dataset using datasets.load_from_disk().")

    # Fields & Prompt
    parser.add_argument("--task_field", type=str, default="task",
                        help="Field name containing the task text. If missing, will try conversations[i].value.")
    parser.add_argument("--conversations_index", type=int, default=None,
                        help="Index to read from example['conversations'][i]['value'] when task_field is absent.")
    parser.add_argument("--id_field", type=str, default="id",
                        help="Field name for unique id. If missing, index will be used.")
    parser.add_argument("--prompt_template", type=str, default="Please provide a general plan to solve this task.\nThe task is: {task}",
                        help="Python format string with '{task}' placeholder.")

    # Generation
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.9, help="Set <=0 to disable nucleus.")
    parser.add_argument("--top_k", type=int, default=0, help="Set <=0 to disable top-k.")
    parser.add_argument("--do_sample", type=str, default="true")

    # Output
    parser.add_argument("--output_dir", type=str, required=True, help="Where to save merged HF Dataset + Parquet.")
    parser.add_argument("--temp_dir", type=str, default="./_tmp_rank_data", help="Temporary dir for per-rank pickles.")
    parser.add_argument("--flush_every", type=int, default=100, help="Local logging frequency.")
    parser.add_argument("--disable_tqdm", type=str, default="false")

    # Misc
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


if __name__ == "__main__":
    args = build_arg_parser()
    run(args)
