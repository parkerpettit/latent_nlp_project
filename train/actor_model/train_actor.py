#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Latent Communication (clean, HF‑style)
=====================================

This is a self‑contained, anonymized training script that:
- Removes private env vars, local paths, and company‑specific code
- Uses standard Hugging Face (transformers + datasets) patterns
- Converts all comments/prints to English
- Moves knobs into dataclass args instead of ad‑hoc overrides
- Keeps your latent‑communication model wrapper and curriculum logic
- Adds simple, optional probes/callbacks (off by default)

Expected data format (JSON/JSONL list of examples):
[
  {
    "id": "sample_0001",
    "conversations": [
      {"from": "human", "value": "<user problem description>"},
      {"from": "gpt",   "value": "<assistant reply>"},
      ... alternating ...
    ],
    "plan": "optional textual plan (string)",
    "hidden_state": [[...], ...]                # 2D list (K,d), OR
    "hidden_state_path": "optional path to .npy or .pt with shape [K,d]"
  },
  ...
]

Usage (single GPU example):
---------------------------
python latent_comm_hf_clean.py \
  --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \
  --train_file ./train.json \
  --eval_file ./eval.json \
  --output_dir ./checkpoints/exp1 \
  --per_device_train_batch_size 2 \
  --save_steps 500 --evaluation_strategy steps --eval_steps 500

Notes
-----
* We add three special tokens: <FIRST_HUMAN_END>, <bop>, <eop>.
* Hidden states are inserted at the end of the first human message (detected via the marker token).
* If an example has no hidden_state/hidden_state_path, it will be skipped for hidden‑insertion and only plan tokens are available for the plan‑only branch.
* Probes/callbacks are disabled by default; enable with flags.
"""

from __future__ import annotations

import os
import json
import math
import time
import random
import logging
import pathlib
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple, Union, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

import datasets
from datasets import load_dataset

import transformers
from transformers import Trainer, TrainerCallback
from transformers.trainer_pt_utils import LabelSmoother
from transformers.trainer_utils import IntervalStrategy

import matplotlib.pyplot as plt

# -----------------------------
# Global constants & utilities
# -----------------------------
IGNORE_TOKEN_ID = LabelSmoother.ignore_index
IGNORE = -100
EPS = 1e-8
LOGGER = logging.getLogger("latent_comm")


def setup_logging(level: str = "INFO") -> None:
    fmt = "%(asctime)s | %(levelname)s | %(message)s"
    logging.basicConfig(format=fmt, level=getattr(logging, level.upper(), logging.INFO))


def set_all_seeds(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# -----------------------------
# Argument dataclasses
# -----------------------------
@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="Qwen/Qwen2.5-0.5B-Instruct",
        metadata={"help": "HF hub path or local path to a CausalLM."},
    )
    trust_remote_code: bool = field(default=False)
    padding_side: str = field(default="right")
    # Latent‑insertion args
    prepended_length: int = field(default=256, metadata={"help": "Max hidden‑state timesteps to insert."})
    prepended_learnable: bool = field(default=False, metadata={"help": "Keep for compatibility; not used by default."})
    prepend_position: str = field(default="first_human", metadata={"help": "Where to inject: 'first_human' only."})
    plan_similarity_weight: float = field(default=1.0)
    random_contrast_weight: float = field(default=2.0)
    num_heads: int = field(default=8, metadata={"help": "MHA heads for hidden‑state processing."})


@dataclass
class DataArguments:
    train_file: str = field(default=None, metadata={"help": "Path to train JSON/JSONL."})
    eval_file: Optional[str] = field(default=None, metadata={"help": "Path to eval JSON/JSONL. If omitted, a split will be made from train."})
    eval_ratio: float = field(default=0.02, metadata={"help": "Hold‑out ratio when eval_file is None."})
    lazy_preprocess: bool = field(default=False)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(default=3072)
    dataloader_num_workers: int = field(default=4)
    dataloader_pin_memory: bool = field(default=True)
    dataloader_prefetch_factor: int = field(default=2)
    save_total_limit: int = field(default=5)
    evaluation_strategy: IntervalStrategy = field(default=IntervalStrategy.STEPS)
    save_strategy: IntervalStrategy = field(default=IntervalStrategy.STEPS)
    eval_steps: int = field(default=500)
    save_steps: int = field(default=500)
    logging_steps: int = field(default=50)
    greater_is_better: bool = field(default=False)
    metric_for_best_model: str = field(default="eval_loss")
    # Optional probes/plots
    enable_loss_plot: bool = field(default=True)
    enable_infoNCE_probe: bool = field(default=False)
    infoNCE_probe_interval: int = field(default=200)
    infoNCE_probe_K: int = field(default=8)
    infoNCE_probe_tau: float = field(default=1.0)


# -----------------------------
# Conversation formatting (ChatML‑like)
# -----------------------------
CHATML_SYS = "You are a helpful assistant."
IM_START = "<|im_start|>"
IM_END = "<|im_end|>\n"
ROLE_MAP = {"human": "user", "gpt": "assistant"}


def format_conversations_chatml(sources: List[List[Dict[str, str]]]) -> Tuple[List[str], List[bool]]:
    """Build ChatML prompts and track whether a first human segment exists.
    Returns (list of full prompts, list of flags for first-human existence).
    """
    prompts: List[str] = []
    has_first_human: List[bool] = []
    for conv in sources:
        # Insert a FIRST_HUMAN_END marker at the end of the first human message.
        first_human_done = False
        parts: List[str] = [f"{IM_START}system\n{CHATML_SYS}{IM_END}"]
        for msg in conv:
            role = ROLE_MAP.get(msg.get("from", "human"), "user")
            text = msg.get("value", "")
            if role == "user" and not first_human_done:
                text = text + "<FIRST_HUMAN_END>"
                first_human_done = True
            parts.append(f"{IM_START}{role}\n{text}{IM_END}")
        prompts.append("".join(parts))
        has_first_human.append(first_human_done)
    return prompts, has_first_human


def tokenize_with_marker(prompts: List[str], has_first_human: List[bool], tokenizer: transformers.PreTrainedTokenizer) -> Dict[str, torch.Tensor]:
    tok = tokenizer(
        prompts,
        return_tensors="pt",
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
    )
    input_ids = tok.input_ids
    labels = input_ids.clone()

    # Mask everything except assistant responses (standard supervised fine‑tuning setting)
    # This is a minimal rule; adapt if you use different labeling.
    # Here we: mask user/system segments, keep assistant tokens.
    # A simple heuristic: anything before the first assistant token is masked.
    # More elaborate masking can be added if needed.
    mask_all = torch.ones_like(labels) * IGNORE_TOKEN_ID
    keep = torch.zeros_like(labels, dtype=torch.bool)
    # Detect assistant segments by tokenizing the marker "assistant\n" and scanning.
    # To keep it robust, we just retain everything after the first occurrence of
    # the assistant header token sequence.
    assistant_hdr_ids = tokenizer("assistant\n", add_special_tokens=False).input_ids
    for b in range(labels.size(0)):
        ids = input_ids[b]
        start_idx = 0
        found = False
        for i in range(0, len(ids) - len(assistant_hdr_ids) + 1):
            if torch.all(ids[i:i + len(assistant_hdr_ids)] == torch.tensor(assistant_hdr_ids)):
                start_idx = i + len(assistant_hdr_ids)
                found = True
                break
        if found:
            keep[b, start_idx:] = True
    labels = torch.where(keep, labels, mask_all)

    # Find <FIRST_HUMAN_END> and record its position (start index of the marker)
    marker_ids = tokenizer("<FIRST_HUMAN_END>", add_special_tokens=False).input_ids
    human_end_positions: List[int] = []
    pad_id = tokenizer.pad_token_id
    for b in range(input_ids.size(0)):
        pos = -1
        if has_first_human[b]:
            seq = input_ids[b]
            for i in range(0, len(seq) - len(marker_ids) + 1):
                if torch.all(seq[i:i + len(marker_ids)] == torch.tensor(marker_ids)):
                    # Record marker start index, then wipe it to pad
                    pos = int(i)
                    seq[i:i + len(marker_ids)] = pad_id
                    break
        human_end_positions.append(pos)

    attention_mask = input_ids.ne(tokenizer.pad_token_id)
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "human_end_positions": torch.tensor(human_end_positions, dtype=torch.long),
    }


# -----------------------------
# Model components
# -----------------------------
class AdaptiveProjection(nn.Module):
    """Simple projection block with residual and calibration."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.2))
        self.output_scale = nn.Parameter(torch.tensor(0.1))
        self.proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size, eps=1e-6),
            nn.Linear(hidden_size, hidden_size),
        )
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.proj[0].weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.proj[0].bias)
        nn.init.xavier_uniform_(self.proj[3].weight, gain=1e-2)
        nn.init.zeros_(self.proj[3].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x * self.scale
        x = self.proj(residual)
        return (residual + x) * self.output_scale


class ModelWithInsertedHiddenState(nn.Module):
    """Wrap a CausalLM and inject a sequence of hidden states between <bop>/<eop>.

    - Hidden states are optionally processed by a small MHA + projection stack.
    - We also support a curriculum path that mixes plan embeddings with hidden states.
    - The forward returns the base model outputs, with loss optionally augmented.
    """

    def __init__(self,
                 base_model: nn.Module,
                 prepended_length: int,
                 hidden_size: int,
                 prepended_learnable: bool = False,
                 num_heads: int = 8,
                 plan_similarity_weight: float = 1.0,
                 random_contrast_weight: float = 2.0):
        super().__init__()
        self.base_model = base_model
        self.hidden_size = hidden_size
        self.prepended_length = prepended_length
        self.prepended_learnable = prepended_learnable
        self.config = getattr(base_model, "config", None)
        self.tokenizer: Optional[transformers.PreTrainedTokenizer] = None
        self.plan_similarity_weight = float(plan_similarity_weight)
        self.random_contrast_weight = float(random_contrast_weight)
        self.ratio_list: List[float] = []  # stores mix ratios per batch item (for curriculum)

        dmodel = hidden_size
        self.hidden_mha = nn.MultiheadAttention(embed_dim=dmodel, num_heads=num_heads, batch_first=True, dropout=0.1)
        self.pre_ln = nn.LayerNorm(dmodel, eps=1e-6)
        self.post_ln = nn.LayerNorm(dmodel, eps=1e-6)
        self.adaptive_proj = AdaptiveProjection(dmodel)
        self.output_projection = nn.Sequential(
            nn.Linear(dmodel, dmodel * 2),
            nn.GLU(dim=-1),
            nn.LayerNorm(dmodel, eps=1e-6),
            nn.Dropout(0.1),
            nn.Linear(dmodel, dmodel),
        )
        self._init_mha()

        if prepended_learnable:
            self.default_prepended_hidden_state = nn.Parameter(torch.randn(prepended_length, dmodel) * 0.02)
        else:
            self.register_buffer("default_prepended_hidden_state", torch.zeros(prepended_length, dmodel))

        # expose scale params from adaptive projection for saving
        self.scale = self.adaptive_proj.scale
        self.output_scale = self.adaptive_proj.output_scale

    def _init_mha(self):
        nn.init.xavier_uniform_(self.hidden_mha.in_proj_weight, gain=1.0 / math.sqrt(3))
        nn.init.xavier_uniform_(self.hidden_mha.out_proj.weight, gain=1.0)
        if self.hidden_mha.in_proj_bias is not None:
            nn.init.constant_(self.hidden_mha.in_proj_bias, 0.0)
            nn.init.constant_(self.hidden_mha.out_proj.bias, 0.0)

    # ---- hidden‑state processing helpers ----
    def process_hidden_states(self, x: torch.Tensor) -> torch.Tensor:
        """MHA + LN + projection. Keeps dtype/device of layer weights."""
        dev = self.pre_ln.weight.device
        dtyp = self.pre_ln.weight.dtype
        x = x.to(device=dev, dtype=dtyp, non_blocking=True).contiguous()
        normed = self.pre_ln(x).contiguous()
        # Run MHA in float32 inside an autocast block for stability
        with torch.cuda.amp.autocast(enabled=False):
            q = normed.float().contiguous()
            k = normed.float().contiguous()
            v = normed.float().contiguous()
            attn_out, _ = self.hidden_mha(q, k, v, need_weights=False)
        attn_out = attn_out.to(dtyp)
        out = self.post_ln(normed + attn_out)
        projected = self.adaptive_proj(out)
        return projected

    def process_hidden_states_list(self, hs_list: List[Optional[torch.Tensor]]) -> List[Optional[torch.Tensor]]:
        out: List[Optional[torch.Tensor]] = []
        for h in hs_list:
            if h is None:
                out.append(None)
            else:
                out.append(self.process_hidden_states(h.unsqueeze(0)).squeeze(0))
        return out

    @staticmethod
    def _first_supervised_pos(labels: torch.Tensor, attn: torch.Tensor) -> Optional[int]:
        mask = (labels != IGNORE) & attn.bool()
        pos = mask.nonzero(as_tuple=False)
        return int(pos[0, 0].item()) if len(pos) else None

    def _avg_ce_masked(self, logits: torch.Tensor, labels: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        mask = (labels.ne(IGNORE) & attn_mask.bool())
        if mask.sum() == 0:
            return torch.tensor(0.0, device=logits.device)
        logit_sel = logits[mask]
        y_sel = labels[mask].long()
        return F.cross_entropy(logit_sel, y_sel, reduction="mean")

    def _js_bits_masked(self, logits_p: torch.Tensor, logits_q: torch.Tensor, labels: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        mask = (labels.ne(IGNORE) & attn_mask.bool())
        if mask.sum() == 0:
            return torch.tensor(0.0, device=logits_p.device)
        p = F.softmax(logits_p[mask], dim=-1).clamp_min(EPS)
        q = F.softmax(logits_q[mask], dim=-1).clamp_min(EPS)
        m = 0.5 * (p + q)
        js_nats = 0.5 * F.kl_div(p.log(), m, reduction="batchmean") + 0.5 * F.kl_div(q.log(), m, reduction="batchmean")
        return js_nats / math.log(2)

    # ---- mix plan embeddings & hidden states (curriculum) ----
    def adaptive_mix(self, hidden_state: torch.Tensor, plan_embeds: torch.Tensor, mix_ratio: float) -> torch.Tensor:
        """Concatenate first portion of hidden_state with the tail of plan_embeds.
        mix_ratio in [0,1].
        """
        if mix_ratio <= 0.0:
            return plan_embeds
        if mix_ratio >= 1.0:
            return hidden_state
        h_len = hidden_state.size(0)
        p_len = plan_embeds.size(0)
        h_take = int(round(h_len * mix_ratio))
        p_skip = int(round(p_len * mix_ratio))
        return torch.cat([hidden_state[:h_take], plan_embeds[p_skip:]], dim=0)

    # ---- core forward helpers ----
    def _forward_with_hidden_states_curriculum(
        self,
        input_ids: torch.Tensor,
        plan_ids: List[List[int]],
        attention_mask: torch.Tensor,
        inputs_embeds: Optional[torch.Tensor],
        labels: Optional[torch.Tensor],
        human_end_positions: Union[List[int], torch.Tensor],
        prepended_hidden_states: Optional[List[Optional[torch.Tensor]]],
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        dev = next(self.base_model.parameters()).device
        dtyp = next(self.base_model.parameters()).dtype

        disable_mix: bool = bool(kwargs.get("disable_mix", False))
        force_no_labels: bool = bool(kwargs.get("force_no_labels", False))

        if isinstance(human_end_positions, torch.Tensor):
            hep_list = human_end_positions.detach().to("cpu", non_blocking=True).tolist()
        else:
            hep_list = list(human_end_positions)

        if input_ids is not None and inputs_embeds is None:
            inputs_embeds = self.base_model.get_input_embeddings()(input_ids).to(dtyp)
        inputs_embeds = inputs_embeds.to(dtype=dtyp)

        # plan embeddings per sample
        plan_embeds_list: List[torch.Tensor] = []
        emb_layer = self.base_model.get_input_embeddings()
        for ids in plan_ids:
            if ids is None or len(ids) == 0:
                plan_embeds_list.append(torch.zeros(0, emb_layer.embedding_dim, device=dev, dtype=dtyp))
            else:
                t = torch.tensor(ids, device=dev)
                plan_embeds_list.append(emb_layer(t).to(dtyp))

        # process hidden states
        if prepended_hidden_states is not None:
            prepended_hidden_states = [h.to(dtyp) if h is not None else None for h in prepended_hidden_states]
            prepended_hidden_states = self.process_hidden_states_list(prepended_hidden_states)

        # special token vectors
        bop_id = self.tokenizer.convert_tokens_to_ids("<bop>")
        eop_id = self.tokenizer.convert_tokens_to_ids("<eop>")
        if bop_id is None or eop_id is None:
            raise RuntimeError("<bop>/<eop> must be added to tokenizer before building datasets.")
        emb_w = emb_layer.weight
        bop_vec = emb_w[bop_id].to(dtyp)
        eop_vec = emb_w[eop_id].to(dtyp)

        B = inputs_embeds.size(0)
        new_inputs, new_masks, new_labels = [], [], []
        for b in range(B):
            insert_pos = int(hep_list[b])
            before = inputs_embeds[b, :insert_pos] if insert_pos >= 0 else inputs_embeds[b]
            after = inputs_embeds[b, insert_pos:] if insert_pos >= 0 else inputs_embeds[b, 0:0]

            if insert_pos < 0 or prepended_hidden_states is None or prepended_hidden_states[b] is None:
                batch_embeds = inputs_embeds[b]
                batch_mask = attention_mask[b]
                batch_labels = labels[b] if labels is not None else None
            else:
                hidden_seq = prepended_hidden_states[b]
                plan_embeds = plan_embeds_list[b]
                if disable_mix:
                    mixed = hidden_seq
                else:
                    if len(self.ratio_list) <= b:
                        self.ratio_list.append(random.choice([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]))
                    mixed = self.adaptive_mix(hidden_seq, plan_embeds, self.ratio_list[b])

                marked = torch.cat([bop_vec.unsqueeze(0), mixed, eop_vec.unsqueeze(0)], dim=0)
                batch_embeds = torch.cat([before, marked, after], dim=0)

                before_mask = attention_mask[b, :insert_pos]
                after_mask = attention_mask[b, insert_pos:]
                marked_mask = torch.ones(marked.size(0), dtype=attention_mask.dtype, device=attention_mask.device)
                batch_mask = torch.cat([before_mask, marked_mask, after_mask], dim=0)

                if labels is not None:
                    before_lab = labels[b, :insert_pos]
                    after_lab = labels[b, insert_pos:]
                    marked_lab = torch.full((marked.size(0),), IGNORE, dtype=labels.dtype, device=labels.device)
                    batch_labels = torch.cat([before_lab, marked_lab, after_lab], dim=0)
                else:
                    batch_labels = None

            new_inputs.append(batch_embeds)
            new_masks.append(batch_mask)
            if labels is not None:
                new_labels.append(batch_labels)

        inputs_embeds = pad_sequence(new_inputs, batch_first=True, padding_value=0)
        attention_mask = pad_sequence(new_masks, batch_first=True, padding_value=0).to(inputs_embeds.device)
        attention_mask = attention_mask.to(dtype=torch.float32, non_blocking=True).contiguous()
        if labels is not None and len(new_labels) > 0:
            labels = pad_sequence(new_labels, batch_first=True, padding_value=IGNORE)

        outputs = self.base_model(
            input_ids=None,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=None if force_no_labels else labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        return outputs, attention_mask, labels

    # ---- public forward ----
    def forward(self,
                input_ids=None,
                attention_mask=None,
                past_key_values=None,
                inputs_embeds=None,
                labels=None,
                plans=None,
                use_cache=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                human_end_positions=None,
                prepended_hidden_states: Optional[List[Optional[torch.Tensor]]] = None,
                **kwargs):
        self._current_batch_hidden_states = prepended_hidden_states
        outputs_n, attn_n, labels_n = self._forward_with_hidden_states_curriculum(
            input_ids, plans, attention_mask, inputs_embeds, labels,
            human_end_positions, prepended_hidden_states,
            past_key_values, use_cache, output_attentions,
            output_hidden_states, return_dict, **kwargs,
        )

        # Build plan‑only branch (no gradients)
        plan_outputs = None
        if plans is not None:
            with torch.no_grad():
                plan_data = self._insert_plan_tokens(input_ids, attention_mask, labels, human_end_positions, plans)
                plan_outputs = self.base_model(
                    input_ids=plan_data["input_ids"],
                    attention_mask=plan_data["attention_mask"],
                    labels=plan_data["labels"],
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
                attn_p, labels_p = plan_data["attention_mask"], plan_data["labels"]

        # Build random/negative hidden branch for contrast (no gradients)
        random_outputs = None
        if prepended_hidden_states is not None and any(h is not None for h in prepended_hidden_states):
            negs = self._cross_gpu_negatives(prepended_hidden_states) or []
            # pad/trim to match lengths
            for i in range(min(len(negs), len(prepended_hidden_states))):
                if negs[i] is None or prepended_hidden_states[i] is None:
                    continue
                tgt_len = prepended_hidden_states[i].size(0)
                if negs[i].size(0) >= tgt_len:
                    negs[i] = negs[i][:tgt_len]
                else:
                    rep = (tgt_len // negs[i].size(0)) + 1
                    negs[i] = negs[i].repeat(rep, 1)[:tgt_len] + torch.randn_like(prepended_hidden_states[i]) * 0.01
            with torch.no_grad():
                random_outputs, attn_r, labels_r = self._forward_with_hidden_states_curriculum(
                    input_ids, plans, attention_mask, inputs_embeds, labels,
                    human_end_positions, negs if len(negs) else prepended_hidden_states,
                    past_key_values, use_cache, output_attentions,
                    output_hidden_states, return_dict, **kwargs,
                )

        # Augment loss with plan‑similarity and random‑contrast terms if available
        total_loss = outputs_n.loss
        if plan_outputs is not None and random_outputs is not None and labels_n is not None:
            ce_pos = self._avg_ce_masked(outputs_n.logits, labels_n, attn_n)
            ce_neg = self._avg_ce_masked(random_outputs.logits, labels_r, attn_r)
            delta_bits = (ce_neg - ce_pos) / math.log(2)
            js_bits = self._js_bits_masked(outputs_n.logits, random_outputs.logits, labels_n, attn_n)

            plan_sim = self._plan_similarity(outputs_n.logits, plan_outputs.logits, attn_n, attn_p, labels_n, labels_p)
            rand_con = self._random_contrast(outputs_n.logits, random_outputs.logits, attn_n, attn_r, labels_n, labels_r)

            total_loss = outputs_n.loss + self.plan_similarity_weight * plan_sim + self.random_contrast_weight * rand_con
            self.last_loss_components = {
                "eval_ce_loss": float(outputs_n.loss.detach().item()),
                "eval_plan_similarity": float(plan_sim.detach().item()),
                "eval_random_contrast": float(rand_con.detach().item()),
                "probe_delta_nll_bits": float(delta_bits.detach().item()),
                "probe_js_bits": float(js_bits.detach().item()),
            }

        outputs_n.loss = total_loss
        return outputs_n

    # ---- plan‑only sequence builder ----
    def _insert_plan_tokens(self, input_ids, attention_mask, labels, human_end_positions, plans) -> Dict[str, torch.Tensor]:
        bop_id = self.tokenizer.convert_tokens_to_ids("<bop>")
        eop_id = self.tokenizer.convert_tokens_to_ids("<eop>")
        emb = self.base_model.get_input_embeddings()
        pad_id = self.tokenizer.pad_token_id

        new_input_ids, new_masks, new_labels = [], [], []
        for b in range(input_ids.size(0)):
            insert_pos = int(human_end_positions[b].item()) if isinstance(human_end_positions, torch.Tensor) else int(human_end_positions[b])
            before_ids = input_ids[b, :insert_pos] if insert_pos >= 0 else input_ids[b]
            after_ids = input_ids[b, insert_pos:] if insert_pos >= 0 else input_ids[b, 0:0]

            if plans[b] is None or len(plans[b]) == 0:
                seq = input_ids[b]
                am = attention_mask[b]
                lab = labels[b] if labels is not None else None
            else:
                plan_toks = torch.tensor(plans[b], device=input_ids.device)
                # surround with bop/eop
                marked = torch.tensor([bop_id], device=input_ids.device)
                marked = torch.cat([marked, plan_toks, torch.tensor([eop_id], device=input_ids.device)], dim=0)
                seq = torch.cat([before_ids, marked, after_ids], dim=0)
                am = torch.cat([
                    attention_mask[b, :insert_pos],
                    torch.ones(marked.size(0), dtype=attention_mask.dtype, device=attention_mask.device),
                    attention_mask[b, insert_pos:]
                ], dim=0)
                if labels is not None:
                    lab = torch.cat([
                        labels[b, :insert_pos],
                        torch.full((marked.size(0),), IGNORE, dtype=labels.dtype, device=labels.device),
                        labels[b, insert_pos:]
                    ], dim=0)
                else:
                    lab = None

            new_input_ids.append(seq)
            new_masks.append(am)
            if labels is not None:
                new_labels.append(lab)

        return {
            "input_ids": pad_sequence(new_input_ids, batch_first=True, padding_value=pad_id),
            "attention_mask": pad_sequence(new_masks, batch_first=True, padding_value=0),
            "labels": pad_sequence(new_labels, batch_first=True, padding_value=IGNORE) if labels is not None else None,
        }

    # ---- contrastive helpers ----
    def _plan_similarity(self, n_logits, p_logits, attn_n, attn_p, labels_n, labels_p, margin_kl=0.7, margin_cos=0.3) -> torch.Tensor:
        losses = []
        B = n_logits.size(0)
        for i in range(B):
            s_n = self._first_supervised_pos(labels_n[i], attn_n[i])
            s_p = self._first_supervised_pos(labels_p[i], attn_p[i])
            if s_n is None or s_p is None:
                continue
            max_len = min(n_logits.size(1) - s_n, p_logits.size(1) - s_p)
            n_slice = n_logits[i, s_n:s_n + max_len]
            p_slice = p_logits[i, s_p:s_p + max_len].detach()
            joint = (attn_n[i, s_n:s_n + max_len].bool() & attn_p[i, s_p:s_p + max_len].bool() &
                     labels_n[i, s_n:s_n + max_len].ne(IGNORE) & labels_p[i, s_p:s_p + max_len].ne(IGNORE))
            if not joint.any():
                continue
            n_l = n_slice[joint]
            p_l = p_slice[joint]
            kl = F.kl_div(F.log_softmax(n_l, dim=-1), F.softmax(p_l, dim=-1).clamp_min(EPS), reduction="batchmean")
            n_prob = F.softmax(n_l, dim=-1).clamp_min(EPS).view(-1)
            p_prob = F.softmax(p_l, dim=-1).clamp_min(EPS).view(-1)
            cos = 1.0 - F.cosine_similarity(n_prob, p_prob, dim=0)
            losses.append(margin_kl * kl + margin_cos * cos)
        return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=n_logits.device, requires_grad=True)

    def _random_contrast(self, n_logits, r_logits, attn_n, attn_r, labels_n, labels_r, margin=0.69) -> torch.Tensor:
        def js(p, q):
            m = 0.5 * (p + q)
            return 0.5 * F.kl_div(p.log(), m, reduction='batchmean') + 0.5 * F.kl_div(q.log(), m, reduction='batchmean')
        losses = []
        B = n_logits.size(0)
        for i in range(B):
            s = self._first_supervised_pos(labels_n[i], attn_n[i])
            if s is None:
                continue
            max_len = min(n_logits.size(1) - s, r_logits.size(1) - s)
            n_slice = n_logits[i, s:s + max_len]
            r_slice = r_logits[i, s:s + max_len].detach()
            joint = (attn_n[i, s:s + max_len].bool() & attn_r[i, s:s + max_len].bool() &
                     labels_n[i, s:s + max_len].ne(IGNORE) & labels_r[i, s:s + max_len].ne(IGNORE))
            if not joint.any():
                continue
            n_probs = F.softmax(n_slice[joint], dim=-1).clamp_min(EPS)
            r_probs = F.softmax(r_slice[joint], dim=-1).clamp_min(EPS)
            loss = torch.clamp(margin - js(n_probs, r_probs), min=0.0)
            losses.append(loss)
        return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=n_logits.device, requires_grad=True)

    # ---- DDP negative sampling pool ----
    def _cross_gpu_negatives(self, prepended_hidden_states: List[Optional[torch.Tensor]]) -> Optional[List[Optional[torch.Tensor]]]:
        if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
            return None
        local = [h.detach().cpu() if (h is not None) else None for h in prepended_hidden_states]
        world = torch.distributed.get_world_size()
        gathered: List[List[Optional[torch.Tensor]]] = [None] * world
        torch.distributed.all_gather_object(gathered, local)
        pool: List[torch.Tensor] = []
        for lst in gathered:
            for x in lst:
                if x is not None:
                    pool.append(x)
        if not pool:
            return None
        device = prepended_hidden_states[0].device if prepended_hidden_states[0] is not None else self.base_model.device
        negs: List[Optional[torch.Tensor]] = []
        for _ in range(len(prepended_hidden_states)):
            x = random.choice(pool).to(device)
            negs.append(x)
        return negs

    # ---- HF compatibility wrappers ----
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        if hasattr(self.base_model, "gradient_checkpointing_enable"):
            self.base_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    def get_input_embeddings(self):
        return self.base_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.base_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.base_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.base_model.set_output_embeddings(new_embeddings)

    def resize_token_embeddings(self, new_num_tokens):
        return self.base_model.resize_token_embeddings(new_num_tokens)

    def prepare_inputs_for_generation(self, *args, **kwargs):
        return self.base_model.prepare_inputs_for_generation(*args, **kwargs)

    @torch.no_grad()
    def generate(self, *args, **kwargs):
        return self.base_model.generate(*args, **kwargs)

    def save_pretrained(self, save_directory: str, **kwargs) -> None:
        os.makedirs(save_directory, exist_ok=True)
        self.base_model.save_pretrained(save_directory, **kwargs)
        # Save custom blocks
        torch.save({
            "hidden_mha": self.hidden_mha.state_dict(),
            "pre_ln": self.pre_ln.state_dict(),
            "post_ln": self.post_ln.state_dict(),
            "output_projection": self.output_projection.state_dict(),
            "scale": float(self.scale.detach().cpu().item()),
            "output_scale": float(self.output_scale.detach().cpu().item()),
            "prepended_length": int(self.prepended_length),
            "prepended_learnable": bool(self.prepended_learnable),
            "hidden_size": int(self.hidden_size),
        }, os.path.join(save_directory, "latent_comm_head.pt"))


# -----------------------------
# Dataset & Collator
# -----------------------------
class SupervisedDataset(Dataset):
    def __init__(self, raw: List[dict], tokenizer: transformers.PreTrainedTokenizer):
        super().__init__()
        sources = [ex["conversations"] for ex in raw]
        prompts, has_first = format_conversations_chatml(sources)
        data = tokenize_with_marker(prompts, has_first, tokenizer)
        self.input_ids = data["input_ids"]
        self.labels = data["labels"]
        self.attention_mask = data["attention_mask"]
        self.human_end_positions = data["human_end_positions"]
        self.tokenizer = tokenizer

        self.plans: List[List[int]] = []
        self.hidden_states: List[Optional[torch.Tensor]] = []
        for ex in raw:
            plan_txt = ex.get("plan", "")
            plan_ids = tokenizer(plan_txt, add_special_tokens=False).input_ids if plan_txt else []
            self.plans.append(plan_ids)
            hs = None
            if "hidden_state" in ex and ex["hidden_state"] is not None:
                hs = torch.tensor(ex["hidden_state"], dtype=torch.float32)
            elif "hidden_state_path" in ex and ex["hidden_state_path"]:
                path = ex["hidden_state_path"]
                if path.endswith(".npy"):
                    hs = torch.from_numpy(np.load(path).astype(np.float32))
                elif path.endswith(".pt") or path.endswith(".pth"):
                    obj = torch.load(path, map_location="cpu")
                    if isinstance(obj, torch.Tensor):
                        hs = obj.float()
                    else:
                        raise ValueError(f"Unsupported tensor object in {path}")
            self.hidden_states.append(hs)
        assert len(self.input_ids) == len(self.plans) == len(self.hidden_states)

    def __len__(self):
        return self.input_ids.size(0)

    def __getitem__(self, i: int) -> Dict[str, Union[torch.Tensor, List[int]]]:
        item = {
            "input_ids": self.input_ids[i],
            "labels": self.labels[i],
            "attention_mask": self.attention_mask[i],
            "human_end_positions": self.human_end_positions[i],
            "plan": self.plans[i],
        }
        if self.hidden_states[i] is not None:
            item["prepended_hidden_states"] = self.hidden_states[i]
        return item


class LazySupervisedDataset(Dataset):
    def __init__(self, raw: List[dict], tokenizer: transformers.PreTrainedTokenizer):
        super().__init__()
        self.raw = raw
        self.tokenizer = tokenizer
        self._cache: Dict[int, dict] = {}

    def __len__(self):
        return len(self.raw)

    def __getitem__(self, idx: int):
        if idx in self._cache:
            return self._cache[idx]
        prompts, has_first = format_conversations_chatml([self.raw[idx]["conversations"]])
        data = tokenize_with_marker(prompts, has_first, self.tokenizer)
        item = {
            "input_ids": data["input_ids"][0],
            "labels": data["labels"][0],
            "attention_mask": data["attention_mask"][0],
            "human_end_positions": data["human_end_positions"][0],
            "plan": self.tokenizer(self.raw[idx].get("plan", ""), add_special_tokens=False).input_ids,
        }
        # load hidden state on demand
        hs = None
        if "hidden_state" in self.raw[idx] and self.raw[idx]["hidden_state"] is not None:
            hs = torch.tensor(self.raw[idx]["hidden_state"], dtype=torch.float32)
        elif "hidden_state_path" in self.raw[idx] and self.raw[idx]["hidden_state_path"]:
            path = self.raw[idx]["hidden_state_path"]
            if path.endswith(".npy"):
                hs = torch.from_numpy(np.load(path).astype(np.float32))
            elif path.endswith(".pt") or path.endswith(".pth"):
                obj = torch.load(path, map_location="cpu")
                if isinstance(obj, torch.Tensor):
                    hs = obj.float()
        if hs is not None:
            item["prepended_hidden_states"] = hs
        self._cache[idx] = item
        return item


class DataCollatorForSupervisedDataset:
    def __init__(self, tokenizer: transformers.PreTrainedTokenizer):
        self.tokenizer = tokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, Union[torch.Tensor, List]]:
        input_ids = pad_sequence([ex["input_ids"] for ex in instances], batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = pad_sequence([ex["labels"] for ex in instances], batch_first=True, padding_value=IGNORE)
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        human_end_positions = torch.tensor([int(ex.get("human_end_positions", -1)) for ex in instances], dtype=torch.long)
        plans = [ex.get("plan", []) for ex in instances]
        hs: List[Optional[torch.Tensor]] = [ex.get("prepended_hidden_states") for ex in instances]
        batch = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "human_end_positions": human_end_positions,
            "plans": plans,
            "prepended_hidden_states": hs,
        }
        return batch


def load_json_or_jsonl(path: str) -> List[dict]:
    p = pathlib.Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    if p.suffix.lower() == ".jsonl":
        rows = []
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    else:
        with p.open("r", encoding="utf-8") as f:
            obj = json.load(f)
            if isinstance(obj, dict):
                # allow {"data": [...]} containers
                obj = obj.get("data", [])
            assert isinstance(obj, list)
            return obj


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args: DataArguments) -> Dict:
    if data_args.train_file is None:
        raise ValueError("--train_file is required")
    train_raw = load_json_or_jsonl(data_args.train_file)

    if data_args.eval_file is None:
        rng = random.Random(42)
        rng.shuffle(train_raw)
        n_eval = max(1, int(len(train_raw) * data_args.eval_ratio))
        eval_raw = train_raw[:n_eval]
        train_raw = train_raw[n_eval:]
    else:
        eval_raw = load_json_or_jsonl(data_args.eval_file)

    ds_cls = LazySupervisedDataset if data_args.lazy_preprocess else SupervisedDataset
    train_ds = ds_cls(train_raw, tokenizer)
    eval_ds = ds_cls(eval_raw, tokenizer)
    collator = DataCollatorForSupervisedDataset(tokenizer)
    return {"train_dataset": train_ds, "eval_dataset": eval_ds, "data_collator": collator}


# -----------------------------
# Callbacks (minimal & optional)
# -----------------------------
class LossRecorderCallback(TrainerCallback):
    def __init__(self, log_path: str = "loss_log.csv", plot_path: str = "loss_curve.png", enable_plot: bool = True):
        self.log_path = log_path
        self.plot_path = plot_path
        self.enable_plot = enable_plot
        self.losses: List[float] = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self.losses.append(float(logs["loss"]))
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(f"{state.global_step},{float(logs['loss'])}\n")

    def on_train_end(self, args, state, control, **kwargs):
        if not self.enable_plot or not self.losses:
            return
        try:
            plt.figure(figsize=(9, 5))
            plt.plot(self.losses, label="training loss")
            plt.xlabel("log steps")
            plt.ylabel("loss")
            plt.title("Training Loss")
            plt.grid(True)
            plt.legend()
            plt.savefig(self.plot_path)
            plt.close()
            LOGGER.info("Saved loss curve to %s", self.plot_path)
        except Exception as e:
            LOGGER.warning("Failed to save loss curve: %s", e)


class InfoNCEProbeCallback(TrainerCallback):
    def __init__(self, eval_dataset, data_collator, interval: int = 200, K: int = 8, tau: float = 1.0):
        self.eval_dataset = eval_dataset
        self.data_collator = data_collator
        self.interval = int(interval)
        self.K = int(K)
        self.tau = float(tau)
        self._last_step = -1

    @torch.no_grad()
    def _run_probe(self, model, device):
        if not (self.eval_dataset and len(self.eval_dataset) >= 2):
            return None
        mdl = getattr(model, "module", model)
        K = min(max(2, self.K), len(self.eval_dataset))
        idxs = random.sample(range(len(self.eval_dataset)), K)
        batch = self.data_collator([self.eval_dataset[i] for i in idxs])
        if "prepended_hidden_states" not in batch or batch["prepended_hidden_states"] is None:
            return None
        all_z = list(batch["prepended_hidden_states"])  # list of Tensors/None
        scores = torch.full((K, K), float("-inf"), device="cpu")
        old_mode = mdl.training
        orig_ratio = mdl.ratio_list.copy() if hasattr(mdl, "ratio_list") else []
        mdl.eval()
        mdl.ratio_list = []
        try:
            for o in range(K):
                z_rot = all_z[o:] + all_z[:o]
                for j in range(K):
                    sb = self.data_collator([self.eval_dataset[idxs[j]]])
                    ids = sb["input_ids"].to(device)
                    attn = sb["attention_mask"].to(device)
                    labs = sb["labels"].to(device)
                    hep = sb["human_end_positions"].to(device)
                    plans = sb["plans"]
                    z = [z_rot[j]]
                    out, attn_o, labs_o = mdl._forward_with_hidden_states_curriculum(
                        ids, plans, attn, None, labs, hep, z,
                        None, None, None, None, True, force_no_labels=True, disable_mix=True,
                    )
                    m = (labs_o.ne(IGNORE) & attn_o.bool())[0]
                    if m.sum() == 0:
                        continue
                    ce = F.cross_entropy(out.logits[0, m], labs_o[0, m].long(), reduction="mean")
                    scores[j, o] = (-ce).item()
        finally:
            mdl.train(old_mode)
            mdl.ratio_list = orig_ratio
        valid = torch.isfinite(scores).all(dim=1)
        if valid.sum() < 2:
            return None
        s = scores[valid] / max(self.tau, 1e-8)
        s -= s.max(dim=1, keepdim=True).values
        loss = (-(s[:, 0] - torch.logsumexp(s, dim=1))).mean().item()
        lb = (math.log(s.size(1)) - loss)
        return {"probe/cInfoNCE_loss": loss, "probe/cInfoNCE_LB": float(lb)}

    def on_log(self, args, state, control, logs=None, **kwargs):
        if self.interval <= 0 or state.global_step == 0 or (state.global_step % self.interval != 0):
            return control
        model = kwargs.get("model")
        if model is None:
            return control
        self._last_step = int(state.global_step)
        device = next(model.parameters()).device
        metrics = self._run_probe(model, device)
        if metrics is not None:
            if logs is None:
                logs = {}
            logs.update(metrics)
        return control


# -----------------------------
# Precision detection
# -----------------------------
def detect_precision() -> str:
    if not torch.cuda.is_available():
        LOGGER.info("CUDA unavailable; running on CPU (no AMP).")
        return "no"
    prop = torch.cuda.get_device_properties(0)
    if getattr(prop, "major", 0) >= 8:
        LOGGER.info("GPU supports bf16; using bf16.")
        return "bf16"
    LOGGER.info("Using fp16.")
    return "fp16"


# -----------------------------
# Train entrypoint
# -----------------------------

def train():
    setup_logging("INFO")
    set_all_seeds(42)

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Mixed precision prefs (auto)
    prec = detect_precision()
    if prec == "bf16":
        training_args.bf16 = True
        training_args.fp16 = False
    elif prec == "fp16":
        training_args.fp16 = True
        training_args.bf16 = False
    else:
        training_args.fp16 = False
        training_args.bf16 = False

    # Load config/model/tokenizer
    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        trust_remote_code=model_args.trust_remote_code,
    )
    if getattr(config, "use_cache", None) is not None:
        config.use_cache = False

    base_model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        trust_remote_code=model_args.trust_remote_code,
        torch_dtype=(torch.bfloat16 if training_args.bf16 else (torch.float16 if training_args.fp16 else torch.float32)),
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side=model_args.padding_side,
        use_fast=False,
        trust_remote_code=model_args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Add special tokens and resize
    special_tokens = {"additional_special_tokens": ["<FIRST_HUMAN_END>", "<bop>", "<eop>"]}
    num_added = tokenizer.add_special_tokens(special_tokens)
    if num_added > 0:
        base_model.resize_token_embeddings(len(tokenizer))
        LOGGER.info("Added %d special tokens.", num_added)

    # Wrap with latent comm head
    use_position_tracking = (model_args.prepended_length > 0 and model_args.prepend_position == "first_human")
    model: nn.Module
    if use_position_tracking:
        hidden_size = int(getattr(base_model.config, "hidden_size", 0))
        model = ModelWithInsertedHiddenState(
            base_model,
            prepended_length=model_args.prepended_length,
            hidden_size=hidden_size,
            prepended_learnable=model_args.prepended_learnable,
            num_heads=model_args.num_heads,
            plan_similarity_weight=model_args.plan_similarity_weight,
            random_contrast_weight=model_args.random_contrast_weight,
        )
        LOGGER.info("Latent head enabled with K=%d", model_args.prepended_length)
    else:
        model = base_model
        LOGGER.info("Latent head disabled (prepended_length=0 or prepend_position!=first_human)")

    model.tokenizer = tokenizer

    # Build data
    data_module = make_supervised_data_module(tokenizer, data_args)

    # Callbacks
    callbacks: List[TrainerCallback] = [
        LossRecorderCallback(
            log_path=os.path.join(training_args.output_dir, "loss_log.csv"),
            plot_path=os.path.join(training_args.output_dir, "loss_curve.png"),
            enable_plot=training_args.enable_loss_plot,
        )
    ]
    if training_args.enable_infoNCE_probe:
        callbacks.append(
            InfoNCEProbeCallback(
                eval_dataset=data_module["eval_dataset"],
                data_collator=data_module["data_collator"],
                interval=training_args.infoNCE_probe_interval,
                K=training_args.infoNCE_probe_K,
                tau=training_args.infoNCE_probe_tau,
            )
        )

    # Trainer
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **data_module,
        callbacks=callbacks,
    )

    trainer.train()

    # Save
    if hasattr(model, "config"):
        model.config.use_cache = True
    trainer.save_state()
    trainer.save_model(training_args.output_dir)
    if isinstance(model, ModelWithInsertedHiddenState):
        model.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    train()
