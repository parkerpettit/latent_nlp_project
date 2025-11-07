import os
import json
import math
import gc
import time
import random
import argparse
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple, Union, List, Any

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

import transformers
from transformers import (
    AutoConfig, AutoTokenizer, AutoModelForCausalLM,
    Trainer, TrainingArguments, PreTrainedTokenizerBase,
    EarlyStoppingCallback, TrainerCallback
)
from transformers.trainer_pt_utils import LabelSmoother

import datasets
from datasets import load_dataset, load_from_disk

# --------------------
# Constants
# --------------------
IGNORE = -100
EPS = 1e-8
IGNORE_TOKEN_ID = LabelSmoother.ignore_index


# =========================
# Utilities & Callbacks
# =========================
class PreCreateCkptDirCallback(TrainerCallback):
    """Ensure checkpoint directory exists at save steps (DDP-safe)."""
    def on_step_end(self, args, state, control, **kwargs):
        if args.save_strategy == "steps" and args.save_steps > 0:
            if state.global_step > 0 and state.global_step % args.save_steps == 0:
                ckpt = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    if torch.distributed.get_rank() == 0:
                        os.makedirs(ckpt, exist_ok=True)
                    torch.distributed.barrier()
                else:
                    os.makedirs(ckpt, exist_ok=True)


class ParamChangeTracker(transformers.TrainerCallback):
    """
    Track parameter deltas and gradient norms for selected modules.
    """
    def __init__(self, model: nn.Module, track_patterns=None, topn: int = 10, skip_first_log: bool = True):
        super().__init__()
        if track_patterns is None:
            track_patterns = ["h2e", "student_lm.model.layers.", "student_lm.lm_head", "lat_bos"]
        self.model = model
        self.track_patterns = track_patterns
        self.topn = topn
        self.skip_first_log = skip_first_log
        self.bound = False
        self.tracked: List[Tuple[str, nn.Parameter]] = []
        self.init_snap: Dict[str, torch.Tensor] = {}
        self.prev_snap: Dict[str, torch.Tensor] = {}
        self.step_grad_norm: Dict[str, float] = {}
        self._did_first_log = False

    def _select_tracked(self):
        tracked = []
        for name, p in self.model.named_parameters():
            if p.requires_grad and any(s in name for s in self.track_patterns):
                tracked.append((name, p))
        return tracked

    def _take_snapshot(self):
        self.init_snap = {n: p.detach().float().cpu().clone() for n, p in self.tracked}
        self.prev_snap = {n: p.detach().float().cpu().clone() for n, p in self.tracked}

    def on_train_begin(self, args, state, control, **kwargs):
        self.tracked = self._select_tracked()
        if not self.tracked:
            print("[ParamChangeTracker] WARNING: no parameters matched:", self.track_patterns)
        self._take_snapshot()
        self.step_grad_norm = {}
        self.bound = True
        print("[ParamChangeTracker] Tracking parameters:")
        for n, _ in self.tracked:
            print("  -", n)

    def on_step_end(self, args, state, control, **kwargs):
        if not self.bound:
            return
        self.step_grad_norm = {}
        for name, p in self.tracked:
            g = p.grad
            if g is not None:
                self.step_grad_norm[name] = g.detach().float().norm().item()

    def _now_stats(self):
        stats = []
        for name, p in self.tracked:
            cur = p.detach().float().cpu()
            theta_norm = cur.norm().item()
            d_prev = (cur - self.prev_snap[name]).norm().item()
            d_init = (cur - self.init_snap[name]).norm().item()
            g_norm = self.step_grad_norm.get(name, 0.0)
            stats.append((name, theta_norm, d_prev, d_init, g_norm))
            self.prev_snap[name] = cur.clone()
        return sorted(stats, key=lambda x: x[2], reverse=True)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not self.bound:
            return
        if self.skip_first_log and not self._did_first_log:
            self._did_first_log = True
            return
        if hasattr(state, "is_local_process_zero") and not state.is_local_process_zero:
            return
        stats = self._now_stats()
        step = int(state.global_step)
        print(f"[param-delta] step={step} (top {self.topn})")
        print(f"{'name':60s}  {'Δprev':>10s}  {'Δinit':>10s}  {'‖θ‖':>10s}  {'‖∇θ‖_step':>12s}")
        for name, theta_norm, d_prev, d_init, g_norm in stats[:self.topn]:
            print(f"{name:60s}  {d_prev:10.4e}  {d_init:10.4e}  {theta_norm:10.4e}  {g_norm:12.4e}")


class EarlyStoppingStatusCallback(transformers.TrainerCallback):
    """
    Print an easy-to-read early stopping status when eval happens.
    """
    def __init__(self, metric_for_best: str, greater_is_better: bool,
                 patience: int, threshold: float, show_last: int = 5):
        self.metric_for_best = metric_for_best
        self.greater_is_better = greater_is_better
        self.patience = patience
        self.threshold = threshold
        self.show_last = show_last
        self.best = None
        self.bad_count = 0
        from collections import deque
        self.history = deque(maxlen=max(5, show_last))

    def _metric_key(self, metrics: dict) -> str:
        key = self.metric_for_best
        if not key.startswith("eval_"):
            key = f"eval_{key}"
        if key not in metrics:
            key = "eval_loss"
        return key

    def _is_improved(self, cur: float) -> bool:
        if self.best is None:
            return True
        if self.greater_is_better:
            return cur > self.best + self.threshold
        else:
            return cur < self.best - self.threshold

    def on_train_begin(self, args, state, control, **kwargs):
        if hasattr(state, "is_local_process_zero") and not state.is_local_process_zero:
            return
        arrow = "↑" if self.greater_is_better else "↓"
        print(f"[early-stop] watching '{self.metric_for_best}' (interpreted as eval_*), "
              f"target {arrow}, patience={self.patience}, threshold={self.threshold}")

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        if hasattr(state, "is_local_process_zero") and not state.is_local_process_zero:
            return
        key = self._metric_key(metrics)
        if key not in metrics:
            print(f"[early-stop] metric '{key}' not found in metrics: {list(metrics.keys())}")
            return
        cur = float(metrics[key])
        step = int(state.global_step)
        improved = self._is_improved(cur)
        if improved:
            self.best = cur
            self.bad_count = 0
        else:
            self.bad_count += 1
        remain = max(0, self.patience - self.bad_count)
        arrow = "↑" if self.greater_is_better else "↓"
        tag = "✅ improved" if improved else "— no improve"
        self.history.append((step, cur, improved))
        print(f"[early-stop] step={step} {key}={cur:.6f} | best={self.best:.6f} (target {arrow}) | "
              f"{tag} | patience used={self.bad_count}/{self.patience} → remaining={remain}")
        try:
            rows = list(self.history)[-self.show_last:]
            header = f"{'step':>8}  {key:>16}  {'improved':>9}"
            print(header)
            for s, v, imp in rows:
                print(f"{s:8d}  {v:16.6f}  {str(imp):>9}")
        except Exception:
            pass


class PrintMetricsCallback(transformers.TrainerCallback):
    """Print last metrics cached on model (optional)."""
    def on_log(self, args, state, control, logs=None, **kwargs):
        model = kwargs.get('model')
        if hasattr(model, "module"):  # DDP
            model = model.module
        m = getattr(model, "last_metrics", None)
        if m and (state.global_step % (args.logging_steps or 1) == 0):
            print(f"[step {state.global_step}] "
                  f"ce={m.get('ce_theta', -1):.6f} "
                  f"kl={m.get('kl', -1):.6f} "
                  f"align={m.get('align', -1):.6f} "
                  f"total={m.get('total', -1):.6f}")


# =========================
# Latent processing head
# =========================
class AdaptiveProjection(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.2))
        self.output_scale = nn.Parameter(torch.tensor(0.1))
        self.proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
        )
        nn.init.normal_(self.proj[0].weight, mean=0, std=0.02); nn.init.zeros_(self.proj[0].bias)
        nn.init.xavier_uniform_(self.proj[3].weight, gain=1e-2); nn.init.zeros_(self.proj[3].bias)

    def forward(self, x):
        if x.dim() == 2: x = x.unsqueeze(0)
        residual = x * self.scale
        x = self.proj(residual)
        out = (residual + x) * self.output_scale
        return out.squeeze(0) if out.size(0) == 1 else out


class HiddenStateHead(nn.Module):
    """
    Process raw (student) hidden states into teacher-space latents.
    A lightweight MHA + GLU + residual stack.
    """
    def __init__(self, hidden_size: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.pre_ln  = nn.LayerNorm(hidden_size, eps=1e-6)
        self.hidden_mha = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=num_heads, batch_first=True, dropout=dropout
        )
        self.post_ln = nn.LayerNorm(hidden_size, eps=1e-6)
        self.adaptive_proj = AdaptiveProjection(hidden_size)
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GLU(dim=-1),
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        single = False
        if x.dim() == 2:
            x = x.unsqueeze(0); single = True
        dev  = self.pre_ln.weight.device
        dtyp = self.pre_ln.weight.dtype
        x = x.to(dev, dtyp).contiguous()
        normed = self.pre_ln(x).contiguous()
        attn_out, _ = self.hidden_mha(normed, normed, normed, need_weights=False)
        out = self.post_ln(normed + attn_out)
        out = self.adaptive_proj(out)
        out = self.output_projection(out)
        return out.squeeze(0) if single else out


def load_hidden_head_from_ckpt(head: HiddenStateHead, ckpt_path: str, to_dtype: torch.dtype):
    """Optional: load a pre-trained head from a ckpt (state dict keys: pre_ln, hidden_mha, post_ln, output_projection, adaptive_proj/scale)."""
    state = torch.load(ckpt_path, map_location="cpu")

    def cast_sd(sd):
        return {k: (v.to(to_dtype) if hasattr(v, "dtype") else v) for k, v in sd.items()}

    head.load_state_dict({
        **head.state_dict(),
        **{f"pre_ln.{k}": v for k, v in cast_sd(state.get("pre_ln", {})).items()},
        **{f"post_ln.{k}": v for k, v in cast_sd(state.get("post_ln", {})).items()},
    }, strict=False)

    if "hidden_mha" in state:
        head.hidden_mha.load_state_dict(cast_sd(state["hidden_mha"]), strict=False)
    if "output_projection" in state:
        head.output_projection.load_state_dict(cast_sd(state["output_projection"]), strict=False)

    if "adaptive_proj" in state:
        head.adaptive_proj.load_state_dict(cast_sd(state["adaptive_proj"]), strict=False)
    else:
        if "scale" in state: head.adaptive_proj.scale.data.fill_(float(state["scale"]))
        if "output_scale" in state: head.adaptive_proj.output_scale.data.fill_(float(state["output_scale"]))


# =========================
# Core Model (Student over Teacher)
# =========================
class StudentOverTeacher(nn.Module):
    """
    Two-model setup:
      - student_lm generates K-step latent states (autoregressively through a small bridge h2e)
      - teacher receives the latent (wrapped by HiddenStateHead) inserted after first human segment
    Losses: CE (teacher on inserted latent), uncertainty-weighted KL (teacher D -> A), cosine alignment.
    """
    def __init__(self, teacher, tokenizer, student_lm, teacher_head: HiddenStateHead,
                 K=128, beta=2.0, lambda_pref=1.0):
        super().__init__()
        self.teacher = teacher
        self.teacher_head = teacher_head
        self.student_lm = student_lm
        self.tokenizer = tokenizer
        self.K = K

        # Basic loss weights (you can expose to args if needed)
        self.lambda_ce    = 1.0
        self.lambda_kl    = 1.0
        self.lambda_align = 1.0

        self.T = 4.0                 # distillation temperature
        self.window_len = 64         # supervised window length after insertion
        self.unc_quantile = 0.95     # clip factor for uncertainty weights

        # freeze teacher fully
        for _, p in self.teacher.named_parameters():
            p.requires_grad = False
        self.teacher.eval()
        assert sum(p.numel() for p in self.teacher.parameters() if p.requires_grad) == 0, "Teacher must be frozen."

        Hs = student_lm.config.hidden_size
        # latent AR bridge on student side
        self.h2e = nn.Sequential(nn.LayerNorm(Hs), nn.Linear(Hs, Hs, bias=False))
        self.lat_bos = nn.Parameter(torch.zeros(Hs))
        nn.init.normal_(self.lat_bos, mean=0.0, std=0.02)

        # special tokens needed for latent block marking
        self.bop_id = tokenizer.convert_tokens_to_ids('<bop>')
        self.eop_id = tokenizer.convert_tokens_to_ids('<eop>')
        assert self.bop_id is not None and self.eop_id is not None, "Missing <bop>/<eop> in tokenizer!"

        self.last_metrics: Dict[str, float] = {}

    # HF Trainer compatibility passthroughs
    def gradient_checkpointing_enable(self, **kwargs):
        if hasattr(self.student_lm, "gradient_checkpointing_enable"):
            self.student_lm.gradient_checkpointing_enable(**kwargs)

    def gradient_checkpointing_disable(self):
        if hasattr(self.student_lm, "gradient_checkpointing_disable"):
            self.student_lm.gradient_checkpointing_disable()

    def enable_input_require_grads(self):
        if hasattr(self.student_lm, "enable_input_require_grads"):
            self.student_lm.enable_input_require_grads()

    def process_hidden_state(self, hidden_state):
        """Map raw student latents into the teacher space using teacher_head (keeps grad)."""
        return self.teacher_head(hidden_state)

    def _assemble_with_latent(self, inputs_embeds, attention_mask, labels, human_end_positions, latent_list):
        """Insert [<bop>, latent..., <eop>] right after the first human segment."""
        emb_weight = self.teacher.get_input_embeddings().weight
        device, dtype = inputs_embeds.device, inputs_embeds.dtype
        bop_vec = emb_weight[self.bop_id].to(device=device, dtype=dtype)
        eop_vec = emb_weight[self.eop_id].to(device=device, dtype=dtype)

        B, T, H = inputs_embeds.shape
        new_embeds, new_masks, new_labels = [], [], []
        for b in range(B):
            pos = int(human_end_positions[b].item()); pos = max(1, min(pos, T))
            before = inputs_embeds[b, :pos]
            after  = inputs_embeds[b, pos:]
            z      = latent_list[b].to(device=device, dtype=dtype)

            marked = torch.cat([bop_vec.unsqueeze(0), z, eop_vec.unsqueeze(0)], dim=0)
            emb_b  = torch.cat([before, marked, after], dim=0)

            m_bef = attention_mask[b, :pos]
            m_aft = attention_mask[b, pos:]
            m_mid = torch.ones(marked.size(0), device=device, dtype=attention_mask.dtype)
            mask_b = torch.cat([m_bef, m_mid, m_aft], dim=0)

            y_bef = labels[b, :pos]
            y_aft = labels[b, pos:]
            y_mid = torch.full((marked.size(0),), IGNORE, device=device, dtype=labels.dtype)
            y_b = torch.cat([y_bef, y_mid, y_aft], dim=0)

            new_embeds.append(emb_b)
            new_masks.append(mask_b)
            new_labels.append(y_b)

        inputs_embeds = pad_sequence(new_embeds, batch_first=True, padding_value=0.0)
        attention_mask = pad_sequence(new_masks, batch_first=True, padding_value=0)
        labels = pad_sequence(new_labels, batch_first=True, padding_value=IGNORE)
        return inputs_embeds, attention_mask, labels

    def _build_window_masks(self, labels_A: torch.Tensor, labels_B: torch.Tensor, human_end_positions: torch.Tensor, window_len: int):
        """Create supervised windows aligned across A/D path vs baseline B."""
        sup_mask_AD = (labels_A != IGNORE)  # [B, LA]
        B, LA = sup_mask_AD.shape
        LB = labels_B.shape[1]

        orig_idx_AD = torch.cumsum(sup_mask_AD.int(), dim=1) - 1  # original token indices
        win_AD = torch.zeros_like(sup_mask_AD, dtype=torch.bool)
        win_B = torch.zeros((B, LB), dtype=torch.bool, device=labels_B.device)

        for b in range(B):
            s = int(human_end_positions[b].item()) + 1
            if s < 0: s = 0
            eB = min(s + window_len, LB)
            if s < eB:
                win_B[b, s:eB] = True
            cond = (orig_idx_AD[b] >= s) & (orig_idx_AD[b] < s + window_len) & sup_mask_AD[b]
            win_AD[b, cond] = True
        return win_AD, win_B

    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                labels: torch.Tensor,
                human_end_positions: torch.Tensor,
                prepended_hidden_states: List[Optional[torch.Tensor]] = None,
                **kwargs):

        device = input_ids.device
        B, T = input_ids.shape
        z_theta_list: List[torch.Tensor] = []
        emb_s = self.student_lm.get_input_embeddings()

        # ---- Student path: produce K-step latents autoregressively
        for b in range(B):
            pos = int(human_end_positions[b].item()); pos = max(1, min(pos, T))

            # Build a short planning prefix before the human prefix (generic prompt)
            require_text = "Please generate a concise plan to solve this task:"
            require_ids = self.tokenizer(require_text, add_special_tokens=True, return_tensors="pt").input_ids.to(device)

            ctx_ids = torch.cat([require_ids, input_ids[b:b+1, :pos]], dim=1)
            ctx_attn = torch.cat([torch.ones_like(require_ids), attention_mask[b:b+1, :pos]], dim=1)

            ctx_out = self.student_lm(
                input_ids=ctx_ids,
                attention_mask=ctx_attn,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True
            )
            past_kv  = ctx_out.past_key_values
            past_len = ctx_ids.size(1)

            step_in = self.lat_bos.to(device, dtype=emb_s.weight.dtype).view(1,1,-1)
            zs = []
            for _ in range(self.K):
                position_ids = torch.tensor([[past_len]], device=device, dtype=torch.long)
                out_k = self.student_lm(
                    inputs_embeds=step_in,
                    attention_mask=torch.ones_like(position_ids),
                    past_key_values=past_kv,
                    position_ids=position_ids,
                    use_cache=True,
                    output_hidden_states=True,
                    return_dict=True
                )
                z_k = out_k.hidden_states[-1][:, -1, :]
                zs.append(z_k.squeeze(0))
                past_kv  = out_k.past_key_values
                past_len += 1
                step_in = self.h2e(z_k).view(1,1,-1)

            z_b = torch.stack(zs, dim=0)  # [K, Hs]
            z_b = self.process_hidden_state(z_b)  # [K, Ht]
            z_theta_list.append(z_b)

        # ---- Teacher paths: A=student latent (grad), D=data latent (no grad), B=baseline (no latent)
        emb_layer = self.teacher.get_input_embeddings()
        inputs_embeds_full = emb_layer(input_ids)

        # A: student latent with gradient
        emb_A, mask_A, label_A = self._assemble_with_latent(
            inputs_embeds_full, attention_mask, labels, human_end_positions, z_theta_list
        )
        out_A = self.teacher(
            input_ids=None, inputs_embeds=emb_A,
            attention_mask=mask_A, labels=label_A,
            use_cache=False, return_dict=True, output_hidden_states=True
        )
        ce_theta = out_A.loss

        # D: data latent reference (no grad)
        dtype = inputs_embeds_full.dtype
        z_data_list: List[torch.Tensor] = []
        for b in range(B):
            z = prepended_hidden_states[b]
            assert z is not None, "Sample is missing 'hidden_state' in dataset."
            if isinstance(z, torch.Tensor) and z.dim() == 3:
                z = z[0]
            z = z.to(device=device, dtype=dtype)
            if z.size(0) >= self.K:
                z = z[:self.K]
            else:
                z = z.repeat((self.K + z.size(0) - 1)//z.size(0), 1)[:self.K]
            with torch.no_grad():
                z = self.process_hidden_state(z)
            z_data_list.append(z)

        with torch.no_grad():
            emb_D, mask_D, label_D = self._assemble_with_latent(
                inputs_embeds_full, attention_mask, labels, human_end_positions, z_data_list
            )
            out_D = self.teacher(
                input_ids=None, inputs_embeds=emb_D,
                attention_mask=mask_D, labels=label_D,
                use_cache=False, return_dict=True, output_hidden_states=True
            )

        # B: baseline (no latent)
        with torch.no_grad():
            out_B = self.teacher(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                use_cache=False, return_dict=True
            )

        # ---- Uncertainty-weighted KL (teacher: D → A) on supervised windows
        win_AD, win_B = self._build_window_masks(label_A, labels, human_end_positions, self.window_len)

        logits_B = out_B.logits.detach()  # [B, LB, V]
        logits_D = out_D.logits.detach()  # [B, LA, V]
        pB = F.softmax(logits_B, dim=-1)
        pD = F.softmax(logits_D, dim=-1)
        ent_B = -(pB * (pB.clamp_min(1e-6)).log()).sum(-1)   # [B, LB]
        ent_D = -(pD * (pD.clamp_min(1e-6)).log()).sum(-1)   # [B, LA]

        sup_AD = (label_A != IGNORE) & win_AD
        sup_B  = (labels  != IGNORE) & win_B

        ent_B_flat = ent_B[sup_B]
        ent_D_flat = ent_D[sup_AD]

        # Fallback to all supervised tokens if the windows are empty/misaligned
        if ent_B_flat.numel() == 0 or ent_D_flat.numel() == 0 or ent_B_flat.numel() != ent_D_flat.numel():
            sup_AD = (label_A != IGNORE)
            sup_B  = (labels  != IGNORE)
            ent_B_flat = ent_B[sup_B]
            ent_D_flat = ent_D[sup_AD]

        dU = ent_B_flat - ent_D_flat                # >0 means latent reduces uncertainty
        w_raw = torch.relu(dU).float()
        if (w_raw > 0).any():
            q_hi = torch.quantile(w_raw[w_raw > 0], 0.95)
            w_raw = torch.clamp(w_raw, 0, q_hi)
        den = w_raw.mean().clamp_min(1e-6)
        W_unc_flat = (w_raw / den).detach()

        T = self.T
        logits_A = out_A.logits
        log_p_t = F.log_softmax(logits_D / T, dim=-1)
        log_p_s = F.log_softmax(logits_A / T, dim=-1)
        p_t     = log_p_t.exp()

        log_p_t_tok = log_p_t[sup_AD]   # [N_tok, V]
        log_p_s_tok = log_p_s[sup_AD]
        p_t_tok     = p_t[sup_AD]

        if p_t_tok.numel() == 0:
            L_kl_unc = logits_A.new_tensor(0.0)
        else:
            kl_tok = (T*T) * (p_t_tok * (log_p_t_tok - log_p_s_tok)).sum(dim=-1)  # [N_tok]
            L_kl_unc = (W_unc_flat * kl_tok).sum() / (W_unc_flat.sum() + 1e-6)

        # ---- Latent direction alignment (cosine)
        def _pool_mean(z_list):
            return torch.stack([z.mean(dim=0) for z in z_list], dim=0)
        Z_theta = _pool_mean(z_theta_list)        # [B, Ht]
        with torch.no_grad():
            Z_data  = _pool_mean(z_data_list)     # [B, Ht]
        L_align = 1.0 - F.cosine_similarity(Z_theta, Z_data, dim=-1).mean()

        # ---- Total loss
        loss = (self.lambda_ce   * ce_theta +
                self.lambda_kl   * L_kl_unc +
                self.lambda_align* L_align)

        # cache for PrintMetricsCallback (optional)
        self.last_metrics = {
            "ce_theta": float(ce_theta.detach().cpu().item()),
            "kl": float(L_kl_unc.detach().cpu().item()),
            "align": float(L_align.detach().cpu().item()),
            "total": float(loss.detach().cpu().item())
        }

        out = out_A
        out.loss = loss
        return out


# =========================
# Data Loading
# =========================
class HiddenStateLoader:
    """
    Load per-task hidden_state/plan from a HF Datasets repo id (hub)
    or a local dataset directory (load_from_disk).
    Expect each record to include 'task_id' (or 'id'), 'hidden_state' (L x H), 'plan' (str).
    """
    def __init__(self, dataset_name_or_path: str):
        self.dataset_name = dataset_name_or_path
        self._load_data()

    def _load_data(self):
        print(f"[HiddenStateLoader] Loading from: {self.dataset_name}")
        if os.path.isdir(self.dataset_name):
            ds = load_from_disk(self.dataset_name)
        else:
            ds = load_dataset(self.dataset_name, split=datasets.Split.TRAIN)
        print(f"[HiddenStateLoader] Loaded {len(ds)} records.")
        id_to_data = {}
        t0 = time.time()
        for row in ds:
            task_id = row.get('task_id') or row.get('id')
            hs = row.get('hidden_state', None)
            plan = row.get('plan', '')
            if task_id is None or hs is None:
                continue
            if isinstance(hs, np.ndarray) and hs.dtype == object:
                hs = np.array(hs.tolist(), dtype=np.float32)
            else:
                hs = np.array(hs, dtype=np.float32)
            tensor = torch.from_numpy(hs)
            id_to_data[task_id] = {'hidden_state': tensor, 'plan': plan if isinstance(plan, str) else str(plan)}
        print(f"[HiddenStateLoader] Built index for {len(id_to_data)} items in {time.time()-t0:.2f}s")
        self.id_to_data = id_to_data

    def get_hidden_state_and_plan(self, task_id):
        if task_id not in self.id_to_data:
            raise KeyError(f"No hidden_state found for task_id: {task_id}")
        d = self.id_to_data[task_id]
        return d['hidden_state'], d['plan']


def tokenize_conversations(
    conversations: List[List[Dict[str, str]]],
    tokenizer: PreTrainedTokenizerBase,
    model_max_length: int,
) -> Dict[str, torch.Tensor]:
    """
    Generic, template-free packing:
      - Iterate messages; append tokens for each message text;
      - Labels = IGNORE for 'human/user/system' segments, tokens for 'assistant' segments;
      - Record token position right after the FIRST human/user message for latent insertion.

    Expected message format per sample:
      [{'from': 'human', 'value': '...'}, {'from': 'gpt', 'value': '...'}, ...]
    """
    role_map = {'human': 'user', 'gpt': 'assistant', 'user': 'user', 'assistant': 'assistant', 'system': 'system'}

    all_input_ids = []
    all_labels = []
    all_attn = []
    human_end_positions = []

    for conv in conversations:
        ids = []
        lbl = []
        pos_counter = 0
        first_human_pos = -1
        for msg in conv:
            role = role_map.get(msg.get('from', '').lower(), 'user')
            text = str(msg.get('value', ''))
            # tokenize message content only (no special tokens)
            tok = tokenizer(text, add_special_tokens=False).input_ids
            if not isinstance(tok, list):
                tok = list(tok)

            # newline separator between messages (makes boundaries clearer)
            nl = tokenizer("\n", add_special_tokens=False).input_ids

            if role in ('user', 'system'):
                ids.extend(tok); lbl.extend([IGNORE]*len(tok))
                ids.extend(nl);  lbl.extend([IGNORE]*len(nl))
                pos_counter += len(tok) + len(nl)
                if first_human_pos < 0 and role == 'user':
                    first_human_pos = pos_counter  # position *after* the first human message
            elif role == 'assistant':
                ids.extend(tok); lbl.extend(tok)  # supervise assistant tokens
                ids.extend(nl);  lbl.extend([IGNORE]*len(nl))  # do not supervise trailing NL
                pos_counter += len(tok) + len(nl)
            else:  # default: treat as user
                ids.extend(tok); lbl.extend([IGNORE]*len(tok))
                ids.extend(nl);  lbl.extend([IGNORE]*len(nl))
                pos_counter += len(tok) + len(nl)

        if first_human_pos < 0:
            # If there was no human message, insert at position 1 (after BOS) for safety
            first_human_pos = 1

        # truncate to model_max_length
        ids = ids[:model_max_length]
        lbl = lbl[:model_max_length]
        attn = [1]*len(ids)

        # pad will be handled by data collator; here keep ragged lists
        all_input_ids.append(torch.tensor(ids, dtype=torch.long))
        all_labels.append(torch.tensor(lbl, dtype=torch.long))
        all_attn.append(torch.tensor(attn, dtype=torch.long))
        human_end_positions.append(first_human_pos if first_human_pos < model_max_length else model_max_length-1)

    return dict(
        input_ids=all_input_ids,
        labels=all_labels,
        attention_mask=all_attn,
        human_end_positions=torch.tensor(human_end_positions, dtype=torch.long),
    )


class SupervisedDataset(Dataset):
    """Pack conversations; attach plan/hidden_state if present."""
    def __init__(self, raw_data, tokenizer: PreTrainedTokenizerBase, model_max_length: int):
        super().__init__()
        sources = [ex["conversations"] for ex in raw_data]
        data_dict = tokenize_conversations(sources, tokenizer, model_max_length)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.attention_mask = data_dict["attention_mask"]
        self.human_end_positions = data_dict["human_end_positions"]

        self.plans: List[List[int]] = []
        self.hidden_states: List[Optional[torch.Tensor]] = []
        self.tokenizer = tokenizer

        for ex in raw_data:
            plan = ex.get("plan", "")
            plan_ids = tokenizer(plan, add_special_tokens=False).input_ids
            self.plans.append(plan_ids)
            if 'hidden_state' in ex and ex['hidden_state'] is not None:
                hs = ex['hidden_state']
                if isinstance(hs, torch.Tensor):
                    self.hidden_states.append(hs)
                else:
                    self.hidden_states.append(torch.tensor(hs, dtype=torch.float32))
            else:
                self.hidden_states.append(None)

        assert len(self.input_ids) == len(self.plans) == len(self.hidden_states)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, Any]:
        ret = dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i],
            attention_mask=self.attention_mask[i],
            human_end_positions=self.human_end_positions[i],
            plan=self.plans[i],
        )
        if self.hidden_states[i] is not None:
            ret['prepended_hidden_states'] = self.hidden_states[i]
        return ret


class DataCollatorForSupervisedDataset:
    """Pad ragged sequences; keep custom fields."""
    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        self.tokenizer = tokenizer

    def __call__(self, instances: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        input_ids, labels = tuple([inst[k] for inst in instances] for k in ("input_ids", "labels"))
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE)
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)

        human_end_positions = torch.tensor(
            [int(inst.get("human_end_positions", -1)) for inst in instances],
            dtype=torch.long,
        )

        prepended_hidden_states = [inst.get("prepended_hidden_states", None) for inst in instances]
        plans = [inst.get("plan", []) for inst in instances]

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "human_end_positions": human_end_positions,
            "plans": plans,
            "prepended_hidden_states": prepended_hidden_states,
        }


def make_supervised_data_module(
        tokenizer: PreTrainedTokenizerBase,
        data_path: str,
        hf_hidden_repo: str,
        eval_ratio: float,
        prepended_length: int,
        model_max_length: int,
) -> Dict[str, Any]:
    """Build datasets and collator; enrich samples with hidden_state/plan from repo."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"--data_path not found: {data_path}")
    if not hf_hidden_repo:
        raise ValueError("--hf_hidden_repo is required (HF repo id or local load_from_disk path)")

    loader_train = HiddenStateLoader(hf_hidden_repo)

    with open(data_path, "r", encoding="utf-8") as f:
        train_json = json.load(f)

    # Attach hidden_state/plan to each item by id; trim to `prepended_length`
    enriched = []
    missing = 0
    for item in train_json:
        item = dict(item)  # shallow copy
        sample_id = item.get('id')
        try:
            hidden_state, plan = loader_train.get_hidden_state_and_plan(sample_id)
            if hidden_state.size(0) > prepended_length:
                hidden_state = hidden_state[:prepended_length, :]
            item['hidden_state'] = hidden_state
            item['plan'] = plan
        except KeyError:
            # leave None if missing; StudentOverTeacher will assert on missing at runtime
            item['hidden_state'] = None
            item['plan'] = item.get('plan', "")
            missing += 1
        enriched.append(item)

    if missing > 0:
        print(f"[Data] WARNING: {missing} samples missing hidden_state in repo; training will assert on those samples.")

    rng = random.Random(42)
    rng.shuffle(enriched)
    n_total = len(enriched)
    n_eval = max(1, int(n_total * eval_ratio))
    eval_json = enriched[:n_eval]
    train_json = enriched[n_eval:]

    train_dataset = SupervisedDataset(train_json, tokenizer=tokenizer, model_max_length=model_max_length)
    eval_dataset  = SupervisedDataset(eval_json,  tokenizer=tokenizer, model_max_length=model_max_length)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator)


# =========================
# Args
# =========================
@dataclass
class Args:
    # Required
    teacher_model_name_or_path: str = field(default=None)   # e.g., "Qwen/Qwen2.5-7B-Instruct"
    student_model_name_or_path: str = field(default=None)   # e.g., "Qwen/Qwen2.5-0.5B"
    data_path: str = field(default=None)                    # path to SFT json
    hf_hidden_repo: str = field(default=None)               # HF repo id or local dataset path (load_from_disk)

    # Optional
    output_dir: str = field(default="./latent_out")
    model_max_length: int = field(default=4096)
    per_device_train_batch_size: int = field(default=2)
    per_device_eval_batch_size: int = field(default=2)
    learning_rate: float = field(default=5e-5)
    num_train_epochs: int = field(default=3)
    logging_steps: int = field(default=10)
    save_steps: int = field(default=200)
    eval_steps: int = field(default=200)
    warmup_ratio: float = field(default=0.03)
    bf16: bool = field(default=False)
    fp16: bool = field(default=False)
    seed: int = field(default=42)
    K: int = field(default=128)
    deepspeed: Optional[str] = field(default=None)
    gradient_checkpointing: bool = field(default=False)
    early_stopping_patience: int = field(default=5)
    early_stopping_threshold: float = field(default=0.0)
    eval_ratio: float = field(default=0.05)
    head_num_heads: int = field(default=8)
    head_dropout: float = field(default=0.1)
    hidden_head_ckpt: Optional[str] = field(default=None)   # optional ckpt file for HiddenStateHead

def parse_args() -> Args:
    ap = argparse.ArgumentParser()
    for k, v in Args().__dict__.items():
        t = type(v)
        if v is None:
            ap.add_argument(f"--{k}", type=str, default=None)
        elif t is bool:
            # store_true toggles from default False → True; store_false from True → False
            ap.add_argument(f"--{k}", action="store_true" if not v else "store_false")
        else:
            ap.add_argument(f"--{k}", type=t, default=v)
    ns = ap.parse_args()
    args = Args(**vars(ns))

    # Validate required
    missing = []
    for key in ["teacher_model_name_or_path", "student_model_name_or_path", "data_path", "hf_hidden_repo"]:
        if not getattr(args, key):
            missing.append(f"--{key}")
    if missing:
        raise ValueError("Missing required arguments: " + ", ".join(missing))

    return args


# =========================
# Main
# =========================
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)

    # Tokenizer (use teacher's tokenizer)
    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model_name_or_path, use_fast=False, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    special_tokens_dict = {'additional_special_tokens': ['<bop>', '<eop>']}
    tokenizer.add_special_tokens(special_tokens_dict)
    tokenizer.model_max_length = args.model_max_length

    # Teacher
    config_t = AutoConfig.from_pretrained(args.teacher_model_name_or_path, trust_remote_code=True)
    config_t.use_cache = False
    attn_impl = None
    try:
        if torch.cuda.is_available():
            attn_impl = "flash_attention_2"
    except Exception:
        attn_impl = None
    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher_model_name_or_path,
        config=config_t,
        trust_remote_code=True,
        torch_dtype=(torch.bfloat16 if args.bf16 else torch.float16 if args.fp16 else torch.float32),
        attn_implementation=attn_impl
    )
    teacher.resize_token_embeddings(len(tokenizer))

    # Student
    config_s = AutoConfig.from_pretrained(args.student_model_name_or_path, trust_remote_code=True)
    config_s.use_cache = False
    student_lm = AutoModelForCausalLM.from_pretrained(
        args.student_model_name_or_path,
        config=config_s,
        trust_remote_code=True,
        torch_dtype=(torch.bfloat16 if args.bf16 else torch.float16 if args.fp16 else torch.float32),
        attn_implementation=attn_impl
    )
    student_lm.resize_token_embeddings(len(tokenizer))

    # Teacher latent head
    hidden_size_t = teacher.config.hidden_size
    teacher_head = HiddenStateHead(hidden_size=hidden_size_t, num_heads=args.head_num_heads, dropout=args.head_dropout)
    teacher_head.to(next(teacher.parameters()).dtype).to(next(teacher.parameters()).device)
    if args.hidden_head_ckpt:
        load_hidden_head_from_ckpt(teacher_head, args.hidden_head_ckpt, to_dtype=next(teacher.parameters()).dtype)

    # Data module
    data_module = make_supervised_data_module(
        tokenizer=tokenizer,
        data_path=args.data_path,
        hf_hidden_repo=args.hf_hidden_repo,
        eval_ratio=args.eval_ratio,
        prepended_length=args.K,
        model_max_length=args.model_max_length
    )

    # Build model wrapper
    model = StudentOverTeacher(
        teacher=teacher,
        tokenizer=tokenizer,
        student_lm=student_lm,
        teacher_head=teacher_head,
        K=args.K,
        beta=2.0,
        lambda_pref=1.0,
    )

    # Training args
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=3,
        warmup_ratio=args.warmup_ratio,
        bf16=args.bf16,
        fp16=args.fp16,
        remove_unused_columns=False,  # IMPORTANT: keep custom inputs
        report_to=[],                 # disable W&B etc. for anonymity
        deepspeed=args.deepspeed,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=8,
        dataloader_pin_memory=True,
    )

    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_threshold=args.early_stopping_threshold
    )
    status_cb = EarlyStoppingStatusCallback(
        metric_for_best=training_args.metric_for_best_model,
        greater_is_better=training_args.greater_is_better,
        patience=args.early_stopping_patience,
        threshold=args.early_stopping_threshold,
        show_last=5,
    )
    tracker_cb = ParamChangeTracker(model, track_patterns=["h2e", "student_lm.lm_head", "student_lm.model.layers"])

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        **data_module,
        callbacks=[PrintMetricsCallback(), early_stopping_callback, status_cb, tracker_cb, PreCreateCkptDirCallback()]
    )

    print("=== Training starts ===")
    trainer.train()
    trainer.save_model(args.output_dir)
    print("=== Training finished ===")


if __name__ == "__main__":
    main()
