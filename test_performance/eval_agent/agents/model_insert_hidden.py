import os
import json
import math
import random
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers.trainer_pt_utils import LabelSmoother

IGNORE_TOKEN_ID = LabelSmoother.ignore_index


class AdaptiveProjection(nn.Module):
    """Lightweight projection with learnable scaling and residual path."""

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
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.proj[0].weight, mean=0, std=0.02)
        nn.init.zeros_(self.proj[0].bias)
        nn.init.xavier_uniform_(self.proj[3].weight, gain=1e-2)
        nn.init.zeros_(self.proj[3].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x * self.scale
        x = self.proj(residual)
        return (residual + x) * self.output_scale


class ModelWithInsertedHiddenState(nn.Module):
    """
    Wrap a base Causal LM to insert latent hidden states (and/or plans) into the
    input embeddings around a specified insertion position (human_end_positions).

    The module supports three heads in training-time forward:
      1) normal_hidden: hidden states inserted
      2) plan_text: plan tokens inserted
      3) random_hidden: randomized hidden states inserted

    Additional losses encourage normal_hidden to be close to plan_text and
    far from random_hidden.
    """

    def __init__(
        self,
        base_model: nn.Module,
        prepended_length: int,
        hidden_size: int,
        prepended_learnable: bool = False,
        num_heads: int = 8,
        plan_similarity_weight: float = 0.5,
        random_contrast_weight: float = 1.5,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.hidden_size = hidden_size
        self.prepended_length = prepended_length
        self.prepended_learnable = prepended_learnable
        self.config = getattr(base_model, "config", None)
        self.tokenizer = None  # must be set externally

        self.plan_similarity_weight = plan_similarity_weight
        self.random_contrast_weight = random_contrast_weight

        # Attention + norms (bfloat16 is common for modern GPUs; keep model dtype-compatible)
        self.hidden_mha = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=num_heads, batch_first=True, dropout=0.1
        )
        self.pre_ln = nn.LayerNorm(hidden_size, eps=1e-6)
        self.post_ln = nn.LayerNorm(hidden_size, eps=1e-6)
        self._init_mha_weights()

        # Small adapter
        self.adaptive_proj = AdaptiveProjection(hidden_size)

        # Optional learnable default hidden state
        if prepended_learnable:
            self.default_prepended_hidden_state = nn.Parameter(
                torch.randn(prepended_length, hidden_size) * 0.02
            )
        else:
            self.register_buffer(
                "default_prepended_hidden_state",
                torch.zeros(prepended_length, hidden_size),
            )

    # ------------------------------------------------------------------ utils
    def _init_mha_weights(self) -> None:
        nn.init.xavier_uniform_(self.hidden_mha.in_proj_weight, gain=1.0 / math.sqrt(3))
        nn.init.xavier_uniform_(self.hidden_mha.out_proj.weight, gain=1.0)
        if self.hidden_mha.in_proj_bias is not None:
            nn.init.constant_(self.hidden_mha.in_proj_bias, 0.0)
            nn.init.constant_(self.hidden_mha.out_proj.bias, 0.0)

    @torch.no_grad()
    def _token_embeds(self, token_ids: torch.Tensor) -> torch.Tensor:
        emb = self.base_model.get_input_embeddings()
        return emb(token_ids)

    def _process_hidden(self, h: torch.Tensor) -> torch.Tensor:
        # LN -> MHA -> residual LN -> small adapter
        normed = self.pre_ln(h)
        attn, _ = self.hidden_mha(normed, normed, normed)
        attn = self.post_ln(normed + attn)
        return self.adaptive_proj(attn)

    def _process_hidden_list(self, hidden_list: List[torch.Tensor]) -> List[torch.Tensor]:
        return [self._process_hidden(x.unsqueeze(0)).squeeze(0) for x in hidden_list]

    # ---------------------------------------------------------- data operations
    def insert_plan_tokens(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        labels: Optional[torch.Tensor],
        human_end_positions: torch.Tensor,
        plans: List[List[int]],
    ) -> Dict[str, Optional[torch.Tensor]]:
        batch = input_ids.size(0)
        new_input_ids, new_attention_mask, new_labels = [], [], []

        bop_id = self.tokenizer.convert_tokens_to_ids("<bop>")
        eop_id = self.tokenizer.convert_tokens_to_ids("<eop>")

        for i in range(batch):
            if human_end_positions[i] >= 0 and plans[i]:
                ins = human_end_positions[i].item()
                plan_tokens = torch.tensor(plans[i], device=input_ids.device)
                before = input_ids[i, :ins]
                after = input_ids[i, ins:]
                marked = torch.cat([
                    torch.tensor([bop_id], device=input_ids.device),
                    plan_tokens,
                    torch.tensor([eop_id], device=input_ids.device),
                ], dim=0)
                new_input_ids.append(torch.cat([before, marked, after], dim=0))

                if attention_mask is not None:
                    new_attention_mask.append(
                        torch.cat([
                            attention_mask[i, :ins],
                            torch.ones(marked.size(0), device=attention_mask.device),
                            attention_mask[i, ins:],
                        ], dim=0)
                    )
                if labels is not None:
                    new_labels.append(
                        torch.cat([
                            labels[i, :ins],
                            torch.full((marked.size(0),), IGNORE_TOKEN_ID, device=labels.device),
                            labels[i, ins:],
                        ], dim=0)
                    )
            else:
                new_input_ids.append(input_ids[i])
                if attention_mask is not None:
                    new_attention_mask.append(attention_mask[i])
                if labels is not None:
                    new_labels.append(labels[i])

        return {
            "input_ids": pad_sequence(new_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id),
            "attention_mask": pad_sequence(new_attention_mask, batch_first=True, padding_value=0)
            if new_attention_mask
            else None,
            "labels": pad_sequence(new_labels, batch_first=True, padding_value=IGNORE_TOKEN_ID)
            if new_labels
            else None,
        }

    def _concat_with_hidden(
        self,
        input_ids: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        inputs_embeds: Optional[torch.Tensor],
        labels: Optional[torch.Tensor],
        human_end_positions: torch.Tensor,
        hidden_seq_list: List[torch.Tensor],
    ) -> Dict[str, Optional[torch.Tensor]]:
        device = next(self.base_model.parameters()).device
        model_dtype = next(self.base_model.parameters()).dtype

        if input_ids is not None and inputs_embeds is None:
            inputs_embeds = self._token_embeds(input_ids).to(model_dtype)
        inputs_embeds = inputs_embeds.to(dtype=model_dtype)

        # process hidden sequences
        if hidden_seq_list is not None:
            hidden_seq_list = [h.to(dtype=model_dtype) for h in hidden_seq_list]
            hidden_seq_list = self._process_hidden_list(hidden_seq_list)

        bop_id = self.tokenizer.convert_tokens_to_ids("<bop>")
        eop_id = self.tokenizer.convert_tokens_to_ids("<eop>")
        bop_embed = self._token_embeds(torch.tensor([bop_id], device=device)).squeeze(0)
        eop_embed = self._token_embeds(torch.tensor([eop_id], device=device)).squeeze(0)

        B = inputs_embeds.size(0)
        new_embeds, new_mask, new_labels = [], [], []

        for i in range(B):
            ins = human_end_positions[i].item()
            if ins < 0:
                new_embeds.append(inputs_embeds[i])
                if attention_mask is not None:
                    new_mask.append(attention_mask[i])
                if labels is not None:
                    new_labels.append(labels[i])
                continue

            before = inputs_embeds[i, :ins]
            after = inputs_embeds[i, ins:]
            hidden_i = hidden_seq_list[i]

            marked = torch.cat([bop_embed.unsqueeze(0), hidden_i, eop_embed.unsqueeze(0)], dim=0)
            cat_embeds = torch.cat([before, marked, after], dim=0)
            new_embeds.append(cat_embeds)

            if attention_mask is not None:
                new_mask.append(
                    torch.cat([
                        attention_mask[i, :ins],
                        torch.ones(marked.size(0), dtype=attention_mask.dtype, device=attention_mask.device),
                        attention_mask[i, ins:],
                    ], dim=0)
                )
            if labels is not None:
                new_labels.append(
                    torch.cat([
                        labels[i, :ins],
                        torch.full((marked.size(0),), IGNORE_TOKEN_ID, dtype=labels.dtype, device=labels.device),
                        labels[i, ins:],
                    ], dim=0)
                )

        out_embeds = pad_sequence(new_embeds, batch_first=True, padding_value=0)
        out_mask = pad_sequence(new_mask, batch_first=True, padding_value=0) if new_mask else None
        out_labels = pad_sequence(new_labels, batch_first=True, padding_value=IGNORE_TOKEN_ID) if new_labels else None

        return {
            "inputs_embeds": out_embeds,
            "attention_mask": out_mask,
            "labels": out_labels,
        }

    # ---------------------------------------------------------- sampling utils
    def _generate_random_hidden(
        self, batch_size: int, device: torch.device, dtype: torch.dtype, lengths: List[int]
    ) -> List[torch.Tensor]:
        out: List[torch.Tensor] = []
        for i in range(batch_size):
            L = int(lengths[i])
            h = torch.randn(L, self.hidden_size, device=device, dtype=dtype) * 0.1
            if L > 4:
                for j in range(0, L, 4):
                    end = min(j + 4, L)
                    if j > 0:
                        h[j:end] = 0.7 * h[j:end] + 0.3 * h[j - 1 : j - 1 + end - j]
            h = F.normalize(h, dim=-1) * 2.0
            out.append(h)
        return out

    # -------------------------------------------------------------- loss terms
    def _plan_similarity_loss(
        self,
        normal_logits: torch.Tensor,
        plan_logits: torch.Tensor,
        human_end_positions: torch.Tensor,
        hidden_lengths: List[int],
        plan_lengths: List[int],
    ) -> torch.Tensor:
        losses = []
        B = normal_logits.size(0)
        for i in range(B):
            pos = int(human_end_positions[i].item())
            if pos < 0:
                continue
            h_end = pos + 1 + hidden_lengths[i] + 1
            p_end = pos + 1 + plan_lengths[i] + 1
            n_rem = normal_logits.size(1) - h_end
            p_rem = plan_logits.size(1) - p_end
            L = min(n_rem, p_rem, 200)
            if L <= 0:
                continue
            n_region = normal_logits[i, h_end : h_end + L]
            p_region = plan_logits[i, p_end : p_end + L].detach()
            n_probs = F.softmax(n_region, dim=-1)
            p_probs = F.softmax(p_region, dim=-1)
            kl = F.kl_div(F.log_softmax(n_region, dim=-1), p_probs, reduction="batchmean")
            cos = 1.0 - F.cosine_similarity(n_probs.view(-1), p_probs.view(-1), dim=0)
            losses.append(0.7 * kl + 0.3 * cos)
        if losses:
            return torch.mean(torch.stack(losses))
        return torch.tensor(0.0, device=normal_logits.device, requires_grad=True)

    def _random_contrast_loss(
        self,
        normal_logits: torch.Tensor,
        random_logits: torch.Tensor,
        human_end_positions: torch.Tensor,
        hidden_lengths: List[int],
    ) -> torch.Tensor:
        def js(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
            m = 0.5 * (p + q)
            return 0.5 * F.kl_div(p.log(), m, reduction="batchmean") + 0.5 * F.kl_div(q.log(), m, reduction="batchmean")

        losses = []
        B = normal_logits.size(0)
        margin = 0.5
        for i in range(B):
            pos = int(human_end_positions[i].item())
            if pos < 0:
                continue
            h_end = pos + 1 + hidden_lengths[i] + 1
            start = h_end
            L = min(normal_logits.size(1) - start, random_logits.size(1) - start, 200)
            if L <= 0:
                continue
            n_region = normal_logits[i, start : start + L]
            r_region = random_logits[i, start : start + L].detach()
            n_probs = F.softmax(n_region, dim=-1) + 1e-8
            r_probs = F.softmax(r_region, dim=-1) + 1e-8
            d = js(n_probs, r_probs)
            losses.append(torch.clamp(margin - d, min=0.0))
        if losses:
            return torch.mean(torch.stack(losses))
        return torch.tensor(0.0, device=normal_logits.device, requires_grad=True)

    # ----------------------------------------------------------------- forwards
    def _forward_with_hidden(
        self,
        input_ids: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        inputs_embeds: Optional[torch.Tensor],
        labels: Optional[torch.Tensor],
        human_end_positions: torch.Tensor,
        hidden_seq_list: List[torch.Tensor],
        past_key_values: Any,
        use_cache: Optional[bool],
        output_attentions: Optional[bool],
        output_hidden_states: Optional[bool],
        return_dict: Optional[bool],
        **kwargs: Any,
    ) -> Any:
        pack = self._concat_with_hidden(
            input_ids, attention_mask, inputs_embeds, labels, human_end_positions, hidden_seq_list
        )
        return self.base_model(
            input_ids=None,
            attention_mask=pack["attention_mask"],
            past_key_values=past_key_values,
            inputs_embeds=pack["inputs_embeds"],
            labels=pack["labels"],
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )

    def _forward_with_hidden_and_plan_mix(
        self,
        input_ids: Optional[torch.Tensor],
        plan_ids: List[List[int]],
        attention_mask: Optional[torch.Tensor],
        inputs_embeds: Optional[torch.Tensor],
        labels: Optional[torch.Tensor],
        human_end_positions: torch.Tensor,
        hidden_seq_list: List[torch.Tensor],
        past_key_values: Any,
        use_cache: Optional[bool],
        output_attentions: Optional[bool],
        output_hidden_states: Optional[bool],
        return_dict: Optional[bool],
        **kwargs: Any,
    ) -> Any:
        device = next(self.base_model.parameters()).device
        model_dtype = next(self.base_model.parameters()).dtype

        if input_ids is not None and inputs_embeds is None:
            inputs_embeds = self._token_embeds(input_ids).to(model_dtype)
        inputs_embeds = inputs_embeds.to(dtype=model_dtype)

        bop_id = self.tokenizer.convert_tokens_to_ids("<bop>")
        eop_id = self.tokenizer.convert_tokens_to_ids("<eop>")
        bop_embed = self._token_embeds(torch.tensor([bop_id], device=device)).squeeze(0)
        eop_embed = self._token_embeds(torch.tensor([eop_id], device=device)).squeeze(0)

        # process hiddens and build plan embeds per sample
        hidden_seq_list = [h.to(dtype=model_dtype) for h in hidden_seq_list]
        hidden_seq_list = self._process_hidden_list(hidden_seq_list)
        plan_embeds_list = []
        for plan_item in plan_ids:
            p = torch.tensor(plan_item, device=device)
            plan_embeds_list.append(self._token_embeds(p).to(model_dtype))

        # choose per-sample mix ratios and splice sequences
        B = inputs_embeds.size(0)
        new_embeds, new_mask, new_labels = [], [], []
        for i in range(B):
            ins = int(human_end_positions[i].item())
            if ins < 0:
                new_embeds.append(inputs_embeds[i])
                if attention_mask is not None:
                    new_mask.append(attention_mask[i])
                if labels is not None:
                    new_labels.append(labels[i])
                continue

            before = inputs_embeds[i, :ins]
            after = inputs_embeds[i, ins:]
            h = hidden_seq_list[i]
            p = plan_embeds_list[i]

            # simple segment mix: take a prefix from h and a suffix from p
            mix_ratio = random.choice([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
            h_len = h.size(0)
            p_len = p.size(0)
            h_part = h[: max(1, int(round(h_len * mix_ratio)))]
            p_part = p[min(p_len, int(round(p_len * mix_ratio))) :]
            mixed = torch.cat([h_part, p_part], dim=0)

            marked = torch.cat([bop_embed.unsqueeze(0), mixed, eop_embed.unsqueeze(0)], dim=0)
            cat_embeds = torch.cat([before, marked, after], dim=0)
            new_embeds.append(cat_embeds)

            if attention_mask is not None:
                new_mask.append(
                    torch.cat([
                        attention_mask[i, :ins],
                        torch.ones(marked.size(0), dtype=attention_mask.dtype, device=attention_mask.device),
                        attention_mask[i, ins:],
                    ], dim=0)
                )
            if labels is not None:
                new_labels.append(
                    torch.cat([
                        labels[i, :ins],
                        torch.full((marked.size(0),), IGNORE_TOKEN_ID, dtype=labels.dtype, device=labels.device),
                        labels[i, ins:],
                    ], dim=0)
                )

        out_embeds = pad_sequence(new_embeds, batch_first=True, padding_value=0)
        out_mask = pad_sequence(new_mask, batch_first=True, padding_value=0) if new_mask else None
        out_labels = pad_sequence(new_labels, batch_first=True, padding_value=IGNORE_TOKEN_ID) if new_labels else None

        return self.base_model(
            input_ids=None,
            attention_mask=out_mask,
            past_key_values=past_key_values,
            inputs_embeds=out_embeds,
            labels=out_labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )

    # ---------------------------------------------------------------- public fwd
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Any] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        plans: Optional[List[List[int]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        human_end_positions: Optional[torch.Tensor] = None,
        prepended_hidden_states: Optional[List[torch.Tensor]] = None,
        **kwargs: Any,
    ) -> Any:
        device = next(self.base_model.parameters()).device
        model_dtype = next(self.base_model.parameters()).dtype

        if input_ids is not None:
            input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        if inputs_embeds is not None:
            inputs_embeds = inputs_embeds.to(device)
        if labels is not None:
            labels = labels.to(device)
        if human_end_positions is not None:
            human_end_positions = human_end_positions.to(device)

        if input_ids is None and inputs_embeds is None:
            raise ValueError("Either input_ids or inputs_embeds must be provided.")
        if prepended_hidden_states is None:
            raise ValueError("prepended_hidden_states must be provided for insertion.")
        if plans is None:
            raise ValueError("plans must be provided (can be empty lists) for plan-related losses.")

        batch_size = input_ids.size(0) if input_ids is not None else inputs_embeds.size(0)
        hidden_lengths = [int(x.shape[0]) for x in prepended_hidden_states]
        plan_lengths = [int(len(x)) for x in plans]

        # main path: mix hidden states with plan embeddings (curriculum-like)
        normal_outputs = self._forward_with_hidden_and_plan_mix(
            input_ids,
            plans,
            attention_mask,
            inputs_embeds,
            labels,
            human_end_positions,
            prepended_hidden_states,
            past_key_values,
            use_cache,
            output_attentions,
            output_hidden_states,
            return_dict,
            **kwargs,
        )

        # plan-only tokens (teacher)
        with torch.no_grad():
            plan_pack = self.insert_plan_tokens(input_ids, attention_mask, labels, human_end_positions, plans)
            plan_outputs = self.base_model(
                input_ids=plan_pack["input_ids"],
                attention_mask=plan_pack["attention_mask"],
                labels=plan_pack["labels"],
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs,
            )

        # random hidden baseline
        with torch.no_grad():
            rand_hidden = self._generate_random_hidden(batch_size, device, model_dtype, hidden_lengths)
            random_outputs = self._forward_with_hidden_and_plan_mix(
                input_ids,
                plans,
                attention_mask,
                inputs_embeds,
                labels,
                human_end_positions,
                rand_hidden,
                past_key_values,
                use_cache,
                output_attentions,
                output_hidden_states,
                return_dict,
                **kwargs,
            )

        # losses
        if human_end_positions is not None:
            sim = self._plan_similarity_loss(
                normal_outputs.logits, plan_outputs.logits, human_end_positions, hidden_lengths, plan_lengths
            )
            ctr = self._random_contrast_loss(
                normal_outputs.logits, random_outputs.logits, human_end_positions, hidden_lengths
            )
            normal_outputs.loss = normal_outputs.loss + self.plan_similarity_weight * sim + self.random_contrast_weight * ctr

        return normal_outputs

    # ------------------------ delegate common helpers to the base model
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs: Optional[Dict[str, Any]] = None):
        return self.base_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    def get_input_embeddings(self):
        return self.base_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        return self.base_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.base_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        return self.base_model.set_output_embeddings(new_embeddings)

    def resize_token_embeddings(self, new_num_tokens: int):
        return self.base_model.resize_token_embeddings(new_num_tokens)

    def prepare_inputs_for_generation(self, *args, **kwargs):
        return self.base_model.prepare_inputs_for_generation(*args, **kwargs)

    @torch.no_grad()
    def generate(self, *args, **kwargs):
        return self.base_model.generate(*args, **kwargs)

    def save_pretrained(self, save_directory: str, **kwargs: Any) -> None:
        os.makedirs(save_directory, exist_ok=True)
        self.base_model.save_pretrained(save_directory, **kwargs)
        if self.prepended_learnable:
            torch.save(self.default_prepended_hidden_state, os.path.join(save_directory, "default_prepended_hidden_state.pt"))
        mha_state = {
            "hidden_mha": self.hidden_mha.state_dict(),
            "pre_ln": self.pre_ln.state_dict(),
            "post_ln": self.post_ln.state_dict(),
            "adaptive_proj": self.adaptive_proj.state_dict(),
        }
        torch.save(mha_state, os.path.join(save_directory, "hidden_mha_state.pt"))
        cfg = {
            "prepended_length": self.prepended_length,
            "prepended_learnable": self.prepended_learnable,
            "hidden_size": int(self.default_prepended_hidden_state.shape[-1]),
            "mha_num_heads": int(self.hidden_mha.num_heads),
            "plan_similarity_weight": float(self.plan_similarity_weight),
            "random_contrast_weight": float(self.random_contrast_weight),
        }
        with open(os.path.join(save_directory, "prepended_config.json"), "w") as f:
            json.dump(cfg, f, indent=2)

    def verify_gradient_requirements(self):
        mha_total = 0
        mha_requires_grad = 0
        for name, param in self.hidden_mha.named_parameters():
            mha_total += 1
            if param.requires_grad:
                mha_requires_grad += 1
            print(f"  MHA.{name}: requires_grad={param.requires_grad}, is_leaf={param.is_leaf}")

        proj_total = 0
        proj_requires_grad = 0
        for i, layer in enumerate(self.output_projection):
            if hasattr(layer, 'weight'):
                proj_total += 1
                if layer.weight.requires_grad:
                    proj_requires_grad += 1
                print(f"  Proj[{i}].weight: requires_grad={layer.weight.requires_grad}, is_leaf={layer.weight.is_leaf}")
            if hasattr(layer, 'bias') and layer.bias is not None:
                proj_total += 1
                if layer.bias.requires_grad:
                    proj_requires_grad += 1
                print(f"  Proj[{i}].bias: requires_grad={layer.bias.requires_grad}, is_leaf={layer.bias.is_leaf}")
