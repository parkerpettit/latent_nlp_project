# -*- coding: utf-8 -*-
"""
PlanAgent (anonymous-ready)
- Pure Hugging Face transformers (no fastchat dependency)
- No hardcoded local paths
- English-only comments & logs
- User-defined links/models via config/env/CLI
"""

import os
import logging
from typing import List, Dict, Any, Optional

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger("agent_frame")
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))

# -----------------------------------------------------------------------------
# Optional base class: if your project exposes LMAgent, we'll inherit from it.
# Otherwise we fall back to a minimal no-op base class to keep this file portable.
# -----------------------------------------------------------------------------
try:
    from .base import LMAgent  # noqa: F401
except Exception:
    class LMAgent:  # minimal placeholder to keep API compatibility
        def __init__(self, config: Dict[str, Any]) -> None:
            self.config = config


# -----------------------------------------------------------------------------
# Wrapper that keeps a clear seam for (future) plan injection if you need it.
# In inference here we just forward to the underlying model.
# -----------------------------------------------------------------------------
class ModelWithPlanInjection(nn.Module):
    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base_model = base_model
        self.config = getattr(base_model, "config", None)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        plans=None,  # placeholder for future use
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        # Inference path: simply delegate to the underlying model.
        return self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )

    # Delegate common helpers to the base model
    def get_input_embeddings(self):
        return self.base_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.base_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.base_model.get_output_embeddings()

    def set_output_embeddings(self, value):
        self.base_model.set_output_embeddings(value)

    def resize_token_embeddings(self, new_num_tokens):
        return self.base_model.resize_token_embeddings(new_num_tokens)

    def gradient_checkpointing_enable(self, **kwargs):
        if hasattr(self.base_model, "gradient_checkpointing_enable"):
            self.base_model.gradient_checkpointing_enable(**kwargs)

    def prepare_inputs_for_generation(self, *args, **kwargs):
        return self.base_model.prepare_inputs_for_generation(*args, **kwargs)

    @torch.no_grad()
    def generate(self, *args, **kwargs):
        return self.base_model.generate(*args, **kwargs)

    def save_pretrained(self, save_directory, **kwargs):
        return self.base_model.save_pretrained(save_directory, **kwargs)


# -----------------------------------------------------------------------------
# Hugging Face-only loader
# -----------------------------------------------------------------------------
def load_model_and_tokenizer(
    model_id: str,
    dtype: Optional[torch.dtype] = torch.bfloat16,
    device_map: Optional[str] = "auto",
    trust_remote_code: bool = True,
):
    """
    Load a Causal LM and its tokenizer from Hugging Face Hub or a local HF directory.
    Users should set `model_id` to their own model link (HF repo id or local path).
    """
    logger.info(f"Loading model from: {model_id}")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map=device_map,           # requires accelerate for multi-GPU "auto"
        trust_remote_code=trust_remote_code,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        use_fast=False,                  # keep deterministic behavior
        trust_remote_code=trust_remote_code,
    )

    # If pad token is not set, fall back to eos.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = ModelWithPlanInjection(base_model)
    return model, tokenizer


# -----------------------------------------------------------------------------
# Minimal chat prompt builder (no fastchat). You can customize the roles/format.
# -----------------------------------------------------------------------------
def build_chat_prompt(messages: List[Dict[str, str]],
                      user_tag: str = "User",
                      assistant_tag: str = "Assistant") -> str:
    """
    Convert an array of {"role": "user"|"assistant", "content": "..."} into a plain prompt.
    This is intentionally simple and repo-agnostic for anonymous submission.
    """
    lines = []
    for m in messages:
        role = m.get("role", "")
        content = m.get("content", "")
        if role == "user":
            lines.append(f"{user_tag}: {content}")
        elif role == "assistant":
            lines.append(f"{assistant_tag}: {content}")
        else:
            # Unknown role: append as raw content to keep robustness.
            lines.append(content)
    # The final assistant cue:
    lines.append(f"{assistant_tag}:")
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# The agent
# -----------------------------------------------------------------------------
class PlanAgent(LMAgent):
    """
    A lightweight HF-based agent. By default, it does *not* modify generation
    with `plan`, but keeps the signature for future extensions.
    """

    def __init__(self, config: Dict[str, Any], MODEL: Optional[str] = None) -> None:
        super().__init__(config)

        # Model can be provided via:
        # 1) explicit `MODEL` argument,
        # 2) config["model_name"],
        # 3) environment variable MODEL_NAME,
        # 4) fallback public instruct model id placeholder (user should replace).
        self.model_name: str = (
            MODEL
            or config.get("model_name")
            or os.environ.get("MODEL_NAME")
            or "Qwen/Qwen2.5-7B-Instruct"     # <--- replace with your own HF repo if needed
        )
        logger.info(f"Using model: {self.model_name}")

        # Sampling/generation hyperparameters
        self.temperature: float = float(config.get("temperature", 0.8))
        self.top_p: float = float(config.get("top_p", 1.0))
        self.max_new_tokens: int = int(config.get("max_new_tokens", 100))
        self.do_sample: bool = bool(config.get("do_sample", True))

        # Device map: keep HF-accelerate "auto" when available;
        # otherwise place on a single device if possible.
        device_map = config.get("device_map", "auto" if torch.cuda.is_available() else None)
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        self.model, self.tokenizer = load_model_and_tokenizer(
            self.model_name, dtype=dtype, device_map=device_map
        )

    def __call__(self, messages: List[Dict[str, str]], plan: Optional[Any] = None) -> str:
        """
        Generate assistant response given a chat-like `messages` list.
        `plan` is reserved for future plan-aware decoding or logit biasing.
        """
        prompt = build_chat_prompt(messages)

        inputs = self.tokenizer(prompt, return_tensors="pt")
        # Move to the first available device of the model
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        gen_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "do_sample": self.do_sample,
            # Optionally add stop tokens if your tokenizer/model supports them:
            # "eos_token_id": self.tokenizer.eos_token_id,
        }

        input_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)

        # Decode only the newly generated tokens
        generated = self.tokenizer.decode(
            outputs[0][input_len:], skip_special_tokens=True
        )

        # Optional: trim by a simple stop cue if you added any
        # if "<|im_end|>" in generated:
        #     generated = generated.split("<|im_end|>", 1)[0]

        return generated.strip()


# -----------------------------------------------------------------------------
# (Optional) simple CLI for quick local tests (kept minimal for anonymous repos)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Minimal HF PlanAgent demo")
    parser.add_argument("--model_name", type=str, default=os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct"),
                        help="HF repo id or local HF directory")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--no_sample", action="store_true", help="Disable sampling (greedy decoding)")
    args = parser.parse_args()

    cfg = dict(
        model_name=args.model_name,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        do_sample=not args.no_sample,
    )

    agent = PlanAgent(cfg)
    demo_messages = [
        {"role": "user", "content": "Hello! Briefly introduce yourself."}
    ]
    print(agent(demo_messages, plan=None))
