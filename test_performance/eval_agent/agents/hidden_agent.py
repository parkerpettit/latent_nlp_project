import json
import time
import logging
import gc
import warnings
from typing import List, Dict, Union, Any, Optional

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
import argparse
from fastchat.model.model_adapter import get_conversation_template

# --- Placeholder for external dependencies ---
# In a real repository, these would be in separate files.

class LMAgent:
    """Base class for a language model agent."""
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    def __call__(self, messages: List[Dict[str, str]], hidden_state: torch.Tensor) -> str:
        raise NotImplementedError("This method should be implemented by a subclass.")

class AlignBlock(nn.Module):
    """Placeholder for the AlignBlock module."""
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.dummy_layer = nn.Linear(1, 1) # Example layer
    
    def forward(self, *args, **kwargs):
        pass # Placeholder

# The ModelWithInsertedHiddenState class provided in the context would be placed here
# For brevity, assuming it's correctly defined in another file as in the original import.
# from .model_insert_hidden import ModelWithInsertedHiddenState 
# Let's use a simplified placeholder if it's not available.
class ModelWithInsertedHiddenState(nn.Module):
    """
    A wrapper for a base language model to handle the insertion of hidden states.
    NOTE: This is a simplified placeholder. In your actual repository, 
    you would use the full class definition.
    """
    def __init__(self, base_model, prepended_length, hidden_size, prepended_learnable=False, **kwargs):
        super().__init__()
        self.base_model = base_model
        self.config = base_model.config
        
        # Placeholder for custom layers
        self.pre_ln = nn.LayerNorm(hidden_size)
        self.hidden_mha = nn.MultiheadAttention(hidden_size, 8, batch_first=True)
        self.post_ln = nn.LayerNorm(hidden_size)
        self.output_projection = nn.Linear(hidden_size, hidden_size)
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.output_scale = nn.Parameter(torch.tensor(1.0))

    def generate(self, *args, **kwargs):
        return self.base_model.generate(*args, **kwargs)

    def get_input_embeddings(self):
        return self.base_model.get_input_embeddings()

    def resize_token_embeddings(self, new_size: int):
        self.base_model.resize_token_embeddings(new_size)

# ---------------------------------------------

logger = logging.getLogger("agent_frame")

# --- Utility Functions ---

def _get_device_and_dtype() -> (str, torch.dtype):
    """
    Determines the appropriate device and dtype for PyTorch operations.
    Returns:
        A tuple of (device_string, torch_dtype).
    """
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    # Add MPS support for Apple Silicon if needed
    # elif torch.backends.mps.is_available():
    #     return "mps", torch.float16
    return "cpu", torch.float32

def _get_attn_implementation(device: str) -> Optional[str]:
    """
    Returns the attention implementation to use based on the device.
    Flash Attention 2 is only safely enabled for CUDA devices.
    """
    return "flash_attention_2" if device.startswith("cuda") else None

def _clean_state_dict_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Removes common prefixes like 'module.' and 'model.' from state dictionary keys
    to ensure compatibility with weights saved by trainers like DDP.
    """
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[7:]
        if key.startswith("model."):
            key = key[6:]
        new_state_dict[key] = value
    return new_state_dict

def _detect_model_source_type(path_or_id: str) -> str:
    """
    Detects the format of the model source.
    Returns:
      - 'hf_export': A directory containing custom module weights (e.g., 'hidden_mha_state.pt').
      - 'trainer_ckpt': A trainer checkpoint directory where 'pytorch_model.bin' contains the full wrapper state_dict.
      - 'base_only': A standard Hugging Face model directory or Hub ID.
    """
    if os.path.isdir(path_or_id):
        # Check for custom weights file, indicating a full HF export
        if os.path.exists(os.path.join(path_or_id, "hidden_mha_state.pt")):
            return "hf_export"

        # Check for a trainer checkpoint file
        pytorch_model_path = os.path.join(path_or_id, "pytorch_model.bin")
        if os.path.exists(pytorch_model_path):
            try:
                # Read only the keys to avoid high memory usage
                state_dict = torch.load(pytorch_model_path, map_location="cpu")
                if isinstance(state_dict, dict) and "state_dict" in state_dict:
                    state_dict = state_dict["state_dict"]
                # Check for keys unique to the custom wrapper model
                if any("hidden_mha" in k for k in state_dict.keys()):
                    return "trainer_ckpt"
            except Exception as e:
                warnings.warn(f"Could not inspect checkpoint file at {pytorch_model_path}: {e}")
                pass

        # Otherwise, assume it's a base model directory
        return "base_only"

    # If not a directory, assume it's a Hugging Face Hub identifier
    return "base_only"

def _ensure_special_tokens(tokenizer, model, tokens: List[str]):
    """
    Adds special tokens to the tokenizer and resizes model embeddings if they don't exist.
    """
    tokens_to_add = [token for token in tokens if token not in tokenizer.get_vocab()]
    if tokens_to_add:
        tokenizer.add_special_tokens({"additional_special_tokens": tokens_to_add})
        model.resize_token_embeddings(len(tokenizer))
        print(f"Added special tokens {tokens_to_add} and resized model embeddings.")

def _load_tokenizer_with_fallback(dir_or_id: str, config: Optional[AutoConfig]) -> AutoTokenizer:
    """
    Loads a tokenizer with a fallback mechanism.
    It first tries the provided path, then the model name from the config, and finally a default hub ID.
    """
    # 1. Try the primary path/ID
    try:
        return AutoTokenizer.from_pretrained(dir_or_id, trust_remote_code=True, use_fast=False)
    except Exception:
        pass

    # 2. Try the name from the model config
    base_name = getattr(config, "_name_or_path", None) or getattr(config, "name_or_path", None)
    if base_name:
        try:
            return AutoTokenizer.from_pretrained(base_name, trust_remote_code=True, use_fast=False)
        except Exception:
            pass

    # 3. Use a final fallback model (can be changed if necessary)
    fallback_id = "Qwen/Qwen2.5-0.5B-Instruct"
    try:
        return AutoTokenizer.from_pretrained(fallback_id, trust_remote_code=True, use_fast=False)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load tokenizer. Attempts were made with:\n"
            f"  - Primary path: {dir_or_id}\n"
            f"  - Config path: {base_name}\n"
            f"  - Fallback: {fallback_id}\n"
            f"Original error: {e}"
        )

def get_module_size_info(module: nn.Module, dtype: torch.dtype = torch.float32) -> str:
    """Calculates and formats the parameter count and memory size of a module."""
    param_count = sum(p.numel() for p in module.parameters() if p.requires_grad)
    bytes_per_param = torch.finfo(dtype).bits // 8
    size_mb = param_count * bytes_per_param / (1024 * 1024)
    return f"{param_count:,} parameters ({size_mb:.2f} MB)"

# --- Unified Model Loader ---

def load_custom_model_and_tokenizer(model_path: str) -> (ModelWithInsertedHiddenState, AutoTokenizer):
    """
    A unified entry point to load the model and tokenizer, automatically detecting the source format.
    
    Args:
        model_path (str): Path to the model directory or a Hugging Face Hub identifier.
    
    Returns:
        A tuple containing the loaded model and tokenizer.
    """
    source_type = _detect_model_source_type(model_path)
    device, torch_dtype = _get_device_and_dtype()
    attn_impl = _get_attn_implementation(device)

    print(f"[Model Loader] Source type: {source_type}, Device: {device}, DType: {torch_dtype}")

    if source_type == "hf_export":
        # Case 1: Load from a full Hugging Face-style export with custom layers saved separately.
        print("Loading from a standard Hugging Face export with custom modules...")
        
        base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            attn_implementation=attn_impl,
        ).to(device)

        tokenizer = _load_tokenizer_with_fallback(model_path, base_model.config)
        
        # Load custom model configuration
        prep_len = 800  # Default value
        config_path = os.path.join(model_path, "prepended_config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                custom_config = json.load(f)
            prep_len = int(custom_config.get("prepended_length", prep_len))

        model = ModelWithInsertedHiddenState(
            base_model=base_model,
            prepended_length=prep_len,
            hidden_size=base_model.config.hidden_size,
        ).to(device)
        
        # Load custom layer weights
        mha_state_path = os.path.join(model_path, "hidden_mha_state.pt")
        mha_state = torch.load(mha_state_path, map_location="cpu")

        model.hidden_mha.load_state_dict(mha_state["hidden_mha"])
        model.pre_ln.load_state_dict(mha_state["pre_ln"])
        model.post_ln.load_state_dict(mha_state["post_ln"])
        model.output_projection.load_state_dict(mha_state["output_projection"])

        # Load scalar parameters
        if "scale" in mha_state:
            model.scale.data.copy_(torch.tensor(mha_state["scale"], dtype=model.scale.dtype))
        if "output_scale" in mha_state:
            model.output_scale.data.copy_(torch.tensor(mha_state["output_scale"], dtype=model.output_scale.dtype))

    elif source_type == "trainer_ckpt":
        # Case 2: Load from a trainer checkpoint.
        print(f"Loading from a trainer checkpoint at: {model_path}")

        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        
        # Initialize base model from config
        base_model = AutoModelForCausalLM.from_config(config, trust_remote_code=True).to(torch_dtype)
        
        # Wrap the base model
        model = ModelWithInsertedHiddenState(
            base_model=base_model,
            prepended_length=800, # This should ideally be in the config
            hidden_size=base_model.config.hidden_size,
        ).to(device)

        # Load the entire state dict from the checkpoint
        state_dict_path = os.path.join(model_path, "pytorch_model.bin")
        raw_state_dict = torch.load(state_dict_path, map_location="cpu")
        state_dict = _clean_state_dict_prefix(raw_state_dict)
        
        # Resize embeddings before loading to avoid size mismatch
        embed_weight_key = "base_model.model.embed_tokens.weight"
        if embed_weight_key in state_dict:
             vocab_size = state_dict[embed_weight_key].shape[0]
             model.resize_token_embeddings(vocab_size)

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"Checkpoint loaded with {len(missing)} missing and {len(unexpected)} unexpected keys.")
        
        tokenizer = _load_tokenizer_with_fallback(model_path, config)

    else: # base_only
        # Case 3: Load a base model and use randomly initialized custom layers.
        print("Loading a base model. Custom layers will be randomly initialized.")
        base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            attn_implementation=attn_impl,
        ).to(device)

        model = ModelWithInsertedHiddenState(
            base_model=base_model,
            prepended_length=800,
            hidden_size=base_model.config.hidden_size,
        ).to(device)

        tokenizer = _load_tokenizer_with_fallback(model_path, base_model.config)
        
        warnings.warn("Custom modules (MHA, projections) are randomly initialized as no weights were found.")

    # Final setup for all cases
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    tokenizer.padding_side = "right"
    
    _ensure_special_tokens(tokenizer, model, ['<FIRST_HUMAN_END>', '<bop>', '<eop>'])
    
    # Print size info
    print("\n--- Model Size Information ---")
    print(f"Custom MHA Layer:          {get_module_size_info(model.hidden_mha, torch_dtype)}")
    print(f"Custom Projection Layer:   {get_module_size_info(model.output_projection, torch_dtype)}")
    print("-" * 30)

    # Clean up memory
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return model, tokenizer

def get_text_embedding(text: str, model, tokenizer, device, dtype) -> torch.Tensor:
    """Helper function to get the embedding for a given text string."""
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    embedding_layer = model.get_input_embeddings()
    return embedding_layer(input_ids).to(dtype)


class HiddenAgent(LMAgent):
    """
    An agent that communicates using hidden states instead of text.
    It processes user messages and a provided hidden state to generate a response.
    """

    def __init__(self, config: Dict[str, Any], model_path: str) -> None:
        super().__init__(config)
        self.model_name = config.get("model_name", "qwen")
        self.temperature = config.get("temperature", 0.7)
        self.max_new_tokens = config.get("max_new_tokens", 256)
        self.top_p = config.get("top_p", 0.95)
        
        self.device, self.dtype = _get_device_and_dtype()
        self.lm_model, self.tokenizer = load_custom_model_and_tokenizer(model_path)
        
        # Stores embeddings of the conversation history
        self.embedded_messages: List[Dict[str, Any]] = []

    def __call__(self, messages: List[Dict], hidden_state: torch.Tensor) -> str:
        """
        Generates a response based on the conversation history and a provided hidden state.
        
        Args:
            messages: A list of dictionaries representing the conversation, e.g., [{"role": "user", "content": "..."}].
            hidden_state: A tensor representing the latent information to be injected.
        
        Returns:
            The generated response string.
        """
        # --- 1. Process the input hidden state through custom layers ---
        hidden_state = hidden_state.to(self.device).to(self.dtype).unsqueeze(0)
        
        # This sequence should match the forward pass logic during training
        normed = self.lm_model.pre_ln(hidden_state)
        attn_output, _ = self.lm_model.hidden_mha(normed, normed, normed)
        attn_output = self.lm_model.post_ln(normed + attn_output)
        residual = attn_output * self.lm_model.scale
        attn_output = self.lm_model.output_projection(residual)
        processed_hidden_state = (residual + attn_output) * self.lm_model.output_scale

        # --- 2. Build the input embeddings from conversation history ---
        # If the new message history is shorter, reset the cache
        if len(messages) < len(self.embedded_messages):
            self.embedded_messages = []

        # Process new messages and convert them to embeddings
        for i, item in enumerate(messages):
            if i >= len(self.embedded_messages):
                role = item['role']
                content = item['content']
                
                # Apply model-specific chat template formatting
                if role == 'user':
                    if i == 0: # First user turn has a special format
                        prompt_text = (f'<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n'
                                       f'<|im_start|>user\n{content}'
                                       f'Now, you are given a step-by-step plan to complete this task as follow: <bop>')
                        current_embedding = get_text_embedding(prompt_text, self.lm_model, self.tokenizer, self.device, self.dtype)
                        
                        # Get embeddings for special tokens and assemble the final input
                        eop_embed = get_text_embedding('<eop>', self.lm_model, self.tokenizer, self.device, self.dtype)
                        end_embed = get_text_embedding('<|im_end|>\n', self.lm_model, self.tokenizer, self.device, self.dtype)
                        assistant_embed = get_text_embedding('<|im_start|>assistant\n', self.lm_model, self.tokenizer, self.device, self.dtype)
                        
                        # Concatenate: User Prompt + Hidden State + Special Tokens
                        current_embedding = torch.cat([current_embedding, processed_hidden_state, eop_embed, end_embed, assistant_embed], dim=1)
                    else:
                        prompt_text = f'<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n'
                        current_embedding = get_text_embedding(prompt_text, self.lm_model, self.tokenizer, self.device, self.dtype)
                
                elif role == 'assistant':
                    prompt_text = f'{content}<|im_end|>\n'
                    current_embedding = get_text_embedding(prompt_text, self.lm_model, self.tokenizer, self.device, self.dtype)

                self.embedded_messages.append({
                    "role": role,
                    "embedding": current_embedding
                })
        
        # Combine embeddings from all messages
        all_embeddings = [msg["embedding"] for msg in self.embedded_messages]
        combined_embedding = torch.cat(all_embeddings, dim=1)
        
        # Create the attention mask for the combined embedding
        attention_mask = torch.ones(combined_embedding.shape[:2], dtype=torch.long, device=self.device)

        # --- 3. Generate a response from the combined embeddings ---
        eos_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        gen_kwargs = dict(
            max_new_tokens=self.max_new_tokens,
            eos_token_id=eos_id if eos_id is not None else self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            temperature=self.temperature,
            top_p=self.top_p,
            do_sample=self.temperature > 0, # Do not sample if temperature is 0
        )

        with torch.no_grad():
            outputs = self.lm_model.generate(
                inputs_embeds=combined_embedding,
                attention_mask=attention_mask,
                **gen_kwargs
            )
        
        # Decode only the newly generated tokens
        input_length = combined_embedding.shape[1]
        generated_tokens = outputs[0, input_length:]
        response_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        print(f"Generated Response: {response_text}")
        return response_text
