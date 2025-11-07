import argparse
import numpy as np
import torch
from typing import Dict
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
import os
import datasets
import time
from tqdm import tqdm

# ===== Set global style consistent with academic papers (e.g., serif font) =====
def set_matplotlib_style():
    """Sets the global matplotlib style for plots."""
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 220,
        "savefig.dpi": 220,
        "font.family": "serif",
        "font.size": 18,
        "axes.labelsize": 18,
        "axes.titlesize": 20,
        "legend.fontsize": 18,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "axes.linewidth": 0.8,
        "lines.linewidth": 2.6,
        "lines.markersize": 7.5,
        "figure.autolayout": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "path.simplify": True,
        "path.simplify_threshold": 0.5,
        "agg.path.chunksize": 8000,
    })

# Apply the style as early as possible
set_matplotlib_style()

# (Recommendation: Set environment variables like OPENLM_TOKEN externally if needed, do not hardcode them here)

class HiddenStateLoader:
    """
    Loads hidden states from a Hugging Face dataset.
    Note: This class is preserved but currently unused in the main script in favor of `load_latents_any`.
    """
    def __init__(self, dataset_name, split):
        self.dataset_name = dataset_name
        self.split = split
        # Automatically load data upon initialization
        self._load_data()

    def _load_data(self):
        print(f"Loading tensor data from {self.dataset_name}")
        if self.split in ['valid_seen', 'dev']:
            self.dataset = datasets.load_dataset(self.dataset_name, split=datasets.Split.TEST)
        else:
            self.dataset = datasets.load_dataset(self.dataset_name, split=datasets.Split.VALIDATION)
        print(f"Loaded {len(self.dataset)} records.")

        def optimized_convert_nested_arrays_with_plan(df):
            """Optimized nested array conversion (following PyTorch recommendations) + includes plan text."""
            print(f"Optimized converting {len(df)} nested arrays with plan text...")
            start_time = time.time()

            def optimized_nested_convert(nested_array):
                """Optimized nested array conversion."""
                try:
                    # Per PyTorch recommendation: first convert to a single numpy array, then to a tensor
                    if isinstance(nested_array, np.ndarray) and nested_array.dtype == object:
                        list_data = nested_array.tolist()
                        numpy_array = np.array(list_data, dtype=np.float32)
                        return torch.from_numpy(numpy_array)
                    else:
                        return torch.from_numpy(nested_array.astype(np.float32))
                except Exception as e:
                    print(f"Conversion failed: {e}")
                    return None

            # Vectorized conversion of 'hidden_state' column in pandas
            df['tensor_hidden_state'] = df['hidden_state'].apply(optimized_nested_convert)

            # Check conversion results
            success_mask = df['tensor_hidden_state'].notna()
            success_count = success_mask.sum()
            print(f"Successfully converted: {success_count}/{len(df)} arrays")

            # Build a dictionary mapping task ID to hidden_state and plan
            valid_df = df[success_mask]
            id_to_data = {}

            for _, row in tqdm(valid_df.iterrows(), total=len(valid_df), desc="Building id_to_data map"):
                row['task'] = row['task'].replace('\n\n', '\n')
                id_to_data[row['task']] = {
                    'hidden_state': row['tensor_hidden_state'],
                    'plan': row['plan']
                }

            conversion_time = time.time() - start_time
            print(f"Optimized conversion completed in {conversion_time:.2f} seconds")

            # Validate results
            if id_to_data:
                sample_key = next(iter(id_to_data))
                sample_data = id_to_data[sample_key]
                print(f"Sample tensor shape: {sample_data['hidden_state'].shape}")
                print(f"Sample tensor dtype: {sample_data['hidden_state'].dtype}")
                print(f"Sample plan: {sample_data['plan'][:100]}...")  # Display first 100 characters

            return id_to_data

        # Show progress bar for the overall conversion step
        with tqdm(total=1, desc="Converting Dataset to Pandas") as pbar:
            df = self.dataset.to_pandas()
            pbar.update(1)

        self.id_to_data = optimized_convert_nested_arrays_with_plan(df)

    def get_hidden_state_and_plan(self, task_id):
        """Efficiently retrieves hidden state and plan by task ID."""
        if task_id not in self.id_to_data:
            raise KeyError(f"No hidden_state found for task_id: {task_id}")
        return self.id_to_data[task_id]['hidden_state'], self.id_to_data[task_id]['plan']


def build_Z_from_loader(loader, steps):
    """
    Extracts hidden states for all samples from a HiddenStateLoader to build Z[B, K, Hs].
    Note: Requires all hidden_states to have at least len(steps) steps.
    """
    id_to_data = loader.id_to_data
    all_hidden_states = []

    # Iterate in task order to maintain batch consistency
    for task_id, data in id_to_data.items():
        h = data['hidden_state']  # [K_total, Hs], from original model output
        K_total, Hs = h.shape

        # Truncate to specified steps (e.g., steps=[0,1,2])
        if max(steps) >= K_total:
            print(f"Warning: task {task_id} has only {K_total} steps, but need up to step {max(steps)}. Skipping.")
            continue

        selected_h = h[steps]  # [K_selected, Hs]
        all_hidden_states.append(selected_h.unsqueeze(0))  # [1, K, Hs]

    # Concatenate into a single tensor [B, K, Hs]
    Z = torch.cat(all_hidden_states, dim=0)
    B, K, Hs = Z.shape
    print(f"[loader] Built Z from loader: B={B}, K={K}, Hs={Hs}")
    return Z, K, Hs


# ========== Data Loading ==========
def load_latents(parquet_path):
    """Loads latents from a parquet file. Returns Z[B,K,Hs], and K, Hs metadata."""
    tbl = pq.read_table(parquet_path)
    K = int((tbl.schema.metadata or {}).get(b"K", b"0"))
    Hs = int((tbl.schema.metadata or {}).get(b"Hs", b"0"))
    latents_py = tbl["latents"].to_pylist()  # list of [K][Hs]
    Z = np.asarray(latents_py, dtype=np.float32)  # [B, K, Hs]
    return torch.from_numpy(Z), K, Hs

def load_latents_any(parquet_path: str, pad_value=np.nan):
    """
    Universal reader for parquet files.
    - Returns Z [B, K_max, Hs] (ragged arrays are padded in memory to K_max),
    - lengths [B] (true number of steps K_i for each sample),
    - Hs
    Note: If the file is already padded, Z is the padded array; if ragged, it's padded on the fly.
    """
    tbl = pq.read_table(parquet_path)
    meta = tbl.schema.metadata or {}
    Hs = int(meta.get(b"Hs", b"0"))
    padded_flag = meta.get(b"padded", b"1")  # b"1" for padded; b"0" for ragged

    latents_py = tbl["latents"].to_pylist()      # list of [K_i][Hs]
    lengths = np.array([len(x) for x in latents_py], dtype=np.int32)
    B = len(latents_py)
    K_max = int(max(lengths)) if B > 0 else 0

    if padded_flag == b"1":
        # Data in file is already [K_max][Hs]
        Z = np.asarray(latents_py, dtype=np.float32)  # [B, K_max, Hs]
    else:
        # Ragged: pad in memory
        Z = np.full((B, K_max, Hs), pad_value, dtype=np.float32)
        for i, x in enumerate(latents_py):
            if len(x) > 0:
                Z[i, :len(x), :] = np.asarray(x, dtype=np.float32)

    return torch.from_numpy(Z), torch.from_numpy(lengths), Hs


# ========== Basic Utilities ==========
def quantile_curve(values, q_points=None):
    """Computes the quantile curve for a 1D array."""
    if q_points is None:
        q_points = np.linspace(0, 100, 101)
    return np.percentile(values, q_points)


@torch.inference_mode()
def logits_from_latents(Z_step, lm_head):
    """Z_step: [B, Hs] -> logits: [B, V]"""
    B, Hs = Z_step.shape
    return lm_head(Z_step.view(B, Hs))


def renorm_within_topn(probs: torch.Tensor, topn: int) -> torch.Tensor:
    """
    Renormalizes probabilities within the top-N tokens for each sample.
    Returns: sorted_vals [B, topn] (sorted by probability, each row sums to 1).
    """
    topn = min(topn, probs.size(1))
    top_vals, _ = torch.topk(probs, k=topn, dim=1, largest=True, sorted=True)  # [B, topn]
    denom = top_vals.sum(dim=1, keepdim=True) + 1e-12
    top_vals = top_vals / denom
    return top_vals


def _load_state_any(ckpt_dir: str) -> Dict[str, torch.Tensor]:
    """Loads a checkpoint (supports safetensors / bin / pt)."""
    import glob
    from safetensors.torch import safe_open

    st_paths = sorted(glob.glob(os.path.join(ckpt_dir, "model*.safetensors")))
    if st_paths:
        state = {}
        for p in st_paths:
            with safe_open(p, framework="pt", device="cpu") as f:
                for k in f.keys():
                    state[k] = f.get_tensor(k)
        return state

    for name in ["pytorch_model.bin", "model.bin", "model.pt", "pytorch_model.pt"]:
        p = os.path.join(ckpt_dir, name)
        if os.path.isfile(p):
            return torch.load(p, map_location="cpu", weights_only=False)

    raise FileNotFoundError(f"No weights found in {ckpt_dir}")

def compute_topk_data_ragged_aware(Z, lengths, steps, topk_list, topn, lm_head, covN, device, dtype):
    """
    Same as compute_topk_data, but only uses samples where lengths > step for each step.
    This avoids including padded values (NaN/0) from ragged arrays in calculations.
    """
    import numpy as np
    import torch

    result = {}
    qx = np.linspace(0, 100, 101) / 100.0

    lengths_cpu = lengths.cpu().numpy()
    for step in steps:
        valid_mask = lengths_cpu > step
        if valid_mask.sum() == 0:
            # No samples are this long, skip
            continue

        Z_step = Z[valid_mask, step, :].to(device=device, dtype=dtype)  # [B_valid, Hs]
        logits = lm_head(Z_step).to(torch.float32)                      # [B_valid, V]
        probs  = torch.softmax(logits, dim=-1)

        # Coverage (P50)
        k_cov = min(covN, probs.size(1))
        top_cov, _ = torch.topk(probs, k=k_cov, dim=1, largest=True, sorted=True)
        S_cov = top_cov.sum(dim=1).float().detach().cpu().numpy()
        p50_cov = np.percentile(S_cov, 50)

        # Renormalize within Top-N
        topn_eff = min(topn, probs.size(1))
        top_vals, _ = torch.topk(probs, k=topn_eff, dim=1, largest=True, sorted=True)
        denom = top_vals.sum(dim=1, keepdim=True) + 1e-12
        sorted_vals = top_vals / denom  # [B_valid, topn_eff]

        # Collect top-k cumulative curves
        topk_data = {}
        for k in topk_list:
            kk = min(k, topn_eff)
            topk_sum = sorted_vals[:, :kk].sum(dim=1).float().cpu().numpy()
            qy = np.percentile(topk_sum, qx * 100)
            topk_data[k] = qy

        result[step] = {"topk_data": topk_data, "p50_cov": p50_cov}

    return result, qx


def compute_topk_data(Z, steps, topk_list, topn, lm_head, covN, device, dtype):
    """
    Computes top-k distribution data for a given set of latents.
    Returns: A dictionary containing top-k data and P50 coverage for each step.
    """
    result = {}
    qx = np.linspace(0, 100, 101) / 100.0

    for i, step in enumerate(steps):
        Z_step = Z[:, step, :].to(device, dtype=dtype)  # [B, Hs]
        logits = logits_from_latents(Z_step, lm_head).to(torch.float32)  # [B, V]
        probs = torch.softmax(logits, dim=-1)  # [B, V]

        # Calculate P50 coverage
        k_cov = min(covN, probs.size(1))
        top_cov, _ = torch.topk(probs, k=k_cov, dim=1, largest=True, sorted=True)
        S_cov = top_cov.sum(dim=1).float().cpu().numpy()
        p50_cov = np.percentile(S_cov, 50)

        # Renormalize within Top-N tokens
        sorted_vals = renorm_within_topn(probs, topn=topn)

        # Collect top-k data
        topk_data = {}
        for k in topk_list:
            kk = min(k, topn)
            topk_sum = sorted_vals[:, :kk].sum(dim=1).float().cpu().numpy()
            qy = quantile_curve(topk_sum, qx * 100)
            topk_data[k] = qy

        result[step] = {
            'topk_data': topk_data,
            'p50_cov': p50_cov
        }

    return result, qx


# === Color Gradient Utilities ===
def _hex_to_rgb(hx: str):
    """Converts a hex color string to an (R, G, B) tuple."""
    hx = hx.lstrip("#")
    return tuple(int(hx[i:i+2], 16) for i in (0, 2, 4))


def _rgb_to_hex(rgb):
    """Converts an (R, G, B) tuple to a hex color string."""
    return "#{:02X}{:02X}{:02X}".format(*rgb)


def gradient_hex(start_hex: str, end_hex: str, n: int):
    """Generates n evenly spaced gradient colors from start_hex to end_hex."""
    if n <= 1:
        return [end_hex]
    s = np.array(_hex_to_rgb(start_hex), dtype=float)
    e = np.array(_hex_to_rgb(end_hex), dtype=float)
    steps = []
    for t in np.linspace(0.0, 1.0, n):
        rgb = (1 - t) * s + t * e
        steps.append(_rgb_to_hex(tuple(int(round(x)) for x in rgb)))
    return steps


# ========== Main Execution ==========
def main():
    ap = argparse.ArgumentParser(description="Compare the Top-k distributions of two sets of hidden states.")
    
    # --- Required arguments for data and model paths ---
    ap.add_argument("--trained_latents_path", type=str, required=True,
                    help="Path to the .parquet file containing trained latents (e.g., /path/to/trained_hidden_states.parquet)")
    ap.add_argument("--original_latents_path", type=str, required=True,
                    help="Path to the .parquet file containing original hidden states (e.g., /path/to/original_latents.parquet)")
    ap.add_argument("--model_path", type=str, required=True,
                    help="Path to the base model directory (e.g., /path/to/Qwen2.5-7B-Base)")
    
    # --- Optional arguments ---
    ap.add_argument("--ckpt_dir", type=str, default=None,
                    help="Directory of the trained checkpoint to load weights from. If not provided, the base model is used for both datasets.")
    ap.add_argument("--use_ckpt_vocab", action="store_true", 
                    help="If the checkpoint vocabulary size differs, resize the model to match the checkpoint.")
    ap.add_argument("--tokenizer_path", type=str, default=None, 
                    help="Optional path to the tokenizer. If not provided, model_path is used.")
    
    # --- Analysis and Plotting parameters ---
    ap.add_argument("--steps", type=int, default=32, help="Number of steps to analyze.")
    ap.add_argument("--topk", type=int, default=6, help="Maximum Top-k to plot (e.g., 6 means Top-1 to Top-6).")
    ap.add_argument("--topn", type=int, default=10, help="Normalization is performed within the top N tokens.")
    ap.add_argument("--split", type=str, default="dev", choices=["valid_seen", "dev", "test"], 
                    help="Dataset split to use (relevant for the old HiddenStateLoader).")
    ap.add_argument("--covN", type=int, default=10, help="The 'N' value for calculating P50(S_N) coverage.")
    ap.add_argument("--out", type=str, default="comparison_topk.png", help="Output path for the comparison plot.")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = ap.parse_args()

    # Format parameters
    analysis_steps = list(range(0, args.steps))
    topk_list = list(range(1, args.topk + 1))

    # 1) Load trained latents from parquet
    print("=" * 50)
    print("Loading TRAINED data from parquet...")
    Z_trained, K_trained, Hs_trained = load_latents(args.trained_latents_path)
    print(f"[trained] B={Z_trained.shape[0]}, K={K_trained}, Hs={Hs_trained}")

    # 2) Load original latents from dataset
    print("=" * 50)
    print("Loading ORIGINAL data from dataset...")
    Z_original, lengths_original, Hs_original = load_latents_any(args.original_latents_path)
    K_original = int(lengths_original.max().item()) if lengths_original.numel() else 0
    print(f"[original] B={Z_original.shape[0]}, K_max={K_original}, Hs={Hs_original}")

    # Ensure dimensions are compatible
    if max(analysis_steps) >= min(K_trained, K_original):
        raise ValueError(f"Not enough steps: trained K={K_trained}, original K_max={K_original}, need steps up to {max(analysis_steps)}")

    # 3) Load base model
    print("=" * 50)
    print("Loading model...")
    tok_path = args.tokenizer_path or args.model_path
    _ = AutoTokenizer.from_pretrained(tok_path, use_fast=False, trust_remote_code=True)

    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    torch_dtype = torch.float16 if args.device.startswith("cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, config=config, trust_remote_code=True, torch_dtype=torch_dtype
    ).to(args.device).eval()

    # 4) Load trained weights for the 'trained' data if a checkpoint is provided
    trained_model = None
    if args.ckpt_dir:
        print(f"Loading trained checkpoint from {args.ckpt_dir}")
        try:
            # Create a copy of the model to load trained weights into
            trained_model = AutoModelForCausalLM.from_pretrained(
                args.model_path, config=config, trust_remote_code=True, torch_dtype=torch_dtype
            ).to(args.device).eval()

            state = _load_state_any(args.ckpt_dir)
            lm_head_weight = state.get("lm_head.weight")
            ckpt_vocab_size = lm_head_weight.shape[0] if lm_head_weight is not None else None
            current_vocab_size = trained_model.lm_head.weight.shape[0]

            if ckpt_vocab_size is not None and ckpt_vocab_size != current_vocab_size:
                if args.use_ckpt_vocab:
                    print(f"[vocab] resizing model to {ckpt_vocab_size}")
                    trained_model.resize_token_embeddings(ckpt_vocab_size)
                else:
                    # Discard mismatched weights
                    for k in list(state.keys()):
                        if ("lm_head.weight" in k or
                            "model.embed_tokens.weight" in k or
                            "transformer.wte.weight" in k):
                            state.pop(k, None)

            # Map state dict keys, removing common prefixes
            mapped = {}
            for k, v in state.items():
                k2 = k
                if k2.startswith("student_lm."):
                    k2 = k2[len("student_lm."):]
                if k2.startswith("model."):
                    k2 = k2[len("model."):]
                mapped[k2] = v

            missing, unexpected = trained_model.load_state_dict(mapped, strict=False)
            print(f"[ckpt] loaded: missing={len(missing)}, unexpected={len(unexpected)}")
        except Exception as e:
            print(f"[ckpt] failed to load: {e}. Falling back to the base model.")
            trained_model = model

    if trained_model is None:
        trained_model = model

    # 5) Compute top-k distributions for both datasets
    print("=" * 50)
    print("Computing top-k distributions...")
    dtype = model.lm_head.weight.dtype

    # Compute distribution for original data (using the base model)
    print("Computing ORIGINAL data distribution...")
    original_results, qx = compute_topk_data_ragged_aware(
        Z_original, lengths_original, analysis_steps, topk_list, args.topn,
        model.lm_head, args.covN, args.device, dtype
    )

    # Compute distribution for trained data (using the trained model)
    print("Computing TRAINED data distribution...")
    trained_results, _ = compute_topk_data(
        Z_trained, analysis_steps, topk_list, args.topn,
        trained_model.lm_head, args.covN, args.device, dtype
    )

    # 6) Plot the comparison
    print("=" * 50)
    print("Plotting comparison...")

    # === Use arXiv Gray (#7F7F7F) and Red (#B31B1B) gradients ===
    ks_sorted = sorted(set(topk_list))
    n_levels = len(ks_sorted)
    original_colors = gradient_hex("#F2F2F2", "#7F7F7F", n_levels)  # Gray gradient (light to dark)
    trained_colors  = gradient_hex("#F6D6D6", "#B31B1B", n_levels)  # Red gradient (light to dark)

    n_steps = len(analysis_steps)
    ncols = 4
    nrows = (n_steps + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), dpi=180)
    fig.patch.set_facecolor('white')

    if n_steps == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, step in enumerate(analysis_steps):
        ax = axes[i]
        
        # Check if data exists for this step
        if step not in original_results or step not in trained_results:
             ax.set_title(f"Step {step+1}\n(No data)", fontsize=14, pad=15)
             ax.text(0.5, 0.5, "Skipped", ha='center', va='center', fontsize=16, color='gray')
             continue

        orig_data = original_results[step]
        train_data = trained_results[step]

        # Plot layered fills for original and trained data
        prev_y_orig = np.zeros_like(qx)
        prev_y_train = np.zeros_like(qx)

        for j, k in enumerate(ks_sorted):
            # Original data (Gray gradient)
            qy_orig = orig_data['topk_data'][k]
            ax.fill_between(
                qx, prev_y_orig, qy_orig,
                color=original_colors[j], alpha=0.6,
                label=f'Original Top {k}' if i == 0 else "",
                edgecolor='white', linewidth=1
            )
            prev_y_orig = qy_orig.copy()

            # Trained data (Red gradient)
            qy_train = train_data['topk_data'][k]
            ax.fill_between(
                qx, prev_y_train, qy_train,
                color=trained_colors[j], alpha=0.6,
                label=f'Trained Top {k}' if i == 0 else "",
                edgecolor='white', linewidth=1
            )
            prev_y_train = qy_train.copy()

        # Add P50 coverage info
        orig_p50 = orig_data['p50_cov']
        train_p50 = train_data['p50_cov']
        label_txt = f"P50(S{args.covN})\nOrig: {orig_p50:.3f}\nTrained: {train_p50:.3f}"
        ax.text(
            0.98, 0.98, label_txt,
            transform=ax.transAxes, ha="right", va="top",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.75", alpha=0.9)
        )

        # Prettify axes and title
        ax.set_title(f"Step {step+1}", fontsize=14, pad=15)
        ax.set_xlabel("Percentile", fontsize=12)
        ax.set_ylabel("Value", fontsize=12)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.set_axisbelow(True)
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_xticklabels(['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'])
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1)
        ax.spines['bottom'].set_linewidth(1)

        # Add legend only to the first subplot
        if i == 0:
            import matplotlib.patches as patches
            legend_handles = []
            for j, k in enumerate(ks_sorted):
                legend_handles.append(patches.Patch(color=original_colors[j], alpha=0.6, label=f'Original Top {k}'))
            for j, k in enumerate(ks_sorted):
                legend_handles.append(patches.Patch(color=trained_colors[j], alpha=0.6, label=f'Trained Top {k}'))
            ax.legend(handles=legend_handles, loc='lower right', frameon=True, fancybox=True, shadow=True, fontsize=8, ncol=2)

    # Hide unused subplots
    for i in range(n_steps, len(axes)):
        fig.delaxes(axes[i])

    # Set overall title
    fig.suptitle(
        f"Comparison: Untrained vs Trained Hidden States\nTop-k Cumulative Distribution (Top-{args.topn} Normalization)",
        fontsize=16, y=1.02
    )

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)

    # Save the figure
    output_dir = os.path.dirname(args.out)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    fig.savefig(args.out, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"[saved] Output plot saved to {args.out}")
    plt.close()


if __name__ == "__main__":
    main()