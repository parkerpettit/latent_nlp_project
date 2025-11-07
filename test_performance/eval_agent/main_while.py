# filename: run_interactive.py
# Description:
#   Clean, anonymized, and reproducible interactive runner for your agent-env loop.
#   - No secrets or hardcoded local paths.
#   - All paths/models via CLI args.
#   - English logs & comments.
#   - HF Datasets standard loading for hidden states.
#   - Non-standard toggles (noise/prune/truncate) are exposed as args and OFF by default.

import os
import re
import json
import math
import time
import logging
import pathlib
import argparse
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

import torch
import datasets

# Optional: nice colored logs; skip if not installed
try:
    from colorama import Fore
except Exception:
    class _F:
        RED = GREEN = BLUE = YELLOW = RESET = ""
    Fore = _F()  # type: ignore

# Project-specific modules (kept as in your original repo structure)
import tasks as tasks
import agents as agents
import envs as envs
from utils.datatypes import State

logger = logging.getLogger("agent_frame")


# ----------------------------
# Hidden-state dataset loader
# ----------------------------
class HiddenStateLoader:
    """
    Load a HF dataset (hub repo id or local path via datasets.load_from_disk) that contains:
      - 'task_id' or 'id' or 'task' (used as key; configurable via --hidden_key_field)
      - 'hidden_state' (array-like, [T, D])
      - 'plan' (optional str)
    The loader builds a dict: key -> {'hidden_state': torch.FloatTensor[T, D], 'plan': str}
    """
    def __init__(self, dataset_name_or_path: str, split: str, key_field: str = "task_id"):
        self.dataset_name = dataset_name_or_path
        self.split = split
        self.key_field = key_field
        self.id_to_data: Dict[str, Dict[str, Any]] = {}
        self._load_data()

    def _canonical_split(self, s: str) -> str:
        s = s.lower()
        # Common synonyms
        if s in ["dev", "valid", "valid_seen", "validation_seen"]:
            return "validation"
        if s in ["test_seen", "eval"]:
            return "test"
        return s  # train | validation | test | custom

    def _load_data(self):
        print(f"[HiddenStateLoader] Loading from: {self.dataset_name} (split={self.split})")
        # Local directory => load_from_disk (ignores split)
        if os.path.isdir(self.dataset_name) and os.path.exists(os.path.join(self.dataset_name, "dataset_info.json")):
            ds = datasets.load_from_disk(self.dataset_name)
            # If the on-disk dataset is a DatasetDict, try the split; else treat as a single split
            if isinstance(ds, datasets.DatasetDict):
                split = self._canonical_split(self.split)
                if split in ds:
                    ds = ds[split]
                else:
                    raise ValueError(f"Split '{split}' not found in on-disk dataset. Available: {list(ds.keys())}")
        else:
            split = self._canonical_split(self.split)
            ds = datasets.load_dataset(self.dataset_name, split=split)

        print(f"[HiddenStateLoader] Loaded {len(ds)} records.")

        def to_tensor(hs) -> Optional[torch.Tensor]:
            try:
                if isinstance(hs, np.ndarray) and hs.dtype == object:
                    hs = np.array(hs.tolist(), dtype=np.float32)
                elif not isinstance(hs, np.ndarray):
                    hs = np.array(hs, dtype=np.float32)
                else:
                    hs = hs.astype(np.float32, copy=False)
                return torch.from_numpy(hs)
            except Exception as e:
                print(f"[HiddenStateLoader] hidden_state conversion failed: {e}")
                return None

        id_to_data: Dict[str, Dict[str, Any]] = {}
        for row in tqdm(ds, desc="Index hidden_state"):
            # Choose key by priority: explicit key_field → fallback fields
            key: Optional[str] = None
            for cand in [self.key_field, "task_id", "id", "task"]:
                if cand in row and row[cand] is not None and str(row[cand]).strip() != "":
                    key = str(row[cand]).replace("\n\n", "\n")
                    break
            if key is None:
                continue

            hs_t = to_tensor(row.get("hidden_state"))
            if hs_t is None:
                continue
            plan = row.get("plan", "")
            id_to_data[key] = {"hidden_state": hs_t, "plan": plan if isinstance(plan, str) else str(plan)}

        if not id_to_data:
            raise RuntimeError("No valid (key, hidden_state) pairs found in dataset.")
        self.id_to_data = id_to_data

        # Quick sample log
        sample_key = next(iter(self.id_to_data))
        sample = self.id_to_data[sample_key]
        print(f"[HiddenStateLoader] Example key: {sample_key[:60]!r}...")
        print(f"[HiddenStateLoader] Example hidden_state shape: {tuple(sample['hidden_state'].shape)}")

    def get_hidden_state_and_plan(self, key: str) -> Tuple[torch.Tensor, Optional[str]]:
        if key not in self.id_to_data:
            raise KeyError(f"No hidden_state found for key: {key!r}")
        d = self.id_to_data[key]
        return d["hidden_state"], d.get("plan", None)


# ----------------------------
# Hidden-state utilities (EN)
# ----------------------------
EN_STOPWORDS = {
    "a","an","the","in","on","at","to","for","of","from","by","with","as","about","into","through","over",
    "after","before","between","during","without","within","along","across","behind","beyond","under","above",
    "up","down","out","off","over","again","further","then","once","here","there","when","where","why","how",
    "all","any","both","each","few","more","most","other","some","such","no","nor","not","only","own","same",
    "so","than","too","very","can","will","just","don","should","now","and","or","but"
}

def _safe_l2norm_rows(H: torch.Tensor) -> torch.Tensor:
    if H.dim() != 2:
        raise ValueError(f"Expected [T, D], got {H.shape}")
    return torch.linalg.vector_norm(H, ord=2, dim=-1)  # [T]

def _svd_leverage_scores(H: torch.Tensor, rank: int = 16) -> torch.Tensor:
    T, D = H.shape
    r = min(rank, T, D)
    U, S, Vh = torch.linalg.svd(H.float(), full_matrices=False)
    U_r = U[:, :r]  # [T, r]
    lev = (U_r ** 2).sum(dim=1) / r
    return lev.to(H.dtype)

def _token_mask_from_plan(plan: Optional[str], T: int) -> torch.Tensor:
    """
    Coarse alignment: whitespace/token-level weighting from plan text.
    Stopwords down-weighted, punctuation lightly down-weighted.
    """
    if plan is None or len(plan.strip()) == 0:
        return torch.ones(T)
    toks = re.findall(r"[A-Za-z']+|\S", plan.lower())
    scores = []
    for tok in toks:
        if tok.isalpha():
            scores.append(0.3 if tok in EN_STOPWORDS else 1.0)
        else:
            scores.append(0.7)
    if len(scores) >= T:
        scores = scores[:T]
    else:
        scores = scores + [1.0] * (T - len(scores))
    return torch.tensor(scores)

def prune_hidden_state_by_importance(
    hidden_state: torch.Tensor,             # [T, D]
    plan_text: Optional[str] = None,
    keep_ratio: Optional[float] = 0.7,
    threshold: Optional[float] = None,
    min_keep: int = 8,
    leverage_rank: int = 16,
    weights: Tuple[float, float, float] = (0.6, 0.3, 0.1),
    return_mask: bool = False
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Compute per-row importance and filter while preserving original order.
    Score = w1 * z(norm) + w2 * z(leverage) + w3 * plan_mask.
    """
    if hidden_state.dim() != 2:
        raise ValueError(f"hidden_state must be [T, D], got {hidden_state.shape}")
    T, _ = hidden_state.shape
    if T <= min_keep:
        return (hidden_state, torch.ones(T, dtype=torch.bool) if return_mask else None)

    w1, w2, w3 = weights

    def _z(x: torch.Tensor) -> torch.Tensor:
        m = x.mean()
        s = x.std(unbiased=False)
        return (x - m) / (s + 1e-6)

    norms = _safe_l2norm_rows(hidden_state)       # [T]
    norms_z = _z(norms)
    lev = _svd_leverage_scores(hidden_state, rank=leverage_rank)
    lev_z = _z(lev)
    plan_mask = _token_mask_from_plan(plan_text, T).to(hidden_state.dtype)

    score = w1 * norms_z + w2 * lev_z + w3 * plan_mask  # [T]

    if threshold is not None:
        keep = score >= threshold
        if keep.sum().item() < min_keep:
            kr = 0.7 if keep_ratio is None else float(keep_ratio)
            k = max(min_keep, int(math.ceil(T * kr)))
            topk_idx = torch.topk(score, k=k, largest=True).indices
            keep = torch.zeros(T, dtype=torch.bool)
            keep[topk_idx] = True
    else:
        kr = 0.7 if keep_ratio is None else float(keep_ratio)
        k = max(min_keep, int(math.ceil(T * kr)))
        topk_idx = torch.topk(score, k=k, largest=True).indices
        keep = torch.zeros(T, dtype=torch.bool)
        keep[topk_idx] = True

    idx = torch.arange(T)[keep]
    pruned = hidden_state.index_select(dim=0, index=idx)
    return (pruned, keep if return_mask else None)

def remove_last_n_percent(hidden_state: torch.Tensor, n: float) -> torch.Tensor:
    """
    Drop last n% tokens along sequence dimension.
    Accepts [T, D] or [B, T, D].
    """
    if n <= 0:
        return hidden_state
    if n >= 100:
        raise ValueError("n should be < 100 to keep at least one token.")
    if hidden_state.dim() == 2:
        seq_len = hidden_state.size(0)
    elif hidden_state.dim() == 3:
        seq_len = hidden_state.size(1)
    else:
        raise ValueError(f"Unsupported hidden_state shape: {hidden_state.shape}")
    keep_len = max(1, int(seq_len * (1 - n / 100.0)))
    if hidden_state.dim() == 2:
        return hidden_state[:keep_len, :]
    return hidden_state[:, :keep_len, :]


# ----------------------------
# Main interactive loop
# ----------------------------
def _resolve_hidden_key(task_obj: tasks.Task, state: State, prefer_field: str = "auto") -> str:
    """
    Choose a lookup key for hidden_state:
      1) If prefer_field == 'task_id' and task_obj.task_id exists → use it
      2) If prefer_field == 'state_history_2' → use state.history[2]['content']
      3) If 'auto' → prefer task_id, fallback to state.history[2]['content']
    """
    key: Optional[str] = None
    if prefer_field == "task_id":
        key = getattr(task_obj, "task_id", None)
    elif prefer_field == "state_history_2":
        try:
            key = state.history[2]["content"]
        except Exception:
            key = None
    else:  # auto
        key = getattr(task_obj, "task_id", None)
        if not key:
            try:
                key = state.history[2]["content"]
            except Exception:
                key = None
    if not key or str(key).strip() == "":
        raise KeyError("Cannot resolve a hidden_state lookup key from task/state. "
                       "Try --hidden_lookup_field task_id or state_history_2.")
    return str(key)

def interactive_loop(
    task: tasks.Task,
    loader: HiddenStateLoader,
    agent: agents.LMAgent,
    env_config: Dict[str, Any],
    hidden_source: str = "data",   # data | white_noise | noised | random_other | none
    noise_std: float = 0.0,
    prune_keep_ratio: Optional[float] = None,
    prune_threshold: Optional[float] = None,
    prune_min_keep: int = 8,
    prune_rank: int = 16,
    truncate_last_percent: float = 0.0,
    hidden_lookup_field: str = "auto",
    restructure_history: bool = False
) -> State:
    """One episode run on a single task."""
    logger.info(f"Loading environment: {env_config.get('env_class', 'Unknown')}")
    env_cls = env_config["env_class"]
    env: envs.BaseEnv = getattr(envs, env_cls)(task, **env_config)

    game_file = getattr(task, "game_file", None)
    if game_file:
        observation, state = env.reset([game_file])
    else:
        observation, state = env.reset([])

    # ----- hidden_state selection -----
    key = _resolve_hidden_key(task, state, prefer_field=hidden_lookup_field)
    plan_text = None

    if hidden_source == "data":
        hidden_state, plan_text = loader.get_hidden_state_and_plan(key)
    elif hidden_source == "white_noise":
        # shape must match data hidden_state shape; infer from dataset
        data_hs, plan_text = loader.get_hidden_state_and_plan(key)
        hidden_state = torch.randn_like(data_hs)
    elif hidden_source == "noised":
        data_hs, plan_text = loader.get_hidden_state_and_plan(key)
        hidden_state = data_hs + torch.randn_like(data_hs) * float(noise_std)
    elif hidden_source == "random_other":
        # pick another random key (not current)
        all_keys = list(loader.id_to_data.keys())
        if len(all_keys) < 2:
            raise ValueError("random_other requires at least 2 items in hidden dataset.")
        import random as _random
        alt = key
        while alt == key:
            alt = _random.choice(all_keys)
        hidden_state = loader.id_to_data[alt]["hidden_state"]
        plan_text = loader.id_to_data[alt].get("plan", None)
    elif hidden_source == "none":
        hidden_state = torch.empty(0)  # up to agent to handle "no latent"
    else:
        raise ValueError(f"Unsupported hidden_source: {hidden_source}")

    # ----- optional pruning & truncation -----
    if hidden_state.numel() > 0 and hidden_state.dim() == 2:
        if prune_keep_ratio is not None or prune_threshold is not None:
            hidden_state, _ = prune_hidden_state_by_importance(
                hidden_state,
                plan_text=plan_text,
                keep_ratio=prune_keep_ratio,
                threshold=prune_threshold,
                min_keep=prune_min_keep,
                leverage_rank=prune_rank,
                return_mask=False
            )
        if truncate_last_percent and truncate_last_percent > 0:
            hidden_state = remove_last_n_percent(hidden_state, n=float(truncate_last_percent))

    print(f"[Hidden] final hidden_state shape: {tuple(hidden_state.shape)}")

    # ----- optional history restructuring (kept off by default) -----
    if restructure_history and isinstance(state.history, list) and len(state.history) >= 1:
        # Replace the very first user message with the current observation and
        # compact the first turn to only contain the latest observation.
        try:
            state.history[0]["content"] = observation
            # If there was an early assistant reply we want to drop, trim 2 entries
            if len(state.history) >= 3:
                del state.history[1]
                del state.history[1]
        except Exception:
            pass

    # ----- main loop -----
    logger.info(f"\n{Fore.YELLOW}{observation}{Fore.RESET}")
    cur_step = 1
    while not state.finished:
        logger.info(f"\n{Fore.RED}Step {cur_step}{Fore.RESET}\n")
        cur_step += 1
        try:
            llm_output: str = agent(state.history, hidden_state)
            print(f"llm_output: {llm_output}")
            logger.info(f"\n{Fore.GREEN}{llm_output}{Fore.RESET}\n")
        except Exception as e:
            logger.info(f"Agent failed with error: {e}")
            print(f"Agent failed with error: {e}")
            state.success = False
            state.finished = True
            state.terminate_reason = "agent_exception"
            break

        observation, state = env.step(llm_output)
        print(f"Observation: {observation}")
        if not state.finished:
            logger.info(f"\n{Fore.BLUE}{observation}{Fore.RESET}\n")

    # ----- summary -----
    if state.reward is not None:
        logger.info(f"Task finished in {state.steps} steps. Success: {state.success}. Reward: {state.reward}")
    else:
        logger.info(f"Task finished in {state.steps} steps. Success: {state.success}")

    return state


# ----------------------------
# One sweep over tasks
# ----------------------------
def run_single_iteration(args: argparse.Namespace, iteration: int):
    timestamp = int(time.time())

    with open(os.path.join(args.exp_path, f"{args.exp_config}.json"), "r", encoding="utf-8") as f:
        exp_config: Dict[str, Any] = json.load(f)
    with open(os.path.join(args.agent_path, f"{args.agent_config}.json"), "r", encoding="utf-8") as f:
        agent_config: Dict[str, Any] = json.load(f)

    # Optionally override model name in agent config
    if args.model_name:
        agent_config["config"]["model_name"] = args.model_name
        if args.append_timestamp:
            agent_config["config"]["model_name"] += f"-{timestamp}"

    # Output directory
    if args.output_path:
        output_path = args.output_path
    else:
        model_handle = agent_config["config"].get("model_name", "model")
        model_stub = str(model_handle).replace("/", "_")
        output_path = os.path.join(
            args.output_root, args.split, model_stub, args.exp_config + args.exp_name
        )
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)

    # Logging
    file_handler = logging.FileHandler(os.path.join(output_path, "log.txt"), mode="w", encoding="utf-8")
    logging.basicConfig(format="%(message)s", handlers=[logging.StreamHandler(), file_handler])

    env_config = exp_config["env_config"]
    logger.info(f"Experiment config: \n{json.dumps(exp_config, indent=2)}")

    # Optional: environment-specific wiring
    if env_config.get("env_class") == "WebShopEnv":
        from webshop.web_agent_site.envs import WebAgentTextEnv
        env_config["env"] = WebAgentTextEnv(observation_mode="text", human_goals=True)
    elif env_config.get("env_class") == "SciWorldEnv":
        from scienceworld import ScienceWorldEnv
        # If your repo has a monkey patch, import it here; otherwise remove next 2 lines.
        try:
            from eval_agent.utils.replace_sciworld_score import sciworld_monkey_patch
            sciworld_monkey_patch()
        except Exception:
            pass
        env_config["env"] = ScienceWorldEnv("", serverPath=os.path.join(os.getcwd(), env_config["env_jar_path"]), envStepLimit=200)
    elif env_config.get("env_class") == "TextCraftEnv":
        from eval_agent.envs.textcraft_env import TextCraft
        env_config["env"] = TextCraft(minecraft_dir="eval_agent/envs")

    # Tasks
    task_config: Dict[str, Any] = exp_config["task"]
    task_class: tasks.Task = getattr(tasks, task_config["task_class"])
    all_tasks, n_tasks = task_class.load_tasks(args.split, args.part_num, args.part_idx)

    # Hidden-state dataset (HF repo id or local load_from_disk path)
    loader = HiddenStateLoader(args.hidden_dataset, args.hidden_split, key_field=args.hidden_key_field)

    # Agent (your agents module should accept config + optional checkpoint path)
    agent: agents.LMAgent = getattr(agents, agent_config["agent_class"])(
        agent_config["config"],
        args.agent_model_path  # may be HF id or local ckpt dir depending on your agent implementation
    )

    # Resume support: collect done task ids
    state_list: List[State] = []
    done_task_id: List[str] = []
    if os.path.exists(output_path) and not args.override:
        for file in os.listdir(output_path):
            if not file.endswith("json"):
                continue
            try:
                with open(os.path.join(output_path, file), "r", encoding="utf-8") as fh:
                    state = State.load_json(json.load(fh))
                state_list.append(state)
                done_task_id.append(file.split(".")[0])
            except Exception:
                continue
        logger.info(f"Existing outputs found. {len(done_task_id)} tasks already done.")

    if len(done_task_id) == n_tasks:
        logger.info("All tasks done. Exiting.")
        _report_and_log_metrics(state_list)
        return

    # Main loop over remaining tasks
    logging.info(f"Running interactive loop for {n_tasks - len(done_task_id)} tasks. Iteration {iteration}")
    with logging_redirect_tqdm():
        pbar = tqdm(total=(n_tasks - len(done_task_id)))
        for i, task in enumerate(all_tasks):
            if args.debug and i == 10:
                break
            if task.task_id in done_task_id or str(task.task_id) in done_task_id:
                continue

            state = interactive_loop(
                task=task,
                loader=loader,
                agent=agent,
                env_config=env_config,
                hidden_source=args.hidden_source,
                noise_std=args.noise_std,
                prune_keep_ratio=args.prune_keep_ratio,
                prune_threshold=args.prune_threshold,
                prune_min_keep=args.prune_min_keep,
                prune_rank=args.prune_rank,
                truncate_last_percent=args.truncate_last_percent,
                hidden_lookup_field=args.hidden_lookup_field,
                restructure_history=args.restructure_history
            )

            state_list.append(state)
            with open(os.path.join(output_path, f"{task.task_id}.json"), "w", encoding="utf-8") as fh:
                json.dump(state.to_dict(), fh, indent=4, ensure_ascii=False)

            pbar.update(1)
        pbar.close()

    logger.warning(f"Iteration {iteration} completed.")
    logger.warning(f"Output saved to {output_path}")
    _report_and_log_metrics(state_list)


def _report_and_log_metrics(state_list: List[State]) -> None:
    reward_list: List[float] = []
    success_list: List[bool] = []
    for st in state_list:
        if st.reward is not None:
            reward_list.append(st.reward)
        success_list.append(bool(st.success))

    if success_list:
        if reward_list:
            logger.warning(f"Average reward: {sum(reward_list)/len(success_list):.4f}")
        logger.warning(f"Success rate: {sum(success_list)/len(success_list):.4f}")


# ----------------------------
# CLI
# ----------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Run the interactive agent-environment loop (anonymized & standardized).")

    # ==== Required-ish paths (defaults point to repo-relative folders) ====
    p.add_argument("--exp_path", type=str, default="./configs/task", help="Directory containing experiment jsons.")
    p.add_argument("--exp_config", type=str, default="alfworld", help="Experiment json filename (without .json).")
    p.add_argument("--agent_path", type=str, default="./configs/model", help="Directory containing agent jsons.")
    p.add_argument("--agent_config", type=str, default="hidden", help="Agent json filename (without .json).")

    # The agent may need a model checkpoint or HF model id/path
    p.add_argument("--agent_model_path", type=str, required=True,
                   help="HF model id or local checkpoint path used by the agent implementation.")

    # ==== Hidden-state dataset (HF standard) ====
    p.add_argument("--hidden_dataset", type=str, required=True,
                   help="HF datasets repo id or local load_from_disk directory for hidden states.")
    p.add_argument("--hidden_split", type=str, default="validation",
                   help="Dataset split name (train/validation/test/dev/valid_seen...).")
    p.add_argument("--hidden_key_field", type=str, default="task_id",
                   help="Field name used as key in the hidden dataset (e.g., task_id | id | task).")
    p.add_argument("--hidden_lookup_field", type=str, default="auto",
                   choices=["auto", "task_id", "state_history_2"],
                   help="How to derive the lookup key at runtime (task attribute vs state history).")

    # ==== Experiment control ====
    p.add_argument("--split", type=str, default="test", help="Evaluation split for tasks loader (repo-specific).")
    p.add_argument("--part_num", type=int, default=1, help="Total parts to shard the evaluation.")
    p.add_argument("--part_idx", type=int, default=-1, help="Index of the current part (0-based) or -1 for all.")
    p.add_argument("--exp_name", type=str, default="", help="Optional experiment name suffix.")
    p.add_argument("--output_root", type=str, default="./outputs", help="Root folder for outputs.")
    p.add_argument("--output_path", type=str, default="", help="Explicit output folder (overrides output_root).")

    # ==== Hidden-state processing toggles (OFF by default) ====
    p.add_argument("--hidden_source", type=str, default="data",
                   choices=["data", "white_noise", "noised", "random_other", "none"],
                   help="Where to get the hidden_state for each task.")
    p.add_argument("--noise_std", type=float, default=0.0, help="Std for Gaussian noise if --hidden_source noised.")
    p.add_argument("--prune_keep_ratio", type=float, default=None,
                   help="If set: percentage [0,1] to keep after importance pruning.")
    p.add_argument("--prune_threshold", type=float, default=None, help="If set: absolute threshold for pruning score.")
    p.add_argument("--prune_min_keep", type=int, default=8, help="Minimum rows to keep after pruning.")
    p.add_argument("--prune_rank", type=int, default=16, help="SVD rank for leverage scores.")
    p.add_argument("--truncate_last_percent", type=float, default=0.0,
                   help="Drop the last N percent tokens (0-100) after pruning.")

    # ==== History shaping (OFF by default) ====
    p.add_argument("--restructure_history", action="store_true",
                   help="If set, compact the first turn to the latest observation (mimics your previous behavior).")

    # ==== Logging / run mode ====
    p.add_argument("--model_name", type=str, default="",
                   help="Override 'config.model_name' in agent config.")
    p.add_argument("--append_timestamp", action="store_true",
                   help="Append UNIX timestamp to --model_name when overriding.")
    p.add_argument("--verbose", action="store_true", help="Set logger to INFO.")
    p.add_argument("--debug", action="store_true", help="Process only ~10 tasks.")
    p.add_argument("--override", action="store_true", help="Ignore already done tasks in output folder and recompute.")

    # ==== Looping ====
    p.add_argument("--loop", action="true", default=False, help=argparse.SUPPRESS)  # hidden flag (off by default)
    p.add_argument("--loop_sleep", type=int, default=10, help="Seconds to sleep between iterations if --loop is set.")
    return p


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    # Logging level
    if args.verbose:
        logger.setLevel(logging.INFO)
    elif args.debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.WARNING)

    # Single pass (default). If you want periodic runs, set --loop via code or env.
    iteration = 1
    while True:
        try:
            print(f"\nStarting iteration {iteration} at {time.strftime('%Y-%m-%d %H:%M:%S')}")
            run_single_iteration(args, iteration)
            if not getattr(args, "loop", False):
                break
            iteration += 1
            time.sleep(int(args.loop_sleep))
        except KeyboardInterrupt:
            print("\nReceived keyboard interrupt. Stopping.")
            break
        except Exception as e:
            print(f"Error in iteration {iteration}: {str(e)}")
            time.sleep(5)  # brief backoff


if __name__ == "__main__":
    main()
