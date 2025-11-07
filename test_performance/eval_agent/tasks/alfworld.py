import os
import json
import yaml
import logging
from typing import Iterable, Tuple, Iterator, Dict, Any, Optional

try:
    # Prefer your project's Task base if available
    from tasks.base import Task  # noqa: F401
except Exception:
    class Task:  # minimal fallback for anonymous submission
        task_id: int
        def __init__(self, **kwargs) -> None:
            for k, v in kwargs.items():
                setattr(self, k, v)

import alfworld
import alfworld.agents.environment as envs
from alfworld.agents.environment import get_environment

logger = logging.getLogger("agent_frame")

# Optional: action prefix aliases (kept for compatibility if you need them)
PREFIXES = {
    "pick_and_place": "put",
    "pick_clean_then_place": "clean",
    "pick_heat_then_place": "heat",
    "pick_cool_then_place": "cool",
    "look_at_obj": "examine",
    "pick_two_obj": "puttwo",
}


class AlfWorldTask(Task):
    task_name = "alfworld"

    def __init__(self, game_file: str, env, obs: str, **kwargs):
        super().__init__(**kwargs)
        self.game_file = game_file
        self.observation = obs
        self.env = env

    @staticmethod
    def _resolve_paths(
        split: str,
        config_path: Optional[str],
        dataset_root: Optional[str],
    ) -> Dict[str, Any]:
        """Load base YAML config and set dataset path for the requested split.

        Resolution order:
          - config_path argument
          - env var ALFWORLD_CONFIG
          - fallback: raise if not provided (keep anonymous + reproducible)

        Dataset root resolution:
          - dataset_root argument
          - env var ALFWORLD_DATA

        The split subdir names follow ALFWorld 2.1.1 convention:
          train -> train
          dev   -> valid_seen
          test  -> valid_unseen
        """
        split_map = {
            "train": "train",
            "dev": "valid_seen",
            "test": "valid_unseen",
        }
        if split not in split_map:
            raise ValueError(f"Unsupported split: {split} (choose from train/dev/test)")

        cfg_path = (
            config_path
            or os.environ.get("ALFWORLD_CONFIG")
        )
        if not cfg_path or not os.path.isfile(cfg_path):
            raise FileNotFoundError(
                "Base config YAML not provided. Set `config_path` argument or ALFWORLD_CONFIG env var."
            )

        with open(cfg_path, "r") as f:
            config = yaml.safe_load(f)

        root = dataset_root or os.environ.get("ALFWORLD_DATA")
        if not root:
            raise EnvironmentError(
                "Dataset root not provided. Set `dataset_root` argument or ALFWORLD_DATA env var."
            )

        sub = split_map[split]
        config.setdefault("dataset", {})
        config["dataset"]["data_path"] = os.path.join(root, "json_2.1.1", sub)
        return config

    @staticmethod
    def _build_env(config: Dict[str, Any], batch_size: int):
        env_type = config.get("env", {}).get("type")
        if not env_type:
            raise KeyError("Config must include env.type (e.g., 'AlfredTWEnv').")
        env_ctor = get_environment(env_type)
        env = env_ctor(config, train_eval="train")  # train/eval behavior is controlled by data_path
        env = env.init_env(batch_size=batch_size)
        return env

    @staticmethod
    def _infer_num_tasks(env) -> int:
        # Try the common attributes exposed by ALFWorld environments
        for key in ("num_games", "num_tasks"):
            if hasattr(env, key):
                val = getattr(env, key)
                if isinstance(val, int) and val > 0:
                    return val
        # Fallback heuristics
        if hasattr(env, "game_files") and isinstance(env.game_files, list):
            return len(env.game_files)
        if hasattr(env, "data") and isinstance(env.data, dict):
            for k in ("gamefiles", "games"):
                if k in env.data and isinstance(env.data[k], list):
                    return len(env.data[k])
        # If nothing is found, require the caller to shard manually.
        logger.warning("Could not infer number of tasks from environment; defaulting to 0.")
        return 0

    @classmethod
    def load_tasks(
        cls,
        split: str,
        part_num: int,
        part_idx: int = -1,
        batch_size: int = 1,
        *,
        config_path: Optional[str] = None,
        dataset_root: Optional[str] = None,
    ) -> Tuple[Iterable["Task"], int]:
        """Create a generator of AlfWorldTask and the number of tasks in this shard.

        Args:
            split: one of {"train", "dev", "test"}
            part_num: number of shards
            part_idx: current shard index (required when part_num > 1)
            batch_size: environment batch size
            config_path: path to base_config.yaml (user-provided)
            dataset_root: root dir that contains json_2.1.1/{train,valid_seen,valid_unseen}
        """
        config = cls._resolve_paths(split=split, config_path=config_path, dataset_root=dataset_root)
        env = cls._build_env(config, batch_size=batch_size)

        total = cls._infer_num_tasks(env)
        if part_num > 1:
            if part_idx < 0:
                raise ValueError("part_idx must be provided when part_num > 1")
            per = total // part_num + (1 if total % part_num else 0)
            skip = per * part_idx
            if hasattr(env, "skip"):
                env.skip(skip)
            count = max(0, min(per, total - skip))
            index_range = range(skip, skip + count)
        else:
            count = total
            index_range = range(total)

        def generator() -> Iterator["Task"]:
            for idx in index_range:
                obs, info = env.reset()
                # ALFWorld returns a list of observations; strip the instruction header if present
                first = obs[0] if isinstance(obs, (list, tuple)) and obs else obs
                text = "\n".join(str(first).split("\n\n")[1:]) if isinstance(first, str) else str(first)
                game_file = info.get("extra.gamefile", [None])[0] if isinstance(info, dict) else None
                yield cls(task_id=idx, game_file=game_file, env=env, obs=text)

        return generator(), count
