# Interactive Agent Runner (Anonymous Repo)

> **TL;DR**: This repo provides a clean, anonymized, Hugging Face–friendly script to run an LLM **agent–environment loop** with optional **latent (hidden-state) injection**. All paths/models are CLI-configurable; no secrets; no hard-coded local paths.

---

## Contents

* `run_interactive.py` — the main runner (cleaned, parameterized, HF-compatible)
* `configs/`

  * `task/*.json` — task & environment configs (example below)
  * `model/*.json` — agent configs (example below)
* Your project modules (expected):

  * `tasks/`, `agents/`, `envs/`, `utils/datatypes.py` (must provide `State`)

> The runner **does not** depend on any company/local path and **does not** require external logging. It prints to console and to `log.txt` in the output folder.

---

## Requirements

* Python **3.9+**
* `torch`, `datasets`, `tqdm`, `numpy`
* (optional) `colorama` for colored logs

Install:

```bash
pip install "torch>=2.1" "datasets>=2.18" tqdm numpy colorama
```

---

## Quick Start

1. Prepare configs (see minimal examples below) under:

```
configs/
  task/alfworld.json
  model/hidden.json
```

2. Prepare a **hidden-state dataset** (Hugging Face repo id or `load_from_disk` directory).
   Required fields per record:

* `task_id` (or `id` or `task`) → string key
* `hidden_state` → shape `[T, H]` numeric array (float32 convertible)
* `plan` (optional) → string

3. Run:

```bash
python run_interactive.py \
  --agent_model_path /PATH/TO/YOUR/AGENT_OR_MODEL \
  --hidden_dataset /PATH/TO/HIDDEN_STATES_DATASET \
  --hidden_split validation \
  --exp_path ./configs/task \
  --exp_config alfworld \
  --agent_path ./configs/model \
  --agent_config hidden \
  --output_root ./outputs \
  --split test \
  --verbose
```

Outputs go to:

```
outputs/<split>/<model_name_sanitized>/<exp_config+exp_name>/
  ├── log.txt
  └── <task_id>.json   # one file per episode
```

---

## Minimal Config Examples

### `configs/task/alfworld.json`

```json
{
  "env_config": {
    "env_class": "YourEnvClass",
    "max_steps": 50,
    "env_jar_path": "OPTIONAL_PATH_FOR_SCIWORLD_JAR"
  },
  "task": {
    "task_class": "YourTaskLoaderClass"
  }
}
```

* The runner will instantiate `getattr(envs, env_class)(task, **env_config)`.
* For custom envs (e.g., WebShop, SciWorld, TextCraft), wire the objects inside `run_interactive.py` or extend `envs`.

### `configs/model/hidden.json`

```json
{
  "agent_class": "LMAgent",
  "config": {
    "model_name": "YourAgentModelNameOrAlias",
    "max_new_tokens": 128,
    "temperature": 0.7
  }
}
```

* The runner calls:
  `agent = getattr(agents, agent_class)(agent_config["config"], args.agent_model_path)`

---

## Hidden-State Dataset (HF Standard)

You can pass either:

* a HF Hub dataset id (e.g., `yourname/hidden_states_xyz`)
* a local directory created via `datasets.DatasetDict.save_to_disk(...)`

The runner will load:

* Split: `--hidden_split` (e.g., `train`, `validation`, `test`, `dev`, `valid_seen`)
* Key field for lookup: `--hidden_key_field` (default `task_id`; fallbacks: `id`, `task`)

### Create a tiny dataset locally

```python
from datasets import Dataset, DatasetDict
import numpy as np

records = [
  {
    "task_id": "toy-1",
    "hidden_state": np.random.randn(32, 896).astype("float32").tolist(),
    "plan": "Example plan text here."
  }
]

ds = Dataset.from_list(records)
dset = DatasetDict({"validation": ds})
dset.save_to_disk("/tmp/hidden_states")
```

Then run with `--hidden_dataset /tmp/hidden_states --hidden_split validation`.

---

## What the Runner Does

* Loads **tasks** via your `tasks.Task` class (`load_tasks(split, part_num, part_idx)`).
* Initializes the **environment** from `envs` using `env_config["env_class"]`.
* Retrieves a **hidden\_state** per task from the dataset (by `task_id` by default).
* Runs an **agent loop**:

  1. `observation, state = env.reset(...)`
  2. `llm_output = agent(state.history, hidden_state)`
  3. `observation, state = env.step(llm_output)` until `state.finished`
* Saves each episode’s `state.to_dict()` to JSON and logs metrics.

---

## Useful CLI Flags

### Required

* `--agent_model_path` : model or checkpoint path/id consumed by your `agents` module
* `--hidden_dataset` : HF dataset id or local `load_from_disk` folder

### Common

* `--exp_path` / `--exp_config` : where to read the **task/env** config JSON
* `--agent_path` / `--agent_config` : where to read the **agent** config JSON
* `--split` : evaluation split for your task loader
* `--output_root` or `--output_path` : where results go
* `--verbose` : INFO logs; `--debug` : process \~10 tasks
* `--override` : ignore existing per-task JSONs and recompute

### Hidden-State Controls (optional)

* `--hidden_split` : dataset split to read (default `validation`)
* `--hidden_key_field` : which field in dataset to use as key (default `task_id`)
* `--hidden_lookup_field {auto,task_id,state_history_2}` : how to derive key at runtime (default `auto`)
* `--hidden_source {data,white_noise,noised,random_other,none}` : where the hidden comes from (default `data`)
* `--noise_std <float>` : std for Gaussian noise if `--hidden_source noised`
* `--prune_keep_ratio [0..1]` / `--prune_threshold <float>` : importance pruning (both off by default)
* `--prune_min_keep` / `--prune_rank` : pruning knobs (defaults: 8 / 16)
* `--truncate_last_percent [0..100)` : drop last N% tokens (off by default)
* `--restructure_history` : optional history compaction (off by default)

---

## Example Runs

**Baseline (use dataset hidden states)**

```bash
python run_interactive.py \
  --agent_model_path /models/my-agent \
  --hidden_dataset /data/hidden_states \
  --hidden_split validation \
  --exp_path ./configs/task --exp_config alfworld \
  --agent_path ./configs/model --agent_config hidden \
  --output_root ./outputs --split test --verbose
```

**White noise ablation**

```bash
python run_interactive.py ... --hidden_source white_noise
```

**Add Gaussian noise (σ=0.5)**

```bash
python run_interactive.py ... --hidden_source noised --noise_std 0.5
```

**Use hidden from another random task**

```bash
python run_interactive.py ... --hidden_source random_other
```

**Prune & truncate**

```bash
python run_interactive.py ... \
  --prune_keep_ratio 0.7 --truncate_last_percent 20
```

**Resume behavior**

* If you re-run without `--override`, existing `<task_id>.json` files are skipped.
* Use `--override` to recompute all.

---

## Output & Metrics

* Per-episode JSON: `<task_id>.json` (whatever `State.to_dict()` returns in your project)
* `log.txt`: console logs (observations, steps, agent outputs)
* Summary in logs:

  * `Success rate`
  * `Average reward` (if present in states)

---

## Reproducibility Tips

* Fix seeds in your **agent** if it samples stochastically.
* Keep environment versions consistent (e.g., `scienceworld`, custom env jars).
* When comparing ablations, keep **all** CLI flags constant except the ablation knobs.

---

## Troubleshooting

* **“No valid (key, hidden\_state) pairs found”**
  Check your dataset schema: one of `task_id|id|task` must be present and non-empty; `hidden_state` must be 2D numeric.

* **KeyError for a task**
  The runtime lookup key (by default `task.task_id`, fallback to `state.history[2]['content']`) must exist in the dataset.
  Fix with `--hidden_key_field` and/or `--hidden_lookup_field`.

* **Shape mismatch**
  Ensure `[T, H]` matches what your agent expects (H = hidden size). If using white noise or noise, shapes are derived from the dataset entry.

* **Env-specific imports**
  Some envs (e.g., SciWorld) require extra packages/files. Edit the small wiring block in `run_interactive.py` or handle it in `envs`.