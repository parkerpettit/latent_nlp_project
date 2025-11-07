import os
import json
import logging
import pathlib
import argparse
import random
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
import torch
import datasets
import time
import numpy as np
import pandas as pd

import tasks as tasks
import agents as agents
import envs as envs
from utils.datatypes import State

# Set up logging
logger = logging.getLogger("agent_frame")


def build_text2latent(parquet_path: str, to_torch: bool = True, allow_duplicates: bool = True) -> Dict[str, Any]:
    """
    Build a mapping from text to latent vectors from a parquet file.

    Args:
        parquet_path: Path to the parquet file
        to_torch: Whether to convert to torch tensors
        allow_duplicates: Whether to allow duplicate texts

    Returns:
        Dictionary mapping text to latent vectors
    """
    # Read only the necessary columns
    df = pd.read_parquet(parquet_path, columns=["text", "latents"])

    if allow_duplicates:
        text2latent = {}
        for text, lat in zip(df["text"].tolist(), df["latents"].tolist()):
            arr = np.array(lat, dtype=np.float32)  # (K, Hs)
            val = torch.from_numpy(arr) if to_torch else arr
            text2latent.setdefault(text, []).append(val)  # Store multiple entries
        return text2latent
    else:
        # Assert text uniqueness
        counts = df["text"].value_counts()
        dup = counts[counts > 1]
        if not dup.empty:
            raise ValueError(
                f"Duplicate text detected (examples): {dup.index.tolist()[:3]}, total {int(dup.sum())} entries")
        return {t: (torch.from_numpy(np.array(lat, np.float32)) if to_torch else np.array(lat, np.float32))
                for t, lat in zip(df["text"], df["latents"])}


class HiddenStateLoader:
    def __init__(self, dataset_name: str, split: str):
        self.dataset_name = dataset_name
        self.split = split

        # Automatically load data during initialization
        self._load_data()

    def _load_data(self) -> None:
        """Load and process the dataset."""
        print(f"Loading tensor data from {self.dataset_name}")

        # Determine the split to use
        if self.split == 'valid_seen' or self.split == 'dev':
            dataset_split = datasets.Split.TEST
        else:
            dataset_split = datasets.Split.VALIDATION

        self.dataset = datasets.load_dataset(self.dataset_name, split=dataset_split)
        print(f"Loaded {len(self.dataset)} records.")

        # Convert to pandas and process
        with tqdm(total=1, desc="Converting Dataset to Pandas") as pbar:
            df = self.dataset.to_pandas()
            pbar.update(1)

        self.id_to_data = self._optimized_convert_nested_arrays_with_plan(df)

    def _optimized_convert_nested_arrays_with_plan(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """
        Optimized conversion of nested arrays with plan text.

        Args:
            df: Input DataFrame

        Returns:
            Dictionary mapping task IDs to hidden states and plans
        """
        print(f"Optimized converting {len(df)} nested arrays with plan text...")
        start_time = time.time()

        def optimized_nested_convert(nested_array: Any) -> Optional[torch.Tensor]:
            """Convert nested arrays to torch tensors."""
            try:
                if isinstance(nested_array, np.ndarray) and nested_array.dtype == object:
                    # Convert list to single numpy array
                    list_data = nested_array.tolist()
                    numpy_array = np.array(list_data, dtype=np.float32)
                    return torch.from_numpy(numpy_array)
                else:
                    # Direct conversion for non-object arrays
                    return torch.from_numpy(nested_array.astype(np.float32))
            except Exception as e:
                print(f"Conversion failed: {e}")
                return None

        # Apply conversion to hidden_state column
        df['tensor_hidden_state'] = df['hidden_state'].apply(optimized_nested_convert)

        # Check conversion results
        success_mask = df['tensor_hidden_state'].notna()
        success_count = success_mask.sum()

        print(f"Successfully converted: {success_count}/{len(df)} arrays")

        # Build dictionary with hidden_state and plan
        valid_df = df[success_mask]
        id_to_data = {}

        for _, row in tqdm(valid_df.iterrows(), total=len(valid_df), desc="Building id_to_data"):
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
            print(f"Sample plan: {sample_data['plan'][:100]}...")

        return id_to_data

    def get_hidden_state_and_plan(self, task_id: str) -> tuple:
        """
        Get hidden state and plan for a given task ID.

        Args:
            task_id: Task identifier

        Returns:
            Tuple of (hidden_state, plan)
        """
        if task_id not in self.id_to_data:
            raise KeyError(f"No hidden_state found for task_id: {task_id}")
        return self.id_to_data[task_id]['hidden_state'], self.id_to_data[task_id]['plan']


def remove_last_n_percent(hidden_state: torch.Tensor, n: float) -> torch.Tensor:
    """
    Remove the last n% of tokens from the hidden_state sequence dimension.

    Args:
        hidden_state: shape [seq_len, hidden_size] or [batch, seq_len, hidden_size]
        n: Percentage to remove (0 <= n <= 100)

    Returns:
        Truncated hidden_state
    """
    if n <= 0:
        return hidden_state
    if n >= 100:
        raise ValueError("n should be less than 100 to keep at least one token.")

    # Support [seq_len, hidden_size] or [batch, seq_len, hidden_size]
    if hidden_state.dim() == 2:
        seq_len = hidden_state.size(0)
    elif hidden_state.dim() == 3:
        seq_len = hidden_state.size(1)
    else:
        raise ValueError(f"Unsupported hidden_state shape: {hidden_state.shape}")

    # Calculate length to keep
    keep_len = int(seq_len * (1 - n / 100))
    keep_len = max(1, keep_len)  # Keep at least one token

    # Truncate
    if hidden_state.dim() == 2:
        return hidden_state[:keep_len, :]  # Keep first keep_len tokens
    else:
        return hidden_state[:, :keep_len, :]


def interactive_loop(
        task: tasks.Task,
        loader: Dict[str, Any],
        agent: agents.LMAgent,
        n_percent_to_remove: float,
        env_config: Dict[str, Any],
) -> State:
    """
    Run interactive loop for a single task.

    Args:
        task: Task object
        loader: Data loader
        agent: Language model agent
        n_percent_to_remove: Percentage of tokens to remove from end
        env_config: Environment configuration

    Returns:
        Final state after task completion
    """
    logger.info(f"Loading environment: {env_config['env_class']}")
    env: envs.BaseEnv = getattr(envs, env_config["env_class"])(task, **env_config)

    game_file = getattr(task, 'game_file', None)
    if game_file:
        observation, state = env.reset([game_file])

    # Get hidden state for the task
    hidden_state = loader[state.history[2]['content'].replace('\n', '\n\n')]
    hidden_state = torch.from_numpy(hidden_state)

    # Remove specified percentage of tokens from the end
    hidden_state = remove_last_n_percent(hidden_state, n=n_percent_to_remove)
    print(f"Truncated hidden_state shape: {hidden_state.shape}")

    # Merge content from history
    merged_content = state.history[0]['content'] + "\n" + state.history[2][
        'content'] + 'Now, you are given a step by step plan as follow: '

    # Update history
    state.history[0]['content'] = observation
    del state.history[1]
    del state.history[1]

    init_msg = observation
    logger.info(f"\n{init_msg}")

    cur_step = 1
    while not state.finished:
        logger.info(f"\nStep {cur_step}\n")
        cur_step += 1

        # Agent action
        try:
            llm_output: str = agent(state.history, hidden_state)
            print(f"LLM output: {llm_output}")
            logger.info(f"\n{llm_output}\n")
        except Exception as e:
            logger.info(f"Agent failed with error: {e}")
            print(f"Agent failed with error: {e}")
            state.success = False
            state.finished = True
            state.terminate_reason = "exceeding maximum input length"
            break

        # Environment step
        observation, state = env.step(llm_output)
        print(f"Observation: {observation}")

        if not state.finished:
            logger.info(f"\n{observation}\n")

        if state.finished:
            break

    if state.reward is not None:
        logger.info(
            f"Task finished in {state.steps} steps. Success: {state.success}. Reward: {state.reward}"
        )
    else:
        logger.info(
            f"Task finished in {state.steps} steps. Success: {state.success}"
        )

    return state


def run_single_iteration(args: argparse.Namespace, iteration: int) -> None:
    """
    Run a single iteration of evaluation.

    Args:
        args: Command line arguments
        iteration: Current iteration number
    """
    timestamp = int(time.time())

    # Load experiment configuration
    with open(os.path.join(args.exp_path, f"{args.exp_config}.json")) as f:
        exp_config: Dict[str, Any] = json.load(f)

    # Load agent configuration
    with open(os.path.join(args.agent_path, f"{args.agent_config}.json")) as f:
        agent_config: Dict[str, Any] = json.load(f)

    # Update model name if provided
    if args.model_name is not None:
        agent_config['config']['model_name'] = f"hidden/qwen-7b-base-hidden-{timestamp}"
    print(agent_config)

    n_percent_to_remove = 0

    # Set output path
    if args.output_path == "":
        output_dir = "hidden_compress_train_128_ablation/no_align"
        model_name = agent_config['config']['model_name'].replace('/', '_')
        output_path = os.path.join(output_dir, args.split, model_name, f"{args.exp_config}{args.exp_name}")
    else:
        output_path = args.output_path

    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)

    # Set up logging
    file_handler = logging.FileHandler(os.path.join(output_path, "log.txt"), mode='w')
    logging.basicConfig(
        format="%(message)s",
        handlers=[logging.StreamHandler(), file_handler],
    )

    env_config = exp_config["env_config"]
    logger.info(f"Experiment config: \n{json.dumps(exp_config, indent=2)}")

    # Initialize environment based on type
    if env_config['env_class'] == 'WebShopEnv':
        from webshop.web_agent_site.envs import WebAgentTextEnv
        env_config['env'] = WebAgentTextEnv(observation_mode="text", human_goals=True)
    elif env_config['env_class'] == 'SciWorldEnv':
        from scienceworld import ScienceWorldEnv
        from eval_agent.utils.replace_sciworld_score import sciworld_monkey_patch
        sciworld_monkey_patch()
        env_config['env'] = ScienceWorldEnv("", serverPath=os.path.join(os.getcwd(), env_config['env_jar_path']),
                                            envStepLimit=200)
    elif env_config['env_class'] == 'TextCraftEnv':
        from eval_agent.envs.textcraft_env import TextCraft
        env_config['env'] = TextCraft(minecraft_dir="eval_agent/envs")

    # Initialize tasks
    task_config: Dict[str, Any] = exp_config["task"]
    task_class: tasks.Task = getattr(tasks, task_config["task_class"])
    all_tasks, n_tasks = task_class.load_tasks(args.split, args.part_num, args.part_idx)

    # Initialize hidden state loader
    loader = HiddenStateLoader("recommend_gul_mdl/alfworld_test_data_qwen7B", args.split)

    # Initialize agent
    agent: agents.LMAgent = getattr(agents, agent_config["agent_class"])(
        agent_config["config"],
        args.model_path  # Use model path from arguments
    )

    state_list = []
    done_task_id = []

    # Check for existing results
    if os.path.exists(output_path) and not args.override:
        for file in os.listdir(output_path):
            if not file.endswith('json'):
                continue
            state = State.load_json(json.load(open(os.path.join(output_path, file))))
            state_list.append(state)
            done_task_id.append(file.split('.')[0])
        logger.info(f"Existing output file found. {len(done_task_id)} tasks done.")

    if len(done_task_id) == n_tasks:
        logger.info("All tasks done. Exiting.")
        # Calculate metrics
        reward_list = []
        success_list = []
        for state in state_list:
            if state.reward is not None:
                reward_list.append(state.reward)
            success_list.append(state.success)

        if len(reward_list) != 0:
            logger.warning(f"Average reward: {sum(reward_list) / len(success_list):.4f}")
        logger.warning(f"Success rate: {sum(success_list) / len(success_list):.4f}")
        return

    # Run evaluation loop
    logging.info(f"Running interactive loop for {n_tasks} tasks. Iteration {iteration}")
    n_todo_tasks = n_tasks - len(done_task_id)  # Only run remaining tasks

    with logging_redirect_tqdm():
        pbar = tqdm(total=n_todo_tasks)
        for i, task in enumerate(all_tasks):
            # Only test limited tasks in debug mode
            if args.debug and i == 5:
                break

            # Skip done tasks
            if task.task_id in done_task_id or str(task.task_id) in done_task_id:
                continue

            # Load latent data - user should provide this path
            if not hasattr(args, 'latent_path') or args.latent_path is None:
                raise ValueError("Please provide latent data path using --latent_path argument")

            parquet_path = args.latent_path

            # Load data efficiently
            latent_df = pd.read_parquet(parquet_path, columns=["id", "text", "latents"])
            latent_dict = {}

            for _, row in latent_df.iterrows():
                latent_dict[row['text']] = np.stack(row['latents'])

            state = interactive_loop(
                task, latent_dict, agent, n_percent_to_remove, env_config
            )

            state_list.append(state)
            json.dump(state.to_dict(), open(os.path.join(output_path, f"{task.task_id}.json"), 'w'), indent=4)

            pbar.update(1)
        pbar.close()

    logger.warning(f"Iteration {iteration} completed.")
    logger.warning(f"Output saved to {output_path}")

    # Calculate metrics
    reward_list = []
    success_list = []
    for state in state_list:
        if state.reward is not None:
            reward_list.append(state.reward)
        success_list.append(state.success)

    if len(reward_list) != 0:
        logger.warning(f"Average reward: {sum(reward_list) / len(success_list):.4f}")
    logger.warning(f"Success rate: {sum(success_list) / len(success_list):.4f}")


def main(args: argparse.Namespace) -> None:
    """
    Main function to run multiple iterations.

    Args:
        args: Command line arguments
    """
    iteration = 1
    while True:
        try:
            print(f"\nStarting iteration {iteration} at {time.strftime('%Y-%m-%d %H:%M:%S')}")
            run_single_iteration(args, iteration)
            iteration += 1
        except KeyboardInterrupt:
            print("\nReceived keyboard interrupt. Stopping the loop.")
            break
        except Exception as e:
            print(f"Error in iteration {iteration}: {str(e)}")
            time.sleep(10)  # Wait 10 seconds after error


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the interactive evaluation loop.")

    # Experiment configuration
    parser.add_argument(
        "--exp_name",
        type=str,
        default="",
        help="Name of the experiment.",
    )
    parser.add_argument(
        "--exp_path",
        type=str,
        required=True,
        help="Path to experiment configuration files.",
    )
    parser.add_argument(
        "--exp_config",
        type=str,
        default="alfworld",
        help="Experiment configuration name.",
    )

    # Evaluation settings
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Evaluation split (test/valid_seen/dev).",
    )
    parser.add_argument(
        "--part_num",
        type=int,
        default=1,
        help="Number of parts to split evaluation into.",
    )
    parser.add_argument(
        "--part_idx",
        type=int,
        default=-1,
        help="Index of the part to evaluate.",
    )

    # Agent configuration
    parser.add_argument(
        "--agent_path",
        type=str,
        required=True,
        help="Path to agent configuration files.",
    )
    parser.add_argument(
        "--agent_config",
        type=str,
        default="hidden",
        help="Agent configuration name.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="qwen-7b-base-hidden-5",
        help="Model name (overrides agent config).",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the model files.",
    )

    # Data paths
    parser.add_argument(
        "--latent_path",
        type=str,
        required=True,
        help="Path to the latent data parquet file.",
    )

    # Runtime options
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode (limit to 5 examples).",
    )
    parser.add_argument(
        "--override",
        action="store_true",
        help="Ignore existing results and override them.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode for demonstration.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="",
        help="Custom output path (default: auto-generated).",
    )

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logger.setLevel(logging.INFO)
    elif args.debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.WARNING)

    main(args)