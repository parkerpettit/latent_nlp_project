import os
import json
import logging
import pathlib
import argparse
from typing import List, Dict, Any
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from colorama import Fore
import torch
import datasets
import time
import numpy as np

import tasks as tasks
import agents as agents
import envs as envs
from utils.datatypes import State

# Remove hardcoded token - user should set this as environment variable
# os.environ["OPENLM_TOKEN"] = "YOUR_TOKEN_HERE"  # User should set this

# Model path should be provided by user through arguments
# MODEL = "/path/to/your/model"  # Removed hardcoded paths

logger = logging.getLogger("agent_frame")


class HiddenStateLoader:
    def __init__(self, dataset_name, split):
        self.dataset_name = dataset_name
        self.split = split

        # Automatically load data during initialization
        self._load_data()

    def _load_data(self):
        print(f"Loading tensor data from {self.dataset_name}")

        # Determine the split to use
        if self.split == 'valid_seen' or self.split == 'dev':
            dataset_split = datasets.Split.TEST
        else:
            dataset_split = datasets.Split.VALIDATION

        self.dataset = datasets.load_dataset(self.dataset_name, split=dataset_split)
        print(f"Loaded {len(self.dataset)} records.")

        def optimized_convert_nested_arrays_with_plan(df):
            """Optimized nested array conversion (following PyTorch recommendations) + include plan text"""

            print(f"Optimized converting {len(df)} nested arrays with plan text...")
            start_time = time.time()

            def optimized_nested_convert(nested_array):
                """Optimized nested array conversion"""
                try:
                    # Follow PyTorch recommendation: convert to numpy array first, then to tensor
                    if isinstance(nested_array, np.ndarray) and nested_array.dtype == object:
                        # Use numpy.array() to convert list to single numpy array
                        list_data = nested_array.tolist()
                        numpy_array = np.array(list_data, dtype=np.float32)
                        return torch.from_numpy(numpy_array)
                    else:
                        # If not object array, convert directly
                        return torch.from_numpy(numpy_array.astype(np.float32))

                except Exception as e:
                    print(f"Conversion failed: {e}")
                    return None

            # Pandas vectorized conversion for hidden_state
            df['tensor_hidden_state'] = df['hidden_state'].apply(optimized_nested_convert)

            # Check conversion results
            success_mask = df['tensor_hidden_state'].notna()
            success_count = success_mask.sum()

            print(f"Successfully converted: {success_count}/{len(df)} arrays")

            # Build dictionary containing hidden_state and plan
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
                print(f"Sample plan: {sample_data['plan'][:100]}...")  # Show first 100 characters

            return id_to_data

        # Show progress bar (as a single step)
        with tqdm(total=1, desc="Converting Dataset to Pandas") as pbar:
            df = self.dataset.to_pandas()
            pbar.update(1)

        self.id_to_data = optimized_convert_nested_arrays_with_plan(df)

    def get_hidden_state_and_plan(self, task_id):
        if task_id not in self.id_to_data:
            raise KeyError(f"No hidden_state found for task_id: {task_id}")
        return self.id_to_data[task_id]['hidden_state'], self.id_to_data[task_id]['plan']


def interactive_loop(
        task: tasks.Task,
        loader,
        agent: agents.LMAgent,
        env_config: Dict[str, Any],
) -> State:
    logger.info(f"Loading environment: {env_config['env_class']}")
    env: envs.BaseEnv = getattr(envs, env_config["env_class"])(task, **env_config)

    # Reset the environment and set the prompt
    game_file = getattr(task, 'game_file', None)
    if game_file:
        observation, state = env.reset([game_file])

    plan = loader.id_to_data[state.history[2]['content']]['plan']

    # Step 1: Merge content from indices 0 and 2
    merged_content = state.history[0]['content'] + "\n" + state.history[2][
        'content'] + '\nNow, you are given a step by step plan as follow: '

    plan = '<bop>' + plan + '<eop>'

    # Step 2: Replace first user content with merged content
    state.history[0]['content'] = merged_content + plan

    print(f"Instruction:{merged_content + plan}")

    # Step 3: Remove extra content
    del state.history[1]
    del state.history[1]

    init_msg = observation

    logger.info(f"\n{Fore.YELLOW}{init_msg}{Fore.RESET}")

    cur_step = 1
    while not state.finished:
        logger.info(f"\n{Fore.RED}Step {cur_step}{Fore.RESET}\n")
        cur_step += 1

        # Agent act
        try:
            llm_output: str = agent(state.history, plan)
            logger.info(f"\n{Fore.GREEN}{llm_output}{Fore.RESET}\n")
        except Exception as e:
            logger.info(f"Agent failed with error: {e}")
            state.success = False
            state.finished = True
            state.terminate_reason = "exceeding maximum input length"
            break

        # Environment step
        observation, state = env.step(llm_output)
        print(f"LLM output:{llm_output}")
        print(f"Observation:{observation}")

        # Color the state in blue
        if not state.finished:
            logger.info(f"\n{Fore.BLUE}{observation}{Fore.RESET}\n")

        if state.finished:
            break

    if state.reward is not None:
        logger.info(f"Task finished in {state.steps} steps. Success: {state.success}. Reward: {state.reward}")
    else:
        logger.info(f"Task finished in {state.steps} steps. Success: {state.success}")

    return state


def main(args: argparse.Namespace):
    # Load experiment configuration
    with open(os.path.join(args.exp_path, f"{args.exp_config}.json")) as f:
        exp_config: Dict[str, Any] = json.load(f)

    # Load agent configuration
    with open(os.path.join(args.agent_path, f"{args.agent_config}.json")) as f:
        agent_config: Dict[str, Any] = json.load(f)

    # Override model name if provided
    if args.model_name is not None:
        agent_config['config']['model_name'] = args.model_name

    print(agent_config)

    # Set output path
    if args.output_path == "":
        # Use model directory structure for output
        model_name_sanitized = agent_config['config']['model_name'].replace('/', '_')
        output_path = os.path.join(
            args.model_dir,
            args.exp_type,
            args.split,
            model_name_sanitized,
            args.exp_config + args.exp_name
        )
    else:
        output_path = args.output_path

    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)

    # Setup logging
    file_handler = logging.FileHandler(os.path.join(output_path, "log.txt"), mode='w')
    logging.basicConfig(
        format="%(message)s",
        handlers=[logging.StreamHandler(), file_handler],
    )

    env_config = exp_config["env_config"]

    logger.info(f"Experiment config: \n{json.dumps(exp_config, indent=2)}")

    # Environment setup based on env_class
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

    # Initialize all tasks
    task_config: Dict[str, Any] = exp_config["task"]
    task_class: tasks.Task = getattr(tasks, task_config["task_class"])
    all_tasks, n_tasks = task_class.load_tasks(args.split, args.part_num, args.part_idx)

    # Load hidden state data
    loader = HiddenStateLoader(args.dataset_path, args.split)

    # Initialize the agent
    agent: agents.LMAgent = getattr(agents, agent_config["agent_class"])(
        agent_config["config"],
        args.model_dir  # Pass model directory to agent
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

    # Run the loop for all tasks
    logging.info(f"Running interactive loop for {n_tasks} tasks.")
    n_todo_tasks = n_tasks - len(done_task_id)  # Only run remaining tasks

    with logging_redirect_tqdm():
        pbar = tqdm(total=n_todo_tasks)
        for i, task in enumerate(all_tasks):
            # Only test limited tasks in debug mode
            if args.debug and i == args.debug_max_tasks:
                break

            # Skip done tasks
            if task.task_id in done_task_id or str(task.task_id) in done_task_id:
                continue

            state = interactive_loop(task, loader, agent, env_config)
            state_list.append(state)

            # Save result
            json.dump(state.to_dict(), open(os.path.join(output_path, f"{task.task_id}.json"), 'w'), indent=4)
            pbar.update(1)

        pbar.close()

    logger.warning("All tasks done.")
    logger.warning(f"Output saved to {output_path}")

    # Calculate final metrics
    reward_list = []
    success_list = []
    for state in state_list:
        if state.reward is not None:
            reward_list.append(state.reward)
        success_list.append(state.success)

    if len(reward_list) != 0:
        logger.warning(f"Average reward: {sum(reward_list) / len(success_list):.4f}")
    logger.warning(f"Success rate: {sum(success_list) / len(success_list):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Run the interactive loop.")

    # Experiment configuration
    parser.add_argument("--exp_name", type=str, default="", help="Name of the experiment")
    parser.add_argument("--exp_path", type=str, required=True, help="Path to experiment configs")
    parser.add_argument("--exp_config", type=str, default="alfworld", help="Experiment config name")
    parser.add_argument("--exp_type", type=str, default="plan_only", help="Type of experiment")

    # Data configuration
    parser.add_argument("--split", type=str, default="test", help="Evaluation split")
    parser.add_argument("--part_num", type=int, default=1, help="Number of parts")
    parser.add_argument("--part_idx", type=int, default=-1, help="Part index")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to dataset")

    # Model configuration
    parser.add_argument("--agent_path", type=str, required=True, help="Path to agent configs")
    parser.add_argument("--agent_config", type=str, default="plan", help="Agent config name")
    parser.add_argument("--model_name", type=str, help="Model name (overrides agent config)")
    parser.add_argument("--model_dir", type=str, required=True, help="Directory containing model files")

    # Runtime options
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--debug_max_tasks", type=int, default=5, help="Max tasks to run in debug mode")
    parser.add_argument("--override", action="store_true", help="Ignore existing results")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    parser.add_argument("--output_path", type=str, default="", help="Custom output path")

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logger.setLevel(logging.INFO)
    elif args.debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.WARNING)

    main(args)