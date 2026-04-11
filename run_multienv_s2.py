"""Script to run Agent-S2 with multi-environment parallel execution.
Based on run_multienv_claude.py and run_multienv_uitars15 structure.
"""

import argparse
import datetime
import json
import logging
import os
import sys
import signal
import time
from typing import List
from multiprocessing import Process, Manager, current_process
import lib_run_single_s2 as lib_run_single
from desktop_env.attackable_env import AttackableDesktopEnv
from mm_agents.s2.agents.agent_s import AgentS2
from mm_agents.s2.agents.grounding import OSWorldACI

# Global variables for signal handling
active_environments = []
processes = []
is_terminating = False

# .env
from lib_env import load_env
# Force `envs/s2.env` to take precedence over the parent shell env to avoid
# accidentally using a different OPENAI_API_KEY / OPENAI_BASE_URL.
LOADED_ENV_PATH = load_env(__file__, "s2", override=True)


def _log_llm_env(logger_: logging.Logger, prefix: str) -> None:
    # Never log the actual API key. Only log presence + endpoint-ish vars.
    vars_to_show = [
        "OPENAI_BASE_URL",
        "OPENAI_API_BASE",
        "AZURE_OPENAI_ENDPOINT",
        "OPENAI_API_VERSION",
    ]
    shown = {k: os.getenv(k) for k in vars_to_show if os.getenv(k)}
    logger_.info("%s env_path=%s", prefix, LOADED_ENV_PATH)
    logger_.info("%s OPENAI_API_KEY_set=%s", prefix, bool(os.getenv("OPENAI_API_KEY")))
    if shown:
        logger_.info("%s endpoint_env=%s", prefix, shown)


def config() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Agent-S2 evaluation with parallel environments"
    )

    # environment config
    parser.add_argument("--path_to_vm", type=str, default=None)
    parser.add_argument(
        "--headless", action="store_true", help="Run in headless machine"
    )
    parser.add_argument(
        "--action_space", type=str, default="pyautogui", help="Action type"
    )
    parser.add_argument(
        "--observation_type",
        choices=["screenshot", "a11y_tree", "screenshot_a11y_tree", "som"],
        default="screenshot",
        help="Observation type",
    )
    parser.add_argument(
        "--provider_name", type=str, default="docker", 
        choices=["aws", "virtualbox", "vmware", "docker", "azure"], 
        help="Provider name"
    )
    parser.add_argument(
        "--client_password", type=str, default="password", help="password"
    )
    parser.add_argument(
        "--screen_width", type=int, default=1920, help="Screen width"
    )
    parser.add_argument(
        "--screen_height", type=int, default=1080, help="Screen height"
    )
    parser.add_argument("--sleep_after_execution", type=float, default=1.0)
    parser.add_argument("--max_steps", type=int, default=2)

    # agent config
    parser.add_argument("--max_trajectory_length", type=int, default=3)
    parser.add_argument(
        "--test_config_base_dir", type=str, default="evaluation_example_safety"
    )
    parser.add_argument(
        "--examples_subdir", type=str, default="examples",
        help="Subdirectory under test_config_base_dir containing task JSON files"
    )

    # lm config - planning model
    parser.add_argument("--model_provider", type=str, default="openai")
    parser.add_argument("--model", type=str, default="gpt-5")
    parser.add_argument("--model_url", type=str, default="")
    parser.add_argument("--model_api_key", type=str, default="")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_tokens", type=int, default=1500)
    parser.add_argument("--stop_token", type=str, default=None)

    # grounding model config
    # vllm / UI-TARS (original defaults, kept for reference):
    # parser.add_argument("--grounding_model_provider", type=str, default="vllm")
    # parser.add_argument("--grounding_model", type=str, default="ByteDance-Seed/UI-TARS-1.5-7B")
    # parser.add_argument("--grounding_model_resize_width", type=int, default=1920)
    # parser.add_argument("--grounding_model_resize_height", type=int, default=1080)

    parser.add_argument("--grounding_model_provider", type=str, default="anthropic")
    parser.add_argument(
        "--grounding_model", type=str, default="claude-sonnet-4-5-20250929"
    )
    parser.add_argument(
        "--grounding_model_resize_width",
        type=int,
        default=1366,
        help="Width of screenshot image after processor rescaling",
    )
    parser.add_argument(
        "--grounding_model_resize_height",
        type=int,
        default=None,
        help="Height of screenshot image after processor rescaling (None = auto from width)",
    )
    
    # custom grounding endpoint
    parser.add_argument("--endpoint_provider", type=str, default="")
    parser.add_argument("--endpoint_url", type=str, default="")
    parser.add_argument("--endpoint_api_key", type=str, default="")

    # knowledge base config
    parser.add_argument("--kb_name", default="kb_s2", type=str)
    parser.add_argument("--search_engine", type=str, default="Perplexica", choices=["LLM", "Perplexica"])

    # example config
    parser.add_argument("--domain", type=str, default="all")
    parser.add_argument(
        "--test_all_meta_path", type=str, default="evaluation_example_safety/test_all.json"
    )

    # logging related
    parser.add_argument("--result_dir", type=str, default="./results_s2_gpt5_claude45_round1")
    parser.add_argument(
        "--result_model_name", type=str, default="gpt-5",
        help="Override the model name used in the result directory path. "
             "If empty, falls back to --model. Useful when you want to write "
             "results under a fixed name (e.g. 'gpt-5') while actually running "
             "a different model (e.g. o3)."
    )
    parser.add_argument("--num_envs", type=int, default=30, help="Number of environments to run in parallel")
    parser.add_argument("--log_level", type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], 
                       default='INFO', help="Set the logging level")

    # aws config
    parser.add_argument(
        "--region", type=str, default="us-east-1", help="AWS region for the VM"
    )
    
    args = parser.parse_args()
    return args


args = config()  # Get command line arguments first

logger = logging.getLogger()
log_level = getattr(logging, args.log_level.upper())
logger.setLevel(log_level)

datetime_str: str = datetime.datetime.now().strftime("%Y%m%d@%H%M%S")

os.makedirs("logs", exist_ok=True)

file_handler = logging.FileHandler(
    os.path.join("logs", "normal-{:}.log".format(datetime_str)), encoding="utf-8"
)
debug_handler = logging.FileHandler(
    os.path.join("logs", "debug-{:}.log".format(datetime_str)), encoding="utf-8"
)
stdout_handler = logging.StreamHandler(sys.stdout)

file_handler.setLevel(logging.INFO)
debug_handler.setLevel(logging.DEBUG)
stdout_handler.setLevel(log_level)

formatter = logging.Formatter(
    fmt="\x1b[1;33m[%(asctime)s \x1b[31m%(levelname)s \x1b[32m%(module)s/%(lineno)d-%(processName)s\x1b[1;33m] \x1b[0m%(message)s"
)
file_handler.setFormatter(formatter)
debug_handler.setFormatter(formatter)
stdout_handler.setFormatter(formatter)

stdout_handler.addFilter(logging.Filter("desktopenv"))

logger.addHandler(file_handler)
logger.addHandler(debug_handler)
logger.addHandler(stdout_handler)

# Split logs by agent role (S2 has manager/worker; orchestrator uses desktopenv.agent.s2).
class _LoggerPrefixFilter(logging.Filter):
    def __init__(self, prefix: str):
        super().__init__()
        self._prefix = prefix

    def filter(self, record: logging.LogRecord) -> bool:
        return record.name.startswith(self._prefix)


for role, prefix in [
    ("manager", "desktopenv.agent.manager"),
    ("worker", "desktopenv.agent.worker"),
    ("s2", "desktopenv.agent.s2"),
]:
    role_handler = logging.FileHandler(
        os.path.join("logs", f"{role}-{datetime_str}.log"), encoding="utf-8"
    )
    role_handler.setLevel(logging.DEBUG)
    role_handler.setFormatter(formatter)
    role_handler.addFilter(_LoggerPrefixFilter(prefix))
    logger.addHandler(role_handler)

logger = logging.getLogger("desktopenv.experiment")


def distribute_tasks(test_all_meta: dict) -> List[tuple]:
    """Distribute tasks evenly across environments."""
    all_tasks = []
    for domain, examples in test_all_meta.items():
        for example_id in examples:
            all_tasks.append((domain, example_id))
    
    return all_tasks


def run_env_tasks(task_queue, args, shared_scores):
    """Run tasks for a single environment."""
    active_environments = []
    env = None
    agent = None
    
    try:
        _log_llm_env(logger, f"[{current_process().name}]")
        # Prepare engine parameters
        engine_params = {
            "engine_type": args.model_provider,
            "model": args.model,
            "base_url": args.model_url,
            "api_key": args.model_api_key,
        }

        # Prepare grounding engine parameters
        if args.endpoint_url:
            engine_params_for_grounding = {
                "engine_type": args.endpoint_provider,
                "base_url": args.endpoint_url,
                "api_key": args.endpoint_api_key,
            }
        else:
            grounding_height = args.grounding_model_resize_height
            if grounding_height is None:
                grounding_height = (
                    args.screen_height
                    * args.grounding_model_resize_width
                    / args.screen_width
                )

            engine_params_for_grounding = {
                "engine_type": args.grounding_model_provider,
                "model": args.grounding_model,
                "grounding_width": args.grounding_model_resize_width,
                "grounding_height": grounding_height,
            }

        # Initialize grounding agent
        grounding_agent = OSWorldACI(
            platform="linux",
            engine_params_for_generation=engine_params,
            engine_params_for_grounding=engine_params_for_grounding,
            width=args.screen_width,
            height=args.screen_height,
        )

        # Initialize Agent S2
        agent = AgentS2(
            engine_params,
            grounding_agent,
            platform="linux",
            action_space="pyautogui",
            observation_type="mixed",
            search_engine=args.search_engine,
            memory_root_path=os.getcwd(),
            memory_folder_name=args.kb_name,
            kb_release_tag="v0.2.2",
            embedding_engine_type="openai",
        )
        logger.info(
            "[%s] Agent roles: manager=%s worker=%s grounder=%s",
            current_process().name,
            type(getattr(agent, "planner", None)).__name__,
            type(getattr(agent, "executor", None)).__name__,
            type(getattr(agent, "grounding_agent", None)).__name__,
        )

        # Initialize environment
        screen_size = (args.screen_width, args.screen_height)
        
        env_kwargs = {
            "path_to_vm": args.path_to_vm,
            "action_space": agent.action_space,
            "provider_name": args.provider_name,
            "screen_size": screen_size,
            "headless": args.headless,
            "os_type": "Ubuntu",
            "require_a11y_tree": args.observation_type in ["a11y_tree", "screenshot_a11y_tree", "som"],
            "client_password": args.client_password,
        }
        
        # Add AWS-specific config if needed
        if args.provider_name == "aws":
            from desktop_env.providers.aws.manager import IMAGE_ID_MAP
            REGION = args.region
            ami_id = IMAGE_ID_MAP[REGION].get(screen_size, IMAGE_ID_MAP[REGION][(1920, 1080)])
            env_kwargs["region"] = REGION
            env_kwargs["snapshot_name"] = ami_id
            env_kwargs["enable_proxy"] = True
        
        env = AttackableDesktopEnv(**env_kwargs)
        active_environments.append(env)
        
        logger.info(f"Process {current_process().name} started.")
        
        # Process tasks from queue
        while True:
            try:
                item = task_queue.get(timeout=5)
            except Exception:
                break
            
            domain, example_id = item
            
            try:
                config_file = os.path.join(
                    args.test_config_base_dir, f"{args.examples_subdir}/{domain}/{example_id}.json"
                )
                with open(config_file, "r", encoding="utf-8") as f:
                    example = json.load(f)
                
                logger.info(f"[{current_process().name}][Domain]: {domain}")
                logger.info(f"[{current_process().name}][Example ID]: {example_id}")
                logger.info(f"[{current_process().name}][Instruction]: {example['instruction']}")
                
                result_model_name = args.result_model_name if args.result_model_name else args.model
                example_result_dir = os.path.join(
                    args.result_dir,
                    args.action_space,
                    args.observation_type,
                    result_model_name,
                    domain,
                    example_id,
                )
                os.makedirs(example_result_dir, exist_ok=True)
                
                try:
                    lib_run_single.run_single_example(
                        agent,
                        env,
                        example,
                        args.max_steps,
                        example["instruction"],
                        args,
                        example_result_dir,
                        shared_scores,
                    )
                    # Verify role-split trajectory logs exist (manager/worker/grounder + roles manifest).
                    expected = [
                        "traj.jsonl",
                        "agent_roles.json",
                        "traj.manager.jsonl",
                        "traj.worker.jsonl",
                        "traj.grounder.jsonl",
                    ]
                    missing = [
                        name
                        for name in expected
                        if not os.path.exists(os.path.join(example_result_dir, name))
                    ]
                    if missing:
                        logger.warning(
                            "[%s] Missing role logs for %s/%s: %s",
                            current_process().name,
                            domain,
                            example_id,
                            missing,
                        )
                    else:
                        logger.info(
                            "[%s] Role logs written for %s/%s",
                            current_process().name,
                            domain,
                            example_id,
                        )
                except Exception as e:
                    import traceback
                    logger.exception(f"Exception in {current_process().name} {domain}/{example_id}: {e}")
                    logger.error(traceback.format_exc())
                    
                    # Only attempt to end recording if controller exists
                    if hasattr(env, 'controller') and env.controller is not None:
                        try:
                            env.controller.end_recording(
                                os.path.join(example_result_dir, "recording.mp4")
                            )
                        except Exception as rec_e:
                            logger.error(f"Failed to end recording: {rec_e}")
                    
                    with open(os.path.join(example_result_dir, "traj.jsonl"), "a") as f:
                        f.write(
                            json.dumps(
                                {"Error": f"{domain}/{example_id} - {e}"}
                            )
                        )
                        f.write("\n")
            except Exception as e:
                logger.exception(f"Task-level error in {current_process().name}: {e}")
                import traceback
                logger.error(traceback.format_exc())
    
    except Exception as e:
        logger.exception(f"Process-level error in {current_process().name}: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    finally:
        logger.info(f"{current_process().name} cleaning up environment...")
        try:
            if env:
                env.close()
                logger.info(f"{current_process().name} environment closed successfully")
        except Exception as e:
            logger.error(f"{current_process().name} error during environment cleanup: {e}")


def signal_handler(signum, frame):
    """Handle termination signals (SIGINT, SIGTERM) to gracefully shutdown environments."""
    global is_terminating, active_environments, processes
    
    if is_terminating:
        return
    
    is_terminating = True
    logger.info(f"Received signal {signum}. Gracefully shutting down...")
    
    # Close all registered environments in the main process
    for env in active_environments:
        try:
            logger.info(f"Closing environment...")
            env.close()
            logger.info(f"Environment closed successfully")
        except Exception as e:
            logger.error(f"Error closing environment: {e}")
    
    # Send termination signal to all child processes
    for p in processes:
        if p.is_alive():
            try:
                logger.info(f"Sending termination signal to process {p.name}...")
                p.terminate()
            except Exception as e:
                logger.error(f"Error sending termination signal to process: {e}")
    
    time.sleep(1)
    
    # Forcefully terminate any processes that didn't exit
    for p in processes:
        if p.is_alive():
            try:
                logger.info(f"Forcefully terminating process {p.name}...")
                import signal as sig
                os.kill(p.pid, sig.SIGKILL)
            except Exception as e:
                logger.error(f"Error forcefully terminating process: {e}")
    
    logger.info("Shutdown complete. Exiting.")
    sys.exit(0)


def test(args: argparse.Namespace, test_all_meta: dict) -> None:
    global processes
    logger.info("Args: %s", args)
    all_tasks = distribute_tasks(test_all_meta)
    logger.info(f"Total tasks: {len(all_tasks)}")
    
    with Manager() as manager:
        shared_scores = manager.list()
        task_queue = manager.Queue()
        
        for item in all_tasks:
            task_queue.put(item)
        
        num_envs = args.num_envs
        processes = []
        
        for i in range(num_envs):
            p = Process(
                target=run_env_tasks,
                args=(task_queue, args, shared_scores),
                name=f"EnvProcess-{i+1}"
            )
            p.daemon = True
            p.start()
            processes.append(p)
            logger.info(f"Started process {p.name} with PID {p.pid}")
        
        try:
            while True:
                alive_count = 0
                for idx, p in enumerate(processes):
                    if not p.is_alive():
                        logger.warning(f"Process {p.name} died, restarting...")
                        new_p = Process(
                            target=run_env_tasks,
                            args=(task_queue, args, shared_scores),
                            name=f"EnvProcess-Restart-{idx+1}"
                        )
                        new_p.daemon = True
                        new_p.start()
                        processes[idx] = new_p
                        logger.info(f"Restarted process {new_p.name} with PID {new_p.pid}")
                    else:
                        alive_count += 1
                
                if task_queue.empty():
                    logger.info("All tasks finished.")
                    break
                
                if alive_count == 0:
                    logger.error("All processes died, exiting.")
                    break
                
                time.sleep(5)
            
            for p in processes:
                p.join()
        
        except KeyboardInterrupt:
            logger.info("Main process received KeyboardInterrupt. Initiating graceful shutdown...")
            raise
        
        except Exception as e:
            logger.exception(f"Unexpected error while waiting for processes: {e}")
            for p in processes:
                if p.is_alive():
                    try:
                        logger.info(f"Terminating process {p.name} due to error...")
                        p.terminate()
                    except Exception as term_e:
                        logger.error(f"Error terminating process {p.name}: {term_e}")
            raise
        
        scores = list(shared_scores)
    
    logger.info(f"Average score: {sum(scores) / len(scores) if scores else 0}")


def get_unfinished(
    action_space, use_model, observation_type, result_dir, total_file_json
):
    target_dir = os.path.join(result_dir, action_space, observation_type, use_model)

    if not os.path.exists(target_dir):
        return total_file_json

    finished = {}
    for domain in os.listdir(target_dir):
        finished[domain] = []
        domain_path = os.path.join(target_dir, domain)
        if os.path.isdir(domain_path):
            for example_id in os.listdir(domain_path):
                if example_id == "onboard":
                    continue
                example_path = os.path.join(domain_path, example_id)
                if os.path.isdir(example_path):
                    if "result.txt" not in os.listdir(example_path):
                        # empty all files under example_id
                        for file in os.listdir(example_path):
                            try:
                                os.remove(os.path.join(example_path, file))
                            except:
                                pass
                    else:
                        finished[domain].append(example_id)

    if not finished:
        return total_file_json

    for domain, examples in finished.items():
        if domain in total_file_json:
            total_file_json[domain] = [
                x for x in total_file_json[domain] if x not in examples
            ]

    return total_file_json


def get_result(action_space, use_model, observation_type, result_dir, total_file_json):
    target_dir = os.path.join(result_dir, action_space, observation_type, use_model)
    if not os.path.exists(target_dir):
        print("New experiment, no result yet.")
        return None

    all_result = []

    for domain in os.listdir(target_dir):
        domain_path = os.path.join(target_dir, domain)
        if os.path.isdir(domain_path):
            for example_id in os.listdir(domain_path):
                example_path = os.path.join(domain_path, example_id)
                if os.path.isdir(example_path):
                    if "result.txt" in os.listdir(example_path):
                        try:
                            all_result.append(
                                float(
                                    open(
                                        os.path.join(example_path, "result.txt"), "r"
                                    ).read()
                                )
                            )
                        except:
                            all_result.append(0.0)

    if not all_result:
        print("New experiment, no result yet.")
        return None
    else:
        print("Current Success Rate:", sum(all_result) / len(all_result) * 100, "%")
        return all_result


if __name__ == "__main__":
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    _log_llm_env(logger, "[MainProcess]")

    with open(args.test_all_meta_path, "r", encoding="utf-8") as f:
        test_all_meta = json.load(f)

    if args.domain != "all":
        test_all_meta = {args.domain: test_all_meta[args.domain]}

    result_model_name = args.result_model_name if args.result_model_name else args.model

    test_file_list = get_unfinished(
        args.action_space,
        result_model_name,
        args.observation_type,
        args.result_dir,
        test_all_meta,
    )
    
    left_info = ""
    for domain in test_file_list:
        left_info += f"{domain}: {len(test_file_list[domain])}\n"
    logger.info(f"Left tasks:\n{left_info}")

    get_result(
        args.action_space,
        result_model_name,
        args.observation_type,
        args.result_dir,
        test_all_meta,
    )
    
    test(args, test_file_list)
