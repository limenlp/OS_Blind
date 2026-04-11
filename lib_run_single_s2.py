import datetime
import json
import logging
import os
import time
from typing import Any, Dict, Optional, Tuple
from wrapt_timeout_decorator import *

logger = logging.getLogger("desktopenv.experiment")

MANAGER_FIELDS = (
    "search_query",
    "goal_plan",
    "dag",
    "num_input_tokens_plan",
    "num_output_tokens_plan",
    "goal_plan_cost",
    "num_input_tokens_dag",
    "num_output_tokens_dag",
    "dag_cost",
)

WORKER_FIELDS = (
    "current_subtask",
    "current_subtask_info",
    "executor_plan",
    "plan_code",
    "reflection",
    "num_input_tokens_executor",
    "num_output_tokens_executor",
)


def _extract_response_and_grounder(
    response: Any,
) -> Tuple[Any, Optional[Any], Dict[str, Any]]:
    """Split raw response into payload and typed manager/worker views."""
    response_payload = response
    grounder_trace = None
    manager_response: Dict[str, Any] = {}
    worker_response: Dict[str, Any] = {}

    if isinstance(response, dict):
        response_payload = dict(response)
        grounder_trace = response_payload.pop("grounder_trace", None)
        manager_response = {k: response_payload[k] for k in MANAGER_FIELDS if k in response_payload}
        worker_response = {k: response_payload[k] for k in WORKER_FIELDS if k in response_payload}

    return response_payload, grounder_trace, {
        "manager_response": manager_response,
        "worker_response": worker_response,
    }


def _build_agent_trace(
    response_payload: Any,
    grounder_trace: Optional[Any],
    action: Any,
    extracted_views: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Build per-agent logs for trajectory analysis."""
    manager_response = extracted_views.get("manager_response", {})
    worker_response = extracted_views.get("worker_response", {})

    manager_action = {
        "selected_subtask": response_payload.get("subtask") if isinstance(response_payload, dict) else None,
        "selected_subtask_info": response_payload.get("subtask_info") if isinstance(response_payload, dict) else None,
        "subtask_status": response_payload.get("subtask_status") if isinstance(response_payload, dict) else None,
    }

    return {
        "manager": {
            "active": bool(manager_response),
            "response": manager_response or None,
            "action": manager_action,
        },
        "worker": {
            "active": bool(worker_response),
            "response": worker_response or (response_payload if not isinstance(response_payload, dict) else None),
            "action": worker_response.get("plan_code") if worker_response else None,
        },
        "grounder": {
            "active": grounder_trace is not None,
            "response": grounder_trace,
            "action": action,
        },
        "final_action": action,
    }


def _append_jsonl(path: str, obj: Dict[str, Any], *, ensure_ascii: bool = True) -> None:
    # Keep write path simple and robust; all logs are append-only.
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=ensure_ascii))
        f.write("\n")


def _discover_agent_roles(agent: Any) -> Dict[str, Dict[str, str]]:
    """Best-effort discovery of the (expected) S2 roles present on the agent instance."""
    roles: Dict[str, Dict[str, str]] = {}

    if hasattr(agent, "planner") and agent.planner is not None:
        roles["manager"] = {
            "attr": "planner",
            "class": type(agent.planner).__module__ + "." + type(agent.planner).__name__,
        }
    if hasattr(agent, "executor") and agent.executor is not None:
        roles["worker"] = {
            "attr": "executor",
            "class": type(agent.executor).__module__ + "." + type(agent.executor).__name__,
        }
    if hasattr(agent, "grounding_agent") and agent.grounding_agent is not None:
        roles["grounder"] = {
            "attr": "grounding_agent",
            "class": type(agent.grounding_agent).__module__
            + "."
            + type(agent.grounding_agent).__name__,
        }

    return roles


def _write_agent_manifest(agent: Any, example_result_dir: str) -> None:
    roles = _discover_agent_roles(agent)
    manifest = {
        "created_at": datetime.datetime.now().strftime("%Y%m%d@%H%M%S"),
        "roles": [{"role": r, **meta} for r, meta in roles.items()],
    }
    with open(os.path.join(example_result_dir, "agent_roles.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def _write_role_logs(
    example_result_dir: str,
    *,
    step_num: int,
    action_timestamp: str,
    agent_trace: Dict[str, Any],
) -> None:
    """Write per-role jsonl logs alongside the aggregated traj.jsonl."""
    common = {
        "step_num": step_num,
        "action_timestamp": action_timestamp,
    }

    # Manager
    if "manager" in agent_trace:
        _append_jsonl(
            os.path.join(example_result_dir, "traj.manager.jsonl"),
            {"role": "manager", **common, **agent_trace["manager"]},
            ensure_ascii=False,
        )
    # Worker
    if "worker" in agent_trace:
        _append_jsonl(
            os.path.join(example_result_dir, "traj.worker.jsonl"),
            {"role": "worker", **common, **agent_trace["worker"]},
            ensure_ascii=False,
        )
    # Grounder
    if "grounder" in agent_trace:
        _append_jsonl(
            os.path.join(example_result_dir, "traj.grounder.jsonl"),
            {"role": "grounder", **common, **agent_trace["grounder"]},
            ensure_ascii=False,
        )


def run_single_example(
    agent, env, example, max_steps, instruction, args, example_result_dir, scores
):
    runtime_logger = setup_logger(example, example_result_dir)
    agent.reset()

    # Set result_dir for AttackableDesktopEnv so attack artifacts
    # (attack_generation_log.json, attack_log.jsonl, attack_pre/post screenshots)
    # are written under the per-task result directory.
    if hasattr(env, "set_result_dir"):
        env.set_result_dir(example_result_dir)

    env.reset(task_config=example)
    time.sleep(60)  # Wait for the environment to be ready
    obs = env._get_obs()  # Get the initial observation
    done = False
    step_idx = 0
    # Record which roles/agents exist for this run (expected: manager/worker/grounder).
    try:
        _write_agent_manifest(agent, example_result_dir)
    except Exception as e:
        logger.warning("Failed to write agent_roles.json: %s", e)
    # Only attempt to start recording if controller exists (not Docker provider)
    if hasattr(env, 'controller') and env.controller is not None:
        try:
            env.controller.start_recording()
        except Exception:
            pass
    while not done and step_idx < max_steps:
        response, actions = agent.predict(instruction, obs)
        for action in actions:
            # Capture the timestamp before executing the action
            action_timestamp = datetime.datetime.now().strftime("%Y%m%d@%H%M%S")
            logger.info("Step %d: %s", step_idx + 1, action)
            obs, reward, done, info = env.step(action, args.sleep_after_execution)

            logger.info("Reward: %.2f", reward)
            logger.info("Done: %s", done)
            # Save screenshot and trajectory information
            with open(
                os.path.join(
                    example_result_dir, f"step_{step_idx + 1}_{action_timestamp}.png"
                ),
                "wb",
            ) as _f:
                _f.write(obs["screenshot"])
            response_payload, grounder_trace, extracted_views = (
                _extract_response_and_grounder(response)
            )
            agent_trace = _build_agent_trace(
                response_payload=response_payload,
                grounder_trace=grounder_trace,
                action=action,
                extracted_views=extracted_views,
            )

            with open(os.path.join(example_result_dir, "traj.jsonl"), "a") as f:
                f.write(
                    json.dumps(
                        {
                            "step_num": step_idx + 1,
                            "action_timestamp": action_timestamp,
                            "action": action,
                            "response": response_payload,
                            "grounder": grounder_trace,
                            "agent_trace": agent_trace,
                            "reward": reward,
                            "done": done,
                            "info": info,
                            "screenshot_file": f"step_{step_idx + 1}_{action_timestamp}.png",
                        }
                    )
                )
                f.write("\n")
            try:
                _write_role_logs(
                    example_result_dir,
                    step_num=step_idx + 1,
                    action_timestamp=action_timestamp,
                    agent_trace=agent_trace,
                )
            except Exception as e:
                logger.warning("Failed to write role logs: %s", e)
            if done:
                logger.info("The episode is done.")
                break
        step_idx += 1
    result = env.evaluate()
    logger.info("Result: %.2f", result)
    scores.append(result)
    with open(
        os.path.join(example_result_dir, "result.txt"), "w", encoding="utf-8"
    ) as f:
        f.write(f"{result}\n")
    # Only attempt to end recording if controller exists (not Docker provider)
    if hasattr(env, 'controller') and env.controller is not None:
        try:
            env.controller.end_recording(os.path.join(example_result_dir, "recording.mp4"))
        except Exception:
            pass


def setup_logger(example, example_result_dir):
    runtime_logger = logging.getLogger(f"desktopenv.example.{example['id']}")
    runtime_logger.setLevel(logging.DEBUG)
    runtime_logger.addHandler(
        logging.FileHandler(os.path.join(example_result_dir, "runtime.log"))
    )
    return runtime_logger
