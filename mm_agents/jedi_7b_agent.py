import base64
import json
import logging
import os
import re
import time
from io import BytesIO
from typing import Dict, List

import backoff
import openai
import requests
from PIL import Image
from google.api_core.exceptions import (
    InvalidArgument,
    ResourceExhausted,
    InternalServerError,
    BadRequest,
)
from requests.exceptions import SSLError

logger = None



from mm_agents.prompts import JEDI_PLANNER_SYS_PROMPT, JEDI_GROUNDER_SYS_PROMPT
from mm_agents.utils.qwen_vl_utils import smart_resize

def encode_image(image_content):
    return base64.b64encode(image_content).decode("utf-8")

class JediAgent7B:
    def __init__(
        self,
        platform="ubuntu",
        planner_model="gpt-4o",
        executor_model="jedi-7b",
        max_tokens=1500,
        top_p=0.9,
        temperature=0.5,
        action_space="pyautogui",
        observation_type="screenshot",
        max_steps=15,
        openai_api_key=None,
        openai_base_url=None,
        jedi_api_key=None,
        jedi_service_url=None
    ):
        self.platform = platform
        self.planner_model = planner_model
        self.executor_model = executor_model
        assert self.executor_model is not None, "Executor model cannot be None"
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.temperature = temperature
        self.action_space = action_space
        self.observation_type = observation_type
        assert action_space in ["pyautogui"], "Invalid action space"
        assert observation_type in ["screenshot"], "Invalid observation type"
        
        self.openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        self.openai_base_url = (
            openai_base_url
            or os.environ.get("OPENAI_BASE_URL")
            or os.environ.get("OPENAI_API_BASE")
            or "https://api.openai.com/v1"
        )
        self.jedi_api_key = jedi_api_key or os.environ.get("JEDI_API_KEY")
        self.jedi_service_url = jedi_service_url or os.environ.get("JEDI_SERVICE_URL")

        self.thoughts = []
        self.actions = []
        self.observations = []
        self.observation_captions = []
        self.max_image_history_length = 5
        self.current_step = 1
        self.max_steps = max_steps

    def _is_reasoning_model(self, model: str) -> bool:
        """Check if the model is a reasoning model that doesn't support temperature/top_p
        and uses max_completion_tokens instead of max_tokens."""
        reasoning_prefixes = ("o3", "o4", "o1", "gpt-5")
        return any(model.startswith(prefix) for prefix in reasoning_prefixes)

    def _openai_chat_endpoint(self) -> str:
        base_url = self.openai_base_url.rstrip("/")
        if base_url.endswith("/v1"):
            return f"{base_url}/chat/completions"
        return f"{base_url}/v1/chat/completions"

    def _build_planner_payload(self, messages):
        """Build the API payload for the planner model, handling reasoning vs non-reasoning models."""
        payload = {
            "model": self.planner_model,
            "messages": messages,
        }
        if self._is_reasoning_model(self.planner_model):
            payload["max_completion_tokens"] = self.max_tokens
        else:
            payload["max_tokens"] = self.max_tokens
            payload["top_p"] = self.top_p
            payload["temperature"] = self.temperature
        return payload

    def predict(self, instruction: str, obs: Dict) -> List:
        """
        Predict the next action(s) based on the current observation.
        """

        # get the width and height of the screenshot
        image = Image.open(BytesIO(obs["screenshot"]))
        width, height = image.convert("RGB").size

        previous_actions = ("\n".join([
            f"Step {i+1}: {action}" for i, action in enumerate(self.actions)
        ]) if self.actions else "None")

        user_prompt = (
            f"""Please generate the next move according to the UI screenshot and instruction. And you can refer to the previous actions and observations for reflection.\n\nInstruction: {instruction}\n\n""")

        messages = [{
            "role": "system",
            "content": [{
                "type": "text",
                "text": JEDI_PLANNER_SYS_PROMPT.replace("{current_step}", str(self.current_step)).replace("{max_steps}", str(self.max_steps))
            }]
        }]

        # Determine which observations to include images for (only most recent ones)
        obs_start_idx = max(0, len(self.observations) - self.max_image_history_length)
        
        # Add all thought and action history
        for i in range(len(self.thoughts)):
            # For recent steps, include the actual screenshot
            if i >= obs_start_idx:
                messages.append({
                    "role": "user",
                    "content": [{
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{encode_image(self.observations[i]['screenshot'])}",
                            "detail": "high"
                        },
                    }]
                })
            # For older steps, use the observation caption instead of the image
            else:
                messages.append({
                    "role": "user",
                    "content": [{
                        "type": "text",
                        "text": f"Observation: {self.observation_captions[i]}"
                    }]
                })

            thought_messages = f"Thought:\n{self.thoughts[i]}"

            action_messages = f"Action:"
            for action in self.actions[i]:
                action_messages += f"\n{action}"
            messages.append({
                "role": "assistant",
                "content": [{
                    "type": "text",
                    "text": thought_messages + "\n" + action_messages
                }]
            })
            #print(thought_messages + "\n" + action_messages)

        messages.append({
            "role":"user",
            "content": [
                {
                    "type":"image_url",
                    "image_url":{
                        "url":f"data:image/png;base64,{encode_image(obs['screenshot'])}",
                        "detail": "high"
                    },
                },
                {
                    "type": "text",
                    "text": user_prompt
                },
            ],
        })
        
        planner_payload = self._build_planner_payload(messages)
        planner_response = self.call_llm(planner_payload, self.planner_model)

        logger.info(f"Planner Output: {planner_response}")
        codes = self.parse_code_from_planner_response(planner_response)
        retry_count = 0
        max_retries = 5
        while not codes and retry_count < max_retries:
            logger.info(f"No codes parsed from planner response. Retrying ({retry_count+1}/{max_retries})...")
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": "You didn't generate valid actions. Please try again."}
                ]
            })
            planner_payload = self._build_planner_payload(messages)
            planner_response = self.call_llm(planner_payload, self.planner_model)
            logger.info(f"Retry Planner Output: {planner_response}")
            codes = self.parse_code_from_planner_response(planner_response)
            retry_count += 1
            
        thought = self.parse_thought_from_planner_response(planner_response)
        observation_caption = self.parse_observation_caption_from_planner_response(planner_response)
        resized_height, resized_width = smart_resize(height, width, max_pixels= 2700 * 28 * 28)
        pyautogui_actions = []
        for line in codes:
            code = self.convert_action_to_grounding_model_instruction(
                line,
                obs,
                instruction,
                height,
                width,
                resized_height,
                resized_width
            )
            pyautogui_actions.append(code)
        self.actions.append([pyautogui_actions])
        self.observations.append(obs)
        self.thoughts.append(thought)
        self.observation_captions.append(observation_caption)
        self.current_step += 1
        return planner_response, pyautogui_actions
        
    def parse_observation_caption_from_planner_response(self, input_string: str) -> str:
        pattern = r"Observation:\n(.*?)\n"
        matches = re.findall(pattern, input_string, re.DOTALL)
        if matches:
            return matches[0].strip()
        return ""

    def parse_thought_from_planner_response(self, input_string: str) -> str:
        pattern = r"Thought:\n(.*?)\n"
        matches = re.findall(pattern, input_string, re.DOTALL)
        if matches:
            return matches[0].strip()
        return ""

    def parse_code_from_planner_response(self, input_string: str) -> List[str]:

        input_string = "\n".join([line.strip() for line in input_string.split(';') if line.strip()])
        if input_string.strip() in ['WAIT', 'DONE', 'FAIL']:
            return [input_string.strip()]

        pattern = r"```(?:\w+\s+)?(.*?)```"
        matches = re.findall(pattern, input_string, re.DOTALL)
        codes = []

        for match in matches:
            match = match.strip()
            commands = ['WAIT', 'DONE', 'FAIL']

            if match in commands:
                codes.append(match.strip())
            elif match.split('\n')[-1] in commands:
                if len(match.split('\n')) > 1:
                    codes.append("\n".join(match.split('\n')[:-1]))
                codes.append(match.split('\n')[-1])
            else:
                codes.append(match)

        return codes

    def convert_action_to_grounding_model_instruction(self, line: str, obs: Dict, instruction: str, height: int, width: int, resized_height: int, resized_width: int ) -> str:
        pattern = r'(#.*?)\n(pyautogui\.(moveTo|click|rightClick|doubleClick|middleClick|dragTo)\((?:x=)?(\d+)(?:,\s*|\s*,\s*y=)(\d+)(?:,\s*duration=[\d.]+)?\))'
        matches = re.findall(pattern, line, re.DOTALL)
        if not matches:
            return line
        new_instruction = line
        for match in matches:
            comment = match[0].split("#")[1].strip()
            original_action = match[1]
            func_name = match[2].strip()

            if "click()" in original_action.lower():
                continue
            
            messages = []
            messages.append({
                "role": "system",
                "content": [{"type": "text", "text": JEDI_GROUNDER_SYS_PROMPT.replace("{height}", str(resized_height)).replace("{width}", str(resized_width))}]
            })
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{encode_image(obs['screenshot'])}",
                                "detail": "high",
                            },
                        },
                        {
                            "type": "text",
                            "text": '\n' + comment,
                        },
                    ],
                }
            )
            grounding_response = self.call_llm({
                "model": self.executor_model,
                "messages": messages,
                "max_tokens": self.max_tokens,
                "top_p": self.top_p,
                "temperature": self.temperature
            }, self.executor_model)
            logger.info("Grounding comment: %s", comment)
            logger.info("Grounding raw response: %s", grounding_response)
            coordinates = self.parse_jedi_response(grounding_response, width, height, resized_width, resized_height)
            logger.info("Grounding coordinates: %s", coordinates)
            if coordinates == [-1, -1]:
                continue
            action_parts = original_action.split('(')
            new_action = f"{action_parts[0]}({coordinates[0]}, {coordinates[1]}"
            if len(action_parts) > 1 and 'duration' in action_parts[1]:
                duration_part = action_parts[1].split(',')[-1]
                new_action += f", {duration_part}"
            elif len(action_parts) > 1 and 'button' in action_parts[1]:
                button_part = action_parts[1].split(',')[-1]
                new_action += f", {button_part}"
            else:
                new_action += ")"
            logger.info(new_action)
            new_instruction = new_instruction.replace(original_action, new_action)
        return new_instruction
        
    def parse_jedi_response(self, response, width: int, height: int, resized_width: int, resized_height: int) -> List[str]:
        """
        Parse the LLM response and convert it to low level action and pyautogui code.
        """ 

        low_level_instruction = ""
        pyautogui_code = []
        try:
            # 定义可能的标签组合
            start_tags = ["<tool_call>", "⚗"]
            end_tags = ["</tool_call>", "⚗"]

            # 找到有效的开始和结束标签
            start_tag = next((tag for tag in start_tags if tag in response), None)
            end_tag = next((tag for tag in end_tags if tag in response), None)

            if not start_tag or not end_tag:
                logger.warning("Grounding response missing valid start or end tags")
                return [-1, -1]

            parts = response.split(start_tag)
            if len(parts) < 2:
                logger.warning("Grounding response missing the start tag")
                return [-1, -1]

            low_level_instruction = parts[0].strip().replace("Action: ", "")
            tool_call_str = parts[1].split(end_tag)[0].strip()

            try:
                tool_call = json.loads(tool_call_str)
                action = tool_call.get("arguments", {}).get("action", "")
                args = tool_call.get("arguments", {})
            except json.JSONDecodeError as e:
                print(f"JSON parsing error: {e}")
                # 处理解析错误，返回默认值或空值
                action = ""
                args = {}
            
            # convert the coordinate to the original resolution
            x = int(args.get("coordinate", [-1, -1])[0] * width / resized_width)
            y = int(args.get("coordinate", [-1, -1])[1] * height / resized_height)

            return [x, y]
        except Exception as e:
            logger.error(f"Failed to parse response: {e}")
            return [-1, -1]

    @backoff.on_exception(
        backoff.constant,
        # here you should add more model exceptions as you want,
        # but you are forbidden to add "Exception", that is, a common type of exception
        # because we want to catch this kind of Exception in the outside to ensure
        # each example won't exceed the time limit
        (
            # General exceptions
            SSLError,
            # OpenAI exceptions
            openai.RateLimitError,
            openai.BadRequestError,
            openai.InternalServerError,
            # Google exceptions
            InvalidArgument,
            ResourceExhausted,
            InternalServerError,
            BadRequest,
            # Groq exceptions
            # todo: check
        ),
        interval=30,
        max_tries=10,
    )
    def call_llm(self, payload, model):
        if model.startswith(("gpt", "o1", "o3", "o4")):
            if not self.openai_api_key:
                logger.error("OPENAI_API_KEY is not set for planner model %s", model)
                return ""
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.openai_api_key}"
            }
            endpoint = self._openai_chat_endpoint()
            logger.info("Generating content with GPT model: %s", model)
            response = requests.post(
                endpoint,
                headers=headers,
                json=payload,
            )

            if response.status_code != 200:
                logger.error("Failed to call LLM: " + response.text)
                time.sleep(5)
                return ""
            else:
                return response.json()["choices"][0]["message"]["content"]

        elif "jedi" in model.lower():
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.jedi_api_key}"
            }
            base_url = self.jedi_service_url.rstrip('/')
            if base_url.endswith('/v1'):
                endpoint = f"{base_url}/chat/completions"
            else:
                endpoint = f"{base_url}/v1/chat/completions"
            logger.info("Calling Jedi grounding at: %s", endpoint)
            response = requests.post(
                endpoint,
                headers=headers,
                json=payload,
            )
            if response.status_code != 200:
                logger.error("Failed to call Jedi LLM (status %d): %s", response.status_code, response.text)
                time.sleep(5)
                return ""
            else:
                return response.json()["choices"][0]["message"]["content"]

    def reset(self, _logger=None):
        global logger
        logger = (_logger if _logger is not None else
                  logging.getLogger("desktopenv.jedi_7b_agent"))

        self.thoughts = []
        self.action_descriptions = []
        self.actions = []
        self.observations = []
        self.observation_captions = []
