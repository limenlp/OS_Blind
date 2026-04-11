import base64
import os
import time
from typing import Any, cast, Optional, Dict
from PIL import Image
import io

from anthropic import (
    Anthropic,
    AnthropicBedrock,
    AnthropicVertex,
    APIError,
    APIResponseValidationError,
    APIStatusError,
)
from anthropic.types.beta import (
    BetaMessageParam,
    BetaTextBlockParam,
)
from .utils import COMPUTER_USE_BETA_FLAG, PROMPT_CACHING_BETA_FLAG,SYSTEM_PROMPT, SYSTEM_PROMPT_WINDOWS, APIProvider, PROVIDER_TO_DEFAULT_MODEL_NAME
from .utils import _response_to_params, _inject_prompt_caching, _maybe_filter_to_n_most_recent_images

import logging
logger = logging.getLogger("desktopenv.agent")

# MAX_HISTORY = 10
API_RETRY_TIMES = 500  
API_RETRY_INTERVAL = 5

MODEL_ALIASES: dict[str, str] = {
    "claude-sonnet-4-5-20250929": "claude-sonnet-4-5",
    "claude-opus-4-5-20251101": "claude-opus-4-5",
    "opus4.5": "claude-opus-4-5",
    "opus-4.5": "claude-opus-4-5",
}

SUPPORTED_MODELS: set[str] = {
    "claude-3-5-sonnet-20241022",
    "claude-3-7-sonnet-20250219",
    "claude-4-opus-20250514",
    "claude-4-sonnet-20250514",
    "claude-opus-4-5",
    "claude-sonnet-4-5",
    "claude-sonnet-4-6",
}

COMPUTER_20250124_MODELS: set[str] = {
    "claude-3-7-sonnet-20250219",
    "claude-4-opus-20250514",
    "claude-4-sonnet-20250514",
    "claude-sonnet-4-5",
}

COMPUTER_20251124_MODELS: set[str] = {
    "claude-opus-4-5",
    "claude-sonnet-4-6",
}

_PROVIDER_VALUE_TO_ENUM: dict[str, APIProvider] = {
    provider.value: provider for provider in APIProvider
}


def _coerce_provider(value: Optional[str | APIProvider]) -> APIProvider:
    if isinstance(value, APIProvider):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in _PROVIDER_VALUE_TO_ENUM:
            return _PROVIDER_VALUE_TO_ENUM[normalized]
        # Allow enum names such as "ANTHROPIC"
        try:
            return APIProvider[normalized.upper()]
        except KeyError:
            logger.warning(
                f"Unknown Anthropic provider '{value}', defaulting to 'anthropic'. "
                "Expected one of: "
                + ", ".join(sorted(_PROVIDER_VALUE_TO_ENUM.keys()))
            )
    return APIProvider.ANTHROPIC

class AnthropicAgent:
    def __init__(self,
                 platform: str = "Ubuntu",
                 model: str = "claude-3-5-sonnet-20241022",
                 provider: APIProvider | str = APIProvider.ANTHROPIC,
                 max_tokens: int = 4096,
                 api_key: str = None,
                 system_prompt_override: str = "",
                 system_prompt_suffix: str = "",
                 only_n_most_recent_images: Optional[int] = 10,
                 action_space: str = "claude_computer_use",
                 screen_size: tuple[int, int] = (1920, 1080),
                 *args, **kwargs
                 ):
        self.platform = platform
        self.action_space = action_space
        self.logger = logger
        self.class_name = self.__class__.__name__
        provider_override = kwargs.pop("provider_name", None) or os.environ.get("ANTHROPIC_PROVIDER")
        normalized_model = MODEL_ALIASES.get(model, model)
        if normalized_model not in SUPPORTED_MODELS:
            supported = ", ".join(sorted(SUPPORTED_MODELS))
            raise ValueError(f"Unsupported model '{model}'. Supported models: {supported}")
        self.model_name = normalized_model
        self.provider = _coerce_provider(provider_override or provider)
        self.max_tokens = max_tokens
        primary_api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        backup_api_key = os.environ.get("ANTHROPIC_API_KEY_BACKUP")
        self.api_key = primary_api_key or backup_api_key
        if self.provider == APIProvider.ANTHROPIC:
            if not self.api_key:
                logger.error(
                    "No Anthropic API key configured. Set ANTHROPIC_API_KEY "
                    "(or ANTHROPIC_API_KEY_BACKUP as fallback)."
                )
            elif not primary_api_key and backup_api_key:
                logger.warning("ANTHROPIC_API_KEY not found, using ANTHROPIC_API_KEY_BACKUP.")
        self.system_prompt_override = system_prompt_override
        self.system_prompt_suffix = system_prompt_suffix
        self.only_n_most_recent_images = only_n_most_recent_images
        self.messages: list[BetaMessageParam] = []
        self.screen_size = screen_size
        self.resize_factor = (
            screen_size[0] / 1280,  # Assuming 1280 is the base width
            screen_size[1] / 720   # Assuming 720 is the base height
        )

    def add_tool_result(self, tool_call_id: str, result: str, screenshot: bytes = None):
        """Add tool result to message history"""
        tool_result_content = [
            {
                "type": "tool_result",
                "tool_use_id": tool_call_id,
                "content": [{"type": "text", "text": result}]
            }
        ]
        
        # Add screenshot if provided
        if screenshot is not None:
            screenshot_base64 = base64.b64encode(screenshot).decode('utf-8')
            tool_result_content[0]["content"].append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png", 
                    "data": screenshot_base64
                }
            })
        
        self.messages.append({
            "role": "user",
            "content": tool_result_content
        })
    
    def parse_actions_from_tool_call(self, tool_call: Dict) -> str:
        result = ""
        function_args = (
            tool_call["input"]
        )
        
        action = function_args.get("action")
        if not action:
            action = tool_call.function.name
        action_conversion = {
            "left click": "click",
            "right click": "right_click"
        }
        action = action_conversion.get(action, action)
        
        text = function_args.get("text")
        coordinate = function_args.get("coordinate")
        start_coordinate = function_args.get("start_coordinate")
        scroll_direction = function_args.get("scroll_direction")
        scroll_amount = function_args.get("scroll_amount")
        duration = function_args.get("duration")
        
        # resize coordinates if resize_factor is set
        if coordinate and self.resize_factor:
            coordinate = (
                int(coordinate[0] * self.resize_factor[0]),
                int(coordinate[1] * self.resize_factor[1])
            )
        if start_coordinate and self.resize_factor:
            start_coordinate = (
                int(start_coordinate[0] * self.resize_factor[0]),
                int(start_coordinate[1] * self.resize_factor[1])
            )
        
        if action == "left_mouse_down":
            result += "pyautogui.mouseDown()\n"
        elif action == "left_mouse_up":
            result += "pyautogui.mouseUp()\n"
        
        elif action == "hold_key":
            if not isinstance(text, str):
                raise ValueError(f"{text} must be a string")
            
            keys = text.split('+')
            for key in keys:
                key = key.strip().lower()
                result += f"pyautogui.keyDown('{key}')\n"
            expected_outcome = f"Keys {text} held down."

        # Handle mouse move and drag actions
        elif action in ("mouse_move", "left_click_drag"):
            if coordinate is None:
                raise ValueError(f"coordinate is required for {action}")
            if text is not None:
                raise ValueError(f"text is not accepted for {action}")
            if not isinstance(coordinate, (list, tuple)) or len(coordinate) != 2:
                raise ValueError(f"{coordinate} must be a tuple of length 2")
            if not all(isinstance(i, int) for i in coordinate):
                raise ValueError(f"{coordinate} must be a tuple of ints")
            
            x, y = coordinate[0], coordinate[1]
            if action == "mouse_move":
                result += (
                    f"pyautogui.moveTo({x}, {y}, duration={duration or 0.5})\n"
                )
                expected_outcome = f"Mouse moved to ({x},{y})."
            elif action == "left_click_drag":
                # If start_coordinate is provided, validate and move to start before dragging
                if start_coordinate:
                    if not isinstance(start_coordinate, (list, tuple)) or len(start_coordinate) != 2:
                        raise ValueError(f"{start_coordinate} must be a tuple of length 2")
                    if not all(isinstance(i, int) for i in start_coordinate):
                        raise ValueError(f"{start_coordinate} must be a tuple of ints")
                    start_x, start_y = start_coordinate[0], start_coordinate[1]
                    result += (
                        f"pyautogui.moveTo({start_x}, {start_y}, duration={duration or 0.5})\n"
                    )
                result += (
                    f"pyautogui.dragTo({x}, {y}, duration={duration or 0.5})\n"
                )
                expected_outcome = f"Cursor dragged to ({x},{y})."

        # Handle keyboard actions
        elif action in ("key", "type"):
            if text is None:
                raise ValueError(f"text is required for {action}")
            if coordinate is not None:
                raise ValueError(f"coordinate is not accepted for {action}")
            if not isinstance(text, str):
                raise ValueError(f"{text} must be a string")

            if action == "key":
                key_conversion = {
                    "page_down": "pagedown",
                    "page_up": "pageup",
                    "super_l": "win",
                    "super": "command",
                    "escape": "esc"
                }
                keys = text.split('+')
                for key in keys:
                    key = key.strip().lower()
                    key = key_conversion.get(key, key)
                    result += (f"pyautogui.keyDown('{key}')\n")
                for key in reversed(keys):
                    key = key.strip().lower()
                    key = key_conversion.get(key, key)
                    result += (f"pyautogui.keyUp('{key}')\n")
                expected_outcome = f"Key {key} pressed."
            elif action == "type":
                result += (
                    f"pyautogui.typewrite(\"\"\"{text}\"\"\", interval=0.01)\n"
                )
                expected_outcome = f"Text {text} written."

        # Handle scroll actions
        elif action == "scroll":
            if coordinate is None:
                if scroll_direction in ("up", "down"):
                    result += (
                        f"pyautogui.scroll({scroll_amount if scroll_direction == 'up' else -scroll_amount})\n"
                    )
                elif scroll_direction in ("left", "right"):
                    result += (
                        f"pyautogui.hscroll({scroll_amount if scroll_direction == 'right' else -scroll_amount})\n"
                    )
            else:
                if scroll_direction in ("up", "down"):
                    x, y = coordinate[0], coordinate[1]
                    result += (
                        f"pyautogui.scroll({scroll_amount if scroll_direction == 'up' else -scroll_amount}, {x}, {y})\n"
                    )
                elif scroll_direction in ("left", "right"):
                    x, y = coordinate[0], coordinate[1]
                    result += (
                        f"pyautogui.hscroll({scroll_amount if scroll_direction == 'right' else -scroll_amount}, {x}, {y})\n"
                    )
            expected_outcome = "Scroll action finished"

        # Handle click actions
        elif action in ("left_click", "right_click", "double_click", "middle_click", "left_press", "triple_click"):
            # Handle modifier keys during click if specified
            if text:
                keys = text.split('+')
                for key in keys:
                    key = key.strip().lower()
                    result += f"pyautogui.keyDown('{key}')\n"
            if coordinate is not None:
                x, y = coordinate
                if action == "left_click":
                    result += (f"pyautogui.click({x}, {y})\n")
                elif action == "right_click":
                    result += (f"pyautogui.rightClick({x}, {y})\n")
                elif action == "double_click":
                    result += (f"pyautogui.doubleClick({x}, {y})\n")
                elif action == "middle_click":
                    result += (f"pyautogui.middleClick({x}, {y})\n")
                elif action == "left_press":
                    result += (f"pyautogui.mouseDown({x}, {y})\n")
                    result += ("time.sleep(1)\n")
                    result += (f"pyautogui.mouseUp({x}, {y})\n")
                elif action == "triple_click":
                    result += (f"pyautogui.tripleClick({x}, {y})\n")

            else:
                if action == "left_click":
                    result += ("pyautogui.click()\n")
                elif action == "right_click":
                    result += ("pyautogui.rightClick()\n")
                elif action == "double_click":
                    result += ("pyautogui.doubleClick()\n")
                elif action == "middle_click":
                    result += ("pyautogui.middleClick()\n")
                elif action == "left_press":
                    result += ("pyautogui.mouseDown()\n")
                    result += ("time.sleep(1)\n")
                    result += ("pyautogui.mouseUp()\n")
                elif action == "triple_click":
                    result += ("pyautogui.tripleClick()\n")
            # Release modifier keys after click
            if text:
                keys = text.split('+')
                for key in reversed(keys):
                    key = key.strip().lower()
                    result += f"pyautogui.keyUp('{key}')\n"
            expected_outcome = "Click action finished"
            
        elif action == "wait":
            result += "pyautogui.sleep(0.5)\n"
            expected_outcome = "Wait for 0.5 seconds"
        elif action == "fail":
            result += "FAIL"
            expected_outcome = "Finished"
        elif action == "done":
            result += "DONE"
            expected_outcome = "Finished"
        elif action == "call_user":
            result += "CALL_USER"
            expected_outcome = "Call user"
        elif action == "screenshot":
            result += "pyautogui.sleep(0.1)\n"
            expected_outcome = "Screenshot taken"   
        else:
            raise ValueError(f"Invalid action: {action}")
        
        return result
            
    def predict(self, task_instruction: str, obs: Dict = None, system: Any = None):
        if system is None:
            if self.system_prompt_override:
                system_text = self.system_prompt_override
            else:
                system_text = (
                    f"{SYSTEM_PROMPT_WINDOWS if self.platform == 'Windows' else SYSTEM_PROMPT}"
                    f"{' ' + self.system_prompt_suffix if self.system_prompt_suffix else ''}"
                )
            system = BetaTextBlockParam(type="text", text=system_text)
        elif isinstance(system, str):
            system = BetaTextBlockParam(type="text", text=system)
        
        # resize screenshot if resize_factor is set
        if obs and "screenshot" in obs:
            # Convert bytes to PIL Image
            screenshot_bytes = obs["screenshot"]
            screenshot_image = Image.open(io.BytesIO(screenshot_bytes))
            
            # Calculate new size based on resize factor
            new_width, new_height = 1280, 720
            
            # Resize the image
            resized_image = screenshot_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Convert back to bytes
            output_buffer = io.BytesIO()
            resized_image.save(output_buffer, format='PNG')
            obs["screenshot"] = output_buffer.getvalue()
            

        if not self.messages:
            
            init_screenshot = obs
            init_screenshot_base64 = base64.b64encode(init_screenshot["screenshot"]).decode('utf-8')
            self.messages.append({
                "role": "user",
                "content": [
                    {
                    "type": "image",
                    "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": init_screenshot_base64,
                        },
                    },
                    {"type": "text", "text": task_instruction},
                ]
            })
            
        # If the last assistant message contained tool calls, attach a single user message
        # with tool_result blocks for each tool_use (Anthropic requires this ordering).
        # Only the LAST tool_result gets the screenshot to avoid bloating the request.
        if self.messages and self.messages[-1]["role"] == "assistant":
            tool_blocks = [c for c in self.messages[-1]["content"] if c.get("type") == "tool_use"]
            if tool_blocks:
                tool_results = []
                for i, tb in enumerate(tool_blocks):
                    is_last = (i == len(tool_blocks) - 1)
                    content_blocks = [{"type": "text", "text": "Success"}]
                    if is_last and obs and obs.get("screenshot"):
                        content_blocks.append({
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": base64.b64encode(obs["screenshot"]).decode("utf-8"),
                            },
                        })
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tb["id"],
                        "content": content_blocks,
                    })
                self.messages.append({"role": "user", "content": tool_results})
            
        enable_prompt_caching = False
        if self.model_name == "claude-3-5-sonnet-20241022":
            betas = [COMPUTER_USE_BETA_FLAG]
        elif self.model_name in COMPUTER_20251124_MODELS:
            betas = ["computer-use-2025-11-24"]
        else:
            betas = ["computer-use-2025-01-24"]
            
        image_truncation_threshold = 10
        if self.provider == APIProvider.ANTHROPIC:
            if not self.api_key:
                error_response = (
                    "Anthropic API key missing. Configure ANTHROPIC_API_KEY "
                    "or ANTHROPIC_API_KEY_BACKUP."
                )
                logger.error(error_response)
                return error_response, ["FAIL"]
            client = Anthropic(api_key=self.api_key, max_retries=4)
            enable_prompt_caching = True
        elif self.provider == APIProvider.VERTEX:
            client = AnthropicVertex()
        elif self.provider == APIProvider.BEDROCK:
            client = AnthropicBedrock(
                # Authenticate by either providing the keys below or use the default AWS credential providers, such as
                # using ~/.aws/credentials or the "AWS_SECRET_ACCESS_KEY" and "AWS_ACCESS_KEY_ID" environment variables.
                aws_access_key=os.getenv('AWS_ACCESS_KEY_ID'),
                aws_secret_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
                # aws_region changes the aws region to which the request is made. By default, we read AWS_REGION,
                # and if that's not present, we default to us-east-1. Note that we do not read ~/.aws/config for the region.
                aws_region=os.getenv('AWS_DEFAULT_REGION'),
            )

        if enable_prompt_caching:
            betas.append(PROMPT_CACHING_BETA_FLAG)
            _inject_prompt_caching(self.messages)
            image_truncation_threshold = 50
            system["cache_control"] = {"type": "ephemeral"}

        if self.only_n_most_recent_images:
            _maybe_filter_to_n_most_recent_images(
                self.messages,
                self.only_n_most_recent_images,
                min_removal_threshold=image_truncation_threshold,
            )

        for msg in self.messages:
            if isinstance(msg.get("content"), list):
                msg["content"] = [
                    b for b in msg["content"]
                    if not (isinstance(b, dict) and b.get("type") == "text" and not b.get("text"))
                ]

        try:
            if self.model_name == "claude-3-5-sonnet-20241022":
                tools = [
                    {'name': 'computer', 'type': 'computer_20241022', 'display_width_px': 1280, 'display_height_px': 720, 'display_number': 1},
                    # {'type': 'bash_20241022', 'name': 'bash'},
                    # {'name': 'str_replace_editor', 'type': 'text_editor_20241022'}
                ] if self.platform == 'Ubuntu' else [
                    {'name': 'computer', 'type': 'computer_20241022', 'display_width_px': 1280, 'display_height_px': 720, 'display_number': 1},
                ]
            elif self.model_name in COMPUTER_20251124_MODELS:
                tools = [
                    {'name': 'computer', 'type': 'computer_20251124', 'display_width_px': 1280, 'display_height_px': 720, 'display_number': 1},
                ] if self.platform == 'Ubuntu' else [
                    {'name': 'computer', 'type': 'computer_20251124', 'display_width_px': 1280, 'display_height_px': 720, 'display_number': 1},
                ]
            elif self.model_name in COMPUTER_20250124_MODELS:
                tools = [
                    {'name': 'computer', 'type': 'computer_20250124', 'display_width_px': 1280, 'display_height_px': 720, 'display_number': 1},
                    # {'type': 'bash_20250124', 'name': 'bash'},
                    # {'name': 'str_replace_editor', 'type': 'text_editor_20250124'}
                ] if self.platform == 'Ubuntu' else [
                    {'name': 'computer', 'type': 'computer_20250124', 'display_width_px': 1280, 'display_height_px': 720, 'display_number': 1},
                ]
            extra_body = {
                "thinking": {"type": "enabled", "budget_tokens": 1024}
            }
            response = None
            
            for attempt in range(API_RETRY_TIMES):
                try:
                    if self.model_name in COMPUTER_20250124_MODELS or self.model_name in COMPUTER_20251124_MODELS:
                        response = client.beta.messages.create(
                            max_tokens=self.max_tokens,
                            messages=self.messages,
                            model=PROVIDER_TO_DEFAULT_MODEL_NAME[self.provider, self.model_name],
                            system=[system],
                            tools=tools,
                            betas=betas,
                            extra_body=extra_body
                        )
                    elif self.model_name == "claude-3-5-sonnet-20241022":
                        response = client.beta.messages.create(
                            max_tokens=self.max_tokens,
                            messages=self.messages,
                            model=PROVIDER_TO_DEFAULT_MODEL_NAME[self.provider, self.model_name],
                            system=[system],
                            tools=tools,
                            betas=betas,
                        )
                    logger.info(f"Response: {response}")
                    break  
                except (APIError, APIStatusError, APIResponseValidationError, TypeError) as e:
                    error_msg = str(e)
                    logger.warning(f"Anthropic API error (attempt {attempt+1}/{API_RETRY_TIMES}): {error_msg}")
                    
                    if "25000000" in error_msg or "Member must have length less than or equal to" in error_msg or "request_too_large" in error_msg:
                        logger.warning("Detected size limit error, automatically reducing image count")
                        current_image_count = self.only_n_most_recent_images
                        new_image_count = max(1, current_image_count // 2)  # Keep at least 1 image
                        self.only_n_most_recent_images = new_image_count
                        
                        _maybe_filter_to_n_most_recent_images(
                            self.messages,
                            new_image_count,
                            min_removal_threshold=image_truncation_threshold,
                        )
                        logger.info(f"Image count reduced from {current_image_count} to {new_image_count}")
                    
                    if attempt < API_RETRY_TIMES - 1:
                        time.sleep(API_RETRY_INTERVAL)
                    else:
                        raise  # All attempts failed, raise exception to enter existing except logic

        except (APIError, APIStatusError, APIResponseValidationError, TypeError) as e:
            logger.exception(f"Anthropic API error: {str(e)}")
            try:
                logger.warning("Retrying with backup API key...")

                backup_client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY_BACKUP"), max_retries=4)
                if self.model_name in COMPUTER_20250124_MODELS or self.model_name in COMPUTER_20251124_MODELS:
                    response = backup_client.beta.messages.create(
                        max_tokens=self.max_tokens,
                        messages=self.messages,
                        model=PROVIDER_TO_DEFAULT_MODEL_NAME[APIProvider.ANTHROPIC, self.model_name],
                        system=[system],
                        tools=tools,
                        betas=betas,
                        extra_body=extra_body
                    )
                elif self.model_name == "claude-3-5-sonnet-20241022":
                    response = backup_client.beta.messages.create(
                        max_tokens=self.max_tokens,
                        messages=self.messages,
                        model=PROVIDER_TO_DEFAULT_MODEL_NAME[APIProvider.ANTHROPIC, self.model_name],
                        system=[system],
                        tools=tools,
                        betas=betas,
                    )
                logger.info("Successfully used backup API key")
            except Exception as backup_e:
                backup_error_msg = str(backup_e)
                logger.exception(f"Backup API call also failed: {backup_error_msg}")
                
                # Check if backup API also has 25MB limit error
                if "25000000" in backup_error_msg or "Member must have length less than or equal to" in backup_error_msg:
                    logger.warning("Backup API also encountered 25MB limit error, further reducing image count")
                    # Reduce image count by half again
                    current_image_count = self.only_n_most_recent_images
                    new_image_count = max(1, current_image_count // 2)  # Keep at least 1 image
                    self.only_n_most_recent_images = new_image_count
                    
                    # Reapply image filtering
                    _maybe_filter_to_n_most_recent_images(
                        self.messages,
                        new_image_count,
                        min_removal_threshold=image_truncation_threshold,
                    )
                    logger.info(f"Backup API image count reduced from {current_image_count} to {new_image_count}")
                
                error_response = f"Anthropic backup API call failed: {backup_error_msg}"
                logger.error(error_response)
                return error_response, ["FAIL"]

        except Exception as e:
            error_response = f"Error in Anthropic API: {str(e)}"
            logger.exception(error_response)
            return error_response, ["FAIL"]

        response_params = _response_to_params(response, model_name=self.model_name)
        logger.info(f"Received response params: {response_params}")

        # Store response in message history
        self.messages.append({
            "role": "assistant",
            "content": response_params
        })

        max_parse_retry = 3
        for parse_retry in range(max_parse_retry):
            actions: list[Any] = []
            reasonings: list[str] = []
            try:
                for content_block in response_params:
                    if content_block["type"] == "tool_use":
                        actions.append({
                            "name": content_block["name"],
                            "input": cast(dict[str, Any], content_block["input"]),
                            "id": content_block["id"],
                            "action_type": content_block.get("type"),
                            "command": self.parse_actions_from_tool_call(content_block)
                        })
                    elif content_block["type"] == "text":
                        reasonings.append(content_block["text"])
                if isinstance(reasonings, list) and len(reasonings) > 0:
                    reasonings = reasonings[0]
                else:
                    reasonings = ""
                logger.info(f"Received actions: {actions}")
                logger.info(f"Received reasonings: {reasonings}")
                if len(actions) == 0:
                    actions = ["DONE"]
                return reasonings, actions
            except Exception as e:
                logger.warning(f"parse_actions_from_tool_call parsing failed (attempt {parse_retry+1}/3), will retry API request: {e}")
                # Remove the recently appended assistant message to avoid polluting history
                self.messages.pop()
                # Retry API request
                response = None
                for attempt in range(API_RETRY_TIMES):
                    try:
                        if self.model_name in COMPUTER_20250124_MODELS or self.model_name in COMPUTER_20251124_MODELS:
                            response = client.beta.messages.create(
                                max_tokens=self.max_tokens,
                                messages=self.messages,
                                model=PROVIDER_TO_DEFAULT_MODEL_NAME[self.provider, self.model_name],
                                system=[system],
                                tools=tools,
                                betas=betas,
                                extra_body=extra_body
                            )
                        elif self.model_name == "claude-3-5-sonnet-20241022":
                            response = client.beta.messages.create(
                                max_tokens=self.max_tokens,
                                messages=self.messages,
                                model=PROVIDER_TO_DEFAULT_MODEL_NAME[self.provider, self.model_name],
                                system=[system],
                                tools=tools,
                                betas=betas,
                            )
                        logger.info(f"Response: {response}")
                        break  # Success, exit retry loop
                    except (APIError, APIStatusError, APIResponseValidationError) as e2:
                        error_msg = str(e2)
                        logger.warning(f"Anthropic API error (attempt {attempt+1}/{API_RETRY_TIMES}): {error_msg}")
                        if attempt < API_RETRY_TIMES - 1:
                            time.sleep(API_RETRY_INTERVAL)
                        else:
                            raise
                response_params = _response_to_params(response, model_name=self.model_name)
                logger.info(f"Received response params: {response_params}")
                self.messages.append({
                    "role": "assistant",
                    "content": response_params
                })
                if parse_retry == max_parse_retry - 1:
                    logger.error(f"parse_actions_from_tool_call parsing failed 3 times consecutively, terminating: {e}")
                    actions = ["FAIL"]
                    return reasonings, actions
    def reset(self, _logger = None, *args, **kwargs):
        """
        Reset the agent's state.
        """
        global logger
        if _logger:
            logger = _logger
        else:
            logger = logging.getLogger("desktopenv.agent")
        self.messages = []
        logger.info(f"{self.class_name} reset.")
