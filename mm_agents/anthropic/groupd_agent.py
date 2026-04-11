from __future__ import annotations

from .groupc_agent import AnthropicGroupCAgent


COACT_GUI_SYSTEM_PROMPT = (
    "You are controlling a computer with a display resolution of 1280x720 pixels. "
    "When performing actions like clicking, use coordinates within this range."
)


class AnthropicGroupDAgent(AnthropicGroupCAgent):
    """Anthropic agent variant with GroupD system+user prompt templates."""

    def __init__(
        self,
        *args,
        system_prompt_override: str = COACT_GUI_SYSTEM_PROMPT,
        **kwargs,
    ):
        if not system_prompt_override:
            system_prompt_override = COACT_GUI_SYSTEM_PROMPT
        super().__init__(
            *args,
            system_prompt_override=system_prompt_override,
            **kwargs,
        )
