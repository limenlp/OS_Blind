"""
Anthropic agent implementation
"""

from .main import AnthropicAgent
from .modified_prompt_agent import AnthropicModifiedPromptAgent
from .groupc_agent import AnthropicGroupCAgent
from .groupd_agent import AnthropicGroupDAgent
from .tools import (
    BashTool,
    CLIResult,
    ComputerTool,
    EditTool,
    ToolCollection,
    ToolResult
)

__all__ = [
    'AnthropicAgent',
    'AnthropicModifiedPromptAgent',
    'AnthropicGroupCAgent',
    'AnthropicGroupDAgent',
    'BashTool',
    'CLIResult',
    'ComputerTool',
    'EditTool',
    'ToolCollection',
    'ToolResult'
]
