from __future__ import annotations

from typing import Any, Dict

from .main import AnthropicAgent


COACT_GUI_PROMPT_TEMPLATE = """# Task
{instruction}

# Hints
- Sudo password is "{CLIENT_PASSWORD}".
- Keep the windows/applications opened at the end of the task.
- Do not use shortcut to reload the application except for the browser, just close and reopen.
- If "The document has been changed by others" pops out, you should click "cancel" and reopen the file.
- If you need to download an email attachment: scroll to the very bottom of the email to find attachments first. After downloading, do NOT click the browser's top-right download popup. Use the File Manager on the right taskbar or open Terminal to access ~/Downloads/.
- If you have completed the user task, reply with the information you want the user to know along with 'TERMINATE'.
- If you don't know how to continue the task, reply your concern or question along with 'IDK'."""


class AnthropicGroupCAgent(AnthropicAgent):
    """Anthropic agent variant with GroupC user prompt wrapping."""

    def __init__(self, *args, client_password: str = "password", **kwargs):
        super().__init__(*args, **kwargs)
        self.client_password = client_password

    def _wrap_initial_instruction(self, instruction: str) -> str:
        return COACT_GUI_PROMPT_TEMPLATE.format(
            instruction=instruction,
            CLIENT_PASSWORD=self.client_password,
        )

    def predict(self, task_instruction: str, obs: Dict = None, system: Any = None):
        if not self.messages:
            task_instruction = self._wrap_initial_instruction(task_instruction)
        return super().predict(task_instruction, obs=obs, system=system)
