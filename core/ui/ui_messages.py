# /core/ui/ui_messages.py

from typing import Tuple, List, Optional

class UIMessage:
    """Base class for messages sent from the engine to the UI."""
    pass

class AddEventLogMessage(UIMessage):
    def __init__(self, text: str, color: Tuple[int, int, int] = (255, 255, 255)):
        self.text = text
        self.color = color

class StreamStartMessage(UIMessage):
    def __init__(self, speaker: str):
        self.speaker = speaker

class StreamTokenMessage(UIMessage):
    def __init__(self, delta: str, is_retry_clear: bool = False):
        self.delta = delta
        self.is_retry_clear = is_retry_clear

class StreamEndMessage(UIMessage):
    pass

class InputRequestMessage(UIMessage):
    def __init__(self, prompt: str, max_tokens: int, initial_text: str = ""):
        self.prompt = prompt
        self.max_tokens = max_tokens
        self.initial_text = initial_text

class MenuRequestMessage(UIMessage):
    def __init__(self, title: str, options: List[str]):
        self.title = title
        self.options = options

class PlayerTaskTakeoverRequestMessage(UIMessage):
    def __init__(self, handler_kwargs: dict):
        self.handler_kwargs = handler_kwargs

class PrometheusMenuRequestMessage(UIMessage):
    def __init__(self, handler_kwargs: dict):
        self.handler_kwargs = handler_kwargs

class RoleCreatorRequestMessage(UIMessage):
    def __init__(self, handler_kwargs: dict):
        self.handler_kwargs = handler_kwargs