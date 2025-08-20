# /core/ui/player_input_handlers.py

from .input_handler import TextInputHandler
from .views import TextInputView, PrometheusView
from ..common.config_loader import config
from ..common.localization import loc
from ..common.utils import get_text_from_messages


def create_default_takeover_view(ui_manager, on_submit, **handler_kwargs):
    """Creates a standard text input view for a player task takeover, now with full context."""
    task_key = handler_kwargs.get("task_key")
    agent_name = handler_kwargs.get("agent_name")
    messages = handler_kwargs.get("messages", [])
    task_prompt_kwargs = handler_kwargs.get("task_prompt_kwargs", {})

    ui_manager._send_player_notification(task_key)
    task_params = config.task_parameters.get(task_key, {})
    task_prompt_key = task_params.get('task_prompt_key')
    task_instruction = loc(task_prompt_key,
                           **(task_prompt_kwargs or {})) if task_prompt_key else f"Perform task: {task_key}"

    # Assemble the full context string to be used as the prompt/frame title.
    # textwrap will handle making it fit.
    message_history_text = get_text_from_messages(messages[-5:])  # Last 5 messages for brevity

    full_prompt_for_player = (
        f"PLAYER TAKEOVER: {task_key}\n\n"
        f"TASK: {task_instruction}\n\n"
        f"ACTING AS: {agent_name}\n"
        f"--------------------\n"
        f"RECENT HISTORY:\n{message_history_text}"
    )

    handler = TextInputHandler(
        prompt=full_prompt_for_player,
        width=ui_manager.root_console.width - 4,
        prefix="> ",
        max_tokens=ui_manager._get_default_token_limit()
    )
    # The call to TextInputView is now correct, without the bad argument.
    ui_manager.active_view = TextInputView(handler=handler, on_submit=on_submit)


def create_prometheus_menu_view(ui_manager, on_submit, **handler_kwargs):
    """Creates the specialized Prometheus tool selection menu."""
    task_key = handler_kwargs.get("task_key")
    agent_name = handler_kwargs.get("agent_name")
    persona = handler_kwargs.get("persona")
    messages = handler_kwargs.get("messages", [])
    task_prompt_kwargs = handler_kwargs.get("task_prompt_kwargs", {})

    ui_manager._send_player_notification(task_key)
    task_params = config.task_parameters.get(task_key, {})
    task_prompt_key = task_params.get('task_prompt_key')

    instruction_text = loc(task_prompt_key, **(task_prompt_kwargs or {}))
    message_history_text = get_text_from_messages(messages)
    context_text = (
        f"--- Agent: {agent_name} ---\n\n"
        f"--- Persona ---\n{persona}\n\n"
        f"--- Message History ---\n{message_history_text}"
    )

    tool_list = list(ui_manager.atlas_engine.prometheus_manager.tool_dispatch_table.keys())
    ui_manager.active_view = PrometheusView(
        tools=tool_list,
        instruction_text=instruction_text,
        context_text=context_text,
        on_submit=on_submit,
        console_width=ui_manager.root_console.width,
        console_height=ui_manager.root_console.height
    )