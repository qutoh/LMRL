# /core/ui/player_input_handlers.py

from ..common.config_loader import config
from ..common.localization import loc
from ..common.utils import get_text_from_messages, serialize_character_for_prompt
from .views import TextInputView, PrometheusView
from .input_handler import TextInputHandler


def create_default_takeover_view(ui_manager, on_submit, **handler_kwargs):
    """Creates a standard text input view for a player task takeover, now with full context."""
    task_key = handler_kwargs.get("task_key")
    agent_name = handler_kwargs.get("agent_name")
    persona = handler_kwargs.get("persona")
    messages = handler_kwargs.get("messages", [])
    task_prompt_kwargs = handler_kwargs.get("task_prompt_kwargs", {})

    ui_manager._send_player_notification(task_key)
    task_params = config.task_parameters.get(task_key, {})
    task_prompt_key = task_params.get('task_prompt_key')
    task_instruction = loc(task_prompt_key,
                           **(task_prompt_kwargs or {})) if task_prompt_key else f"Perform task: {task_key}"

    # Assemble the full context string, similar to what the LLM would receive
    message_history_text = get_text_from_messages(messages)
    full_context_text = (
        f"--- AGENT ---\n{agent_name}\n\n"
        f"--- PERSONA / INSTRUCTIONS ---\n{persona}\n\n"
        f"--- MESSAGE HISTORY ---\n{message_history_text}"
    )

    # The prompt for the text handler is just the AI's task-specific instruction
    handler_prompt = f"Your Response (as {agent_name})"

    # Calculate height for the input handler
    console_height = ui_manager.root_console.height
    input_height = (console_height - 4) // 3

    handler = TextInputHandler(
        prompt=handler_prompt,
        width=ui_manager.root_console.width - 4,
        height=input_height,
        prefix="> ",
        max_tokens=ui_manager._get_default_token_limit()
    )
    ui_manager.active_view = TextInputView(handler=handler, on_submit=on_submit, context_text=full_context_text)


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