# /core/ui/player_interface_handler.py

from .views import TextInputView, MenuView, PrometheusView, RoleCreatorView
from .input_handler import TextInputHandler
from ..common.config_loader import config
from ..common.localization import loc
from ..common.utils import get_text_from_messages
from .ui_messages import InputRequestMessage, MenuRequestMessage


class PlayerInterfaceHandler:
    """
    Manages all direct player interaction by creating and dispatching the correct
    UI views for input, menus, and task takeovers.
    """

    def __init__(self, ui_manager):
        self.ui_manager = ui_manager
        self.app = ui_manager.app  # Convenience accessor

    def get_player_task_input(self, **kwargs):
        """Synchronously gets player input for a task, blocking the calling engine."""
        task_key = kwargs.get("task_key")
        task_params = config.task_parameters.get(task_key, {})
        if not task_params.get("player_takeover_enabled"):
            return None

        self.ui_manager.sync_input_active = True
        self.ui_manager.sync_input_result = None
        original_view = self.ui_manager.active_view

        def on_submit(result):
            self.ui_manager.sync_input_result = result
            self.ui_manager.sync_input_active = False

        handler_map = {
            "PROMETHEUS_DETERMINE_TOOL_USE": self.create_prometheus_menu_view,
            "DIRECTOR_DEFINE_LEAD_ROLES_FOR_SCENE": self.create_role_creator_view,
            "DIRECTOR_CAST_REPLACEMENT": None,
        }
        creation_func = handler_map.get(task_key, self.create_default_takeover_view)
        if creation_func:
            creation_func(on_submit, **kwargs)
        else:
            self.ui_manager.sync_input_active = False
            return None

        while self.ui_manager.sync_input_active:
            self.ui_manager._sync_text_input_state()
            self.ui_manager.help_bar_handler.update(self.ui_manager.active_view, self.app.app_state)
            self.ui_manager.render()
            import tcod.event
            for event in tcod.event.get():
                converted_event = self.ui_manager.context.convert_event(event)
                if self.ui_manager.active_view: self.ui_manager.active_view.handle_event(converted_event)
                if isinstance(converted_event, tcod.event.Quit):
                    self.app.app_state = self.app.AppState.EXITING
                    self.ui_manager.sync_input_active = False
                    self.ui_manager.sync_input_result = None
        self.ui_manager.active_view = original_view
        return self.ui_manager.sync_input_result if self.ui_manager.sync_input_result is not None else None

    def create_default_takeover_view(self, on_submit, **handler_kwargs):
        task_key = handler_kwargs.get("task_key")
        agent_name = handler_kwargs.get("agent_name")
        messages = handler_kwargs.get("messages", [])
        task_prompt_kwargs = handler_kwargs.get("task_prompt_kwargs", {})
        self.ui_manager._send_player_notification(task_key)
        task_params = config.task_parameters.get(task_key, {})
        task_prompt_key = task_params.get('task_prompt_key')
        task_instruction = loc(task_prompt_key,
                               **(task_prompt_kwargs or {})) if task_prompt_key else f"Perform task: {task_key}"
        message_history_text = get_text_from_messages(messages[-5:])
        full_prompt_for_player = (
            f"PLAYER TAKEOVER: {task_key}\n\n"
            f"TASK: {task_instruction}\n\n"
            f"ACTING AS: {agent_name}\n"
            f"--------------------\n"
            f"RECENT HISTORY:\n{message_history_text}"
        )
        handler = TextInputHandler(
            prompt=full_prompt_for_player,
            width=self.ui_manager.root_console.width - 4,
            prefix="> ",
            max_tokens=self.ui_manager._get_default_token_limit()
        )
        self.ui_manager.active_view = TextInputView(handler=handler, on_submit=on_submit)

    def create_prometheus_menu_view(self, on_submit, **handler_kwargs):
        task_key = handler_kwargs.get("task_key")
        agent_name = handler_kwargs.get("agent_name")
        persona = handler_kwargs.get("persona")
        messages = handler_kwargs.get("messages", [])
        task_prompt_kwargs = handler_kwargs.get("task_prompt_kwargs", {})
        self.ui_manager._send_player_notification(task_key)
        task_params = config.task_parameters.get(task_key, {})
        task_prompt_key = task_params.get('task_prompt_key')
        instruction_text = loc(task_prompt_key, **(task_prompt_kwargs or {}))
        message_history_text = get_text_from_messages(messages)
        context_text = (
            f"--- Agent: {agent_name} ---\n\n"
            f"--- Persona ---\n{persona}\n\n"
            f"--- Message History ---\n{message_history_text}"
        )
        tool_list = list(self.app.atlas_engine.prometheus_manager.tool_dispatch_table.keys())
        self.ui_manager.active_view = PrometheusView(
            tools=tool_list,
            instruction_text=instruction_text,
            context_text=context_text,
            on_submit=on_submit,
            console_width=self.ui_manager.root_console.width,
            console_height=self.ui_manager.root_console.height
        )

    def create_role_creator_view(self, on_submit, **handler_kwargs):
        task_key = handler_kwargs.get("task_key")
        self.ui_manager._send_player_notification(task_key)
        task_prompt_kwargs = handler_kwargs.get("task_prompt_kwargs", {})
        context_text = (
            f"World Theme: {task_prompt_kwargs.get('prompt_substring_world_scene_context', '')}\n\n"
            f"Location: {task_prompt_kwargs.get('location_summary', 'N/A')}\n\n"
            f"Existing NPCs: {task_prompt_kwargs.get('npc_list_summary', 'None')}"
        )
        view = RoleCreatorView(
            on_submit=on_submit,
            context_text=context_text,
            console_width=self.ui_manager.root_console.width,
            console_height=self.ui_manager.root_console.height,
            app_controller=self.app
        )
        self.ui_manager.active_view = view

    def create_in_game_input_view(self, msg: InputRequestMessage):
        def on_submit(text): self.ui_manager.input_queue.put(text); self.ui_manager.active_view = self.ui_manager.game_view
        handler = TextInputHandler(prompt=msg.prompt, width=self.ui_manager.root_console.width - 2, prefix="> ", max_tokens=msg.max_tokens, initial_text=msg.initial_text)
        self.ui_manager.active_view = TextInputView(handler=handler, on_submit=on_submit)

    def create_in_game_menu_view(self, msg: MenuRequestMessage):
        def on_submit(choice: str | None): self.ui_manager.input_queue.put(choice); self.ui_manager.active_view = self.ui_manager.game_view
        self.ui_manager.active_view = MenuView(
            title=msg.title,
            options=msg.options,
            on_choice=on_submit,
            console_width=self.ui_manager.root_console.width,
            console_height=self.ui_manager.root_console.height
        )