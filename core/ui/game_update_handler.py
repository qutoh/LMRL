# /core/ui/game_update_handler.py

from .views import GameView
from .ui_messages import (
    UIMessage, AddEventLogMessage, StreamStartMessage, StreamTokenMessage,
    StreamEndMessage, InputRequestMessage, MenuRequestMessage,
    PlayerTaskTakeoverRequestMessage, PrometheusMenuRequestMessage,
    RoleCreatorRequestMessage
)

class GameUpdateHandler:
    """
    Processes messages from the engine's render queue and updates the UI accordingly.
    """

    def __init__(self, ui_manager, player_interface_handler):
        self.ui_manager = ui_manager
        self.player_interface = player_interface_handler
        self.app = ui_manager.app
        self.render_queue = None

        self.dispatch = {
            AddEventLogMessage: self._handle_add_event_log,
            StreamStartMessage: self._handle_stream_start,
            StreamTokenMessage: self._handle_stream_token,
            StreamEndMessage: self._handle_stream_end,
            InputRequestMessage: self._handle_input_request,
            MenuRequestMessage: self._handle_menu_request,
            PlayerTaskTakeoverRequestMessage: self._handle_task_takeover,
            PrometheusMenuRequestMessage: self._handle_prometheus_menu,
            RoleCreatorRequestMessage: self._handle_role_creator
        }

    def set_queue(self, queue):
        self.render_queue = queue

    def process_queue(self):
        """Processes all pending messages from the engine process queue."""
        if self.render_queue and not self.render_queue.empty():
            try:
                message = self.render_queue.get(timeout=0.01)
                if message is None:
                    self.app.app_state = self.app.AppState.SHUTTING_DOWN
                elif isinstance(message, UIMessage):
                    handler = self.dispatch.get(type(message))
                    if handler:
                        handler(message)
                else:  # Assumes it's a GameState object
                    if isinstance(self.ui_manager.active_view, GameView):
                        self.ui_manager.active_view.update_state(message)
            except Exception:
                pass

    def _handle_add_event_log(self, msg: AddEventLogMessage):
        self.ui_manager.event_log.add_message(msg.text, fg=msg.color)

    def _handle_stream_start(self, msg: StreamStartMessage):
        if isinstance(self.ui_manager.active_view, GameView):
            self.ui_manager.active_view.start_new_log_entry(speaker=msg.speaker)

    def _handle_stream_token(self, msg: StreamTokenMessage):
        if isinstance(self.ui_manager.active_view, GameView):
            self.ui_manager.active_view.append_to_active_log(text_delta=msg.delta, is_retry_clear=msg.is_retry_clear)

    def _handle_stream_end(self, msg: StreamEndMessage):
        if isinstance(self.ui_manager.active_view, GameView):
            self.ui_manager.active_view.finalize_active_log()

    def _handle_input_request(self, msg: InputRequestMessage):
        self.player_interface.create_in_game_input_view(msg)

    def _handle_menu_request(self, msg: MenuRequestMessage):
        self.player_interface.create_in_game_menu_view(msg)

    def _handle_task_takeover(self, msg: PlayerTaskTakeoverRequestMessage):
        def on_submit(text): self.ui_manager.input_queue.put(text); self.ui_manager.active_view = self.ui_manager.game_view
        self.player_interface.create_default_takeover_view(on_submit, **msg.handler_kwargs)

    def _handle_prometheus_menu(self, msg: PrometheusMenuRequestMessage):
        def on_submit(choices): self.ui_manager.input_queue.put(choices); self.ui_manager.active_view = self.ui_manager.game_view
        self.player_interface.create_prometheus_menu_view(on_submit, **msg.handler_kwargs)

    def _handle_role_creator(self, msg: RoleCreatorRequestMessage):
        def on_submit(result): self.ui_manager.input_queue.put(result); self.ui_manager.active_view = self.ui_manager.game_view
        self.player_interface.create_role_creator_view(on_submit, **msg.handler_kwargs)