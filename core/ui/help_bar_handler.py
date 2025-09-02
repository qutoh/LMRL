# /core/ui/help_bar_handler.py

from ..common.localization import loc
from .ui_framework import HelpBar, View
from .app_states import AppState

class HelpBarHandler:
    """Manages the state and rendering of the context-aware help bar."""

    def __init__(self, console_width: int, console_height: int):
        self.help_texts = {
            "DEFAULT": "help_bar_default",
            AppState.WORLD_SELECTION: "help_bar_world_select",
            AppState.SETTINGS: "help_bar_settings_nav",
            "SETTINGS_EDIT": "help_bar_settings_edit",
            "SETTINGS_CONFIRM": "help_bar_settings_confirm_exit",
            AppState.LOAD_OR_NEW: "help_bar_load_or_new",
            AppState.LOAD_GAME_SELECTION: "help_bar_load_game",
            AppState.SCENE_SELECTION: "help_bar_scene_select",
            AppState.AWAITING_THEME_INPUT: "help_bar_text_input",
            AppState.AWAITING_SCENE_INPUT: "help_bar_text_input",
            "PLAYER_TAKEOVER": "help_bar_player_takeover",
            "PROMETHEUS_TAKEOVER": "help_bar_prometheus",
            "ROLE_CREATOR_TAKEOVER": "help_bar_role_creator",
            AppState.GAME_RUNNING: "help_bar_game_running",
            "IN_GAME_INPUT": "help_bar_ingame_input",
            "IN_GAME_MENU": "help_bar_ingame_menu",
            AppState.CALIBRATING: "help_bar_calibrating",
            AppState.PEG_V3_TEST: "help_bar_peg_test",
            AppState.CHARACTER_GENERATION_TEST: "help_bar_char_gen_test",
            AppState.NOISE_VISUALIZER_TEST: "help_bar_noise_test",
            AppState.SHUTTING_DOWN: "help_bar_shutting_down"
        }
        self.widget = HelpBar(help_texts=self.help_texts, width=console_width, height=1, y=console_height - 1)

    def update(self, active_view: View | None, app_state: AppState):
        """Sets the correct context key for the help bar."""
        if active_view and active_view.help_context_key:
            self.widget.set_context(active_view.help_context_key)
        else:
            self.widget.set_context(app_state)

    def render(self, console):
        self.widget.render(console)