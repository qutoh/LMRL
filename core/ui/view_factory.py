# /core/ui/view_factory.py

from .app_states import AppState
from . import ui_data_provider
from .views import (
    WorldSelectionView, SceneSelectionView, LoadOrNewView, SaveSelectionView,
    TabbedSettingsView, TextInputView, CalibrationView
)
from .input_handler import TextInputHandler
from ..common import file_io


class ViewFactory:
    """Creates UI view instances based on the application state."""

    def __init__(self, app_controller, ui_manager):
        self.app = app_controller
        self.ui = ui_manager
        self.view_map = {
            AppState.WORLD_SELECTION: self._create_world_selection_view,
            AppState.SETTINGS: self._create_settings_view,
            AppState.LOAD_OR_NEW: self._create_load_or_new_view,
            AppState.LOAD_GAME_SELECTION: self._create_save_selection_view,
            AppState.SCENE_SELECTION: self._create_scene_selection_view,
            AppState.AWAITING_THEME_INPUT: self._create_theme_input_view,
            AppState.AWAITING_SCENE_INPUT: self._create_scene_input_view,
            AppState.CALIBRATING: self._create_calibration_view,
            AppState.WORLD_CREATING: self._create_world_graph_view,
            AppState.CHARACTER_GENERATION_TEST: self._create_char_gen_test_input_view
        }

    def create_view(self, app_state: AppState):
        """Creates and returns the appropriate view for the given state."""
        # The NOISE_VISUALIZER_TEST state is handled by the SpecialModeManager, so it does not need a factory entry.
        creator_func = self.view_map.get(app_state)
        if creator_func:
            return creator_func()
        # States that manage their own views or have no view return None
        return None

    def _create_world_selection_view(self):
        return WorldSelectionView(
            worlds=ui_data_provider.get_available_worlds(),
            on_choice=self.app.on_world_selection,
            console_width=self.ui.root_console.width,
            console_height=self.ui.root_console.height
        )

    def _create_settings_view(self):
        def on_back():
            self.app.app_state = AppState.WORLD_SELECTION

        return TabbedSettingsView(
            on_back=on_back,
            console_width=self.ui.root_console.width,
            console_height=self.ui.root_console.height
        )

    def _create_load_or_new_view(self):
        return LoadOrNewView(
            on_choice=self.app.on_load_new_choice,
            console_width=self.ui.root_console.width,
            console_height=self.ui.root_console.height
        )

    def _create_save_selection_view(self):
        saved_runs = file_io.get_saved_runs_for_world(self.app.selected_world_name)
        return SaveSelectionView(
            saves=saved_runs,
            on_choice=self.app.on_save_selected,
            on_back=lambda: self.app.on_load_new_choice("BACK"),
            console_width=self.ui.root_console.width,
            console_height=self.ui.root_console.height
        )

    def _create_scene_selection_view(self):
        return SceneSelectionView(
            scenes=ui_data_provider.get_available_scenes(),
            anachronisms=ui_data_provider.get_anachronisms(self.app.selected_world_name),
            on_choice=self.app.on_scene_selection,
            on_new=self.app.on_new_scene,
            on_back=lambda: self.app.on_world_selection(self.app.selected_world_name),
            console_width=self.ui.root_console.width,
            console_height=self.ui.root_console.height
        )

    def _create_theme_input_view(self):
        prompt = "Enter a theme for the new world (or leave blank for a generated one)."
        handler = TextInputHandler(prompt=prompt, width=self.ui.root_console.width - 2, prefix="> ",
                                   max_tokens=self.ui._get_default_token_limit())
        return TextInputView(handler=handler, on_submit=self.app.handle_theme_submission)

    def _create_scene_input_view(self):
        prompt = "Write an opening scene to begin the story (or leave blank for a generated one)."
        handler = TextInputHandler(prompt=prompt, width=self.ui.root_console.width - 2, prefix="> ",
                                   max_tokens=self.ui._get_default_token_limit())
        return TextInputView(handler=handler, on_submit=self.app.on_new_scene_submit)

    def _create_calibration_view(self):
        return CalibrationView(
            console_width=self.ui.root_console.width,
            console_height=self.ui.root_console.height
        )

    def _create_world_graph_view(self):
        # This view is created dynamically during the worldgen process
        return None

    def _create_char_gen_test_input_view(self):
        def on_submit(context: str):
            if context:
                self.ui.special_modes.start_character_generation_test(context)
            else:
                self.app.app_state = AppState.WORLD_SELECTION

        prompt = "Enter a scene context for character generation."
        handler = TextInputHandler(prompt=prompt, width=self.ui.root_console.width - 2, prefix="> ",
                                   max_tokens=self.ui._get_default_token_limit())
        return TextInputView(handler=handler, on_submit=on_submit)