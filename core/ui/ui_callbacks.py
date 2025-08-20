# /core/ui/ui_callbacks.py

import copy
import random

from .app_states import AppState
from ..common import file_io
from ..common.config_loader import config
from ..llm.llm_api import execute_task
from ..worldgen.atlas_manager import AtlasManager


class UICallbacks:
    """
    Manages the application's response to user interactions from the UI views.
    This class acts as the 'controller' for the UI state machine.
    """

    def __init__(self, ui_manager):
        self.ui_manager = ui_manager

    def initiate_game_start(self):
        """Prepares the application to transition into the main game state."""
        self.ui_manager.event_log.add_message("Loading game models...", (200, 200, 255))
        self.ui_manager.model_manager.transition_to_state('GAME_RUNNING')
        self.ui_manager.app_state = AppState.LOADING_GAME
        self.ui_manager.active_view = None

    def on_world_selection(self, choice: str):
        """Callback for when a user selects an option on the main world selection screen."""
        self.ui_manager.active_view = None
        if choice == "EXIT":
            self.ui_manager.app_state = AppState.EXITING
            return

        if choice == "Build new...":
            self.ui_manager.app_state = AppState.AWAITING_THEME_INPUT
            return
        elif choice == "Settings":
            self.ui_manager.app_state = AppState.SETTINGS
            return
        elif choice == "Calibrate Task Temperatures":
            self.ui_manager.app_state = AppState.CALIBRATING
            self.ui_manager.special_modes.start_calibration()
            return
        elif choice == "PEG V3 Iterative Test":
            self.ui_manager.app_state = AppState.PEG_V3_TEST
            return
        elif choice == "Character Generation Test":
            self.ui_manager.app_state = AppState.CHARACTER_GENERATION_TEST
            return
        else:
            self.ui_manager.selected_world_name = choice
            config.load_world_data(self.ui_manager.selected_world_name)
            root_key = next(iter(config.world), None)
            self.ui_manager.world_theme = config.world[root_key].get('theme',
                                                                     'A generic fantasy world.') if root_key else 'A generic fantasy world.'
            saved_runs = file_io.get_saved_runs_for_world(self.ui_manager.selected_world_name)
            if saved_runs:
                self.ui_manager.app_state = AppState.LOAD_OR_NEW
            else:
                self.ui_manager.app_state = AppState.SCENE_SELECTION

    def handle_theme_submission(self, theme_text: str):
        """Callback for when a user submits a theme for a new world."""
        ui = self.ui_manager
        ui.active_view = None
        if not ui.model_manager.models_loaded.is_set():
            ui.event_log.add_message("World creation models are loading, please wait...", (255, 255, 100))
            ui.app_state = AppState.WORLD_SELECTION
            return
            
        # Ensure the atlas_engine has the loaded embedding model before use.
        ui.atlas_engine.embedding_model = ui.model_manager.embedding_model

        if not theme_text:
            atlas = AtlasManager(ui.atlas_engine)
            ui.world_theme = atlas.content_generator.get_world_name(
                execute_task(ui.atlas_engine, config.agents['ATLAS'], 'WORLDGEN_CREATE_THEME',
                             []) or "A generic fantasy world."
            )
        else:
            ui.world_theme = theme_text

        ui.app_state = AppState.WORLD_CREATING

    def _continue_to_game_start(self, is_newly_written_scene: bool):
        """Helper function to handle the final steps before starting the game engine process."""
        ui = self.ui_manager
        
        # Ensure the atlas_engine has the loaded embedding model before use.
        ui.atlas_engine.embedding_model = ui.model_manager.embedding_model
        
        if ui.starting_location is None:
            atlas = AtlasManager(ui.atlas_engine)
            ui.event_log.add_message("Finding a home for your scene...", (200, 200, 255))
            ui.starting_location, ui.location_breadcrumb = atlas.find_or_create_location_for_scene(
                ui.selected_world_name, ui.world_theme, ui.scene_prompt
            )
            if is_newly_written_scene and ui.starting_location:
                scenes_path = file_io.join_path(ui.atlas_engine.config.data_dir, 'worlds', ui.selected_world_name,
                                                'generated_scenes.json')
                all_scenes = file_io.read_json(scenes_path, default=[])

                # Create a sanitized copy of the location data for the scene file.
                # This prevents dynamic inhabitants from being saved into the reusable scene.
                sanitized_location = copy.deepcopy(ui.starting_location)
                if 'inhabitants' in sanitized_location:
                    del sanitized_location['inhabitants']

                new_scene_entry = {
                    "scene_prompt": ui.scene_prompt,
                    "source_location_name": ui.starting_location.get("Name"),
                    "source_location": sanitized_location
                }

                if not any(s.get('scene_prompt') == new_scene_entry['scene_prompt'] for s in all_scenes):
                    all_scenes.append(new_scene_entry)
                    file_io.write_json(scenes_path, all_scenes)
                    ui.event_log.add_message(f"Saved new scene to '{ui.selected_world_name}'.", (150, 255, 150))
                    config.generated_scenes = all_scenes

        if not ui.location_breadcrumb:
            ui.location_breadcrumb = [next(iter(config.world), None)]

        self.initiate_game_start()

    def on_scene_selection(self, choice: dict | str):
        """Callback for when a user selects a scene from the menu."""
        ui = self.ui_manager
        ui.active_view = None

        if isinstance(choice, str) and "RANDOM" in choice:
            from . import ui_data_provider
            if choice == "RANDOM_NORMAL":
                scenes = ui_data_provider.get_available_scenes()
                if not scenes: self.on_new_scene(); return
                random_choice = random.choice(scenes)
                ui.scene_prompt = random_choice.get('scene_prompt')
                ui.starting_location = random_choice.get('source_location')
            elif choice == "RANDOM_ANACHRONISM":
                anachronisms = ui_data_provider.get_anachronisms(ui.selected_world_name)
                if not anachronisms: self.on_new_scene(); return
                ui.scene_prompt = random.choice(anachronisms).get('scene_prompt')
                ui.starting_location = None
        elif isinstance(choice, dict):
            ui.scene_prompt = choice.get('scene_prompt')
            ui.starting_location = choice.get('source_location')

        self._continue_to_game_start(is_newly_written_scene=False)

    def on_new_scene(self):
        """Callback to switch to the new scene input view."""
        self.ui_manager.active_view = None
        self.ui_manager.app_state = AppState.AWAITING_SCENE_INPUT

    def on_new_scene_submit(self, scene_text: str):
        """Callback after the player has written a new scene."""
        ui = self.ui_manager
        ui.active_view = None
        is_newly_written = True

        if not scene_text:  # Player left it blank, so generate one instead
            from ..worldgen.scene_maker import SceneMakerManager
            ui.atlas_engine.world_theme = ui.world_theme
            scene_maker = SceneMakerManager(ui.atlas_engine)
            generated_scene = scene_maker.generate_scene(world_name=ui.selected_world_name)
            ui.scene_prompt = generated_scene.get('scene_prompt')
            ui.starting_location = generated_scene.get('source_location')
            is_newly_written = False
        else:  # Player wrote one
            ui.scene_prompt = scene_text
            ui.starting_location = None

        self._continue_to_game_start(is_newly_written_scene=is_newly_written)

    def on_load_new_choice(self, choice: str):
        """Callback for the 'Load Game' / 'New Game' screen."""
        self.ui_manager.active_view = None
        if choice == "LOAD":
            self.ui_manager.app_state = AppState.LOAD_GAME_SELECTION
        elif choice == "NEW":
            self.ui_manager.app_state = AppState.SCENE_SELECTION
        elif choice == "BACK":
            self.ui_manager.app_state = AppState.WORLD_SELECTION

    def on_save_selected(self, save_name: str):
        """Callback for when a specific save file is chosen."""
        self.ui_manager.active_view = None
        self.ui_manager.load_path = file_io.join_path(file_io.SAVE_DATA_ROOT, self.ui_manager.selected_world_name,
                                                      save_name)
        self.initiate_game_start()