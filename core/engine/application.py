# /core/engine/application.py

import copy
import random
import tcod
import threading
from multiprocessing import Queue, Process

from .story_engine import StoryEngine
from .setup_manager import SetupManager
from ..common import file_io, utils
from ..common.config_loader import config
from ..common.game_state import GameState
from ..llm.llm_api import execute_task
from ..llm.model_manager import ModelManager
from ..ui.app_states import AppState
from ..ui.ui_manager import UIManager, engine_process
from ..worldgen.atlas_manager import AtlasManager


class Application:
    """
    The main application class, responsible for orchestrating the entire program,
    managing state transitions, and handling core logic.
    """

    def __init__(self, context, root_console, tileset, load_path):
        self.context = context
        self.root_console = root_console
        self.tileset = tileset
        self.load_path = load_path

        self.app_state = AppState.WORLD_SELECTION
        self.model_manager = ModelManager()
        self.ui_manager = UIManager(self, context, root_console, self.model_manager, tileset)

        self.model_manager.transition_to_state('WORLD_CREATION')

        # Game State attributes
        self.engine_proc = None
        self.input_queue = None

        self.selected_world_name = None
        self.world_theme = "A generic fantasy world."
        self.scene_prompt = None
        self.starting_location = None
        self.location_breadcrumb = None

        # World Creation attributes
        self.atlas_engine = StoryEngine(world_name=None, ui_manager=self.ui_manager.player_interface_handler)
        self.atlas_engine.embedding_model = None
        self.worldgen_generator = None

        # --- Calibration Attributes ---
        self.calibration_thread = None
        self.calibration_jobs = []
        self.current_job_index = 0
        self.calibration_update_queue = Queue()

    def run(self):
        """The main application loop."""
        while self.app_state != AppState.EXITING:
            self.ui_manager.set_view_for_state(self.app_state)
            self._process_logic()
            self.ui_manager.render()
            self.ui_manager.process_events()

            if self.app_state == AppState.SHUTTING_DOWN:
                if self.engine_proc and self.engine_proc.is_alive():
                    self.engine_proc.join(timeout=5.0)
                self.app_state = AppState.EXITING
                continue

        self.shutdown()

    def _process_logic(self):
        """Handles the primary logic for the current application state."""
        if self.app_state == AppState.LOADING_GAME:
            self._handle_game_loading()
        elif self.app_state == AppState.GAME_RUNNING:
            self.ui_manager.game_update_handler.process_queue()
        elif self.app_state == AppState.WORLD_CREATING:
            self._handle_world_creation_step()
        elif self.app_state == AppState.CALIBRATING:
            self._handle_calibration_updates()
            self._handle_calibration_step()
        elif self.app_state in [AppState.PEG_V3_TEST, AppState.CHARACTER_GENERATION_TEST]:
            self.ui_manager.special_modes.process_logic(self.app_state)

    def _handle_game_loading(self):
        """Checks if models are ready and then starts the engine process."""
        if self.model_manager.models_loaded.is_set():
            self.start_engine_process()
            self.app_state = AppState.GAME_RUNNING

    def _handle_world_creation_step(self):
        if not self.model_manager.models_loaded.is_set(): return

        if self.worldgen_generator is None:
            atlas = AtlasManager(self.atlas_engine)
            self.worldgen_generator = atlas.create_new_world_generator(self.world_theme)

        try:
            result = next(self.worldgen_generator)
            if result['status'] == 'update':
                self.ui_manager.update_world_graph_view(result['world_data'])
            elif result['status'] == 'complete':
                self.selected_world_name = result['world_name']
                if self.selected_world_name:
                    config.load_world_data(self.selected_world_name)
                    self.app_state = AppState.SCENE_SELECTION
                else:
                    self.ui_manager.event_log.add_message("FATAL: World creation failed.", (255, 50, 50))
                    self.app_state = AppState.WORLD_SELECTION
                self.worldgen_generator = None
        except StopIteration:
            self.worldgen_generator = None
            if self.app_state == AppState.WORLD_CREATING:
                self.app_state = AppState.WORLD_SELECTION

    def start_engine_process(self):
        """Initializes queues and starts the StoryEngine in a separate process."""
        render_queue, self.input_queue = Queue(), Queue()
        self.ui_manager.set_engine_queues(render_queue, self.input_queue)
        self.engine_proc = Process(target=engine_process, args=(
            render_queue, self.input_queue, self.load_path,
            self.selected_world_name, self.starting_location,
            self.world_theme, self.scene_prompt,
            self.model_manager.embedding_model,
            self.model_manager.active_llm_models,
            self.location_breadcrumb
        ))
        self.engine_proc.start()

    def shutdown(self):
        """Handles graceful shutdown of the application and engine process."""
        self.ui_manager.shutdown()
        if self.engine_proc and self.engine_proc.is_alive():
            self.engine_proc.terminate()
            self.engine_proc.join()
        print("\n[SYSTEM] Application has finished. Exiting.")

    def _handle_calibration_updates(self):
        """Processes messages from the calibration thread queue to update the UI."""
        from ..ui.views import CalibrationView
        if self.calibration_update_queue and not self.calibration_update_queue.empty():
            try:
                while not self.calibration_update_queue.empty():
                    message = self.calibration_update_queue.get_nowait()
                    if isinstance(self.ui_manager.active_view, CalibrationView):
                        if message.get('type') == 'phase_change':
                            self.ui_manager.active_view.set_phase(message['phase'], message['text'])
                        else:
                            self.ui_manager.active_view.update_data(message)
            except Exception:
                pass

    def _handle_calibration_step(self):
        """Checks calibration progress and starts the next job if ready."""
        from ..llm.calibration_manager import CalibrationManager
        if self.calibration_thread is None or not self.calibration_thread.is_alive():
            if self.current_job_index >= len(self.calibration_jobs):
                if self.calibration_jobs:
                    self.calibration_update_queue.put(
                        {'type': 'status', 'text': "Calibration complete for all models."})
                    import time
                    time.sleep(3)
                self.app_state = AppState.WORLD_SELECTION
            else:
                job = self.calibration_jobs[self.current_job_index]
                self.calibration_thread = threading.Thread(target=self._run_single_calibration_job, args=(job,),
                                                           daemon=True)
                self.calibration_thread.start()
                self.current_job_index += 1

    def _run_single_calibration_job(self, job):
        """The target function for the calibration thread. Runs one job."""
        from ..llm.calibration_manager import CalibrationManager
        model_id = job['model_id']
        self.calibration_update_queue.put({'type': 'status', 'text': f"Loading model for calibration: {model_id}..."})
        self.model_manager.set_active_models([model_id])
        self.model_manager.models_loaded.wait()
        cal_manager = CalibrationManager(self.atlas_engine, self.model_manager.embedding_model,
                                         self.ui_manager.event_log, self.calibration_update_queue)
        cal_manager.run_calibration_test(job)

    # --- CALLBACKS (Moved from UICallbacks) ---

    def initiate_game_start(self):
        """Prepares the application to transition into the main game state."""
        self.ui_manager.event_log.add_message("Loading game models...", (200, 200, 255))
        self.model_manager.transition_to_state('GAME_RUNNING')
        self.app_state = AppState.LOADING_GAME

    def on_world_selection(self, choice: str):
        if choice == "EXIT":
            self.app_state = AppState.EXITING
            return
        if choice == "Build new...":
            self.app_state = AppState.AWAITING_THEME_INPUT
        elif choice == "Settings":
            self.app_state = AppState.SETTINGS
        elif choice == "Calibrate Task Temperatures":
            from ..llm.calibration_manager import CalibrationManager
            cal_manager = CalibrationManager(self.atlas_engine, self.model_manager.embedding_model,
                                             self.ui_manager.event_log)
            self.calibration_jobs = cal_manager.get_calibration_plan()
            self.current_job_index = 0
            self.app_state = AppState.CALIBRATING
        elif choice == "PEG V3 Iterative Test":
            self.app_state = AppState.PEG_V3_TEST
        elif choice == "Character Generation Test":
            self.app_state = AppState.CHARACTER_GENERATION_TEST
        else:
            self.selected_world_name = choice
            config.load_world_data(self.selected_world_name)
            root_key = next(iter(config.world), None)
            self.world_theme = config.world[root_key].get('theme',
                                                          'A generic fantasy world.') if root_key else 'A generic fantasy world.'
            saved_runs = file_io.get_saved_runs_for_world(self.selected_world_name)
            if saved_runs:
                self.app_state = AppState.LOAD_OR_NEW
            else:
                self.app_state = AppState.SCENE_SELECTION

    def handle_theme_submission(self, theme_text: str):
        if not self.model_manager.models_loaded.is_set():
            self.ui_manager.event_log.add_message("World creation models are loading, please wait...", (255, 255, 100))
            self.app_state = AppState.WORLD_SELECTION
            return

        self.atlas_engine.embedding_model = self.model_manager.embedding_model

        if not theme_text:
            self.world_theme = execute_task(self.atlas_engine, config.agents['ATLAS'], 'WORLDGEN_CREATE_THEME',
                                            []) or "A generic fantasy world."
        else:
            self.world_theme = theme_text

        self.app_state = AppState.WORLD_CREATING

    def _continue_to_game_start(self, is_newly_written_scene: bool):
        self.atlas_engine.embedding_model = self.model_manager.embedding_model
        if self.starting_location is None:
            atlas = AtlasManager(self.atlas_engine)
            self.ui_manager.event_log.add_message("Finding a home for your scene...", (200, 200, 255))
            self.starting_location, self.location_breadcrumb = atlas.find_or_create_location_for_scene(
                self.selected_world_name, self.world_theme, self.scene_prompt
            )
            if is_newly_written_scene and self.starting_location:
                scenes_path = file_io.join_path(self.atlas_engine.config.data_dir, 'worlds', self.selected_world_name,
                                                'generated_scenes.json')
                all_scenes = file_io.read_json(scenes_path, default=[])
                sanitized_location = copy.deepcopy(self.starting_location)
                if 'inhabitants' in sanitized_location: del sanitized_location['inhabitants']
                new_scene_entry = {
                    "scene_prompt": self.scene_prompt,
                    "source_location_name": self.starting_location.get("Name"),
                    "source_location": sanitized_location
                }
                if not any(s.get('scene_prompt') == new_scene_entry['scene_prompt'] for s in all_scenes):
                    all_scenes.append(new_scene_entry)
                    file_io.write_json(scenes_path, all_scenes)
                    self.ui_manager.event_log.add_message(f"Saved new scene to '{self.selected_world_name}'.",
                                                          (150, 255, 150))
                    config.generated_scenes = all_scenes
        if not self.location_breadcrumb:
            self.location_breadcrumb = [next(iter(config.world), None)]
        self.initiate_game_start()

    def on_scene_selection(self, choice: dict | str):
        from ..ui import ui_data_provider
        if isinstance(choice, str) and "RANDOM" in choice:
            if choice == "RANDOM_NORMAL":
                scenes = ui_data_provider.get_available_scenes()
                if not scenes: self.on_new_scene(); return
                random_choice = random.choice(scenes)
                self.scene_prompt = random_choice.get('scene_prompt')
                self.starting_location = random_choice.get('source_location')
            elif choice == "RANDOM_ANACHRONISM":
                anachronisms = ui_data_provider.get_anachronisms(self.selected_world_name)
                if not anachronisms: self.on_new_scene(); return
                self.scene_prompt = random.choice(anachronisms).get('scene_prompt')
                self.starting_location = None
        elif isinstance(choice, dict):
            self.scene_prompt = choice.get('scene_prompt')
            self.starting_location = choice.get('source_location')
        self._continue_to_game_start(is_newly_written_scene=False)

    def on_new_scene(self):
        self.app_state = AppState.AWAITING_SCENE_INPUT

    def on_new_scene_submit(self, scene_text: str):
        is_newly_written = True
        if not scene_text:
            from ..worldgen.scene_maker import SceneMakerManager
            self.atlas_engine.world_theme = self.world_theme
            scene_maker = SceneMakerManager(self.atlas_engine)
            generated_scene = scene_maker.generate_scene(world_name=self.selected_world_name)
            self.scene_prompt = generated_scene.get('scene_prompt')
            self.starting_location = generated_scene.get('source_location')
            is_newly_written = False
        else:
            self.scene_prompt = scene_text
            self.starting_location = None
        self._continue_to_game_start(is_newly_written_scene=is_newly_written)

    def on_load_new_choice(self, choice: str):
        if choice == "LOAD":
            self.app_state = AppState.LOAD_GAME_SELECTION
        elif choice == "NEW":
            self.app_state = AppState.SCENE_SELECTION
        elif choice == "BACK":
            self.app_state = AppState.WORLD_SELECTION

    def on_save_selected(self, save_name: str):
        self.load_path = file_io.join_path(file_io.SAVE_DATA_ROOT, self.selected_world_name, save_name)
        self.initiate_game_start()