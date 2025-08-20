# /core/ui/ui_manager.py

from multiprocessing import Process, Queue

import numpy as np
import tcod
import tcod.constants

try:
    from windows_toasts import WindowsToaster, Toast
except ImportError:
    Toast = None
    WindowsToaster = None

from .app_states import AppState
from . import ui_data_provider
from .ui_callbacks import UICallbacks
from .ui_special_modes import SpecialModeManager
from . import player_input_handlers

from .ui_framework import EventLog, DynamicTextBox, HelpBar
from ..engine.story_engine import StoryEngine
from ..engine.setup_manager import SetupManager
from ..common import file_io
from ..common.config_loader import config
from ..common import utils
from ..common.localization import loc

from ..common.game_state import GameState
from .views import WorldSelectionView, GameView, SceneSelectionView, TextInputView, LoadOrNewView, SaveSelectionView, \
    TabbedSettingsView, MenuView, PrometheusView, CalibrationView
from .input_handler import TextInputHandler
from ..llm.model_manager import ModelManager


def engine_process(render_queue: Queue, input_queue: Queue, load_path: str, world_name: str,
                   starting_location: dict, world_theme: str, scene_prompt: str,
                   embedding_model, model_context_limits: dict, location_breadcrumb: list):
    engine = StoryEngine(load_path=load_path, game_state=GameState(), render_queue=render_queue,
                         input_queue=input_queue, world_name=world_name, starting_location=starting_location,
                         world_theme=world_theme, scene_prompt=scene_prompt, embedding_model=embedding_model,
                         model_context_limits=model_context_limits, location_breadcrumb=location_breadcrumb)
    setup_manager = SetupManager(engine)
    if setup_manager.initialize_run():
        engine.run()
    else:
        render_queue.put(None)


class UIManager:
    def __init__(self, context, root_console, load_path, model_manager: ModelManager, tileset):
        self.context, self.root_console, self.load_path, self.model_manager, self.tileset = context, root_console, load_path, model_manager, tileset
        self.app_state = AppState.WORLD_SELECTION
        self.active_view = None
        self.text_input_active = False
        self.sync_input_active = False
        self.sync_input_result = None
        self.toaster = WindowsToaster('AI Storyteller') if WindowsToaster else None

        self.model_manager.transition_to_state('WORLD_CREATION')
        self.event_log = EventLog(max_lines=10)

        self.engine_proc, self.render_queue, self.input_queue = None, None, None
        self.selected_world_name, self.world_theme, self.scene_prompt, self.starting_location, self.location_breadcrumb = None, "A generic fantasy world.", None, None, None

        self.atlas_engine = StoryEngine(world_name=None, ui_manager=self)
        self.atlas_engine.embedding_model = None

        self.callbacks = UICallbacks(self)
        self.special_modes = SpecialModeManager(self)
        self.special_mode_active = None

        self.calibration_thread = None
        self.calibration_jobs = []
        self.current_job_index = 0
        self.calibration_update_queue = Queue()

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
            AppState.GAME_RUNNING: "help_bar_game_running",
            "IN_GAME_INPUT": "help_bar_ingame_input",
            "IN_GAME_MENU": "help_bar_ingame_menu",
            AppState.CALIBRATING: "help_bar_calibrating",
            AppState.PEG_V3_TEST: "help_bar_peg_test",
            AppState.CHARACTER_GENERATION_TEST: "help_bar_char_gen_test",
            AppState.SHUTTING_DOWN: "help_bar_shutting_down"
        }
        self.help_bar = HelpBar(help_texts=self.help_texts)

    def _send_player_notification(self, task_key: str):
        """Sends a toast notification for a player takeover event."""
        if not self.toaster: return
        try:
            task_params = config.task_parameters.get(task_key, {})
            message = task_params.get("player_takeover_notification", "Player action required.")
            toast = Toast()
            toast.text_fields = [message]
            self.toaster.show_toast(toast)
        except Exception as e:
            print(f"[UI NOTIFICATION ERROR] {e}")

    def _get_default_token_limit(self) -> int:
        return config.settings.get("DEFAULT_INPUT_TOKEN_LIMIT", 4096)

    def _process_logic(self):
        """The main state machine for the UI. Determines which view to show or action to take."""
        if self.app_state == AppState.CALIBRATING:
            if not isinstance(self.active_view, CalibrationView):
                self.active_view = CalibrationView(self.root_console.width, self.root_console.height)
            self.special_modes.handle_calibration_step()
            self._handle_calibration_updates()
            return

        if self.active_view is None:
            self.special_mode_active = None
            if self.app_state == AppState.WORLD_SELECTION:
                self.active_view = WorldSelectionView(
                    worlds=ui_data_provider.get_available_worlds(),
                    on_choice=self.callbacks.on_world_selection,
                    console_width=self.root_console.width, console_height=self.root_console.height
                )
            elif self.app_state == AppState.SETTINGS:
                def on_settings_back():
                    self.app_state = AppState.WORLD_SELECTION
                    self.active_view = None

                self.active_view = TabbedSettingsView(on_back=on_settings_back, console_width=self.root_console.width, console_height=self.root_console.height)

            elif self.app_state == AppState.AWAITING_THEME_INPUT:
                user_prompt = "Enter a theme for the new world (or leave blank for a generated one)."
                handler = TextInputHandler(prompt=user_prompt, width=self.root_console.width - 2, prefix="> ",
                                           max_tokens=self._get_default_token_limit())
                self.active_view = TextInputView(handler=handler, on_submit=self.callbacks.handle_theme_submission)

            elif self.app_state == AppState.AWAITING_SCENE_INPUT:
                user_prompt = "Write an opening scene to begin the story (or leave blank for a generated one)."
                handler = TextInputHandler(prompt=user_prompt, width=self.root_console.width - 2, prefix="> ",
                                           max_tokens=self._get_default_token_limit())
                self.active_view = TextInputView(handler=handler, on_submit=self.callbacks.on_new_scene_submit)

            elif self.app_state == AppState.LOAD_OR_NEW:
                self.active_view = LoadOrNewView(on_choice=self.callbacks.on_load_new_choice,
                                                 console_width=self.root_console.width,
                                                 console_height=self.root_console.height)

            elif self.app_state == AppState.LOAD_GAME_SELECTION:
                saved_runs = file_io.get_saved_runs_for_world(self.selected_world_name)
                self.active_view = SaveSelectionView(saves=saved_runs, on_choice=self.callbacks.on_save_selected,
                                                     on_back=lambda: self.callbacks.on_load_new_choice("BACK"),
                                                     console_width=self.root_console.width,
                                                     console_height=self.root_console.height)

            elif self.app_state == AppState.SCENE_SELECTION:
                self.active_view = SceneSelectionView(scenes=ui_data_provider.get_available_scenes(),
                                                      anachronisms=ui_data_provider.get_anachronisms(
                                                          self.selected_world_name),
                                                      on_choice=self.callbacks.on_scene_selection,
                                                      on_new=self.callbacks.on_new_scene,
                                                      on_back=lambda: self.callbacks.on_world_selection(
                                                          self.selected_world_name),
                                                      console_width=self.root_console.width,
                                                      console_height=self.root_console.height)

            elif self.app_state == AppState.CHARACTER_GENERATION_TEST:
                def on_context_submit(context: str):
                    self.active_view = None
                    if context:
                        self.special_modes.start_character_generation_test(context)
                    else:
                        self.app_state = AppState.WORLD_SELECTION

                user_prompt = "Enter a scene context for character generation."
                handler = TextInputHandler(prompt=user_prompt, width=self.root_console.width - 2, prefix="> ",
                                           max_tokens=self._get_default_token_limit())
                self.active_view = TextInputView(handler=handler, on_submit=on_context_submit)

        if self.app_state == AppState.WORLD_CREATING:
            if not self.model_manager.models_loaded.is_set(): return
            self.active_view = None
            from ..worldgen.atlas_manager import AtlasManager
            atlas = AtlasManager(self.atlas_engine)
            self.selected_world_name = atlas.create_new_world(self.world_theme, ui_manager=self)
            if self.selected_world_name:
                config.load_world_data(self.selected_world_name)
                self.app_state = AppState.SCENE_SELECTION
            else:
                self.event_log.add_message("FATAL: World creation failed.", (255, 50, 50))
                self.app_state = AppState.WORLD_SELECTION
        elif self.app_state == AppState.PEG_V3_TEST:
            if self.special_modes.peg_test_generator is None:
                self.active_view = None
                self.special_modes.run_peg_v3_test()
            self.special_mode_active = "PEG_V3_TEST"
        elif self.app_state == AppState.CHARACTER_GENERATION_TEST:
            if self.special_mode_active == "CHAR_GEN_TEST":
                self.special_modes.handle_character_generation_updates()
        elif self.app_state == AppState.LOADING_GAME:
            self._handle_game_loading()
        elif self.app_state == AppState.GAME_RUNNING:
            self._handle_game_updates()
        elif self.app_state == AppState.SHUTTING_DOWN:
            self.active_view = None

    def get_player_task_input(self, **kwargs):
        """Synchronously gets player input for a task, blocking the calling engine."""
        task_key = kwargs.get("task_key")

        # --- Redundant safety check to prevent deadlocks ---
        task_params = config.task_parameters.get(task_key, {})
        if not task_params.get("player_takeover_enabled"):
            utils.log_message('debug', f"[UI] get_player_task_input called for disabled task '{task_key}'. Ignoring.")
            return None

        self.sync_input_active = True
        self.sync_input_result = None
        original_view = self.active_view

        def on_submit(result):
            self.sync_input_result = result
            self.sync_input_active = False

        handler_map = {
            "PROMETHEUS_DETERMINE_TOOL_USE": player_input_handlers.create_prometheus_menu_view,
            "DIRECTOR_CAST_REPLACEMENT": None,
        }

        creation_func = handler_map.get(task_key, player_input_handlers.create_default_takeover_view)

        if creation_func:
            creation_func(self, on_submit, **kwargs)
        else:
            self.sync_input_active = False
            return None

        while self.sync_input_active:
            self._sync_text_input_state()
            self._update_help_context()
            self._render()
            for event in tcod.event.get():
                converted_event = self.context.convert_event(event)
                if self.active_view:
                    self.active_view.handle_event(converted_event)
                if isinstance(converted_event, tcod.event.Quit):
                    self.app_state = AppState.EXITING
                    self.sync_input_active = False
                    self.sync_input_result = None

        self.active_view = original_view
        return self.sync_input_result if self.sync_input_result is not None else None

    def _handle_calibration_updates(self):
        """Processes messages from the calibration thread queue to update the UI."""
        if self.calibration_update_queue and not self.calibration_update_queue.empty():
            try:
                while not self.calibration_update_queue.empty():
                    message = self.calibration_update_queue.get_nowait()
                    if isinstance(self.active_view, CalibrationView):
                        if message.get('type') == 'phase_change':
                            self.active_view.set_phase(message['phase'], message['text'])
                        else:
                            self.active_view.update_data(message)
            except Exception:
                pass

    def _handle_game_loading(self):
        """Checks if models are ready and then starts the engine process."""
        self.active_view = None
        if self.model_manager.models_loaded.is_set():
            if not self.scene_prompt:
                from ..worldgen.scene_maker import SceneMakerManager
                self.atlas_engine.embedding_model = self.model_manager.embedding_model
                scene_maker = SceneMakerManager(self.atlas_engine)
                generated_scene = scene_maker.generate_scene(world_name=self.selected_world_name)
                self.scene_prompt, self.starting_location = generated_scene.get('scene_prompt'), generated_scene.get(
                    'source_location')
                self.location_breadcrumb = [next(iter(config.world))]

            self.render_queue, self.input_queue = Queue(), Queue()
            self.engine_proc = Process(target=engine_process, args=(self.render_queue, self.input_queue, self.load_path,
                                                                    self.selected_world_name, self.starting_location,
                                                                    self.world_theme, self.scene_prompt,
                                                                    self.model_manager.embedding_model,
                                                                    self.model_manager.active_llm_models,
                                                                    self.location_breadcrumb))
            self.engine_proc.start()
            self.app_state = AppState.GAME_RUNNING

    def _handle_game_updates(self):
        """Processes messages from the engine process queue to update the game view."""
        if self.active_view is None:
            game_log_box = DynamicTextBox(title="Game Log", text="", x=0, y=self.root_console.height - 10,
                                          max_width=self.root_console.width, max_height=9)
            self.active_view = GameView(self.event_log, game_log_box)

        if self.render_queue and not self.render_queue.empty():
            try:
                message = self.render_queue.get(timeout=0.01)
                if message is None:
                    self.app_state = AppState.SHUTTING_DOWN
                elif isinstance(message, tuple):
                    msg_type = message[0]
                    if msg_type == 'ADD_EVENT_LOG':
                        self.event_log.add_message(message[1], fg=message[2] if len(message) > 2 else (255, 255, 255))
                    elif msg_type == 'GAME_LOG_UPDATE':
                        self.active_view.game_log_box.set_text(message[1])
                    elif msg_type == 'INPUT_REQUEST':
                        self._create_in_game_input_view(message)
                    elif msg_type == 'MENU_REQUEST':
                        self._create_in_game_menu_view(message)
                    elif msg_type == 'PLAYER_TASK_TAKEOVER_REQUEST':
                        def on_submit(text):
                            self.input_queue.put(text)
                            game_log_box = DynamicTextBox(title="Game Log", text="", x=0,
                                                          y=self.root_console.height - 10,
                                                          max_width=self.root_console.width, max_height=9)
                            self.active_view = GameView(self.event_log, game_log_box)

                        player_input_handlers.create_default_takeover_view(self, on_submit, **message[1])
                    elif msg_type == 'PROMETHEUS_MENU_REQUEST':
                        def on_submit(choices):
                            self.input_queue.put(choices)
                            game_log_box = DynamicTextBox(title="Game Log", text="", x=0,
                                                          y=self.root_console.height - 10,
                                                          max_width=self.root_console.width, max_height=9)
                            self.active_view = GameView(self.event_log, game_log_box)

                        player_input_handlers.create_prometheus_menu_view(self, on_submit, **message[1])

                else:
                    if isinstance(self.active_view, GameView): self.active_view.update_state(message)
            except Exception:
                pass

    def _create_in_game_input_view(self, message: tuple):
        """Creates the TextInputView for in-game player actions."""
        prompt, max_tokens = message[1], message[2] if len(message) > 2 else self._get_default_token_limit()
        initial_text = message[3] if len(message) > 3 else ""

        def on_submit(text):
            self.input_queue.put(text)
            game_log_box = DynamicTextBox(title="Game Log", text="", x=0, y=self.root_console.height - 10,
                                          max_width=self.root_console.width, max_height=9)
            self.active_view = GameView(self.event_log, game_log_box)

        handler = TextInputHandler(prompt=prompt, width=self.root_console.width - 2, prefix="> ", max_tokens=max_tokens,
                                   initial_text=initial_text)
        self.active_view = TextInputView(handler=handler, on_submit=on_submit)

    def _create_in_game_menu_view(self, message: tuple):
        """Creates the MenuView for in-game player choices."""
        title, options = message[1], message[2] if len(message) > 2 else []

        def on_submit(choice: str | None):
            self.input_queue.put(choice)
            game_log_box = DynamicTextBox(title="Game Log", text="", x=0, y=self.root_console.height - 10,
                                          max_width=self.root_console.width, max_height=9)
            self.active_view = GameView(self.event_log, game_log_box)

        self.active_view = MenuView(
            title=title,
            options=options,
            on_choice=on_submit,
            console_width=self.root_console.width,
            console_height=self.root_console.height
        )

    def _update_help_context(self):
        """Determines the correct context key for the help bar based on the current UI state."""
        context_key = self.app_state

        if isinstance(self.active_view, TextInputView):
            if self.app_state == AppState.GAME_RUNNING:
                if "Your turn" in self.active_view.handler.prompt:
                    context_key = "IN_GAME_INPUT"
                else:
                    context_key = "PLAYER_TAKEOVER"
            elif self.app_state in [AppState.AWAITING_THEME_INPUT, AppState.AWAITING_SCENE_INPUT]:
                context_key = self.app_state

        elif isinstance(self.active_view, TabbedSettingsView):
            if self.active_view.show_exit_confirmation:
                context_key = "SETTINGS_CONFIRM"
            elif getattr(self.active_view.pages[self.active_view.active_page_index], 'is_editing', False):
                context_key = "SETTINGS_EDIT"
            else:
                context_key = AppState.SETTINGS

        elif isinstance(self.active_view, PrometheusView):
            context_key = "PROMETHEUS_TAKEOVER"
        elif isinstance(self.active_view, MenuView):
            context_key = "IN_GAME_MENU"

        self.help_bar.set_context(context_key)

    def _render(self):
        """Clears the console and renders the current view or status message."""
        self.root_console.clear()
        if self.active_view:
            self.active_view.render(self.root_console)
        elif self.app_state == AppState.WORLD_CREATING:
            status = "Forging new world..." if self.model_manager.models_loaded.is_set() else "Loading world creation models..."
            self.root_console.print(self.root_console.width // 2, 20, status, alignment=tcod.constants.CENTER,
                                    fg=(200, 200, 255))
        elif self.app_state == AppState.LOADING_GAME:
            self.root_console.print(self.root_console.width // 2, 20, "Please Wait... Loading Game Models",
                                    alignment=tcod.constants.CENTER, fg=(200, 200, 255))
        elif self.app_state == AppState.SHUTTING_DOWN:
            self.root_console.print(self.root_console.width // 2, 20, "Story finished. Saving run and exiting...",
                                    alignment=tcod.constants.CENTER, fg=(220, 220, 200))

        self.help_bar.render(self.root_console)
        self.context.present(self.root_console)

        if self.active_view and self.active_view.sdl_primitives:
            with self.context.renderer.color as color:
                for prim in self.active_view.sdl_primitives:
                    if prim['type'] == 'line':
                        color[...] = prim['color']
                        self.context.renderer.line(
                            x=np.array([prim['start'][0], prim['end'][0]]),
                            y=np.array([prim['start'][1], prim['end'][1]])
                        )
            self.active_view.sdl_primitives.clear()

    def _sync_text_input_state(self):
        if not isinstance(self.active_view, TabbedSettingsView):
            is_text_view = isinstance(self.active_view, TextInputView)
        else:
            active_page = self.active_view.pages[self.active_view.active_page_index]
            is_text_view = getattr(active_page, 'is_editing', False)

        if is_text_view and not self.text_input_active:
            if self.context.sdl_window: self.context.sdl_window.start_text_input()
            self.text_input_active = True
        elif not is_text_view and self.text_input_active:
            if self.context.sdl_window: self.context.sdl_window.stop_text_input()
            self.text_input_active = False

    def run(self):
        """The main application loop."""
        while self.app_state != AppState.EXITING:
            if not self.sync_input_active:
                self._process_logic()

            self._sync_text_input_state()
            self._update_help_context()
            self._render()

            if self.app_state == AppState.SHUTTING_DOWN:
                if self.engine_proc and self.engine_proc.is_alive():
                    self.engine_proc.join(timeout=5.0)
                self.app_state = AppState.EXITING
                continue

            for event in tcod.event.get():
                converted_event = self.context.convert_event(event)

                if self.active_view: self.active_view.handle_event(converted_event)

                if isinstance(converted_event, tcod.event.Quit): self.app_state = AppState.EXITING; break

                if isinstance(converted_event, tcod.event.KeyDown):
                    if self.special_mode_active == "PEG_V3_TEST":
                        key = converted_event.sym
                        phase = self.special_modes.current_peg_phase

                        if key in (tcod.event.KeySym.LEFT, tcod.event.KeySym.RIGHT, tcod.event.KeySym.ESCAPE):
                            current_app_state = self.app_state
                            self.special_modes.stop_peg_patches()
                            if key == tcod.event.KeySym.LEFT:
                                self.special_modes.peg_v3_scenario_index = (
                                                                                   self.special_modes.peg_v3_scenario_index - 1) % len(
                                    self.special_modes.peg_v3_scenarios)
                            elif key == tcod.event.KeySym.RIGHT:
                                self.special_modes.peg_v3_scenario_index = (
                                                                                   self.special_modes.peg_v3_scenario_index + 1) % len(
                                    self.special_modes.peg_v3_scenarios)
                            elif key == tcod.event.KeySym.ESCAPE:
                                self.app_state = AppState.WORLD_SELECTION
                                self.active_view = None
                                continue

                            self.app_state = current_app_state
                            self.active_view = None
                            continue

                        if phase in ('DONE', 'FINAL') and key in (tcod.event.KeySym.RETURN, tcod.event.KeySym.KP_ENTER):
                            self.active_view = None
                        elif phase == 'INITIAL_PLACEMENT' and key in (tcod.event.KeySym.RETURN,
                                                                      tcod.event.KeySym.KP_ENTER):
                            self.special_modes.advance_peg_test_step(key)
                        elif phase in ('SUBFEATURE_STEP', 'INTERIOR_PLACEMENT_STEP') and key in (
                                tcod.event.KeySym.RETURN, tcod.event.KeySym.KP_ENTER, tcod.event.KeySym.SPACE):
                            self.special_modes.advance_peg_test_step(key)
                        elif phase in ('PRE_JITTER', 'PRE_INTERIOR_PLACEMENT', 'PRE_CONNECT',
                                       'POST_CONNECT') and key in (tcod.event.KeySym.SPACE, tcod.event.KeySym.RETURN,
                                                                   tcod.event.KeySym.KP_ENTER):
                            self.special_modes.advance_peg_test_step(key)

                    elif self.special_mode_active == "CHAR_GEN_TEST":
                        if converted_event.sym == tcod.event.KeySym.ESCAPE:
                            self.special_modes.stop_character_generation_test()
                            continue

                    if self.input_queue:
                        if converted_event.sym == tcod.event.KeySym.F7:
                            self.input_queue.put('__INTERRUPT_CAST_MANAGER__')
                        elif converted_event.sym == tcod.event.KeySym.F8:
                            self.input_queue.put('__FLAG_LAST_RESPONSE__')
                        elif converted_event.sym == tcod.event.KeySym.F9:
                            self.input_queue.put('__INTERRUPT_PLAYER__')
                        elif converted_event.sym in (tcod.event.KeySym.F10, tcod.event.KeySym.ESCAPE):
                            self.input_queue.put('__INTERRUPT_SAVE__')
        self.shutdown()

    def shutdown(self):
        if self.text_input_active:
            if self.context.sdl_window: self.context.sdl_window.stop_text_input()
        if self.engine_proc and self.engine_proc.is_alive():
            self.engine_proc.terminate()
            self.engine_proc.join()
        self.special_modes.stop_peg_patches()
        print("\n[SYSTEM] Application has finished. Exiting.")