# /core/ui/ui_manager.py

from multiprocessing import Process, Queue

import numpy as np
import tcod
import tcod.constants
from tcod.render import SDLTilesetAtlas, SDLConsoleRender

try:
    from windows_toasts import WindowsToaster, Toast
except ImportError:
    Toast = None
    WindowsToaster = None

from .app_states import AppState
from .ui_special_modes import SpecialModeManager
from .view_factory import ViewFactory
from .ui_framework import EventLog, HelpBar
from ..common.config_loader import config
from ..llm.model_manager import ModelManager
from .views import GameView, TextInputView, MenuView, PrometheusView, RoleCreatorView, WorldGraphView, \
    TabbedSettingsView
from .player_interface_handler import PlayerInterfaceHandler
from .game_update_handler import GameUpdateHandler


def engine_process(render_queue: Queue, input_queue: Queue, load_path: str, world_name: str,
                   starting_location: dict, world_theme: str, scene_prompt: str,
                   embedding_model, model_context_limits: dict, location_breadcrumb: list):
    from ..engine.story_engine import StoryEngine
    from ..engine.setup_manager import SetupManager
    from ..common.game_state import GameState
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
    def __init__(self, app_controller, context, root_console, model_manager: ModelManager, tileset):
        self.app = app_controller
        self.context = context
        self.root_console = root_console
        self.model_manager = model_manager
        self.tileset = tileset

        self.active_view = None
        self.current_app_state = None
        self.text_input_active = False
        self.sync_input_active = False
        self.sync_input_result = None
        self.toaster = WindowsToaster('AI Storyteller') if WindowsToaster else None

        self.event_log = EventLog(max_lines=10)
        self.input_queue = None

        # Instantiate handlers
        self.player_interface_handler = PlayerInterfaceHandler(self)
        self.game_update_handler = GameUpdateHandler(self, self.player_interface_handler)
        self.view_factory = ViewFactory(app_controller, self)
        self.special_modes = SpecialModeManager(self)

        self.atlas = None
        self.console_render = None
        if self.context.sdl_renderer and self.tileset:
            self.atlas = SDLTilesetAtlas(self.context.sdl_renderer, self.tileset)
            self.console_render = SDLConsoleRender(self.atlas)

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
            AppState.SHUTTING_DOWN: "help_bar_shutting_down"
        }
        self.help_bar = HelpBar(help_texts=self.help_texts)
        self.game_view = GameView(
            event_log=self.event_log,
            console_width=self.root_console.width,
            console_height=self.root_console.height
        )

    def set_engine_queues(self, render_queue, input_queue):
        self.game_update_handler.set_queue(render_queue)
        self.input_queue = input_queue

    def set_view_for_state(self, new_app_state: AppState):
        if new_app_state != self.current_app_state:
            self.current_app_state = new_app_state
            if new_app_state == AppState.GAME_RUNNING:
                self.active_view = self.game_view
            elif new_app_state in [AppState.LOADING_GAME, AppState.SHUTTING_DOWN]:
                self.active_view = None
            else:
                self.active_view = self.view_factory.create_view(new_app_state)
            self.special_modes.set_active_mode(new_app_state)

    def update_world_graph_view(self, world_data):
        self.active_view = WorldGraphView(world_data, self.root_console.width,
                                          self.root_console.height, self.tileset)

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

    def _update_help_context(self):
        """Determines the correct context key for the help bar based on the current UI state."""
        context_key = self.app.app_state
        if self.active_view and self.active_view.help_context_key:
            context_key = self.active_view.help_context_key

        if isinstance(self.active_view, TabbedSettingsView):
            if self.active_view.show_exit_confirmation:
                context_key = "SETTINGS_CONFIRM"
            elif getattr(self.active_view.pages[self.active_view.active_page_index], 'is_editing', False):
                context_key = "SETTINGS_EDIT"
            else:
                context_key = AppState.SETTINGS

        self.help_bar.set_context(context_key)

    def render(self):
        """Prepares all console content and renders it and all SDL primitives in the correct order."""
        self.root_console.clear()
        if self.active_view:
            self.active_view.render(self.root_console)
        elif self.app.app_state == AppState.LOADING_GAME:
            self.root_console.print(self.root_console.width // 2, 20, "Please Wait... Loading Game Models",
                                    alignment=tcod.constants.CENTER, fg=(200, 200, 255))
        elif self.app.app_state == AppState.SHUTTING_DOWN:
            self.root_console.print(self.root_console.width // 2, 20, "Story finished. Saving run and exiting...",
                                    alignment=tcod.constants.CENTER, fg=(220, 220, 200))
        self.help_bar.render(self.root_console)
        renderer = self.context.sdl_renderer
        if not renderer or not self.console_render:
            self.context.present(self.root_console)
            return
        renderer.clear()
        root_texture = self.console_render.render(self.root_console)
        renderer.copy(root_texture)
        if self.active_view and self.active_view.sdl_primitives:
            lines = [p for p in self.active_view.sdl_primitives if p['type'] == 'line']
            consoles = [p for p in self.active_view.sdl_primitives if p['type'] == 'console']
            for prim in lines:
                points = tcod.los.bresenham(prim['start'], prim['end']).tolist()
                points_array = np.array(points, dtype=np.intc)
                if points_array.size > 0:
                    renderer.draw_color = (*prim['color'], 255)
                    renderer.draw_points(points=points_array)
            for prim in consoles:
                temp_console = prim['console']
                if temp_console.width <= 0 or temp_console.height <= 0: continue
                texture = self.console_render.render(temp_console)
                texture.blend_mode = tcod.sdl.render.BlendMode.BLEND
                renderer.copy(texture, dest=(prim['x'], prim['y'], texture.width, texture.height))
            self.active_view.sdl_primitives.clear()
        renderer.present()

    def _sync_text_input_state(self):
        is_text_view = isinstance(self.active_view, TextInputView) or \
                       (isinstance(self.active_view, TabbedSettingsView) and \
                        getattr(self.active_view.pages[self.active_view.active_page_index], 'is_editing', False))
        if is_text_view and not self.text_input_active:
            if self.context.sdl_window: self.context.sdl_window.start_text_input()
            self.text_input_active = True
        elif not is_text_view and self.text_input_active:
            if self.context.sdl_window: self.context.sdl_window.stop_text_input()
            self.text_input_active = False

    def process_events(self):
        self._sync_text_input_state()
        self._update_help_context()
        for event in tcod.event.get():
            converted_event = self.context.convert_event(event)
            if isinstance(converted_event, tcod.event.Quit):
                self.app.app_state = AppState.EXITING
                return

            if isinstance(converted_event, tcod.event.KeyDown):
                if self.app.app_state == AppState.GAME_RUNNING and self.input_queue:
                    key = converted_event.sym
                    if key == tcod.event.KeySym.F7:
                        self.input_queue.put('__INTERRUPT_CAST_MANAGER__')
                        continue
                    elif key == tcod.event.KeySym.F8:
                        self.input_queue.put('__FLAG_LAST_RESPONSE__')
                        continue
                    elif key == tcod.event.KeySym.F9:
                        self.input_queue.put('__INTERRUPT_PLAYER__')
                        continue
                    elif key == tcod.event.KeySym.F10:
                        self.input_queue.put('__INTERRUPT_SAVE__')
                        continue

            if self.special_modes.active_mode and self.special_modes.handle_event(converted_event):
                continue

            if self.active_view:
                self.active_view.handle_event(converted_event)

    def shutdown(self):
        if self.text_input_active:
            if self.context.sdl_window: self.context.sdl_window.stop_text_input()
        self.special_modes.stop_all()