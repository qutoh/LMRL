# /core/engine/story_engine.py

import os
import queue
from .game_modes.narrative_simulation_mode import NarrativeSimulationMode
from datetime import datetime, timedelta
from multiprocessing import Queue

from core.common.annotation_manager import AnnotationManager
from core.common.config_loader import config
from core.common.game_state import GameState
from core.common.localization import loc
from core.components import roster_manager
from core.components.character_manager import CharacterManager
from core.components.dm_manager import DMManager
from core.components.item_manager import ItemManager
from core.components.player import PlayerInterface
from core.components.summary_manager import SummaryManager
from core.llm import llm_api
from .director import DirectorManager
from .prometheus_manager import PrometheusManager
from .turn_manager import TurnManager
from ..common import file_io, utils
from ..ui.ui_messages import (
    AddEventLogMessage, StreamStartMessage, StreamTokenMessage, StreamEndMessage,
    InputRequestMessage, MenuRequestMessage, PlayerTaskTakeoverRequestMessage,
    PrometheusMenuRequestMessage, RoleCreatorRequestMessage
)


class StoryEngine:
    def __init__(self, load_path=None, game_state: GameState = None, render_queue: Queue = None,
                 input_queue: Queue = None, world_name: str = None,
                 starting_location: dict = None, world_theme: str = None,
                 scene_prompt: str = None, embedding_model=None, model_context_limits=None,
                 location_breadcrumb: list = None, ui_manager=None):

        self.game_state = game_state
        self.characters = []
        self.dialogue_log = []
        self.summaries = []
        self.lead_character_summary = ""
        self.lead_roster_changed = True  # Start dirty to force initial generation
        self.dehydrated_npcs = []
        self.current_game_time = datetime(1486, 6, 12, 14, 0, 0)
        self.token_context_limit = 8192
        self.token_summary_threshold = 6192

        self.embedding_model = embedding_model
        self.model_context_limits = model_context_limits or {}

        self.memory_manager = None
        self.json_incompatible_models = set()
        self.run_path = None
        self.load_path = load_path
        self.last_interaction_log = {}
        self.character_log_sessions = {}

        self.world_name = world_name
        self.world_theme = world_theme
        self.starting_location = starting_location
        self.location_breadcrumb = location_breadcrumb
        self.scene_prompt = scene_prompt

        self.interrupted = False
        self.player_interrupted = False
        self.player_input_active = False
        self.cast_manager_interrupted = False

        self.generation_state = None
        self.layout_graph = None
        self.current_scene_key = None
        self.render_queue = render_queue
        self.input_queue = input_queue

        # This ui_manager is the PlayerInterfaceHandler for synchronous pre-game input
        self.ui_manager = ui_manager

        utils.configure_logger(self.render_queue)

        self.player_interface = PlayerInterface(self) if render_queue else None

        self.turn_manager = TurnManager(self, self.player_interface)
        self.director_manager = DirectorManager(self)
        self.summary_manager = SummaryManager(self)
        self.prometheus_manager = PrometheusManager(self)
        self.annotation_manager = AnnotationManager(self)
        self.item_manager = ItemManager(self)
        self.character_manager = CharacterManager()
        self.dm_manager = DMManager(self)
        self.config = config

        self.game_mode = None

        if world_name:
            self.config.load_world_data(world_name)

    def _print_final_recap(self):
        utils.log_message('debug', loc('log_recap_header'))
        if self.summaries:
            utils.log_message('debug', loc('log_recap_summary_header'))
            utils.log_message('debug', self.summaries[-1])
            utils.log_message('debug', loc('log_recap_dialogue_header'))
        for entry in self.dialogue_log:
            if entry.get('speaker', '').lower() == 'system': continue
            utils.log_message('debug', f"{entry.get('speaker', 'Unknown')}:\n{entry.get('content', '')}\n")

    def advance_time(self, seconds: int):
        if seconds > 0:
            self.current_game_time += timedelta(seconds=seconds)
            time_str = self.current_game_time.strftime('%H:%M on %A, %B %d')
            utils.log_message('debug',
                              f"[TIME] Advanced by {seconds}s. New time: {self.current_game_time.strftime('%Y-%m-%d %H:%M:%S')}")
            self.render_queue.put(AddEventLogMessage(loc('log_time_update', time_str=time_str)))

    def save_state(self, save_path):
        if not save_path: return
        file_io.create_directory(save_path)
        utils.log_message('debug', loc('log_serialization_start', path=save_path))
        state_data = {
            "dialogue_log": self.dialogue_log,
            "summaries": self.summaries,
            "lead_character_summary": self.lead_character_summary,
            "current_game_time": self.current_game_time.isoformat(),
            "current_scene_key": self.current_scene_key
        }
        state_path = file_io.join_path(save_path, 'story_state.json')
        file_io.write_json(state_path, state_data)
        file_io.save_active_character_files(self)
        utils.log_message('debug', loc('log_serialization_success'))

    def _check_for_interrupts(self):
        if self.player_input_active:
            return

        requeue_messages = []
        try:
            while True:
                message = self.input_queue.get_nowait()
                if message == '__INTERRUPT_SAVE__':
                    if not self.interrupted: self.interrupted = True
                elif message == '__INTERRUPT_PLAYER__':
                    if not self.player_interrupted: self.player_interrupted = True
                elif message == '__INTERRUPT_CAST_MANAGER__':
                    if not self.cast_manager_interrupted: self.cast_manager_interrupted = True
                elif message == '__FLAG_LAST_RESPONSE__':
                    if self.last_interaction_log:
                        self.annotation_manager.annotate_last_log_as_failure('PLAYER_FLAGGED',
                                                                             'Player marked this response as low quality.')
                        self.render_queue.put(
                            AddEventLogMessage("Last response flagged as low quality.", (255, 200, 100)))
                else:
                    requeue_messages.append(message)
        except queue.Empty:
            pass
        finally:
            for msg in requeue_messages:
                self.input_queue.put(msg)

    def _generate_save_name(self):
        fallback_name = f"run_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

        previous_summary = self.summaries[-1] if self.summaries else ""
        dialogue_text = "\n".join(f"{e['speaker']}: {e['content']}" for e in self.dialogue_log)
        story_context = f"{previous_summary}\n\n--- RECENT DIALOGUE ---\n\n{dialogue_text}".strip()

        if not story_context: return fallback_name

        summarizer_agent = self.config.agents['SUMMARIZER']
        previous_attempts = []

        for _ in range(3):
            task_kwargs = {
                "previous_attempts": ", ".join(f"'{a}'" for a in previous_attempts) if previous_attempts else "None"}

            name = llm_api.execute_task(
                self,
                summarizer_agent,
                'GET_SAVE_NAME',
                [{"role": "user", "content": f"--- STORY TEXT TO ANALYZE ---\n{story_context}"}],
                task_prompt_kwargs=task_kwargs
            )

            if utils.is_valid_filename(name): return name
            if name: previous_attempts.append(name.strip())
        return fallback_name

    def run(self):
        self.game_mode = NarrativeSimulationMode(self)

        for i in range(self.config.settings['MAX_CYCLES']):
            utils.log_message('debug',
                              loc('log_cycle_header', cycle_num=i + 1, max_cycles=self.config.settings['MAX_CYCLES']))

            self.summary_manager.check_and_perform_summary()
            if self.lead_roster_changed:
                self.dm_manager.update_lead_character_summary()

            roster_manager.spawn_entities_from_roster(self, self.game_state)

            self.game_mode.run_cycle()

            if self.player_interrupted:
                self.player_interface.initiate_takeover_menu()
                self.player_interrupted = False
                continue

            if self.cast_manager_interrupted:
                self.player_interface.initiate_cast_management_menu()
                self.cast_manager_interrupted = False
                continue

            if self.interrupted:
                break

            self.director_manager.execute_phase()

        initial_name = ""
        if self.load_path:
            basename = os.path.basename(self.load_path)
            if not basename.startswith("run_"):
                initial_name = basename

        prompt_message = "Enter a name for this save file.\nLeave blank for an AI-generated title."
        user_input = self.player_interface._get_player_input(prompt_message, initial_text=initial_name).strip()

        final_name = user_input if user_input else self._generate_save_name()

        new_run_path, error = file_io.finalize_run_directory(self.run_path, final_name)
        self.run_path = new_run_path
        if error: self.render_queue.put(AddEventLogMessage(loc('error_run_rename_fail', error=error), (255, 100, 100)))
        utils.log_message('debug', f"Run directory finalized as: {self.run_path}")

        self.render_queue.put(None)
        self.save_state(self.run_path)