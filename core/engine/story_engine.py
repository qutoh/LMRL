# /core/story_engine.py

import random
import os
from multiprocessing import Queue
import queue
from datetime import datetime, timedelta
from core.common.config_loader import config
from core.common.localization import loc
from core.llm import llm_api
from ..common import file_io, utils
from core.components import roster_manager
from .director import DirectorManager
from core.components.player import PlayerInterface
from .turn_manager import TurnManager
from core.common.game_state import GameState
from core.components.summary_manager import SummaryManager
from .prometheus_manager import PrometheusManager
from core.common.annotation_manager import AnnotationManager
from core.components.item_manager import ItemManager
from core.components.character_manager import CharacterManager


class StoryEngine:
    def __init__(self, load_path=None, game_state: GameState = None, render_queue: Queue = None,
                 input_queue: Queue = None, world_name: str = None,
                 starting_location: dict = None, world_theme: str = None,
                 scene_prompt: str = None, embedding_model=None, model_context_limits=None,
                 location_breadcrumb: list = None, ui_manager=None):

        self.characters = []
        self.dialogue_log = []
        self.summaries = []
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

        self.game_state = game_state
        self.generation_state = None
        self.layout_graph = None
        self.current_scene_key = None
        self.render_queue = render_queue
        self.input_queue = input_queue
        self.ui_manager = ui_manager  # For synchronous, pre-game player input

        # Configure the logger service with the render queue
        utils.configure_logger(self.render_queue)

        # PlayerInterface is only fully functional for the async main engine
        self.player_interface = PlayerInterface(self, self.game_state) if render_queue else None

        self.turn_manager = TurnManager(self, self.game_state, self.player_interface)
        self.director_manager = DirectorManager(self)
        self.summary_manager = SummaryManager(self)
        self.prometheus_manager = PrometheusManager(self)
        self.annotation_manager = AnnotationManager(self)
        self.item_manager = ItemManager(self)
        self.character_manager = CharacterManager()
        self.config = config

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
            utils.log_message('game', loc('log_time_update', time_str=time_str))

    def save_state(self, save_path):
        if not save_path: return
        file_io.create_directory(save_path)
        utils.log_message('debug', loc('log_serialization_start', path=save_path))
        state_data = {
            "dialogue_log": self.dialogue_log,
            "summaries": self.summaries,
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
            while True:  # Loop until queue is confirmed empty
                message = self.input_queue.get_nowait()
                if message == '__INTERRUPT_SAVE__':
                    if not self.interrupted: self.interrupted = True
                elif message == '__INTERRUPT_PLAYER__':
                    if not self.player_interrupted: self.player_interrupted = True
                elif message == '__FLAG_LAST_RESPONSE__':
                    if self.last_interaction_log:
                        self.annotation_manager.annotate_last_log_as_failure('PLAYER_FLAGGED',
                                                                             'Player marked this response as low quality.')
                        self.render_queue.put(
                            ('ADD_EVENT_LOG', "Last response flagged as low quality.", (255, 200, 100)))
                else:
                    requeue_messages.append(message)
        except queue.Empty:
            pass  # This is the expected way to exit the loop
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
        for i in range(self.config.settings['MAX_CYCLES']):
            utils.log_message('debug',
                              loc('log_cycle_header', cycle_num=i + 1, max_cycles=self.config.settings['MAX_CYCLES']))

            self.summary_manager.check_and_perform_summary()
            roster_manager.spawn_entities_from_roster(self, self.game_state)

            turn_order = self.characters[:]
            random.shuffle(turn_order)

            if not turn_order:
                utils.log_message('debug', loc('log_no_roles_left'))
                break

            turn_queue = turn_order[:]
            while turn_queue:
                self._check_for_interrupts()
                if self.interrupted or self.player_interrupted:
                    break

                current_actor = turn_queue.pop(0)
                next_actor_choice = self.turn_manager.execute_turn_for(current_actor, turn_queue)
                self.render_queue.put(self.game_state)

                if next_actor_choice:
                    utils.log_message('debug', loc('log_turn_pass', actor_name=current_actor['name'],
                                                   next_actor_name=next_actor_choice['name']))
                    turn_queue = [r for r in turn_queue if r.get('name') != next_actor_choice.get('name')]
                    turn_queue.insert(0, next_actor_choice)

            if self.player_interrupted:
                self.player_interface.initiate_takeover_menu()
                self.player_interrupted = False
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
        if error: utils.log_message('debug', loc('error_run_rename_fail', error=error))
        utils.log_message('debug', f"Run directory finalized as: {self.run_path}")

        self.render_queue.put(None)
        self.save_state(self.run_path)