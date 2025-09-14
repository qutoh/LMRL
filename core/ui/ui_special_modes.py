# /core/ui/ui_special_modes.py

import copy
import json
import random
import re
import threading
import time
from collections import defaultdict
from datetime import datetime
from queue import Queue
from unittest.mock import patch

import tcod

from .app_states import AppState
from .ui_framework import DynamicTextBox
from .views.character_generation_test_view import CharacterGenerationTestView
from .views.game_view import GameView
from .views.noise_visualizer_view import NoiseVisualizerView
from ..common import file_io, utils, command_parser
from ..common.config_loader import config
from ..common.game_state import GameState, GenerationState
from ..components import game_functions, character_factory
from ..llm.calibration_manager import CalibrationManager
from ..llm.llm_api import execute_task
from ..worldgen.procgen_manager import ProcGenManager
from ..worldgen.map_architect_v3 import MapArchitectV3
from ..worldgen.v3_components.feature_node import FeatureNode


class SpecialModeManager:
    """
    Manages the logic and state for non-standard application modes like
    Calibration and PEG testing.
    """

    def __init__(self, ui_manager):
        self.ui_manager = ui_manager
        self.active_mode = None
        self.plausible_subfeatures = [
            ("Bookshelf", "GENERIC_INTERACTABLE"), ("Small Alcove", "ALCOVE"),
            ("Weapon Rack", "GENERIC_INTERACTABLE"), ("Fireplace", "GENERIC_INTERACTABLE"),
            ("Circular room", "CIRCULAR_ROOM"), ("Window", "ARCHWAY"),
            ("Storage Closet", "BUILT_ROOM"), ("Support Pillar", "SOLID_BARRIER"),
            ("Connecting Tunnel", "GENERIC_HALL"), ("Dirt Trail", "GENERIC_TRAIL"),
            ("Paved Road", "GENERIC_PATH")
        ]
        self.peg_test_generator = None
        self.current_peg_phase = 'INIT'
        self.active_patchers = []
        self.scenario_subfeatures = []
        self.last_peg_state = None
        self.last_path_tracer = []
        self.original_find_path = None
        self.char_gen_context = None
        self.char_gen_thread = None
        self.char_gen_queue = None
        self.char_gen_stop_event = None
        self.generated_characters = []
        self.peg_v3_scenarios = []
        self.peg_v3_scenario_index = 0
        self.noise_visualizer_active = False

    def set_active_mode(self, app_state: AppState):
        """Sets the active special mode based on the application state."""
        if app_state == AppState.PEG_V3_TEST:
            self.active_mode = "PEG_V3_TEST"
        elif app_state == AppState.CHARACTER_GENERATION_TEST:
            self.active_mode = "CHAR_GEN_TEST"
        elif app_state == AppState.CALIBRATING:
            self.active_mode = "CALIBRATING"
        elif app_state == AppState.NOISE_VISUALIZER_TEST:
            self.active_mode = "NOISE_VISUALIZER"
        else:
            self.active_mode = None

    def process_logic(self, app_state: AppState):
        """Handles the logic tick for the currently active special mode."""
        if app_state == AppState.PEG_V3_TEST:
            if self.peg_test_generator is None:
                self.run_peg_v3_test()
        elif app_state == AppState.CHARACTER_GENERATION_TEST:
            self.handle_character_generation_updates()
        elif app_state == AppState.NOISE_VISUALIZER_TEST:
            if not self.noise_visualizer_active:
                self.run_noise_visualizer_test()

    def handle_event(self, event: tcod.event.Event) -> bool:
        """
        Handles events for active special modes. Returns True if the event was
        consumed, False otherwise.
        """
        if not self.active_mode:
            return False

        if self.active_mode == "PEG_V3_TEST":
            if isinstance(event, tcod.event.KeyDown):
                key = event.sym
                phase = self.current_peg_phase

                if key in (tcod.event.KeySym.LEFT, tcod.event.KeySym.RIGHT, tcod.event.KeySym.ESCAPE):
                    self.stop_peg_patches()
                    if key == tcod.event.KeySym.LEFT:
                        self.peg_v3_scenario_index = (self.peg_v3_scenario_index - 1) % len(self.peg_v3_scenarios)
                        self.run_peg_v3_test()
                    elif key == tcod.event.KeySym.RIGHT:
                        self.peg_v3_scenario_index = (self.peg_v3_scenario_index + 1) % len(self.peg_v3_scenarios)
                        self.run_peg_v3_test()
                    elif key == tcod.event.KeySym.ESCAPE:
                        self.ui_manager.app.app_state = AppState.WORLD_SELECTION
                        self.ui_manager.active_view = None
                    return True

                if key in (tcod.event.KeySym.RETURN, tcod.event.KeySym.KP_ENTER, tcod.event.KeySym.SPACE):
                    if phase in ('DONE', 'FINAL'):
                        self.run_peg_v3_test()
                    else:
                        self.advance_peg_test_step(key)
                    return True

        elif self.active_mode == "CHAR_GEN_TEST":
            if isinstance(event, tcod.event.KeyDown) and event.sym == tcod.event.KeySym.ESCAPE:
                self.stop_character_generation_test()
                return True

        elif self.active_mode == "NOISE_VISUALIZER":
            if self.ui_manager.active_view:
                self.ui_manager.active_view.handle_event(event)
                return True

        return False

    def stop_all(self):
        """Stops all active special mode processes (threads, patches)."""
        self.stop_peg_patches()
        self.stop_character_generation_test()
        self.noise_visualizer_active = False

    def start_character_generation_test(self, context: str):
        """Initializes and kicks off the character generation test."""
        self.char_gen_context = context
        self.char_gen_queue = Queue()
        self.char_gen_stop_event = threading.Event()
        self.generated_characters = []

        self.char_gen_thread = threading.Thread(target=self._run_char_gen_loop, daemon=True)
        self.char_gen_thread.start()

        self.ui_manager.active_view = CharacterGenerationTestView(
            context_text=context,
            console_width=self.ui_manager.root_console.width,
            console_height=self.ui_manager.root_console.height
        )
        self.active_mode = "CHAR_GEN_TEST"

    def _run_char_gen_loop(self):
        """The target function for the character generation thread."""
        director_agent = config.agents['DIRECTOR']
        while not self.char_gen_stop_event.is_set():
            initial_char = character_factory.create_lead_from_role_and_scene(
                self.ui_manager.app.atlas_engine, director_agent,
                self.char_gen_context, "A compelling character suitable for the scene."
            )
            if not initial_char:
                time.sleep(1)
                continue
            self.char_gen_queue.put(copy.deepcopy(initial_char))
            time.sleep(3)

    def handle_character_generation_updates(self):
        """Checks the queue for newly generated characters and updates the view."""
        if self.char_gen_queue and not self.char_gen_queue.empty():
            new_char = self.char_gen_queue.get()
            self.generated_characters.insert(0, new_char)
            if isinstance(self.ui_manager.active_view, CharacterGenerationTestView):
                self.ui_manager.active_view.update_characters(self.generated_characters)

    def stop_character_generation_test(self):
        """Stops the generation thread and resets state."""
        if self.char_gen_stop_event: self.char_gen_stop_event.set()
        if self.char_gen_thread and self.char_gen_thread.is_alive():
            self.char_gen_thread.join(timeout=2.0)
        self.char_gen_context = None
        self.char_gen_thread = None
        self.char_gen_queue = None
        self.char_gen_stop_event = None
        self.generated_characters = []
        self.ui_manager.app.app_state = AppState.WORLD_SELECTION
        self.active_mode = None

    def _mock_execute_task_for_peg(self, engine, agent, task_key, messages, **kwargs):
        """A stateful mock function to replace llm_api.execute_task for the PEG tests."""
        utils.log_message('debug', f"[MOCK] Received call for task: {task_key}")
        response_data = None

        if task_key == 'PROCGEN_GENERATE_NARRATIVE_BEAT':
            combined_features = self.plausible_subfeatures + self.scenario_subfeatures
            feature_name, _ = random.choice(combined_features)
            self.call_counters[feature_name] += 1
            unique_name = f"{feature_name} {self.call_counters[feature_name]}"
            response_data = f"There is a {unique_name} here."

        elif task_key == 'PEG_CREATE_FEATURE_JSON':
            sentence = kwargs['task_prompt_kwargs']['sentence']
            match = re.search(r'is a (.+?) here\.', sentence)
            if not match:
                response_data = {}
            else:
                feature_name = match.group(1).strip()
                base_name = re.sub(r'\s\d+$', '', feature_name)
                all_feature_types = self.plausible_subfeatures + self.scenario_subfeatures
                feature_type = next((ftype for name, ftype in all_feature_types if name == base_name), "BUILT_ROOM")
                feature_def = config.features.get(feature_type, {})
                response_data = {
                    "name": feature_name,
                    "description": f"A standard {base_name.lower()} as described.",
                    "type": feature_type,
                    "dimensions": "medium"
                }

        elif task_key == 'PEG_V3_DECIDE_CONNECTOR_STRATEGY':
            targets_str = kwargs['task_prompt_kwargs']['nearby_targets_list']
            targets = [line.strip().lstrip('- ') for line in targets_str.split('\n') if line.strip()]
            if targets and random.random() > 0.5:
                response_data = random.choice(targets)
            else:
                response_data = "CREATE_NEW"

        elif task_key == 'PEG_V3_CREATE_CONNECTOR_CHILD':
            response_data = "At the end of the passage is a small guard post."

        elif task_key in ['PEG_V3_CHOOSE_PARENT', 'PEG_V3_CHOOSE_PATH_TARGET']:
            option_list_str = kwargs['task_prompt_kwargs'].get('parent_options_list') or kwargs[
                'task_prompt_kwargs'].get('target_options_list', '')
            options = [line.strip().lstrip('- ') for line in option_list_str.split('\n') if
                       line.strip() and "_BORDER" not in line]
            response_data = random.choice(options) if options else "NONE"

        else:
            mock_response_template = self.mock_data_source.get(task_key)
            if mock_response_template is None:
                utils.log_message('debug', f"  [MOCK WARNING] No mock data found for task: {task_key}")
                return ""
            if task_key in ['PEG_V3_GET_NARRATIVE_SEED', 'PEG_V2_DEFINE_AREA_DATA']:
                area_name = kwargs['task_prompt_kwargs']['area_name']
                if isinstance(mock_response_template, dict):
                    response_data = mock_response_template.get(area_name)
            else:
                response_data = mock_response_template

        utils.log_message('debug', f"  [MOCK] Responding for {task_key} with: {response_data}")
        if response_data is None: return ""
        if isinstance(response_data, str): return response_data
        return json.dumps(response_data)

    def _start_peg_patches(self):
        """Starts and tracks manual mock patches for the PEG test modes."""
        self.stop_peg_patches()
        llm_patcher = patch('core.worldgen.v3_components.v3_llm.execute_task',
                            side_effect=self._mock_execute_task_for_peg)
        llm_patcher.start()
        self.active_patchers.append(llm_patcher)

    def stop_peg_patches(self):
        """Stops any active mock patches."""
        for p in self.active_patchers:
            p.stop()
        self.active_patchers.clear()
        self.peg_test_generator = None
        self.current_peg_phase = 'INIT'

    def run_peg_v3_test(self):
        """Sets up and runs the first step of the PEG V3 test view."""
        ui = self.ui_manager
        self.last_peg_state = None

        if not self.peg_v3_scenarios:
            scenarios_path = file_io.join_path(config.data_dir, 'peg_v3_test_scenarios.json')
            self.peg_v3_scenarios = file_io.read_json(scenarios_path, default=[])
            self.peg_v3_scenario_index = 0
        if not self.peg_v3_scenarios:
            ui.app.app_state = AppState.WORLD_SELECTION;
            return

        scenario = self.peg_v3_scenarios[self.peg_v3_scenario_index]
        self.mock_data_source = scenario.get('mock_llm_calls', {})
        self.call_counters = defaultdict(int)

        game_state = GameState()

        # Correctly instantiate the manager first, then the architect
        procgen_manager = ProcGenManager(ui.app.atlas_engine)
        procgen_manager._initialize_components(game_state.game_map)

        architect = MapArchitectV3(procgen_manager, game_state.game_map, "PEG V3 Test", scenario['name'])

        ui.active_view = ui.game_view
        ui.active_view.update_state(game_state)

        def ui_render_callback(gen_state):
            if isinstance(ui.active_view, GameView):
                ui.active_view.update_state(game_state, gen_state)
                ui.active_view.set_overlay_mask(getattr(gen_state, 'clearance_mask', None))

        self._start_peg_patches()
        self.peg_test_generator = architect.generate_layout_in_steps(ui_render_callback)
        self.advance_peg_test_step()

    def advance_peg_test_step(self, key_sym=None):
        """
        Runs the PEG generator, looping internally until it hits a "pause" state.
        """
        if self.peg_test_generator is None: return

        animation_phases = ['INITIAL_GROWTH_STEP', 'SUBFEATURE_GROWTH_STEP', 'PATH_DRAW_STEP']
        major_decision_phases = ['POST_GROWTH', 'PRE_JITTER', 'PRE_INTERIOR_PLACEMENT', 'PRE_CONNECT', 'POST_CONNECT',
                                 'PRE_PATH_DRAW']

        for _ in range(500):
            try:
                phase, state = next(self.peg_test_generator)
                self.current_peg_phase = phase
                self.last_peg_state = state
            except StopIteration:
                self.current_peg_phase = 'DONE';
                break
            if self.current_peg_phase in animation_phases: break
            if key_sym == tcod.event.KeySym.SPACE and self.current_peg_phase in major_decision_phases:
                break
            elif self.current_peg_phase in major_decision_phases:
                break

        if self.current_peg_phase in ('FINAL', 'DONE', 'VERTEX_DATA'):
            self.stop_peg_patches()

    def run_noise_visualizer_test(self):
        """Sets up and runs the noise visualizer test."""
        self.noise_visualizer_active = True
        ui = self.ui_manager

        game_state = GameState()
        procgen_manager = ProcGenManager(ui.app.atlas_engine)
        procgen_manager._initialize_components(game_state.game_map)

        architect = MapArchitectV3(procgen_manager, game_state.game_map, "Noise Test", "Noise Test")

        possible_features = [
            key for key, data in config.features.items()
            if data.get('placement_strategy') == 'EXTERIOR'
        ]
        if len(possible_features) < 5:
            possible_features.extend(['GENERIC_ROOM'] * (5 - len(possible_features)))

        initial_specs = [
            {'name': f'Area {i}', 'type': random.choice(possible_features),
             'size_tier': random.choice(['small', 'medium', 'large'])}
            for i in range(5)
        ]

        # Note: This part might need adjustment as place_and_grow_initial_features is not on architect
        # For now, we assume a simplified initial placement for visualization
        for spec in initial_specs:
            procgen_manager.placement.place_new_root_branch(spec, architect.initial_feature_branches)

        architect.map_ops.run_refinement_phase(architect.initial_feature_branches, on_iteration_end=lambda: None)

        root_nodes = architect.initial_feature_branches
        for i in range(len(root_nodes)):
            for j in range(i + 1, len(root_nodes)):
                node_a, node_b = root_nodes[i], root_nodes[j]
                start_points = architect.pathing.get_valid_connection_points(node_a, 0, root_nodes)
                end_points = architect.pathing.get_valid_connection_points(node_b, 0, root_nodes)
                if start_points and end_points:
                    connection = architect.pathing.find_first_valid_connection(start_points, end_points, 0,
                                                                               root_nodes)
                    if connection and connection != (None, None):
                        start_pos, end_pos = connection
                        path = architect.pathing.find_path_with_clearance(start_pos, end_pos, 0, root_nodes, {})
                        if path:
                            path_node = FeatureNode(f"Path {i}-{j}", "GENERIC_PATH", 0, 0, 0, 0)
                            path_node.path_coords = path
                            node_a.subfeatures.append(path_node)

        gen_state = GenerationState(game_state.game_map)
        architect.converter.populate_generation_state(gen_state, architect.initial_feature_branches)

        from ..common.game_state import MapArtist
        artist = MapArtist()
        artist.draw_map(game_state.game_map, gen_state, config.features)

        clearance_mask = architect.pathing._create_clearance_mask(1,
                                                                  architect.initial_feature_branches)

        def on_exit():
            self.noise_visualizer_active = False
            self.ui_manager.app.app_state = AppState.WORLD_SELECTION

        view = NoiseVisualizerView(
            on_exit=on_exit,
            console_width=ui.root_console.width,
            console_height=ui.root_console.height
        )
        view.set_base_map(game_state, gen_state, clearance_mask)
        ui.active_view = view