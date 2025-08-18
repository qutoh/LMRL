# /core/ui/ui_special_modes.py

import threading
import time
import json
import random
import re
import copy
from collections import defaultdict
from unittest.mock import patch
import tcod
from datetime import datetime
from queue import Queue

from ..llm.calibration_manager import CalibrationManager
from ..llm.llm_api import execute_task
from ..common.game_state import GameState, GenerationState, MapArtist
from ..worldgen.map_architect import MapArchitect
from ..worldgen.map_architect_v2 import MapArchitectV2
from ..worldgen.map_architect_v3 import MapArchitectV3
from ..common import file_io, utils, command_parser
from ..common.config_loader import config
from .app_states import AppState
from .ui_framework import DynamicTextBox
from .views.game_view import GameView
from .views.character_generation_test_view import CharacterGenerationTestView
from ..components import game_functions, character_factory


class SpecialModeManager:
    """
    Manages the logic and state for non-standard application modes like
    Calibration and PEG testing.
    """

    def __init__(self, ui_manager):
        self.ui_manager = ui_manager
        self.plausible_subfeatures = [
            ("Bookshelf", "GENERIC_INTERACTABLE"), ("Small Alcove", "GENERIC_CONTAINER"),
            ("Weapon Rack", "GENERIC_INTERACTABLE"), ("Fireplace", "GENERIC_INTERACTABLE"),
            ("Archway", "GENERIC_PORTAL"), ("Window", "GENERIC_PORTAL"),
            ("Storage Closet", "GENERIC_CONTAINER"), ("Support Pillar", "SOLID_BARRIER")
        ]
        # PEG Test attributes
        self.peg_test_generator = None
        self.current_peg_phase = 'INIT'
        self.active_patchers = []
        self.scenario_subfeatures = []
        self.last_peg_state = None
        self.last_path_tracer = []
        self.original_find_path = None

        # Character Generation Test attributes
        self.char_gen_context = None
        self.char_gen_thread = None
        self.char_gen_queue = None
        self.char_gen_stop_event = None
        self.generated_characters = []

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
        self.ui_manager.special_mode_active = "CHAR_GEN_TEST"

    def _generate_initial_equipment_stepwise_for_test(self, equipment_agent, physical_description: str) -> list[dict]:
        """A copy of the stepwise fallback for use in the isolated test environment."""
        utils.log_message('debug', "[CHAR_GEN_TEST] JSON generation failed, using stepwise fallback.")
        names_kwargs = {"physical_description": physical_description}
        names_response = execute_task(self.ui_manager.atlas_engine, equipment_agent,
                                      'EQUIPMENT_GET_ITEM_NAMES_FROM_DESC', [], task_prompt_kwargs=names_kwargs)

        item_names = [name.strip() for name in names_response.split(';') if name.strip()]
        if not item_names:
            return []

        desc_kwargs = {"item_names_list": "; ".join(item_names), "context": physical_description}
        raw_desc_response = execute_task(self.ui_manager.atlas_engine, equipment_agent, 'EQUIPMENT_DESCRIBE_ITEMS', [],
                                         task_prompt_kwargs=desc_kwargs)

        command = command_parser.parse_structured_command(
            self.ui_manager.atlas_engine, raw_desc_response, equipment_agent.get('name'),
            fallback_task_key='CH_FIX_INITIAL_EQUIPMENT_JSON'
        )

        final_items = []
        if command:
            for name, details in command.items():
                if isinstance(details, dict) and 'description' in details:
                    final_items.append({"name": name, "description": details["description"]})
        return final_items

    def _run_char_gen_loop(self):
        """The target function for the character generation thread."""
        director_agent = config.agents['DIRECTOR']
        equipment_agent = config.agents['EQUIPMENT_MANAGER']
        generic_role = "A compelling character suitable for the scene."

        while not self.char_gen_stop_event.is_set():
            initial_char = character_factory.create_lead_from_role_and_scene(
                self.ui_manager.atlas_engine, director_agent,
                self.char_gen_context, generic_role
            )
            if not initial_char:
                time.sleep(1)
                continue

            self.char_gen_queue.put(copy.deepcopy(initial_char))
            utils.log_message('debug', f"[CHAR_GEN_TEST] Generated initial character: {initial_char.get('name')}")
            time.sleep(1)

            if not initial_char.get('equipment', {}).get('equipped'):
                continue

            utils.log_message('debug', f"[CHAR_GEN_TEST] Starting REMOVAL phase for {initial_char['name']}.")
            char_state = copy.deepcopy(initial_char)
            items_to_remove = list(char_state['equipment']['equipped'])

            for item_to_remove in items_to_remove:
                if self.char_gen_stop_event.is_set(): return
                char_state['equipment']['equipped'].remove(item_to_remove)
                char_state['equipment']['removed'].append(item_to_remove)
                utils.log_message('debug', f"  - Removing: {item_to_remove['name']}")

                current_equipped_names = sorted([item['name'] for item in char_state['equipment']['equipped']])

                # OUTFIT SHORT-CIRCUIT
                matched_outfit = next((outfit for outfit in char_state['equipment']['outfits'].values() if
                                       sorted(outfit.get('items', [])) == current_equipped_names), None)
                if matched_outfit:
                    char_state['physical_description'] = matched_outfit['description']
                    utils.log_message('debug', f"  - Matched existing outfit. Short-circuiting.")
                else:
                    update_desc_kwargs = {"original_description": char_state.get('physical_description', ''),
                                          "items_added": "None", "items_removed": item_to_remove['name']}
                    new_desc = execute_task(self.ui_manager.atlas_engine, director_agent,
                                            'DIRECTOR_UPDATE_PHYSICAL_DESCRIPTION', [],
                                            task_prompt_kwargs=update_desc_kwargs)
                    if new_desc: char_state['physical_description'] = new_desc.strip()

                    outfit_name = f"OUTFIT_{len(char_state['equipment']['outfits'])}"
                    char_state['equipment']['outfits'][outfit_name] = {"items": current_equipped_names,
                                                                       "description": char_state[
                                                                           'physical_description']}
                    utils.log_message('debug', f"  - Saved new outfit: {outfit_name}")

                self.char_gen_queue.put(copy.deepcopy(char_state))
                time.sleep(1)

            utils.log_message('debug', f"[CHAR_GEN_TEST] Starting ADDITION phase for {initial_char['name']}.")
            items_to_add_back = list(char_state['equipment']['removed'])
            items_to_add_back.sort(key=lambda x: x['name'])

            for item_to_add in items_to_add_back:
                if self.char_gen_stop_event.is_set(): return
                char_state['equipment']['removed'].remove(item_to_add)
                char_state['equipment']['equipped'].append(item_to_add)
                utils.log_message('debug', f"  - Adding back: {item_to_add['name']}")

                current_equipped_names = sorted([item['name'] for item in char_state['equipment']['equipped']])

                # OUTFIT SHORT-CIRCUIT
                matched_outfit_data = next((data for name, data in char_state['equipment']['outfits'].items() if
                                            sorted(data.get('items', [])) == current_equipped_names), None)
                if matched_outfit_data:
                    char_state['physical_description'] = matched_outfit_data['description']
                    utils.log_message('debug', f"  - Matched existing outfit. Short-circuiting.")
                else:
                    utils.log_message('debug', "[CHAR_GEN_TEST_WARNING] Outfit hashing failed on re-add.")
                    update_desc_kwargs = {"original_description": char_state.get('physical_description', ''),
                                          "items_added": item_to_add['name'], "items_removed": "None"}
                    new_desc = execute_task(self.ui_manager.atlas_engine, director_agent,
                                            'DIRECTOR_UPDATE_PHYSICAL_DESCRIPTION', [],
                                            task_prompt_kwargs=update_desc_kwargs)
                    if new_desc: char_state['physical_description'] = new_desc.strip()

                self.char_gen_queue.put(copy.deepcopy(char_state))
                time.sleep(1)

            utils.log_message('debug',
                              f"[CHAR_GEN_TEST] Cycle for {initial_char['name']} complete. Waiting before next character.")
            time.sleep(3)

    def handle_character_generation_updates(self):
        """Checks the queue for newly generated characters and updates the view."""
        if self.char_gen_queue and not self.char_gen_queue.empty():
            new_char = self.char_gen_queue.get()
            self.generated_characters.insert(0, new_char)  # Add to the top of the list
            if isinstance(self.ui_manager.active_view, CharacterGenerationTestView):
                self.ui_manager.active_view.update_characters(self.generated_characters)

    def stop_character_generation_test(self):
        """Stops the generation thread and resets state."""
        if self.char_gen_stop_event:
            self.char_gen_stop_event.set()
        if self.char_gen_thread and self.char_gen_thread.is_alive():
            self.char_gen_thread.join(timeout=2.0)

        self.char_gen_context = None
        self.char_gen_thread = None
        self.char_gen_queue = None
        self.char_gen_stop_event = None
        self.generated_characters = []

        self.ui_manager.app_state = AppState.WORLD_SELECTION
        self.ui_manager.active_view = None
        self.ui_manager.special_mode_active = None

    def start_calibration(self):
        """Initializes and kicks off the calibration process."""
        ui = self.ui_manager
        cal_manager = CalibrationManager(ui.atlas_engine, ui.model_manager.embedding_model, ui.event_log)
        ui.calibration_jobs = cal_manager.get_calibration_plan()
        ui.current_job_index = 0

    def _run_single_calibration_job(self, job):
        """The target function for the calibration thread. Runs one job."""
        ui = self.ui_manager
        model_id = job['model_id']
        task = job['task']

        ui.calibration_update_queue.put({'type': 'status', 'text': f"Loading model for calibration: {model_id}..."})
        ui.model_manager.set_active_models([model_id])
        ui.model_manager.models_loaded.wait()

        cal_manager = CalibrationManager(ui.atlas_engine, ui.model_manager.embedding_model, ui.event_log,
                                         ui.calibration_update_queue)
        cal_manager.run_calibration_test(job)

    def handle_calibration_step(self):
        """Checks calibration progress and starts the next job if ready."""
        ui = self.ui_manager
        if ui.calibration_thread is None or not ui.calibration_thread.is_alive():
            if ui.current_job_index >= len(ui.calibration_jobs):
                ui.calibration_update_queue.put({'type': 'status', 'text': "Calibration complete for all models."})
                # Short delay to allow the final message to be seen
                time.sleep(3)
                ui.app_state = AppState.WORLD_SELECTION
            else:
                job = ui.calibration_jobs[ui.current_job_index]
                ui.calibration_thread = threading.Thread(target=self._run_single_calibration_job, args=(job,),
                                                         daemon=True)
                ui.calibration_thread.start()
                ui.current_job_index += 1

    def _mock_execute_task_for_peg(self, engine, agent, task_key, messages, **kwargs):
        """A stateful mock function to replace llm_api.execute_task for the PEG tests."""
        response_data = None

        log_message = f"[MOCK] Received call for task: {task_key}"
        utils.log_message('debug', log_message)

        # --- Mocks for V2 & V3 Initial Phase ---
        if task_key == 'PEG_V2_DEFINE_AREA_DATA':
            area_name = kwargs['task_prompt_kwargs']['area_name']
            response_data = self.mock_data_source[task_key].get(area_name)

        # --- Mocks for V3 Conversational Subfeature Phase ---
        elif task_key == 'PEG_V3_GET_NARRATIVE_SEED':
            area_name = kwargs['task_prompt_kwargs']['area_name']
            response_data = f"The {area_name} is a vast and open space, defining the center of the complex."

        elif task_key == 'PROCGEN_GENERATE_NARRATIVE_BEAT':
            # This now picks from a combined list of generic and scenario-specific features for stress testing.
            combined_features = self.plausible_subfeatures + self.scenario_subfeatures
            feature_name, _ = random.choice(combined_features)

            self.call_counters[feature_name] += 1
            unique_name = f"{feature_name} {self.call_counters[feature_name]}"
            response_data = f"Nearby, there is a {unique_name} that is medium-sized."

        elif task_key == 'PEG_CREATE_FEATURE_JSON':
            sentence = kwargs['task_prompt_kwargs']['sentence']
            match = re.search(r'there is a (.+?) that is', sentence)
            if not match:
                return json.dumps({})

            feature_name = match.group(1).strip()
            base_name = re.sub(r'\s\d+$', '', feature_name)

            all_feature_types = self.plausible_subfeatures + self.scenario_subfeatures
            feature_type = "GENERIC_CONTAINER"  # Default
            for name, f_type in all_feature_types:
                if name == base_name:
                    feature_type = f_type
                    break

            feature_def = config.features.get(feature_type, {})
            mock_json = {
                "name": feature_name,
                "description": f"A standard {base_name.lower()} as described.",
                "type": feature_type,
                "dimensions": "1x1 tile" if feature_def.get('feature_type') == "INTERACTABLE" else "medium"
            }
            response_data = mock_json

        elif task_key == 'PEG_V3_CHOOSE_PARENT':
            parent_list_str = kwargs['task_prompt_kwargs']['parent_options_list']
            options = [line.strip().lstrip('- ') for line in parent_list_str.split('\n') if line.strip()]
            response_data = random.choice(options) if options else "NONE"

        # --- Default V2/V3 mocks ---
        else:
            response_data = self.mock_data_source.get(task_key)

        utils.log_message('debug', f"  [MOCK] Responding for {task_key} with: {response_data}")
        if response_data is None: return ""
        if isinstance(response_data, str): return response_data
        return json.dumps(response_data)

    def _find_path_wrapper(self, *args, **kwargs):
        """A wrapper for the real find_path function to capture its output for the tracer."""
        path = self.original_find_path(*args, **kwargs)
        if path and self.ui_manager.active_view:
            tileset = self.ui_manager.tileset
            tile_w, tile_h = tileset.tile_width, tileset.tile_height

            for i in range(len(path) - 1):
                start_tx, start_ty = path[i]
                end_tx, end_ty = path[i + 1]

                start_px = start_tx * tile_w + tile_w // 2
                start_py = start_ty * tile_h + tile_h // 2
                end_px = end_tx * tile_w + tile_w // 2
                end_py = end_ty * tile_h + tile_h // 2

                self.ui_manager.active_view.add_line((start_px, start_py), (end_px, end_py), (255, 255, 0))
        return path

    def _start_peg_patches(self):
        """Starts and tracks manual mock patches for the PEG test modes."""
        self.stop_peg_patches()
        # Patch LLM calls
        llm_patcher = patch('core.worldgen.v3_components.v3_llm.execute_task',
                            side_effect=self._mock_execute_task_for_peg)
        llm_patcher.start()
        self.active_patchers.append(llm_patcher)

        # Monkey-patch the pathfinding function to intercept its results
        if self.original_find_path is None:
            self.original_find_path = game_functions.find_path
            game_functions.find_path = self._find_path_wrapper

    def stop_peg_patches(self):
        """Stops any active mock patches."""
        for p in self.active_patchers:
            p.stop()
        self.active_patchers.clear()

        # Restore the original pathfinding function
        if self.original_find_path:
            game_functions.find_path = self.original_find_path
            self.original_find_path = None

        self.peg_test_generator = None
        self.current_peg_phase = 'INIT'

    def _save_peg_test_output(self, gen_state: GenerationState, scenario_name: str):
        """Saves the complete output of a PEG test run to a debug world and save file."""
        self.ui_manager.event_log.add_message("Saving PEG test output to debug world...", (200, 200, 100))

        DEBUG_WORLD_NAME = "_peg_debug_world"
        DEBUG_SCENE_PROMPT = f"PEG V3 Test Scenario: {scenario_name}"
        DEBUG_SAVE_NAME = f"peg_v3_{file_io.sanitize_filename(scenario_name)}"

        # 1. Create World structure
        world_dir = file_io.join_path(file_io.PROJECT_ROOT, 'data', 'worlds', DEBUG_WORLD_NAME)
        file_io.create_directory(world_dir)
        world_json_path = file_io.join_path(world_dir, 'world.json')
        world_data = {
            DEBUG_WORLD_NAME: {
                "Name": DEBUG_WORLD_NAME, "Type": "Debug",
                "Description": "World for PEG test outputs.", "theme": "Procedural Generation Debugging"
            }
        }
        file_io.write_json(world_json_path, world_data)
        file_io.write_json(file_io.join_path(world_dir, 'casting_npcs.json'), [])

        # 2. Save generated scene to world's scene list
        scenes_path = file_io.join_path(world_dir, 'generated_scenes.json')
        scenes_data = file_io.read_json(scenes_path, default=[])
        scenes_data = [s for s in scenes_data if s.get('scene_prompt') != DEBUG_SCENE_PROMPT]
        new_scene = {
            "scene_prompt": DEBUG_SCENE_PROMPT,
            "source_location_name": DEBUG_WORLD_NAME,
            "source_location": world_data[DEBUG_WORLD_NAME]
        }
        scenes_data.append(new_scene)
        file_io.write_json(scenes_path, scenes_data)

        # 3. Cache the generated level to world's levels.json
        levels_path = file_io.join_path(world_dir, 'levels.json')
        levels_cache = file_io.read_json(levels_path, default={})

        serializable_features = {}
        for tag, feature in gen_state.placed_features.items():
            feat_copy = feature.copy()
            if 'bounding_box' in feat_copy and feat_copy.get('bounding_box'):
                feat_copy['bounding_box'] = [int(v) for v in feat_copy['bounding_box']]
            serializable_features[tag] = feat_copy

        level_data_to_cache = {
            "placed_features": serializable_features,
            "narrative_log": gen_state.narrative_log,
            "door_locations": gen_state.door_locations
        }
        if gen_state.layout_graph:
            level_data_to_cache['layout_graph'] = {
                'nodes': gen_state.layout_graph.nodes,
                'edges': gen_state.layout_graph.edges
            }
        if gen_state.physics_layout:
            level_data_to_cache['physics_layout'] = gen_state.physics_layout

        levels_cache[DEBUG_SCENE_PROMPT] = level_data_to_cache
        file_io.write_json(levels_path, levels_cache)
        self.ui_manager.event_log.add_message(f"Saved level data for '{scenario_name}'.")

        # 4. Create a save game directory (overwrite old one)
        world_save_path = file_io.join_path(file_io.SAVE_DATA_ROOT, DEBUG_WORLD_NAME)
        final_save_path = file_io.join_path(world_save_path, DEBUG_SAVE_NAME)
        file_io.remove_directory(final_save_path)  # Overwrite logic

        temp_run_path = file_io.setup_new_run_directory(config.data_dir, DEBUG_WORLD_NAME)
        final_run_path, error = file_io.finalize_run_directory(temp_run_path, DEBUG_SAVE_NAME)
        if error:
            self.ui_manager.event_log.add_message(f"Error creating save directory: {error}", (255, 100, 100))
            return

        # 5. Save a minimal story_state.json
        story_state_data = {
            "dialogue_log": [{"speaker": "Scene Setter", "content": gen_state.narrative_log or DEBUG_SCENE_PROMPT}],
            "summaries": [],
            "current_game_time": datetime.now().isoformat(),
            "current_scene_key": DEBUG_SCENE_PROMPT
        }
        state_path = file_io.join_path(final_run_path, 'story_state.json')
        file_io.write_json(state_path, story_state_data)

        self.ui_manager.event_log.add_message(f"Created save game '{DEBUG_SAVE_NAME}'.", (150, 255, 150))

    def run_peg_v3_test(self):
        """Sets up and runs the first step of the PEG V3 test view."""
        ui = self.ui_manager
        self.last_peg_state = None
        self.last_path_tracer.clear()

        if not hasattr(self, 'peg_v3_scenarios'):
            scenarios_path = file_io.join_path(config.data_dir, 'peg_v2_test_scenarios.json')
            self.peg_v3_scenarios = file_io.read_json(scenarios_path, default=[])
            self.peg_v3_scenario_index = 0

        if not self.peg_v3_scenarios:
            ui.event_log.add_message("peg_v2_test_scenarios.json not found or empty.", (255, 100, 100))
            ui.app_state = AppState.WORLD_SELECTION
            return

        scenario = self.peg_v3_scenarios[self.peg_v3_scenario_index]
        self.mock_data_source = scenario.get('mock_llm_calls', {})
        self.call_counters = defaultdict(int)

        area_names_str = self.mock_data_source.get("PEG_V2_GET_AREA_NAMES", "")
        area_names = [name.strip() for name in area_names_str.split(';') if name.strip()]
        area_data = self.mock_data_source.get("PEG_V2_DEFINE_AREA_DATA", {})
        self.scenario_subfeatures = [(name, area_data.get(name, {}).get("type", "GENERIC_CONTAINER")) for name in
                                     area_names]

        game_state = GameState()
        architect = MapArchitectV3(ui.atlas_engine, game_state.game_map, "PEG V3 Test", scenario['name'])
        title = "PEG V3 (Iterative Placement) Debug"

        ui.active_view = GameView(ui.event_log, None, is_debug_mode=True)
        ui.active_view.update_state(game_state)

        def ui_render_callback(gen_state):
            if isinstance(ui.active_view, GameView):
                ui.active_view.update_state(game_state, gen_state)
                ui._render()

        self._start_peg_patches()
        self.peg_test_generator = architect.generate_layout_in_steps(ui_render_callback)
        self.advance_peg_test_step()

    def advance_peg_test_step(self, key_sym=None):
        """Runs the next step of the current PEG test generator, driven by user input."""
        ui = self.ui_manager
        if self.peg_test_generator is None:
            return

        # Handle fast-forwarding
        if self.current_peg_phase in ('SUBFEATURE_STEP',
                                      'INTERIOR_PLACEMENT_STEP') and key_sym == tcod.event.KeySym.SPACE:
            target_phase = self.current_peg_phase
            while self.current_peg_phase == target_phase:
                try:
                    phase, state = next(self.peg_test_generator)
                    self.current_peg_phase = phase
                    self.last_peg_state = state
                except StopIteration:
                    self.current_peg_phase = 'DONE'
                    break
        else:
            # Handle a normal single step
            try:
                phase, state = next(self.peg_test_generator)
                self.current_peg_phase = phase
                self.last_peg_state = state
            except StopIteration:
                self.current_peg_phase = 'DONE'

        # Determine the prompt text based on the new phase
        scenario = self.peg_v3_scenarios[self.peg_v3_scenario_index]
        title_str = "PEG V3 (Iterative)"
        base_text = f"Scenario: {scenario['name']} | [LEFT/RIGHT] Change | [ESC] Back"
        prompt_text = ""

        if self.current_peg_phase == 'INITIAL_PLACEMENT':
            prompt_text = "\n[ENTER] Begin placing sub-features"
        elif self.current_peg_phase == 'SUBFEATURE_STEP':
            prompt_text = "\n[ENTER] Place next sub-feature | [SPACE] Skip to Jitter"
        elif self.current_peg_phase == 'PRE_JITTER':
            prompt_text = "\n[ENTER] Apply Jitter"
        elif self.current_peg_phase == 'PRE_INTERIOR_PLACEMENT':
            prompt_text = "\n[ENTER] Begin placing interior features"
        elif self.current_peg_phase == 'INTERIOR_PLACEMENT_STEP':
            prompt_text = "\n[ENTER] Place next interior feature | [SPACE] Skip to Connections"
        elif self.current_peg_phase == 'PRE_CONNECT':
            prompt_text = "\n[ENTER] Create Connections"
        elif self.current_peg_phase == 'POST_CONNECT':
            prompt_text = "\nConnections complete. [ENTER] to finalize."
        else:  # FINAL or DONE
            prompt_text = "\nGeneration Complete. [ENTER] to regenerate."
            self.stop_peg_patches()
            if self.last_peg_state:
                self._save_peg_test_output(self.last_peg_state, scenario['name'])

        # Update the UI
        log_text = f"{title_str}: {base_text}{prompt_text}"
        game_log_box = DynamicTextBox(
            title=title_str, text=log_text,
            x=0, y=ui.root_console.height - 10, max_width=ui.root_console.width, max_height=9
        )
        if ui.active_view:
            ui.active_view.game_log_box = game_log_box
            ui.active_view.widgets = [game_log_box]