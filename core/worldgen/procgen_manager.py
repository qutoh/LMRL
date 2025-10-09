import re
import numpy as np
from collections import deque
import random
import queue

from core.common.config_loader import config
from core.common.game_state import GenerationState
from .map_architect import MapArchitect
from .map_architect_v2 import MapArchitectV2
from .map_architect_v3 import MapArchitectV3
from ..common import utils, file_io
from ..ui.ui_messages import AddEventLogMessage
from .v3_components.v3_llm import V3_LLM
from .v3_components.pathing import Pathing
from .v3_components.placement import Placement
from .v3_components.map_ops import MapOps
from .v3_components.interior import Interior
from .v3_components.converter import Converter
from .semantic_search import SemanticSearch
from .v3_components.feature_node import FeatureNode
from . import procgen_utils
from typing import List, Optional, Callable, Tuple, Generator, Set


class ProcGenManager:
    """
    A high-level manager that orchestrates the procedural generation process
    by delegating to a MapArchitect. It also owns the core service components
    like Pathing and Placement.
    """

    def __init__(self, engine):
        self.engine = engine
        self.level_generator = config.agents['LEVEL_GENERATOR']
        self.llm = V3_LLM(engine)
        self.semantic_search = SemanticSearch(engine.embedding_model)
        self.converter = Converter()

        # Components will be initialized with map dimensions in `generate`
        self.pathing: Optional[Pathing] = None
        self.placement: Optional[Placement] = None
        self.map_ops: Optional[MapOps] = None
        self.interior: Optional[Interior] = None

    def _initialize_components(self, game_map):
        """Creates instances of the core generation components."""
        map_width, map_height = game_map.width, game_map.height
        self.pathing = Pathing(game_map, self._get_temp_grid, procgen_utils.get_door_placement_mask_for_pathing)
        self.placement = Placement(map_width, map_height, self._get_temp_grid, self.pathing, self.llm,
                                   self.semantic_search)
        self.map_ops = MapOps(map_width, map_height, self.pathing, self.placement)
        self.interior = Interior(self.placement, map_width, map_height)

    def _get_temp_grid(self, all_branches: List[FeatureNode],
                       exclude_nodes: Optional[Set[FeatureNode]] = None) -> np.ndarray:
        void_space_index = config.tile_type_map.get("VOID_SPACE", -1)
        grid = np.full((self.pathing.map_height, self.pathing.map_width), void_space_index, dtype=np.int8)

        all_nodes_in_map = [node for branch in all_branches for node in branch.get_all_nodes_in_branch()]

        if exclude_nodes:
            nodes_to_draw = [node for node in all_nodes_in_map if node not in exclude_nodes]
        else:
            nodes_to_draw = all_nodes_in_map

        for node in nodes_to_draw:
            feature_def = config.features.get(node.feature_type, {})

            footprint = node.get_absolute_footprint()
            interior_footprint = node.get_absolute_interior_footprint()
            if not footprint: continue

            if border_tile_type := feature_def.get('border_tile_type'):
                if config.tile_type_map.get(border_tile_type):
                    border_coords = np.array(list(footprint - interior_footprint))
                    if border_coords.size > 0:
                        valid_mask = (border_coords[:, 0] >= 0) & (border_coords[:, 0] < self.pathing.map_width) & (
                                border_coords[:, 1] >= 0) & (border_coords[:, 1] < self.pathing.map_height)
                        valid_coords = border_coords[valid_mask]
                        if valid_coords.size > 0: grid[valid_coords[:, 1], valid_coords[:, 0]] = config.tile_type_map[
                            border_tile_type]
            if tile_type := feature_def.get('tile_type'):
                if config.tile_type_map.get(tile_type):
                    floor_coords = np.array(list(interior_footprint))
                    if floor_coords.size > 0:
                        valid_mask = (floor_coords[:, 0] >= 0) & (floor_coords[:, 0] < self.pathing.map_width) & (
                                floor_coords[:, 1] >= 0) & (floor_coords[:, 1] < self.pathing.map_height)
                        valid_coords = floor_coords[valid_mask]
                        if valid_coords.size > 0: grid[valid_coords[:, 1], valid_coords[:, 0]] = config.tile_type_map[
                            tile_type]
        return grid

    def _assign_op_budgets(self, node: FeatureNode):
        """
        Assigns cumulative operation budgets to a node based on its instance-level natures.
        """
        footprint_size = len(node.footprint)

        # Reset budgets to 0 before accumulation
        node.jitter_budget = 0
        node.erosion_budget = 0
        node.organic_op_budget = 0

        for nature_name in node.natures:
            nature_def = config.natures.get(nature_name)
            if not nature_def:
                continue

            for op_name, op_data in nature_def.get('operations', {}).items():
                scaling_factor = op_data.get('budget_scaling_factor', 0.0)
                budget_increase = int(footprint_size * scaling_factor)

                if op_name == 'jitter':
                    node.jitter_budget += budget_increase
                elif op_name == 'erosion':
                    node.erosion_budget += budget_increase
                elif op_name == 'organic_reshaping':
                    node.organic_op_budget += budget_increase

    def _grow_subfeature_coroutine(self, node: FeatureNode, draw_callback: Callable):
        final_footprint = node.footprint.copy()
        if not final_footprint or len(final_footprint) <= 1: return
        center_rx = node.current_abs_width // 2
        center_ry = node.current_abs_height // 2
        current_footprint = {(center_rx, center_ry)}
        node.footprint = current_footprint
        draw_callback()
        yield "SUBFEATURE_GROWTH_STEP", None
        q = deque(list(current_footprint))
        while current_footprint != final_footprint:
            made_change = False
            q_len = len(q)
            if q_len == 0: break
            for _ in range(q_len):
                cx, cy = q.popleft()
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                    neighbor = (cx + dx, cy + dy)
                    if neighbor in final_footprint and neighbor not in current_footprint:
                        current_footprint.add(neighbor)
                        q.append(neighbor)
                        made_change = True
            if made_change:
                node.footprint = current_footprint
                draw_callback()
                yield "SUBFEATURE_GROWTH_STEP", None
            else:
                break
        node.footprint = final_footprint
        draw_callback()
        yield "SUBFEATURE_GROWTH_STEP", None

    def _draw_path_coroutine(self, node: FeatureNode, draw_callback: Callable) -> Generator[
        Tuple[str, None], None, None]:
        if not node.path_coords: return
        full_path = list(node.path_coords)
        node.path_coords.clear()
        chunk_size = max(1, len(full_path) // 10)
        for i in range(0, len(full_path), chunk_size):
            node.path_coords.extend(full_path[i:i + chunk_size])
            draw_callback()
            yield "PATH_DRAW_STEP", None
        node.path_coords = full_path
        draw_callback()
        yield "PATH_DRAW_STEP", None

    def place_path_feature(self, feature_data: dict, parent_branch: FeatureNode, target_name: str,
                           all_branches: List[FeatureNode]) -> Optional[FeatureNode]:
        """
        Handles the reusable logic for placing a path feature from a source to a named target.
        The target can be another feature branch or a map border.
        """
        if not target_name or 'none' in target_name.lower():
            return None

        if "_BORDER" in target_name.upper():
            direction = target_name.upper().replace("_BORDER", "")
            border_coords = self.pathing.get_border_coordinates_for_direction(direction)
            if border_coords:
                return self.placement.place_subfeature_path_to_border(feature_data, parent_branch,
                                                                      border_coords, all_branches)
        else:
            # Find the target branch among all initial branches
            target_branch = next((b for b in all_branches if b.name.lower() == target_name.lower().strip()), None)
            if target_branch:
                return self.placement.place_subfeature_path_between_branches(feature_data, parent_branch,
                                                                             target_branch, all_branches)
            else:
                utils.log_message('debug', f"  Could not resolve target branch '{target_name}' for pathing.")

        return None

    def _process_newly_placed_nodes(self, newly_placed_nodes: List[FeatureNode], all_branches: List[FeatureNode],
                                    update_and_draw: Callable) -> Generator[Tuple[str, None], None, None]:
        """Helper to handle common tasks for newly placed nodes."""
        for node in newly_placed_nodes:
            self._assign_op_budgets(node)
            update_and_draw()
            if node.path_coords:
                yield from self._draw_path_coroutine(node, update_and_draw)
            else:
                yield from self._grow_subfeature_coroutine(node, update_and_draw)

        op_choice = random.choice(['jitter', 'erosion', 'organic'])
        all_nodes = [n for b in all_branches for n in b.get_all_nodes_in_branch()]
        if all_nodes:
            if op_choice == 'jitter':
                node_to_jitter = random.choice([n for n in all_nodes if n.jitter_budget > 0] or all_nodes)
                if self.map_ops.apply_jitter(node_to_jitter, all_branches):
                    update_and_draw()
            elif op_choice == 'erosion':
                erodible_nodes = [n for n in all_nodes if n.erosion_budget > 0]
                if erodible_nodes:
                    node_to_erode = random.choice(erodible_nodes)
                    if self.map_ops.apply_erosion(node_to_erode, all_branches):
                        update_and_draw()
            else:  # organic
                organic_nodes = [n for n in all_nodes if n.organic_op_budget > 0]
                if organic_nodes:
                    node_to_reshape = random.choice(organic_nodes)
                    if self.map_ops.apply_organic_reshaping(node_to_reshape, all_branches):
                        update_and_draw()

    def generate(self, scene_prompt: str, game_map, ui_callback: callable = None) -> GenerationState:
        utils.log_message('debug', "\n--- [SYSTEM] Starting Procedural Environment Generation (PEG) ---")

        # --- Initialize components with map dimensions ---
        self._initialize_components(game_map)

        # --- Exterior Tile Selection ---
        tiles = {name: data for name, data in self.engine.config.tile_types.items() if name != "VOID_SPACE"}
        tile_options_str = "\n".join(
            f"- `{name}`: {data.get('description', 'No description.')}" for name, data in tiles.items())
        raw_llm_choice = self.llm.choose_exterior_tile(scene_prompt, tile_options_str)

        chosen_tile = None
        valid_tile_keys = tiles.keys()

        for key in valid_tile_keys:
            if key.lower() == raw_llm_choice.strip().lower():
                chosen_tile = key
                break

        if not chosen_tile:
            for key in valid_tile_keys:
                if re.search(rf"\b{re.escape(key)}\b", raw_llm_choice, re.IGNORECASE):
                    chosen_tile = key
                    utils.log_message('debug',
                                      f"[PEG Setup] Extracted tile keyword '{chosen_tile}' from chatty response.")
                    break

        if not chosen_tile or chosen_tile == "VOID_SPACE":
            utils.log_message('debug', "[PEG Setup] Invalid or no tile chosen. Triggering new tile generation.")
            self.engine.render_queue.put(
                AddEventLogMessage("That's an interesting idea... creating a new ground tile for this scene..."))

            new_tile_data = self.llm.create_new_tile_type(scene_prompt)
            if new_tile_data:
                world_tiles_path = file_io.join_path(self.engine.config.data_dir, 'worlds', self.engine.world_name,
                                                     'generated_tiles.json')
                existing_world_tiles = file_io.read_json(world_tiles_path, default={})
                existing_world_tiles.update(new_tile_data)
                file_io.write_json(world_tiles_path, existing_world_tiles)

                self.engine.config.tile_types.update(new_tile_data)
                self.engine.config.tile_type_map = {name: i for i, name in
                                                    enumerate(self.engine.config.tile_types.keys())}
                self.engine.config.tile_type_map_reverse = {i: name for name, i in
                                                            self.engine.config.tile_type_map.items()}

                chosen_tile = list(new_tile_data.keys())[0]
                utils.log_message('debug', f"[PEG Setup] Successfully created and selected new tile: {chosen_tile}")
                self.engine.render_queue.put(
                    AddEventLogMessage(f"New tile '{chosen_tile}' created for {self.engine.world_name}!",
                                       (150, 255, 150)))
            else:
                chosen_tile = "DEFAULT_FLOOR"
                utils.log_message('debug', "[PEG Setup] New tile generation failed. Falling back to default.")

        utils.log_message('debug', f"[PEG] LLM chose '{chosen_tile}' as the exterior tile.")

        # --- Algorithm Selection & Execution ---
        algorithm = config.settings.get("PEG_RECONCILIATION_METHOD", "CONVERSATIONAL").strip().upper()
        utils.log_message('debug', f"[PEG] Using algorithm: {algorithm}")

        def default_callback(state):
            pass

        callback = ui_callback if ui_callback else default_callback

        state = None
        if algorithm == "PARTITIONING":
            architect_v2 = MapArchitectV2(self.engine, game_map, getattr(self.engine, 'world_theme', 'fantasy'),
                                          scene_prompt)
            state = architect_v2.generate_layout(callback)
        elif algorithm == "ITERATIVE_PLACEMENT":
            architect_v3 = MapArchitectV3(self, game_map, self.engine.game_state,
                                          getattr(self.engine, 'world_theme', 'fantasy'),
                                          scene_prompt)
            gen = architect_v3.generate_layout_in_steps(callback)
            for _, final_state_obj in gen:
                state = final_state_obj
                try:
                    message = self.engine.input_queue.get_nowait()
                    if message == '__INTERRUPT_GENERATION__':
                        utils.log_message('debug', "[ProcGen] Manual interrupt received. Finalizing level generation.")
                        architect_v3.manual_stop_generation = True
                except queue.Empty:
                    pass
        else:
            architect = MapArchitect(self.engine, game_map, getattr(self.engine, 'world_theme', 'fantasy'),
                                     scene_prompt)
            state = architect.generate_layout(callback)

        if state:
            state.exterior_tile_type = chosen_tile

        utils.log_message('debug', f"--- [SYSTEM] PEG process ({algorithm}) complete. ---")
        return state