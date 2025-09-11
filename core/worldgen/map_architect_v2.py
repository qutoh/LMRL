# /core/worldgen/map_architect_v2.py

import random
from typing import List, Dict, Tuple, Callable

import numpy as np

from .v3_components.converter import Converter
from .v3_components.feature_node import FeatureNode
from .v3_components.map_ops import MapOps
from .v3_components.pathing import Pathing
from .v3_components.placement import Placement as PlacementV3  # For shrink helper
from .v3_components.v3_llm import V3_LLM
from ..common import utils
from ..common.config_loader import config
from ..common.game_state import GenerationState, MapArtist

# --- Algorithm Constants ---
MIN_FEATURE_SIZE = 3


class MapArchitectV2:
    def __init__(self, engine, game_map, world_theme, scene_prompt):
        self.engine = engine
        self.game_map = game_map
        self.map_width = game_map.width
        self.map_height = game_map.height

        # --- Shared V3 Components ---
        self.llm = V3_LLM(engine)
        self.placement_utils = PlacementV3(self.map_width, self.map_height)
        self.pathing = Pathing(self.game_map)
        self.map_ops = MapOps(self.map_width, self.map_height, self.pathing) # THIS IS THE FIX
        self.converter = Converter()

        self.initial_feature_branches: List[FeatureNode] = []

    def _get_temp_grid(self) -> np.ndarray:
        """Renders all currently placed features to a simple grid for collision detection."""
        grid = np.zeros((self.map_height, self.map_width), dtype=int)
        for branch in self.initial_feature_branches:
            x, y, w, h = branch.get_rect()
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(self.map_width, x + w), min(self.map_height, y + h)
            if x2 > x1 and y2 > y1:
                grid[y1:y2, x1:x2] = 1
        return grid

    def _find_placement_in_slice(self, slice_rect: Tuple[int, int, int, int], new_w: int, new_h: int) -> Tuple[
                                                                                                               int, int] | None:
        """Finds a random valid placement within a rectangular slice of empty space."""
        sx, sy, sw, sh = slice_rect
        if new_w > sw or new_h > sh:
            return None
        x = sx + random.randint(0, sw - new_w)
        y = sy + random.randint(0, sh - new_h)
        return x, y

    def _place_initial_features(self, initial_specs: List[Dict], shrink_factor: float) -> bool:
        """
        The core partitioning algorithm. Places a set of features by iteratively
        shrinking the existing layout to make space for new ones.
        """
        self.initial_feature_branches.clear()
        if not initial_specs: return True

        w, h = self.map_width - 2, self.map_height - 2
        x, y = 1, 1
        spec = initial_specs[0]
        root_node = FeatureNode(spec['name'], spec['type'], w, h, x, y)
        self.initial_feature_branches.append(root_node)

        scaling_cycle = ['NORTH', 'NORTHEAST', 'EAST', 'SOUTHEAST', 'SOUTH', 'SOUTHWEST', 'WEST', 'NORTHWEST']
        size_tier_map = {'large': 0.8, 'medium': 0.5, 'small': 0.25}

        for i, spec in enumerate(initial_specs[1:]):
            scale_direction = scaling_cycle[i % len(scaling_cycle)]

            for branch in self.initial_feature_branches:
                self.placement_utils._apply_shrink_transform_to_branch(branch, scale_direction, shrink_factor)

            shrunk_x1 = min(b.current_x for b in self.initial_feature_branches)
            shrunk_y1 = min(b.current_y for b in self.initial_feature_branches)
            shrunk_x2 = max(b.current_x + b.current_abs_width for b in self.initial_feature_branches)
            shrunk_y2 = max(b.current_y + b.current_abs_height for b in self.initial_feature_branches)

            empty_slices = []
            if shrunk_x1 > 1: empty_slices.append((1, 1, shrunk_x1 - 1, self.map_height - 2))  # Left slice
            if shrunk_x2 < self.map_width - 1: empty_slices.append(
                (shrunk_x2, 1, self.map_width - 1 - shrunk_x2, self.map_height - 2))  # Right
            if shrunk_y1 > 1: empty_slices.append((1, 1, self.map_width - 2, shrunk_y1 - 1))  # Top
            if shrunk_y2 < self.map_height - 1: empty_slices.append(
                (1, shrunk_y2, self.map_width - 2, self.map_height - 1 - shrunk_y2))  # Bottom

            if not empty_slices: return False
            biggest_slice = max(empty_slices, key=lambda r: r[2] * r[3])
            rel_dim = size_tier_map.get(spec['size_tier'], 0.5)
            new_w = max(MIN_FEATURE_SIZE, int(biggest_slice[2] * rel_dim))
            new_h = max(MIN_FEATURE_SIZE, int(biggest_slice[3] * rel_dim))

            placement = self._find_placement_in_slice(biggest_slice, new_w, new_h)
            if not placement:
                placement_found = False
                for s in sorted(empty_slices, key=lambda r: r[2] * r[3], reverse=True):
                    placement = self._find_placement_in_slice(s, new_w, new_h)
                    if placement:
                        placement_found = True
                        break
                if not placement_found: return False

            px, py = placement
            new_node = FeatureNode(spec['name'], spec['type'], new_w, new_h, px, py)
            self.initial_feature_branches.append(new_node)
        return True

    def generate_layout(self, ui_callback: Callable) -> GenerationState:
        """The main orchestrator for the V2 algorithm."""
        gen_state = GenerationState(self.game_map)
        artist = MapArtist()

        def update_and_draw():
            self.converter.populate_generation_state(gen_state, self.initial_feature_branches)
            artist.draw_map(self.game_map, gen_state, config.features)
            ui_callback(gen_state)
            import time
            time.sleep(0.05)

        initial_specs = self.llm.get_initial_features()
        if not initial_specs: return gen_state

        shrink_factor, success = 0.5, False
        for _ in range(5):  # Max 5 retries
            if self._place_initial_features(initial_specs, shrink_factor):
                success = True
                break
            shrink_factor *= 0.8
        if not success:
            utils.log_message('debug', "[PEG V2 FATAL] Could not place initial features.")
            return gen_state
        update_and_draw()

        self.map_ops.apply_jitter(self.initial_feature_branches, on_iteration_end=update_and_draw)
        update_and_draw()

        all_connections, all_door_placements, all_hallways = self.pathing.create_all_connections(self.initial_feature_branches)
        gen_state.door_locations = all_door_placements
        gen_state.blended_hallways = all_hallways

        gen_state.layout_graph = self.converter.serialize_feature_tree_to_graph(self.initial_feature_branches)
        self.converter.populate_generation_state(gen_state, self.initial_feature_branches)
        gen_state.physics_layout = self.converter.convert_to_vertex_representation(gen_state.layout_graph,
                                                                                    all_connections)
        artist.draw_map(self.game_map, gen_state, config.features)
        ui_callback(gen_state)
        gen_state.narrative_log = "The area was generated via the Partitioning algorithm."
        return gen_state