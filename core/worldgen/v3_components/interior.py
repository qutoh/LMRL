# /core/worldgen/v3_components/interior.py

import random
from typing import Generator, Tuple

import numpy as np

from core.common import utils
from core.common.game_state import GenerationState

from .feature_node import FeatureNode
from .placement import Placement
from ...common.config_loader import config


class Interior:
    """
    Handles the placement of deferred interior features within parent containers,
    including logic for scaling and adaptation.
    """

    def __init__(self, placement: Placement, map_width: int, map_height: int):
        self.placement = placement
        self.map_width = map_width
        self.map_height = map_height

    def finalize_placements(self, all_branches: list[FeatureNode], gen_state: GenerationState) -> Generator[
        None, None, None]:
        """After jitter, place all deferred INTERIOR features, yielding after each placement."""
        utils.log_message('debug', "[PEGv3 Interior] Finalizing interior feature placement...")
        all_nodes = [node for branch in all_branches for node in branch.get_all_nodes_in_branch()]
        for parent_node in all_nodes:
            if not parent_node.interior_features_to_place:
                continue
            features_to_place_this_node = list(parent_node.interior_features_to_place)
            for feature_data in features_to_place_this_node:
                self._place_or_adapt_feature(feature_data, parent_node, all_branches)
                yield

    def _place_or_adapt_feature(self, feature_data: dict, parent: FeatureNode, all_branches: list[FeatureNode]):
        """Attempts to place an interior feature, scaling it down or scaling its parent up on failure."""
        size_map = {'large': (3, 3), 'medium': (2, 2), 'small': (1, 1)}
        size_tiers = ['large', 'medium', 'small']
        initial_tier = feature_data.get('size_tier', 'medium')
        start_index = size_tiers.index(initial_tier) if initial_tier in size_tiers else 1
        feature_def = config.features.get(feature_data.get('type'), {})

        for i in range(start_index, len(size_tiers)):
            tier = size_tiers[i]
            w, h = size_map[tier]
            if placement := self._find_placement((w, h), parent, feature_def):
                x, y = placement
                feature_data['bounding_box'] = (x, y, x + w, y + h)
                parent.placed_interior_features.append(feature_data)
                utils.log_message('debug',
                                  f"  Placed interior feature '{feature_data['name']}' ({w}x{h}) in '{parent.name}' at ({x},{y}).")
                return

        final_w, final_h = size_map['small']
        if self._attempt_scale_parent_for_child(parent, (final_w, final_h), all_branches):
            if placement := self._find_placement((final_w, final_h), parent, feature_def):
                x, y = placement
                feature_data['bounding_box'] = (x, y, x + final_w, y + final_h)
                parent.placed_interior_features.append(feature_data)
                utils.log_message('debug',
                                  f"  Placed interior feature '{feature_data['name']}' in scaled-up parent '{parent.name}'.")
                return

        utils.log_message('debug',
                          f"  [DISCARDED] Could not place interior feature '{feature_data['name']}' in '{parent.name}' after all attempts.")

    def _find_placement(self, size: Tuple[int, int], parent: FeatureNode, feature_def: dict) -> Tuple[int, int] | None:
        """Finds a valid spot for an interior feature within the parent's usable space."""
        w, h = size
        px, py, pw, ph = parent.get_rect()

        # Build a temporary grid representing just the parent's interior
        temp_interior_grid = np.full((ph, pw), config.tile_type_map.get(
            config.features.get(parent.feature_type, {}).get('tile_type', 'DEFAULT_FLOOR')), dtype=int)

        # Carve out already placed interior features
        for f in parent.placed_interior_features:
            if 'bounding_box' not in f: continue
            bb = f['bounding_box']
            x1, y1, x2, y2 = bb
            rel_x1, rel_y1 = x1 - px, y1 - py
            rel_x2, rel_y2 = x2 - px, y2 - py

            interior_feature_def = config.features.get(f.get('type'), {})
            tile_type_index = config.tile_type_map.get(interior_feature_def.get('tile_type', 'DEFAULT_WALL'))

            temp_interior_grid[rel_y1:rel_y2, rel_x1:rel_x2] = tile_type_index

        interior_x_offset, interior_y_offset = 1, 1
        interior_w, interior_h = pw - 2, ph - 2

        if w > interior_w or h > interior_h:
            return None

        possible_positions = []
        if interior_w >= w and interior_h >= h:
            for x in range(interior_x_offset, interior_x_offset + interior_w - w + 1):
                for y in range(interior_y_offset, interior_y_offset + interior_h - h + 1):
                    possible_positions.append((x, y))

        if not possible_positions:
            return None

        random.shuffle(possible_positions)

        for rel_x, rel_y in possible_positions:
            # We are checking a slice of the parent's *interior* grid.
            # We don't need a full map grid.
            placement_slice = temp_interior_grid[rel_y:rel_y + h, rel_x:rel_x + w]

            allowed_indices = {config.tile_type_map.get(
                config.features.get(parent.feature_type, {}).get('tile_type', 'DEFAULT_FLOOR'))}
            my_tile_type_index = config.tile_type_map.get(feature_def.get('tile_type'))
            if my_tile_type_index is not None:
                allowed_indices.add(my_tile_type_index)

            invalid_collisions = np.isin(placement_slice, list(allowed_indices), invert=True)
            if not np.any(invalid_collisions):
                return rel_x + px, rel_y + py  # Return absolute map coordinates

        return None

    def _attempt_scale_parent_for_child(self, parent: FeatureNode, child_size: Tuple[int, int],
                                        all_branches: list[FeatureNode]) -> bool:
        """Tries to expand a parent feature to make space for a child."""
        child_w, child_h = child_size
        px, py, pw, ph = parent.get_rect()
        required_w, required_h = child_w + 2, child_h + 2

        if pw >= required_w and ph >= required_h:
            return False

        new_w, new_h = max(pw, required_w), max(ph, required_h)
        new_x, new_y = px - (new_w - pw) // 2, py - (new_h - ph) // 2
        proposal_rect = (new_x, new_y, new_w, new_h)

        parent_root = parent
        while parent_root.parent:
            parent_root = parent_root.parent

        other_branches = [b for b in all_branches if b is not parent_root]
        temp_grid_others = self.placement._get_temp_grid(other_branches)
        parent_def = config.features.get(parent.feature_type, {})

        if self.placement._is_placement_valid(proposal_rect, temp_grid_others, parent_def):
            parent.current_x, parent.current_y = new_x, new_y
            parent.current_abs_width, parent.current_abs_height = new_w, new_h
            utils.log_message('debug',
                              f"  Parent '{parent.name}' scaled up from ({pw}x{ph}) to ({new_w}x{new_h}) to fit child.")
            return True
        return False