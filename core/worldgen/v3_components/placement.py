# /core/worldgen/v3_components/placement.py

import numpy as np
import random
from typing import List, Tuple, Dict, Optional

from .feature_node import FeatureNode
from core.common import utils
from ...common.config_loader import config

# --- Algorithm Constants ---
MIN_FEATURE_SIZE = 3

# A map to easily find the opposite direction for scaling.
OPPOSITE_DIRECTION_MAP = {
    'NORTH': 'SOUTH', 'SOUTH': 'NORTH', 'EAST': 'WEST', 'WEST': 'EAST',
    'NORTHEAST': 'SOUTHWEST', 'SOUTHWEST': 'NORTHEAST',
    'NORTHWEST': 'SOUTHEAST', 'SOUTHEAST': 'NORTHWEST'
}


class Placement:
    """
    Handles the geometric placement, collision detection, and shrinking algorithms
    for the V3 map architect.
    """

    def __init__(self, map_width: int, map_height: int):
        self.map_width = map_width
        self.map_height = map_height

    def _get_temp_grid(self, all_branches: List[FeatureNode]) -> np.ndarray:
        """Renders all currently placed features to a simple grid for collision detection."""
        grid = np.zeros((self.map_height, self.map_width), dtype=int)
        all_nodes = [node for branch in all_branches for node in branch.get_all_nodes_in_branch()]
        for node in all_nodes:
            x, y, w, h = node.get_rect()
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(self.map_width, x + w), min(self.map_height, y + h)
            if x2 > x1 and y2 > y1:
                grid[y1:y2, x1:x2] = 1
        return grid

    def _is_placement_valid(self, rect_to_check: tuple, temp_grid: np.ndarray) -> bool:
        """Checks if a rectangle is within bounds and doesn't overlap existing features on a grid."""
        x, y, w, h = rect_to_check
        if not (x >= 1 and y >= 1 and (x + w) <= (self.map_width - 1) and (y + h) <= (self.map_height - 1)):
            return False
        if np.any(temp_grid[y:y + h, x:x + w] == 1):
            return False
        return True

    def _find_valid_placements(self, w: int, h: int, temp_grid: np.ndarray, all_branches: List[FeatureNode]) -> list:
        """Finds all valid (non-overlapping) positions adjacent to existing features."""
        placements = []
        all_nodes = [node for branch in all_branches for node in branch.get_all_nodes_in_branch()]
        for parent in all_nodes:
            px, py, pw, ph = parent.get_rect()
            positions_to_check = {
                'N': (px + (pw - w) // 2, py - h),
                'S': (px + (pw - w) // 2, py + ph),
                'W': (px - w, py + (ph - h) // 2),
                'E': (px + pw, py + (ph - h) // 2),
            }
            for face, (x, y) in positions_to_check.items():
                if self._is_placement_valid((x, y, w, h), temp_grid):
                    placements.append((x, y, parent, face))
        return placements

    def place_initial_features(self, feature_specs: List[Dict]) -> List[FeatureNode]:
        """Iteratively places and shrinks initial features, returning a list of root FeatureNode branches."""
        utils.log_message('debug', "[PEGv3 Placement] Placing initial features...")
        initial_feature_branches = []
        scaling_cycle = ['NORTH', 'NORTHEAST', 'EAST', 'SOUTHEAST', 'SOUTH', 'SOUTHWEST', 'WEST', 'NORTHWEST']
        size_tier_map = {'large': 0.8, 'medium': 0.5, 'small': 0.25}

        spec = feature_specs[0]
        rel_dim = size_tier_map.get(spec['size_tier'], 0.5)
        w = int((self.map_width - 2) * rel_dim)
        h = int((self.map_height - 2) * rel_dim)
        x = (self.map_width - w) // 2
        y = (self.map_height - h) // 2
        root_node = FeatureNode(spec['name'], spec['type'], rel_dim, rel_dim, w, h, x, y)
        root_node.narrative_log = spec.get('description_sentence', '')
        initial_feature_branches.append(root_node)
        utils.log_message('debug', f"  Placed root feature '{spec['name']}' at ({x},{y}) size ({w},{h}).")

        for i, spec in enumerate(feature_specs[1:]):
            rel_dim = size_tier_map.get(spec['size_tier'], 0.5)
            w = int((self.map_width - 2) * rel_dim)
            h = int((self.map_height - 2) * rel_dim)
            placement_found = False
            current_shrink_factor = 0.25
            max_retries = 5

            for retry_count in range(max_retries):
                temp_grid = self._get_temp_grid(initial_feature_branches)
                valid_placements = self._find_valid_placements(w, h, temp_grid, initial_feature_branches)
                if valid_placements:
                    x, y, adjacent_node, face = random.choice(valid_placements)
                    new_node = FeatureNode(spec['name'], spec['type'], rel_dim, rel_dim, w, h, x, y)
                    new_node.narrative_log = spec.get('description_sentence', '')
                    initial_feature_branches.append(new_node)
                    placement_found = True
                    break
                else:
                    scale_dir = scaling_cycle[(i + retry_count) % len(scaling_cycle)]
                    utils.log_message('debug',
                                      f"  [RETRY {retry_count + 1}] No space for '{spec['name']}'. Shrinking towards {scale_dir} (factor: {current_shrink_factor:.2f}).")
                    for branch in initial_feature_branches:
                        self._apply_shrink_transform_to_branch(branch, scale_dir, current_shrink_factor)
                    current_shrink_factor *= 0.8
            if not placement_found:
                utils.log_message('debug',
                                  f"[PEGv3 FATAL] Could not place '{spec['name']}' after {max_retries} retries. Skipping.")
        return initial_feature_branches

    def find_and_place_subfeature(self, feature_data: dict, parent_branch: FeatureNode, all_branches: List[FeatureNode],
                                  chosen_parent_name: str, shrink_factor: float) -> Optional[FeatureNode]:
        """
        Finds a valid spot for a new subfeature, creates its node, attaches it,
        and shrinks the parent branch.
        """
        temp_grid = self._get_temp_grid(all_branches)
        possible_placements = []
        size_tier_map = {'large': 0.75, 'medium': 0.5, 'small': 0.25}
        size_ratio = size_tier_map.get(feature_data.get('size_tier', 'medium'), 0.5)

        for node_in_branch in parent_branch.get_all_nodes_in_branch():
            px, py, pw, ph = node_in_branch.get_rect()
            sub_w = max(MIN_FEATURE_SIZE, int(pw * size_ratio))
            sub_h = sub_w
            if sub_w < pw:
                for x_offset in range(pw - sub_w + 1):
                    rect_n = (px + x_offset, py - sub_h - 1, sub_w, sub_h)
                    if self._is_placement_valid(rect_n, temp_grid): possible_placements.append(
                        {'parent': node_in_branch, 'face': 'N', 'rect': rect_n})
                    rect_s = (px + x_offset, py + ph + 1, sub_w, sub_h)
                    if self._is_placement_valid(rect_s, temp_grid): possible_placements.append(
                        {'parent': node_in_branch, 'face': 'S', 'rect': rect_s})
            sub_h = max(MIN_FEATURE_SIZE, int(ph * size_ratio))
            sub_w = sub_h
            if sub_h < ph:
                for y_offset in range(ph - sub_h + 1):
                    rect_w = (px - sub_w - 1, py + y_offset, sub_w, sub_h)
                    if self._is_placement_valid(rect_w, temp_grid): possible_placements.append(
                        {'parent': node_in_branch, 'face': 'W', 'rect': rect_w})
                    rect_e = (px + pw + 1, py + y_offset, sub_w, sub_h)
                    if self._is_placement_valid(rect_e, temp_grid): possible_placements.append(
                        {'parent': node_in_branch, 'face': 'E', 'rect': rect_e})

        if not possible_placements: return None

        valid_placements_for_chosen_parent = [p for p in possible_placements if
                                              p['parent'].name.lower() in chosen_parent_name.lower()]
        if not valid_placements_for_chosen_parent:
            chosen_parent_node = next((p['parent'] for p in possible_placements), None)
            valid_placements_for_chosen_parent = [p for p in possible_placements if p['parent'] == chosen_parent_node]

        if not valid_placements_for_chosen_parent: return None

        placement = random.choice(valid_placements_for_chosen_parent)
        parent_node = placement['parent']
        x, y, w, h = placement['rect']
        new_subfeature = FeatureNode(feature_data['name'], feature_data['type'], 0.0, 0.0, w, h, x, y,
                                     parent=parent_node, anchor_face=placement['face'])
        parent_node.subfeatures.append(new_subfeature)

        parent_def = config.features.get(parent_node.feature_type, {})
        if parent_def.get('is_shrinkable', True):
            self._apply_shrink_transform_to_branch(parent_node,
                                                   OPPOSITE_DIRECTION_MAP.get(placement['face'], placement['face']),
                                                   shrink_factor)

        return new_subfeature

    def _apply_shrink_transform_to_branch(self, feature: FeatureNode, direction: str, shrink_factor: float,
                                          parent_new_rect: Optional[Tuple[int, int, int, int]] = None,
                                          parent_old_rect: Optional[Tuple[int, int, int, int]] = None):
        """Recursively shrinks a feature and its sub-branch, maintaining anchors."""
        old_x, old_y, old_w, old_h = feature.get_rect()
        new_w, new_h = old_w, old_h
        if 'E' in direction or 'W' in direction: new_w = max(MIN_FEATURE_SIZE, int(old_w * (1.0 - shrink_factor)))
        if 'N' in direction or 'S' in direction: new_h = max(MIN_FEATURE_SIZE, int(old_h * (1.0 - shrink_factor)))

        if parent_new_rect is None:
            new_x, new_y = old_x, old_y
            if 'E' in direction: new_x = (old_x + old_w) - new_w
            if 'S' in direction: new_y = (old_y + old_h) - new_h
        else:
            ppx_new, ppy_new, ppw_new, pph_new = parent_new_rect
            ppx_old, ppy_old, ppw_old, pph_old = parent_old_rect
            face = feature.anchor_to_parent_face
            if face in ('N', 'S'):
                ratio = (old_x - ppx_old) / ppw_old if ppw_old > 0 else 0.5
                new_x = ppx_new + int(ppw_new * ratio)
                new_y = ppy_new - new_h if face == 'N' else ppy_new + pph_new
            elif face in ('E', 'W'):
                ratio = (old_y - ppy_old) / pph_old if pph_old > 0 else 0.5
                new_y = ppy_new + int(pph_new * ratio)
                new_x = ppx_new - new_w if face == 'W' else ppx_new + ppw_new
            else:
                ratio_x = (old_x - ppx_old + old_w / 2) / ppw_old if ppw_old > 0 else 0.5
                ratio_y = (old_y - ppy_old + old_h / 2) / pph_old if pph_old > 0 else 0.5
                new_center_x = ppx_new + int(ppw_new * ratio_x)
                new_center_y = ppy_new + int(pph_new * ratio_y)
                new_x = new_center_x - new_w // 2
                new_y = new_center_y - new_h // 2

        feature.current_x, feature.current_y = new_x, new_y
        feature.current_abs_width, feature.current_abs_height = new_w, new_h
        my_new_rect = (new_x, new_y, new_w, new_h)
        my_old_rect = (old_x, old_y, old_w, old_h)
        for sub in feature.subfeatures:
            self._apply_shrink_transform_to_branch(sub, direction, shrink_factor, my_new_rect, my_old_rect)