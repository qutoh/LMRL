# /core/worldgen/v3_components/placement.py

import math
import random
from typing import List, Tuple, Dict, Optional, Generator

import numpy as np

from core.common import utils
from .feature_node import FeatureNode
from ...common.config_loader import config

# --- Algorithm Constants ---
MIN_FEATURE_SIZE = 3

# --- Growth Algorithm Constants ---
GROWTH_COVERAGE_THRESHOLD = 0.40
MAX_GROWTH_ITERATIONS = 200
SIZE_TIER_TARGET_AREA_FACTOR = {'large': 0.15, 'medium': 0.08, 'small': 0.04}
RECTANGLE_ASPECT_RATIOS = [(4, 3), (3, 2), (5, 3), (16, 9), (2, 1), (3, 4), (2, 3), (3, 5), (9, 16), (1, 2)]

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
        self.void_space_index = config.tile_type_map.get("VOID_SPACE", -1)

    def _get_temp_grid(self, all_branches: List[FeatureNode], exclude_node: Optional[FeatureNode] = None) -> np.ndarray:
        """
        Renders all currently placed features to a grid containing their terrain_type index.
        """
        grid = np.full((self.map_height, self.map_width), self.void_space_index, dtype=np.int8)
        all_nodes = [node for branch in all_branches for node in branch.get_all_nodes_in_branch()]

        for node in all_nodes:
            if node is exclude_node:
                continue

            feature_def = config.features.get(node.feature_type, {})

            if node.path_coords:
                path_set = set(map(tuple, node.path_coords))
                tile_type_name = feature_def.get('tile_type', 'DEFAULT_FLOOR')
                tile_type_index = config.tile_type_map.get(tile_type_name, self.void_space_index)
                border_type_name = feature_def.get('border_tile_type')
                border_type_index = config.tile_type_map.get(border_type_name) if border_type_name else -1

                for x, y in path_set:
                    if 0 <= y < self.map_height and 0 <= x < self.map_width:
                        grid[y, x] = tile_type_index
                    if border_type_index != -1:
                        for nx in range(x - 1, x + 2):
                            for ny in range(y - 1, y + 2):
                                if (nx, ny) == (x, y): continue
                                if 0 <= ny < self.map_height and 0 <= nx < self.map_width:
                                    if (nx, ny) not in path_set:
                                        grid[ny, nx] = border_type_index

            else:
                tile_type_name = feature_def.get('tile_type', 'DEFAULT_FLOOR')
                tile_type_index = config.tile_type_map.get(tile_type_name, self.void_space_index)
                x, y, w, h = node.get_rect()
                x1, y1 = max(0, x), max(0, y)
                x2, y2 = min(self.map_width, x + w), min(self.map_height, y + h)
                if x2 > x1 and y2 > y1:
                    grid[y1:y2, x1:x2] = tile_type_index

                border_thickness = feature_def.get('border_thickness', 0)
                if border_thickness > 0:
                    border_tile_type = feature_def.get('border_tile_type')
                    if border_tile_type and border_tile_type in config.tile_type_map:
                        border_tile_index = config.tile_type_map[border_tile_type]
                        effective_border = min(border_thickness, (x2 - x1) // 2, (y2 - y1) // 2)
                        if effective_border > 0:
                            grid[y1:y1 + effective_border, x1:x2] = border_tile_index
                            grid[y2 - effective_border:y2, x1:x2] = border_tile_index
                            grid[y1:y2, x1:x1 + effective_border] = border_tile_index
                            grid[y1:y2, x2 - effective_border:x2] = border_tile_index

        return grid

    def _is_placement_valid(self, rect_to_check: tuple, terrain_grid: np.ndarray, feature_def: dict) -> bool:
        """
        Checks if a rectangle's placement is valid based on terrain intersection rules,
        using fast, vectorized NumPy operations. Overlaps are only allowed if explicitly
        defined in the feature's 'intersects_ok' list.
        """
        x, y, w, h = rect_to_check
        if not (x >= 1 and y >= 1 and (x + w) <= (self.map_width - 1) and (y + h) <= (self.map_height - 1)):
            return False

        allowed_indices = {self.void_space_index}
        for rule in feature_def.get('intersects_ok', []):
            if intersect_type := rule.get('type'):
                if intersect_type in config.tile_type_map:
                    allowed_indices.add(config.tile_type_map[intersect_type])
        placement_slice = terrain_grid[y:y + h, x:x + w]
        invalid_collisions_mask = np.isin(placement_slice, list(allowed_indices), invert=True)

        if np.any(invalid_collisions_mask):
            return False

        return True

    def _attempt_aspect_aware_growth(self, node: FeatureNode, all_branches: List[FeatureNode]) -> bool:
        """
        Calculates the node's next size based on its aspect ratio and attempts to find a valid
        position for the new, larger rectangle that contains the original rectangle.
        """
        ratio_w, ratio_h = node.target_aspect_ratio
        next_k = node.growth_multiplier + 1
        prop_w, prop_h = int(ratio_w * next_k), int(ratio_h * next_k)

        if prop_w == node.current_abs_width and prop_h == node.current_abs_height:
            next_k += 1
            prop_w, prop_h = int(ratio_w * next_k), int(ratio_h * next_k)
            if prop_w == node.current_abs_width and prop_h == node.current_abs_height:
                return False

        old_x, old_y, old_w, old_h = node.get_rect()
        dw, dh = prop_w - old_w, prop_h - old_h

        possible_x_starts = list(range(old_x - dw, old_x + 1))
        possible_y_starts = list(range(old_y - dh, old_y + 1))
        random.shuffle(possible_x_starts)
        random.shuffle(possible_y_starts)

        temp_grid = self._get_temp_grid(all_branches, exclude_node=node)
        feature_def = config.features.get(node.feature_type, {})

        for x in possible_x_starts:
            for y in possible_y_starts:
                if self._is_placement_valid((x, y, prop_w, prop_h), temp_grid, feature_def):
                    node.current_x, node.current_y = x, y
                    node.current_abs_width, node.current_abs_height = prop_w, prop_h
                    node.growth_multiplier = next_k
                    return True

        return False

    def place_and_grow_initial_features(self, feature_specs: List[Dict]) -> Generator[
        List[FeatureNode], None, List[FeatureNode]]:
        """
        Robustly places initial features using a growth algorithm. Features start as 1x1 seeds
        and iteratively expand while maintaining a target aspect ratio, relocating if they get stuck.
        Yields the state after each "tick" where all active features have attempted to grow.
        """
        if not feature_specs:
            yield []
            return

        initial_feature_branches: List[FeatureNode] = []
        total_map_area = self.map_width * self.map_height

        start_x = self.map_width // 2 - len(feature_specs) // 2
        start_y = self.map_height // 2

        for i, spec in enumerate(feature_specs):
            node = FeatureNode(spec['name'], spec['type'], 0.0, 0.0, 1, 1, start_x + i, start_y)
            node.target_area = total_map_area * SIZE_TIER_TARGET_AREA_FACTOR.get(spec.get('size_tier'), 0.08)
            node.size_tier = spec.get('size_tier', 'medium')
            node.is_stuck = False
            node.growth_multiplier = 1
            node.relocation_history.add((start_x + i, start_y))
            feature_def = config.features.get(spec['type'], {})
            shape = feature_def.get('default_shape', 'rectangle')
            if shape == 'rectangle':
                node.target_aspect_ratio = random.choice(RECTANGLE_ASPECT_RATIOS)
            else:
                node.target_aspect_ratio = (1, 1)
            initial_feature_branches.append(node)

        yield initial_feature_branches

        for i in range(MAX_GROWTH_ITERATIONS):
            current_area = sum(n.current_abs_width * n.current_abs_height for n in initial_feature_branches)
            if (current_area / total_map_area) >= GROWTH_COVERAGE_THRESHOLD:
                utils.log_message('debug',
                                  f"[PEGv3 Growth] Reached coverage threshold ({GROWTH_COVERAGE_THRESHOLD:.0%}).")
                break

            growable_features = [n for n in initial_feature_branches if
                                 not n.is_stuck and (n.current_abs_width * n.current_abs_height) < n.target_area]
            if not growable_features:
                utils.log_message('debug', "[PEGv3 Growth] All features are grown or stuck.")
                break

            random.shuffle(growable_features)
            tick_changed = False

            for feature in growable_features:
                if self._attempt_aspect_aware_growth(feature, initial_feature_branches):
                    tick_changed = True
                    feature.relocation_history.clear()
                    feature.relocation_history.add((feature.current_x, feature.current_y))
                    continue

                temp_grid_others = self._get_temp_grid(initial_feature_branches, exclude_node=feature)
                other_branches = [b for b in initial_feature_branches if b is not feature]
                w, h = feature.current_abs_width, feature.current_abs_height
                feature_def = config.features.get(feature.feature_type, {})

                all_relocation_spots = self._find_valid_placements(w, h, temp_grid_others, other_branches, feature_def)

                unvisited_spots = [
                    spot for spot in all_relocation_spots
                    if (spot[0], spot[1]) not in feature.relocation_history
                ]

                if unvisited_spots:
                    new_x, new_y, _, _ = random.choice(unvisited_spots)
                    feature.current_x, feature.current_y = new_x, new_y
                    feature.relocation_history.add((new_x, new_y))
                    utils.log_message('full', f"  Relocated stuck feature '{feature.name}' to {new_x},{new_y}.")
                    tick_changed = True
                else:
                    feature.is_stuck = True
                    utils.log_message('debug',
                                      f"  Feature '{feature.name}' is permanently stuck after trying all relocation options.")

            if tick_changed:
                yield initial_feature_branches
            else:
                utils.log_message('debug', "[PEGv3 Growth] No change in a full tick. Finalizing layout.")
                break
        else:
            utils.log_message('debug', f"[PEGv3 Growth] Reached max iterations ({MAX_GROWTH_ITERATIONS}).")

        return initial_feature_branches

    def _find_valid_placements(self, w: int, h: int, temp_grid: np.ndarray, all_branches: List[FeatureNode],
                               feature_def_to_place: dict) -> list:
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
                if self._is_placement_valid((x, y, w, h), temp_grid, feature_def_to_place):
                    placements.append((x, y, parent, face))
        return placements

    def find_and_place_subfeature(self, feature_data: dict, parent_branch: FeatureNode, all_branches: List[FeatureNode],
                                  chosen_parent_name: str, shrink_factor: float) -> Optional[FeatureNode]:
        """
        Finds a valid spot for a new subfeature, creates its node as a 1x1 'seed' on the anchor face,
        attaches it, and shrinks the parent branch. The final bounding box is stored on the
        node for the architect to animate its growth.
        """
        temp_grid = self._get_temp_grid(all_branches)
        possible_placements = []
        size_tier_map = {'large': 0.75, 'medium': 0.5, 'small': 0.25}
        size_ratio = size_tier_map.get(feature_data.get('size_tier', 'medium'), 0.5)

        feature_def = config.features.get(feature_data['type'], {})
        shape = feature_def.get('default_shape', 'rectangle')
        if shape == 'rectangle':
            aspect_ratio = random.choice(RECTANGLE_ASPECT_RATIOS)
        else:
            aspect_ratio = (1, 1)
        rw, rh = aspect_ratio

        for node_in_branch in parent_branch.get_all_nodes_in_branch():
            px, py, pw, ph = node_in_branch.get_rect()

            parent_area = pw * ph
            target_sub_area = parent_area * size_ratio

            sub_w = max(MIN_FEATURE_SIZE, round(math.sqrt(target_sub_area * rw / rh)))
            sub_h = max(MIN_FEATURE_SIZE, round(math.sqrt(target_sub_area * rh / rw)))

            if sub_w < pw:
                for x_offset in range(pw - sub_w + 1):
                    rect_n = (px + x_offset, py - sub_h, sub_w, sub_h)
                    if self._is_placement_valid(rect_n, temp_grid, feature_def): possible_placements.append(
                        {'parent': node_in_branch, 'face': 'N', 'rect': rect_n})
                    rect_s = (px + x_offset, py + ph, sub_w, sub_h)
                    if self._is_placement_valid(rect_s, temp_grid, feature_def): possible_placements.append(
                        {'parent': node_in_branch, 'face': 'S', 'rect': rect_s})
            if sub_h < ph:
                for y_offset in range(ph - sub_h + 1):
                    rect_w = (px - sub_w, py + y_offset, sub_w, sub_h)
                    if self._is_placement_valid(rect_w, temp_grid, feature_def): possible_placements.append(
                        {'parent': node_in_branch, 'face': 'W', 'rect': rect_w})
                    rect_e = (px + pw, py + y_offset, sub_w, sub_h)
                    if self._is_placement_valid(rect_e, temp_grid, feature_def): possible_placements.append(
                        {'parent': node_in_branch, 'face': 'E', 'rect': rect_e})

        if not possible_placements: return None

        valid_placements_for_chosen_parent = [p for p in possible_placements if
                                              p['parent'].name.lower() in chosen_parent_name.lower()]
        if not valid_placements_for_chosen_parent:
            chosen_parent_node = next((p['parent'] for p in possible_placements), None)
            if not chosen_parent_node: return None
            valid_placements_for_chosen_parent = [p for p in possible_placements if p['parent'] == chosen_parent_node]

        if not valid_placements_for_chosen_parent: return None

        placement = random.choice(valid_placements_for_chosen_parent)
        parent_node = placement['parent']
        anchor_face = placement['face']
        final_x, final_y, final_w, final_h = placement['rect']

        if anchor_face == 'N':
            seed_x, seed_y = final_x + final_w // 2, final_y + final_h - 1
        elif anchor_face == 'S':
            seed_x, seed_y = final_x + final_w // 2, final_y
        elif anchor_face == 'W':
            seed_x, seed_y = final_x + final_w - 1, final_y + final_h // 2
        else:
            seed_x, seed_y = final_x, final_y + final_h // 2

        new_subfeature = FeatureNode(feature_data['name'], feature_data['type'], 0.0, 0.0, 1, 1, seed_x, seed_y,
                                     parent=parent_node, anchor_face=anchor_face)

        new_subfeature.target_growth_rect = (final_x, final_y, final_w, final_h)

        parent_node.subfeatures.append(new_subfeature)

        parent_def = config.features.get(parent_node.feature_type, {})
        if parent_def.get('is_shrinkable', True):
            self._apply_shrink_transform_to_branch(parent_node,
                                                   OPPOSITE_DIRECTION_MAP.get(anchor_face, anchor_face),
                                                   shrink_factor)

        return new_subfeature

    def handle_connector_placement(self, feature_data: dict, parent_branch: FeatureNode,
                                   all_branches: List[FeatureNode], llm, pathing, semantic_search, grow_coroutine,
                                   draw_callback) -> Optional[List[FeatureNode]]:
        """Handles the 'bridging' and 'seeding' logic for a CONNECTOR feature."""
        feature_def = config.features.get(feature_data['type'], {})
        min_len_ft, max_len_ft = feature_def.get('min_dimensions_ft', [5, 5])[1], \
        feature_def.get('max_dimensions_ft', [10, 80])[1]
        min_len_tiles, max_len_tiles = int(min_len_ft / 5), int(max_len_ft / 5)

        # --- Bridging Method ---
        valid_targets = []
        all_other_nodes = [node for branch in all_branches if branch is not parent_branch for node in
                           branch.get_all_nodes_in_branch()]

        for target_node in all_other_nodes:
            px, py, _, _ = parent_branch.get_rect()
            tx, ty, _, _ = target_node.get_rect()
            dist = math.hypot(tx - px, ty - py)
            if dist > max_len_tiles * 1.5: continue

            parent_points = pathing.get_valid_connection_points(parent_branch, 1, all_branches)
            target_points = pathing.get_valid_connection_points(target_node, 1, all_branches)
            if not parent_points or not target_points: continue

            start_pos, end_pos = pathing.find_first_valid_connection(parent_points, target_points, 1, all_branches)
            if start_pos and end_pos:
                path = pathing.find_path_with_clearance(start_pos, end_pos, 1, all_branches, feature_def)
                if path and min_len_tiles <= len(path) <= max_len_tiles:
                    valid_targets.append({'node': target_node, 'path': path})

        target_options = sorted(list(set(t['node'].name for t in valid_targets)))
        choice = llm.decide_connector_strategy(feature_data['name'], feature_data['description'], target_options)

        if "CREATE_NEW" not in choice:
            chosen_target_node = next((t['node'] for t in valid_targets if t['node'].name == choice), None)
            if chosen_target_node:
                chosen_path = next(t['path'] for t in valid_targets if t['node'] == chosen_target_node)
                new_node = FeatureNode(name=feature_data['name'], feature_type=feature_data['type'], rel_w=0, rel_h=0,
                                       abs_w=0, abs_h=0, x=0, y=0, parent=parent_branch)
                new_node.path_coords = chosen_path
                new_node.is_blended_portal = True
                new_node.blend_source_a = parent_branch.feature_type
                new_node.blend_source_b = chosen_target_node.feature_type
                parent_branch.subfeatures.append(new_node)
                return [new_node]

        # --- Seeding Method ---
        placement_result = self._find_placement_for_seed_connector(feature_data, feature_data.get('size_tier', 'small'),
                                                                   parent_branch, all_branches)
        if not placement_result:
            return None

        placement, child_growth_rect = placement_result
        parent_node, face, conn_rect = placement['parent'], placement['face'], placement['rect']
        cx, cy, cw, ch = conn_rect

        connector_node = FeatureNode(feature_data['name'], feature_data['type'], 0, 0, cw, ch, cx, cy,
                                     parent=parent_node, anchor_face=face)
        connector_node.narrative_log = feature_data.get('description_sentence', feature_data.get('description', ''))

        child_feature_data = llm.create_connector_child(parent_node, connector_node)
        if not child_feature_data:
            return None

        child_node = FeatureNode(child_feature_data['name'], child_feature_data['type'], 0, 0, 1, 1,
                                 child_growth_rect[0], child_growth_rect[1], parent=connector_node,
                                 anchor_face=OPPOSITE_DIRECTION_MAP.get(face))
        child_node.target_growth_rect = child_growth_rect
        child_node.narrative_log = child_feature_data.get('description_sentence', '')

        parent_node.subfeatures.append(connector_node)
        connector_node.subfeatures.append(child_node)

        growth_anim = grow_coroutine(child_node, draw_callback)
        for _ in growth_anim:
            pass

        return [connector_node, child_node]

    def _find_placement_for_seed_connector(self, connector_data: dict, child_size_tier: str, parent_node: FeatureNode,
                                           all_branches: List[FeatureNode]) -> Optional[
        Tuple[dict, Tuple[int, int, int, int]]]:
        """Finds a valid placement for a connector and its guaranteed child space."""
        temp_grid = self._get_temp_grid(all_branches)
        connector_def = config.features.get(connector_data.get('type'), {})

        c_min_w, c_min_h = [d // 5 for d in connector_def.get('min_dimensions_ft', [5, 5])]

        size_tier_map = {'large': (12, 12), 'medium': (8, 8), 'small': (4, 4)}
        child_w, child_h = size_tier_map.get(child_size_tier, (4, 4))

        px, py, pw, ph = parent_node.get_rect()

        faces_to_check = list(OPPOSITE_DIRECTION_MAP.keys())
        random.shuffle(faces_to_check)

        for face in faces_to_check:
            if face in ('N', 'S'):
                for x_offset in range(pw - c_min_w + 1):
                    conn_x, child_x = px + x_offset, px + x_offset + (c_min_w - child_w) // 2
                    conn_y = py - c_min_h if face == 'N' else py + ph
                    child_y = py - c_min_h - child_h if face == 'N' else py + ph + c_min_h

                    conn_rect = (conn_x, conn_y, c_min_w, c_min_h)
                    child_rect = (child_x, child_y, child_w, child_h)

                    if self._is_placement_valid(conn_rect, temp_grid, connector_def) and \
                            self._is_placement_valid(child_rect, temp_grid, {}):
                        return {'parent': parent_node, 'face': face, 'rect': conn_rect}, child_rect
            else:
                for y_offset in range(ph - c_min_h + 1):
                    conn_y, child_y = py + y_offset, py + y_offset + (c_min_h - child_h) // 2
                    conn_x = px - c_min_w if face == 'W' else px + pw
                    child_x = px - c_min_w - child_w if face == 'W' else px + pw + c_min_w

                    conn_rect = (conn_x, conn_y, c_min_w, c_min_h)
                    child_rect = (child_x, child_y, child_w, child_h)

                    if self._is_placement_valid(conn_rect, temp_grid, connector_def) and \
                            self._is_placement_valid(child_rect, temp_grid, {}):
                        return {'parent': parent_node, 'face': face, 'rect': conn_rect}, child_rect
        return None

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