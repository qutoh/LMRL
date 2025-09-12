# /core/worldgen/v3_components/placement.py

import math
import random
from typing import List, Tuple, Dict, Optional, Set, Generator, Callable
from collections import defaultdict

import numpy as np

from core.common import utils
from .feature_node import FeatureNode
from ...common.config_loader import config
from . import shape_utils, geometry_probes
from .pathing import Pathing
from .v3_llm import V3_LLM
from ..semantic_search import SemanticSearch

MIN_FEATURE_SIZE = 3
GROWTH_COVERAGE_THRESHOLD = 0.40
MAX_GROWTH_ITERATIONS = 200
SIZE_TIER_TARGET_AREA_FACTOR = {'large': 0.75, 'medium': 0.5, 'small': 0.25}


class Placement:
    def __init__(self, map_width: int, map_height: int, get_temp_grid_func: Callable, pathing_instance: 'Pathing',
                 llm: V3_LLM, semantic_search: SemanticSearch):
        self.map_width = map_width
        self.map_height = map_height
        self.void_space_index = config.tile_type_map.get("VOID_SPACE", -1)
        self._get_temp_grid_external = get_temp_grid_func
        self.pathing = pathing_instance
        self.llm = llm
        self.semantic_search = semantic_search

    def _get_temp_grid(self, all_branches: List[FeatureNode],
                       exclude_nodes: Optional[Set[FeatureNode]] = None) -> np.ndarray:
        return self._get_temp_grid_external(all_branches, exclude_nodes)

    def _is_placement_valid(self, footprint: Set[Tuple[int, int]], collision_mask: np.ndarray) -> bool:
        for x, y in footprint:
            if not (0 <= y < self.map_height and 0 <= x < self.map_width) or collision_mask[y, x]:
                return False
        return True

    def _get_rotated_footprints(self, node: FeatureNode, feature_def: Dict) -> List[
        Tuple[Set[Tuple[int, int]], Set[Tuple[int, int]]]]:
        """
        Generates a list of unique footprint pairs (total, interior) for a feature
        based on the angles specified in its definition. If the feature is symmetrical
        (square, circle), only one orientation (0-degree rotation) is returned.
        """
        if not node.footprint:
            return []

        base_total_fp = node.footprint
        base_interior_fp = node.interior_footprint

        # Check for symmetry
        is_symmetrical_shape = False
        shape_type = feature_def.get('shape', 'rectangle')
        if shape_type == 'ellipse' and node.current_abs_width == node.current_abs_height:
            is_symmetrical_shape = True
        elif shape_type == 'rectangle' and node.current_abs_width == node.current_abs_height:
            is_symmetrical_shape = True

        # If symmetrical, only calculate 0-degree rotation
        if is_symmetrical_shape:
            # Normalize the 0-degree footprint
            if not base_total_fp: return []
            min_x = min(p[0] for p in base_total_fp)
            min_y = min(p[1] for p in base_total_fp)
            normalized_total = {(x - min_x, y - min_y) for x, y in base_total_fp}
            normalized_interior = {(x - min_x, y - min_y) for x, y in base_interior_fp}
            return [(normalized_total, normalized_interior)]

        # For asymmetrical shapes, proceed with allowed rotations
        w = node.current_abs_width
        h = node.current_abs_height
        cx, cy = (w - 1) / 2.0, (h - 1) / 2.0

        unique_footprints = set()
        footprint_pairs = []

        allowed_rotations = feature_def.get("allowed_rotations", [0, 90])

        for angle_deg in allowed_rotations:
            angle_rad = math.radians(angle_deg)
            cos_a = math.cos(angle_rad)
            sin_a = math.sin(angle_rad)

            rotated_total = {((x - cx) * cos_a - (y - cy) * sin_a, (x - cx) * sin_a + (y - cy) * cos_a) for x, y in
                             base_total_fp}
            rotated_interior = {((x - cx) * cos_a - (y - cy) * sin_a, (x - cx) * sin_a + (y - cy) * cos_a) for x, y in
                                base_interior_fp}

            if not rotated_total: continue

            min_x = min(p[0] for p in rotated_total)
            min_y = min(p[1] for p in rotated_total)

            normalized_total = frozenset({(round(x - min_x), round(y - min_y)) for x, y in rotated_total})

            if normalized_total not in unique_footprints:
                unique_footprints.add(normalized_total)
                normalized_interior = {(round(x - min_x), round(y - min_y)) for x, y in rotated_interior}
                footprint_pairs.append((set(normalized_total), normalized_interior))

        return footprint_pairs

    def _find_placement_logic(self, feature_data: Dict, parent_node: FeatureNode,
                              all_branches: List[FeatureNode], dry_run: bool) -> Optional[FeatureNode | bool]:
        feature_def = config.features.get(feature_data['type'], {})
        size_tier_map = {'large': 0.75, 'medium': 0.5, 'small': 0.25}
        size_ratio = size_tier_map.get(feature_data.get('size_tier', 'medium'), 0.5)
        parent_area = len(parent_node.footprint)
        target_sub_area = parent_area * size_ratio if parent_area > 0 else 25
        shape = feature_def.get('shape', 'rectangle')
        rw, rh = (1, 1) if shape == 'ellipse' else (random.randint(1, 16), random.randint(1, 16))
        sub_w = max(MIN_FEATURE_SIZE, round(math.sqrt(target_sub_area * rw / rh)))
        sub_h = max(MIN_FEATURE_SIZE, round(math.sqrt(target_sub_area * rh / rw)))

        temp_child_node_for_size = FeatureNode("temp", feature_data['type'], sub_w, sub_h)
        rotated_footprint_pairs = self._get_rotated_footprints(temp_child_node_for_size, feature_def)
        if not rotated_footprint_pairs: return None
        random.shuffle(rotated_footprint_pairs)

        parent_connection_points = geometry_probes.find_potential_connection_points(parent_node.footprint)
        if not parent_connection_points: return None

        shuffled_parent_points = list(parent_connection_points.items())
        random.shuffle(shuffled_parent_points)

        base_collision_mask = self._get_temp_grid(all_branches) != self.void_space_index
        parent_footprint_abs_base = parent_node.get_absolute_footprint()

        for total_fp, interior_fp in rotated_footprint_pairs:
            child_connection_points = geometry_probes.find_potential_connection_points(total_fp)
            if not child_connection_points: continue

            shuffled_child_points = list(child_connection_points.items())
            random.shuffle(shuffled_child_points)

            for (px, py), p_data in shuffled_parent_points:
                for (cx, cy), _ in shuffled_child_points:
                    adjacency_dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)]
                    random.shuffle(adjacency_dirs)
                    for dx, dy in adjacency_dirs:
                        abs_px, abs_py = px + parent_node.current_x, py + parent_node.current_y

                        adj_x, adj_y = abs_px + dx, abs_py + dy
                        origin_x = adj_x - cx
                        origin_y = adj_y - cy

                        footprint_abs = {(x + origin_x, y + origin_y) for x, y in total_fp}

                        if not self._is_placement_valid(footprint_abs, base_collision_mask):
                            continue

                        p_recon = p_data.get('reconciliation')
                        parent_footprint_for_check = parent_footprint_abs_base

                        if p_recon and 'extrude' in p_recon:
                            rel_extrude = p_recon['extrude']
                            abs_extrude = (
                                rel_extrude[0] + parent_node.current_x, rel_extrude[1] + parent_node.current_y)

                            if abs_extrude in footprint_abs or base_collision_mask[abs_extrude[1]][abs_extrude[0]]:
                                continue

                            parent_footprint_for_check = parent_footprint_abs_base.union({abs_extrude})

                        if not footprint_abs.isdisjoint(parent_footprint_for_check):
                            continue

                        if dry_run:
                            return True

                        # --- SUCCESS: A valid placement was found ---
                        if p_recon and 'extrude' in p_recon:
                            parent_node.footprint.add(p_recon['extrude'])
                            parent_node.update_bounding_box_from_footprint()

                        new_subfeature = FeatureNode(feature_data['name'], feature_data['type'], sub_w, sub_h, origin_x,
                                                     origin_y, parent=parent_node)
                        new_subfeature.footprint = total_fp
                        new_subfeature.interior_footprint = interior_fp
                        new_subfeature.update_bounding_box_from_footprint()
                        parent_node.subfeatures.append(new_subfeature)
                        return new_subfeature

        return None if not dry_run else False

    def can_place_subfeature(self, feature_data: Dict, parent_node: FeatureNode,
                             all_branches: List[FeatureNode]) -> bool:
        """
        A non-destructive 'dry run' to check if a subfeature can be placed on a parent.
        """
        return self._find_placement_logic(feature_data, parent_node, all_branches, dry_run=True)

    def find_and_place_subfeature(self, feature_data: Dict, parent_node: FeatureNode,
                                  all_branches: List[FeatureNode]) -> Optional[FeatureNode]:
        """
        Finds a valid placement for a subfeature and attaches it to the parent.
        """
        return self._find_placement_logic(feature_data, parent_node, all_branches, dry_run=False)

    def place_new_root_branch(self, feature_data: Dict, all_branches: List[FeatureNode]) -> Optional[FeatureNode]:
        feature_def = config.features.get(feature_data['type'], {})
        size_tier_map = {'large': 0.8, 'medium': 0.5, 'small': 0.25}
        rel_dim = size_tier_map.get(feature_data.get('size_tier', 'medium'), 0.5)
        prop_w = max(MIN_FEATURE_SIZE, int((self.map_width) * rel_dim * 0.5))
        prop_h = max(MIN_FEATURE_SIZE, int((self.map_height) * rel_dim * 0.5))
        temp_node = FeatureNode("temp_root", feature_data['type'], prop_w, prop_h)
        map_grid = self._get_temp_grid(all_branches)
        allowed_indices = {self.void_space_index}
        collision_mask = ~np.isin(map_grid, list(allowed_indices))
        possible_origins = []
        for y in range(1, self.map_height - prop_h - 1):
            for x in range(1, self.map_width - prop_w - 1):
                proposed_footprint = {(rx + x, ry + y) for rx, ry in temp_node.footprint}
                if self._is_placement_valid(proposed_footprint, collision_mask):
                    possible_origins.append((x, y))
        if not possible_origins:
            return None
        x, y = random.choice(possible_origins)
        new_root_node = FeatureNode(feature_data['name'], feature_data['type'], prop_w, prop_h, x, y, parent=None)
        all_branches.append(new_root_node)
        return new_root_node

    def handle_connector_placement(self, feature_data: Dict, parent_branch: FeatureNode,
                                   all_branches: List[FeatureNode]) -> Optional[List[FeatureNode]]:
        """
        Handles the placement of a 'CONNECTOR' type feature, using the "probe, then ask"
        method to ensure geometric validity before calling the LLM.
        """
        valid_targets = []
        for target_branch in all_branches:
            if target_branch is parent_branch:
                continue
            if self.place_subfeature_path_between_branches(feature_data, parent_branch, target_branch, all_branches,
                                                           dry_run=True):
                valid_targets.append(target_branch)

        target_options = [b.name for b in valid_targets]

        choice = self.llm.decide_connector_strategy(feature_data['name'], feature_data.get('description', ''),
                                                    target_options)

        if "CREATE_NEW" not in choice.upper() and valid_targets:
            target_name = self.semantic_search.find_best_match(choice, target_options)
            target_branch = next((b for b in valid_targets if b.name == target_name), None)
            if target_branch:
                path_node = self.place_subfeature_path_between_branches(feature_data, parent_branch, target_branch,
                                                                        all_branches, dry_run=False)
                if path_node:
                    return [path_node]
                else:
                    utils.log_message('debug',
                                      f"  [Placement FAIL] Connector bridge failed on commit for '{feature_data['name']}' from '{parent_branch.name}' to '{target_branch.name}'.")

        connector_node = self.find_and_place_subfeature(feature_data, parent_branch, all_branches)
        if not connector_node:
            utils.log_message('debug',
                              f"  [Placement FAIL] Could not place seed connector '{feature_data['name']}' ('{feature_data['type']}') on '{parent_branch.name}'.")
            return None
        child_feature_data = self.llm.create_connector_child(grandparent_node=parent_branch,
                                                             connector_node=connector_node)
        if not child_feature_data:
            if connector_node.parent:
                connector_node.parent.subfeatures.remove(connector_node)
            else:
                all_branches.remove(connector_node)
            utils.log_message('debug',
                              f"  [Placement FAIL] LLM failed to create child for connector '{connector_node.name}'.")
            return None
        child_node = self.find_and_place_subfeature(child_feature_data, connector_node, all_branches)
        if not child_node:
            if connector_node.parent:
                connector_node.parent.subfeatures.remove(connector_node)
            else:
                all_branches.remove(connector_node)
            utils.log_message('debug',
                              f"  [Placement FAIL] Could not place child '{child_feature_data['name']}' for connector '{connector_node.name}'.")
            return None
        return [connector_node, child_node]

    def _create_path_node_from_centerline(self, centerline: List[Tuple[int, int]], feature_data: Dict,
                                          clearance: int) -> FeatureNode:
        brush = {(dx, dy) for dx in range(-clearance, clearance + 1) for dy in range(-clearance, clearance + 1) if
                 dx * dx + dy * dy <= clearance * clearance}
        path_footprint = {(x + dx, y + dy) for x, y in centerline for dx, dy in brush}
        new_path_node = FeatureNode(name=feature_data['name'], feature_type=feature_data['type'], abs_w=0, abs_h=0,
                                    x=0, y=0)
        new_path_node.path_coords = list(path_footprint)
        new_path_node.footprint = path_footprint
        new_path_node.interior_footprint = path_footprint
        return new_path_node

    def place_subfeature_path_between_branches(self, feature_data: Dict, parent_branch: FeatureNode,
                                               target_branch: FeatureNode,
                                               all_branches: List[FeatureNode], dry_run: bool = False) -> Optional[
        FeatureNode]:
        feature_def = config.features.get(feature_data['type'], {})
        size_tier = feature_data.get('size_tier', 'medium')
        clearance = {'small': 0, 'medium': 1, 'large': 2}.get(size_tier, 0)
        clearance = max(clearance, 1)
        start_points = [p for n in parent_branch.get_all_nodes_in_branch() for p in
                        self.pathing.get_valid_connection_points(n, clearance, all_branches)]
        end_points = [p for n in target_branch.get_all_nodes_in_branch() for p in
                      self.pathing.get_valid_connection_points(n, clearance, all_branches)]
        if not start_points or not end_points: return None
        start_pos, end_pos = self.pathing.find_first_valid_connection(start_points, end_points, clearance,
                                                                      all_branches, path_feature_def=feature_def)
        if start_pos and end_pos:
            if dry_run: return FeatureNode("probe", "PROBE", 1, 1)
            centerline = self.pathing.find_path_with_clearance(start_pos, end_pos, clearance, all_branches,
                                                               feature_def)
            if centerline:
                new_path_node = self._create_path_node_from_centerline(centerline, feature_data, clearance)
                new_path_node.parent = parent_branch
                parent_branch.subfeatures.append(new_path_node)
                return new_path_node
        return None

    def _attempt_aspect_aware_growth(self, node: FeatureNode, all_branches: List[FeatureNode]) -> bool:
        ratio_w, ratio_h = node.target_aspect_ratio
        next_k = node.growth_multiplier + 1
        prop_w = max(node.current_abs_width + 1, int(ratio_w * next_k))
        prop_h = max(node.current_abs_height + 1, int(ratio_h * next_k))
        if prop_w == node.current_abs_width and prop_h == node.current_abs_height:
            return False
        exclude_nodes = set(node.get_all_nodes_in_branch())
        temp_grid = self._get_temp_grid(all_branches, exclude_nodes=exclude_nodes)
        collision_mask = temp_grid != self.void_space_index
        temp_growth_node = FeatureNode(node.name, node.feature_type, prop_w, prop_h)
        PROBE_COUNT = 100
        possible_origins = []
        if self.map_width > prop_w + 1 and self.map_height > prop_h + 1:
            for _ in range(PROBE_COUNT):
                x = random.randint(1, self.map_width - prop_w - 1)
                y = random.randint(1, self.map_height - prop_h - 1)
                possible_origins.append((x, y))
        possible_origins.insert(0, (node.current_x, node.current_y))
        for x, y in possible_origins:
            if not (0 <= y < self.map_height - prop_h and 0 <= x < self.map_width - prop_w):
                continue
            proposed_footprint = {(rx + x, ry + y) for rx, ry in temp_growth_node.footprint}
            if self._is_placement_valid(proposed_footprint, collision_mask):
                node.current_x, node.current_y = x, y
                node.footprint = temp_growth_node.footprint
                node.interior_footprint = temp_growth_node.interior_footprint
                node.update_bounding_box_from_footprint()
                node.growth_multiplier = next_k
                return True
        return False

    def place_initial_path_between_branches(self, feature_data: Dict, source_node: FeatureNode, dest_node: FeatureNode,
                                            all_branches: List[FeatureNode]) -> Optional[FeatureNode]:
        feature_def = config.features.get(feature_data['type'], {})
        size_tier = feature_data.get('size_tier', 'medium')
        clearance = {'small': 0, 'medium': 1, 'large': 2}.get(size_tier, 0)
        clearance = max(clearance, 1)
        start_points = self.pathing.get_valid_connection_points(source_node, clearance, all_branches)
        end_points = self.pathing.get_valid_connection_points(dest_node, clearance, all_branches)
        if not start_points or not end_points: return None
        start_pos, end_pos = self.pathing.find_first_valid_connection(start_points, end_points, clearance,
                                                                      all_branches, path_feature_def=feature_def)
        if start_pos and end_pos:
            centerline = self.pathing.find_path_with_clearance(start_pos, end_pos, clearance, all_branches,
                                                               feature_def)
            if centerline:
                new_path_node = self._create_path_node_from_centerline(centerline, feature_data, clearance)
                all_branches.append(new_path_node)
                return new_path_node
        return None

    def place_initial_path_to_border(self, feature_data: Dict, source_node: FeatureNode,
                                     border_coords: List[Tuple[int, int]],
                                     all_branches: List[FeatureNode]) -> Optional[FeatureNode]:
        feature_def = config.features.get(feature_data['type'], {})
        size_tier = feature_data.get('size_tier', 'medium')
        clearance = {'small': 0, 'medium': 1, 'large': 2}.get(size_tier, 0)
        clearance = max(clearance, 1)
        start_points = self.pathing.get_valid_connection_points(source_node, clearance, all_branches)
        end_points = border_coords
        if not start_points or not end_points: return None
        start_pos, end_pos = self.pathing.find_first_valid_connection(start_points, end_points, clearance,
                                                                      all_branches, path_feature_def=feature_def)
        if start_pos and end_pos:
            centerline = self.pathing.find_path_with_clearance(start_pos, end_pos, clearance, all_branches,
                                                               feature_def)
            if centerline:
                new_path_node = self._create_path_node_from_centerline(centerline, feature_data, clearance)
                all_branches.append(new_path_node)
                return new_path_node
        return None

    def place_subfeature_path_to_border(self, feature_data: Dict, parent_branch: FeatureNode,
                                        border_coords: List[Tuple[int, int]],
                                        all_branches: List[FeatureNode]) -> Optional[FeatureNode]:
        feature_def = config.features.get(feature_data['type'], {})
        size_tier = feature_data.get('size_tier', 'medium')
        clearance = {'small': 0, 'medium': 1, 'large': 2}.get(size_tier, 0)
        clearance = max(clearance, 1)
        start_points = [p for n in parent_branch.get_all_nodes_in_branch() for p in
                        self.pathing.get_valid_connection_points(n, clearance, all_branches)]
        end_points = border_coords
        if not start_points or not end_points: return None
        start_pos, end_pos = self.pathing.find_first_valid_connection(start_points, end_points, clearance,
                                                                      all_branches, path_feature_def=feature_def)
        if start_pos and end_pos:
            centerline = self.pathing.find_path_with_clearance(start_pos, end_pos, clearance, all_branches,
                                                               feature_def)
            if centerline:
                new_path_node = self._create_path_node_from_centerline(centerline, feature_data, clearance)
                new_path_node.parent = parent_branch
                parent_branch.subfeatures.append(new_path_node)
                return new_path_node
        return None

    def place_initial_path_between_borders(self, feature_data: Dict, source_border_coords: List[Tuple[int, int]],
                                           dest_border_coords: List[Tuple[int, int]],
                                           all_branches: List[FeatureNode]) -> Optional[FeatureNode]:
        feature_def = config.features.get(feature_data['type'], {})
        size_tier = feature_data.get('size_tier', 'medium')
        clearance = {'small': 0, 'medium': 1, 'large': 2}.get(size_tier, 0)
        clearance = max(clearance, 1)
        start_points = source_border_coords
        end_points = dest_border_coords
        if not start_points or not end_points: return None
        start_pos, end_pos = self.pathing.find_first_valid_connection(start_points, end_points, clearance,
                                                                      all_branches, path_feature_def=feature_def)
        if start_pos and end_pos:
            centerline = self.pathing.find_path_with_clearance(start_pos, end_pos, clearance, all_branches,
                                                               feature_def)
            if centerline:
                new_path_node = self._create_path_node_from_centerline(centerline, feature_data, clearance)
                all_branches.append(new_path_node)
                return new_path_node
        return None