import math
import random
from typing import List, Tuple, Dict, Optional, Set, Generator, Callable
from collections import defaultdict, deque

import numpy as np
import tcod
from scipy.spatial import ConvexHull
from scipy.ndimage import distance_transform_edt

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

    def _calculate_feature_size(self, feature_data: Dict, parent_node: Optional[FeatureNode] = None,
                                is_root: bool = False) -> Tuple[int, int]:
        """
        Calculates width and height for new features, respecting size_tier.
        - is_root: Proportional to the whole map.
        - parent_node provided: Proportional to the parent.
        - Neither: Proportional to a small fraction of the map (for 'nearby').
        """
        size_tier_map = {'large': 0.7, 'medium': 0.45, 'small': 0.2}
        size_tier = feature_data.get('size_tier', 'medium')
        size_ratio = size_tier_map.get(size_tier, 0.45)

        if is_root:
            ref_w, ref_h = self.map_width, self.map_height
        elif parent_node:
            ref_w, ref_h = parent_node.current_abs_width, parent_node.current_abs_height
        else:  # 'nearby' case
            ref_w, ref_h = self.map_width * 0.25, self.map_height * 0.25

        # Use area to determine scale, then apply a random aspect ratio
        ref_area = ref_w * ref_h
        target_area = ref_area * size_ratio
        aspect_ratio_tweak = random.uniform(0.7, 1.4)

        w = max(MIN_FEATURE_SIZE, int(round(math.sqrt(target_area * aspect_ratio_tweak))))
        h = max(MIN_FEATURE_SIZE, int(round(math.sqrt(target_area / aspect_ratio_tweak))))

        return w, h

    def _is_proposal_valid(self, node: FeatureNode, proposal_x: int, proposal_y: int,
                           proposal_footprint: Set[Tuple[int, int]], all_branches: List[FeatureNode]) -> bool:
        """Validates a single proposed transformation for a single node."""
        if not proposal_footprint:
            return False

        temp_grid_others = self.pathing._get_temp_grid(all_branches, exclude_nodes={node})
        absolute_proposal_footprint = {(rx + proposal_x, ry + proposal_y) for rx, ry in proposal_footprint}

        for px, py in absolute_proposal_footprint:
            if not (0 <= px < self.map_width and 0 <= py < self.map_height) or \
                    temp_grid_others[py, px] != self.pathing.void_space_index:
                return False

        original_x, original_y, original_footprint = node.current_x, node.current_y, node.footprint
        node.current_x, node.current_y, node.footprint = proposal_x, proposal_y, proposal_footprint

        if node.parent:
            if not self.pathing.reconcile_connection(node.parent, node, all_branches,
                                                     {node: node.get_absolute_footprint(),
                                                      node.parent: node.parent.get_absolute_footprint()}):
                node.current_x, node.current_y, node.footprint = original_x, original_y, original_footprint
                return False

        for child in node.subfeatures:
            if not self.pathing.reconcile_connection(node, child, all_branches,
                                                     {node: node.get_absolute_footprint(),
                                                      child: child.get_absolute_footprint()}):
                node.current_x, node.current_y, node.footprint = original_x, original_y, original_footprint
                return False

        node.current_x, node.current_y, node.footprint = original_x, original_y, original_footprint
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

    def _place_outside_logic(self, feature_data: Dict, parent_node: FeatureNode,
                             all_branches: List[FeatureNode], dry_run: bool) -> Optional[FeatureNode | bool]:
        feature_def = config.features.get(feature_data['type'], {})
        sub_w, sub_h = self._calculate_feature_size(feature_data, parent_node=parent_node)

        temp_child_node_for_size = FeatureNode("temp", feature_data['type'], sub_w, sub_h)
        temp_child_node_for_size.natures = feature_data.get('natures', [])
        rotated_footprint_pairs = self._get_rotated_footprints(temp_child_node_for_size, feature_def)
        if not rotated_footprint_pairs: return None
        random.shuffle(rotated_footprint_pairs)

        parent_connection_points = geometry_probes.find_potential_connection_points(parent_node.footprint)
        if not parent_connection_points: return None

        shuffled_parent_points = list(parent_connection_points.items())
        random.shuffle(shuffled_parent_points)

        base_collision_mask = self._get_temp_grid(all_branches) != self.void_space_index

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

                        if dry_run:
                            return True

                        # --- SUCCESS: A valid placement was found ---
                        p_recon = p_data.get('reconciliation')
                        if p_recon and 'extrude' in p_recon:
                            parent_node.footprint.add(p_recon['extrude'])
                            parent_node.update_bounding_box_from_footprint()

                        new_subfeature = FeatureNode(feature_data['name'], feature_data['type'], sub_w, sub_h, origin_x,
                                                     origin_y, parent=parent_node)
                        new_subfeature.natures = feature_data.get('natures', [])
                        new_subfeature.footprint = total_fp
                        new_subfeature.interior_footprint = interior_fp
                        new_subfeature.update_bounding_box_from_footprint()
                        parent_node.subfeatures.append(new_subfeature)
                        return new_subfeature

        return None if not dry_run else False

    def can_place_outside(self, feature_data: Dict, parent_node: FeatureNode,
                          all_branches: List[FeatureNode]) -> bool:
        """
        A non-destructive 'dry run' to check if a subfeature can be placed on a parent's exterior.
        """
        return self._place_outside_logic(feature_data, parent_node, all_branches, dry_run=True)

    def place_outside(self, feature_data: Dict, parent_node: FeatureNode,
                      all_branches: List[FeatureNode]) -> Optional[FeatureNode]:
        """
        Finds a valid placement for a subfeature on a parent's exterior and attaches it.
        """
        return self._place_outside_logic(feature_data, parent_node, all_branches, dry_run=False)

    def place_nearby(self, feature_data: Dict, target_node: FeatureNode,
                     all_branches: List[FeatureNode]) -> Optional[FeatureNode]:
        """
        Places a feature near a target. First tries direct adjacency, then searches outwards.
        """
        # 1. Attempt to place directly adjacent using 'place_outside' logic
        if new_node := self.place_outside(feature_data, target_node, all_branches):
            return new_node

        # 2. If adjacency fails, perform a BFS search from the target's perimeter
        q = deque(list(target_node.get_absolute_border_footprint()))
        visited = set(q)
        search_limit = 5000  # Practical limit to prevent freezing
        iterations = 0

        # Create a template child node to get its dimensions and footprint variants
        feature_def = config.features.get(feature_data['type'], {})
        sub_w, sub_h = self._calculate_feature_size(feature_data, parent_node=None)

        temp_child_node = FeatureNode("temp", feature_data['type'], sub_w, sub_h)
        rotated_footprint_pairs = self._get_rotated_footprints(temp_child_node, feature_def)
        if not rotated_footprint_pairs: return None

        collision_mask = self._get_temp_grid(all_branches) != self.void_space_index

        while q and iterations < search_limit:
            iterations += 1
            x, y = q.popleft()

            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                neighbor = (nx, ny)

                if neighbor in visited or not (0 <= ny < self.map_height and 0 <= nx < self.map_width):
                    continue
                visited.add(neighbor)

                if not collision_mask[ny, nx]:
                    # Found an empty spot, try to place the feature here
                    for total_fp, interior_fp in rotated_footprint_pairs:
                        footprint_abs = {(rx + nx, ry + ny) for rx, ry in total_fp}
                        if self._is_placement_valid(footprint_abs, collision_mask):
                            # SUCCESS!
                            new_node = FeatureNode(feature_data['name'], feature_data['type'], sub_w, sub_h, nx, ny,
                                                   parent=target_node)
                            new_node.natures = feature_data.get('natures', [])
                            new_node.footprint = total_fp
                            new_node.interior_footprint = interior_fp
                            new_node.update_bounding_box_from_footprint()
                            target_node.subfeatures.append(new_node)
                            return new_node
                    q.append(neighbor)  # Add to queue to continue searching from here
        return None

    def _repartition_interior_space(self, parent_node: FeatureNode) -> bool:
        """
        Subdivides a parent's interior space among its children using a Voronoi-like algorithm.
        This is called every time a new child is placed inside the parent.
        """
        parent_interior_abs = parent_node.get_absolute_interior_footprint()
        if not parent_interior_abs: return False

        children = [c for c in parent_node.subfeatures]
        if not children: return True

        # Clear all previous footprints to ensure a clean slate
        for child in children:
            child.footprint.clear()
            child.interior_footprint.clear()

        # 1. Establish unique seed points for each child
        seeds = {}
        available_seed_points = list(parent_interior_abs)
        random.shuffle(available_seed_points)

        for child in children:
            if not available_seed_points:
                utils.log_message('debug', f"  [Repartition FAIL] Ran out of seed points in '{parent_node.name}'.")
                return False
            seeds[child] = available_seed_points.pop()

        # 2. Partition the parent's interior space based on distance to seeds
        child_footprints = defaultdict(set)
        for px, py in parent_interior_abs:
            min_dist = float('inf')
            closest_child = None
            for child, (sx, sy) in seeds.items():
                dist = math.hypot(px - sx, py - sy)
                if dist < min_dist:
                    min_dist = dist
                    closest_child = child
            if closest_child:
                child_footprints[closest_child].add((px, py))

        # 3. Update geometry for all children
        for child in children:
            new_abs_fp = child_footprints.get(child)
            if not new_abs_fp or len(new_abs_fp) < MIN_FEATURE_SIZE ** 2:
                return False

            min_x = min(p[0] for p in new_abs_fp)
            min_y = min(p[1] for p in new_abs_fp)
            child.current_x, child.current_y = min_x, min_y
            child.footprint = {(x - min_x, y - min_y) for x, y in new_abs_fp}

            # 4. Calculate borders only between siblings
            if len(children) == 1:
                child.interior_footprint = child.footprint.copy()
            else:
                sibling_fps = {other_child: other_fp for other_child, other_fp in child_footprints.items() if
                               other_child is not child}
                child_border_abs = set()
                for x, y in new_abs_fp:
                    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        neighbor = (x + dx, y + dy)
                        for sibling_fp in sibling_fps.values():
                            if neighbor in sibling_fp:
                                child_border_abs.add((x, y))
                                break
                    if (x, y) in child_border_abs: continue

                child_border_rel = {(x - min_x, y - min_y) for x, y in child_border_abs}
                child.interior_footprint = child.footprint - child_border_rel

            child.update_bounding_box_from_footprint()

        return True

    def _reconcile_container_growth(self, parent_node: FeatureNode, all_branches: List[FeatureNode]) -> bool:
        """Attempts to grow a container parent by one tile in the most open direction."""
        original_footprint = parent_node.footprint.copy()
        original_interior = parent_node.interior_footprint.copy()
        original_x, original_y = parent_node.current_x, parent_node.current_y

        collision_mask = self._get_temp_grid(all_branches, exclude_nodes={parent_node}) != self.void_space_index

        direction_scores = defaultdict(int)
        parent_abs_footprint = parent_node.get_absolute_footprint()
        for x, y in parent_abs_footprint:
            for dx, dy, direction in [(0, -1, "N"), (0, 1, "S"), (-1, 0, "W"), (1, 0, "E")]:
                nx, ny = x + dx, y + dy
                if 0 <= ny < self.map_height and 0 <= nx < self.map_width and not collision_mask[ny, nx]:
                    direction_scores[direction] += 1

        if not direction_scores: return False

        best_direction = max(direction_scores, key=direction_scores.get)

        proposal_footprint = parent_node.footprint.copy()
        if best_direction == "N":
            min_y = min(p[1] for p in proposal_footprint)
            for x in range(parent_node.current_abs_width):
                if (x, min_y) in proposal_footprint: proposal_footprint.add((x, min_y - 1))
        elif best_direction == "S":
            max_y = max(p[1] for p in proposal_footprint)
            for x in range(parent_node.current_abs_width):
                if (x, max_y) in proposal_footprint: proposal_footprint.add((x, max_y + 1))
        elif best_direction == "W":
            min_x = min(p[0] for p in proposal_footprint)
            for y in range(parent_node.current_abs_height):
                if (min_x, y) in proposal_footprint: proposal_footprint.add((min_x - 1, y))
        elif best_direction == "E":
            max_x = max(p[0] for p in proposal_footprint)
            for y in range(parent_node.current_abs_height):
                if (max_x, y) in proposal_footprint: proposal_footprint.add((max_x + 1, y))

        if self._is_proposal_valid(parent_node, parent_node.current_x, parent_node.current_y, proposal_footprint,
                                   all_branches):
            parent_node.footprint = proposal_footprint
            parent_node.interior_footprint = original_footprint  # The old footprint is the new interior
            parent_node.update_bounding_box_from_footprint()
            return True

        parent_node.footprint, parent_node.interior_footprint = original_footprint, original_interior
        parent_node.current_x, parent_node.current_y = original_x, original_y
        parent_node.update_bounding_box_from_footprint()
        return False

    def _reconcile_path_bulge(self, parent_node: FeatureNode, child_data: Dict, all_branches: List[FeatureNode]) -> \
            Optional[FeatureNode]:
        """Systematically searches for a valid way to bulge a path to fit a new interior feature."""
        if not parent_node.centerline_coords: return None

        w, h = self._calculate_feature_size(child_data, parent_node=parent_node)
        temp_child = FeatureNode("temp", child_data['type'], w, h)
        rotated_fps = self._get_rotated_footprints(temp_child, config.features.get(child_data['type'], {}))

        candidates = []
        base_collision_mask = self._get_temp_grid(all_branches, exclude_nodes={parent_node}) != self.void_space_index
        parent_abs_fp = parent_node.get_absolute_footprint()

        for i, (cx, cy) in enumerate(parent_node.centerline_coords):
            dx, dy = 0, 0
            if i > 0:
                px, py = parent_node.centerline_coords[i - 1]
                dx += cx - px
                dy += cy - py
            if i < len(parent_node.centerline_coords) - 1:
                nx, ny = parent_node.centerline_coords[i + 1]
                dx += nx - cx
                dy += ny - cy

            outward_vectors = [(-dy, dx), (dy, -dx)] if dx != 0 or dy != 0 else [(1, 0), (-1, 0), (0, 1), (0, -1)]

            for odx, ody in outward_vectors:
                for child_fp_rel, child_int_fp_rel in rotated_fps:
                    child_w = max(p[0] for p in child_fp_rel) + 1 if child_fp_rel else 0
                    child_h = max(p[1] for p in child_fp_rel) + 1 if child_fp_rel else 0
                    niche_child_x, niche_child_y = cx + odx, cy + ody
                    niche_child_abs_fp = {(rx + niche_child_x, ry + niche_child_y) for rx, ry in child_fp_rel}

                    if self._is_placement_valid(niche_child_abs_fp, base_collision_mask):
                        new_parent_fp_abs = parent_abs_fp.union(niche_child_abs_fp)
                        candidates.append(
                            {'parent_fp': new_parent_fp_abs, 'child_fp': child_fp_rel, 'child_int_fp': child_int_fp_rel,
                             'child_x': niche_child_x, 'child_y': niche_child_y, 'w': child_w, 'h': child_h})

        random.shuffle(candidates)
        for cand in candidates:
            min_x_parent = min(p[0] for p in cand['parent_fp'])
            min_y_parent = min(p[1] for p in cand['parent_fp'])
            parent_fp_rel = {(x - min_x_parent, y - min_y_parent) for x, y in cand['parent_fp']}

            if self._is_proposal_valid(parent_node, min_x_parent, min_y_parent, parent_fp_rel, all_branches):
                parent_node.footprint = parent_fp_rel
                parent_node.current_x, parent_node.current_y = min_x_parent, min_y_parent
                parent_node.update_bounding_box_from_footprint()

                new_node = FeatureNode(child_data['name'], child_data['type'], cand['w'], cand['h'], x=cand['child_x'],
                                       y=cand['child_y'], parent=parent_node)
                new_node.footprint = cand['child_fp']
                new_node.interior_footprint = cand['child_int_fp']
                new_node.update_bounding_box_from_footprint()
                new_node.is_interior = True
                parent_node.subfeatures.append(new_node)
                return new_node

        return None

    def _reconcile_parent_for_interior_placement(self, parent_node: FeatureNode, child_data: Dict,
                                                 all_branches: List[FeatureNode]) -> Optional[FeatureNode]:
        """Controller for parent reconciliation logic."""
        parent_def = config.features.get(parent_node.feature_type, {})
        feature_type = parent_def.get('feature_type')

        if feature_type == 'CONTAINER':
            if self._reconcile_container_growth(parent_node, all_branches):
                return parent_node
            return None
        elif feature_type in ['PATH', 'PORTAL']:
            return self._reconcile_path_bulge(parent_node, child_data, all_branches)
        return None

    def place_inside(self, feature_data: Dict, parent_node: FeatureNode,
                     all_branches: List[FeatureNode]) -> Optional[FeatureNode]:
        """
        Places a feature inside a parent. For CONTAINERs, it repartitions the space.
        For REGIONs, it finds an empty plot of land within the region's boundaries.
        """
        parent_def = config.features.get(parent_node.feature_type, {})
        feature_type = parent_def.get('feature_type')

        if feature_type not in ['CONTAINER', 'PATH', 'PORTAL', 'REGION']:
            return None

        if feature_type in ['PATH', 'PORTAL']:
            return self._reconcile_parent_for_interior_placement(parent_node, feature_data, all_branches)

        if feature_type == 'REGION':
            # Use the same logic as placing a new root branch, but confined to the region's footprint.
            existing_nodes_in_region = parent_node.get_all_nodes_in_branch()
            w, h = self._calculate_feature_size(feature_data, parent_node=parent_node)

            # Create a temporary spec with the calculated size to pass to the placement function
            temp_spec = feature_data.copy()
            temp_spec['temp_w'] = w
            temp_spec['temp_h'] = h

            nodes, success = self._place_initial_non_path_seeds([temp_spec], existing_nodes_in_region,
                                                                parent_node.get_absolute_footprint())
            if success and nodes:
                new_node = nodes[0]
                new_node.parent = parent_node
                parent_node.subfeatures.append(new_node)
                return new_node
            else:
                utils.log_message('debug',
                                  f"[Placement.place_inside] Failed to place seed '{feature_data['name']}' in REGION '{parent_node.name}'.")
                return None

        # --- ORIGINAL LOGIC FOR 'CONTAINER' ---
        new_node = FeatureNode(
            name=feature_data['name'], feature_type=feature_data['type'],
            abs_w=1, abs_h=1, parent=parent_node
        )
        new_node.natures = feature_data.get('natures', [])
        parent_node.subfeatures.append(new_node)

        if self._repartition_interior_space(parent_node):
            new_node.is_interior = True
            return new_node
        else:
            # Repartitioning failed, attempt to reconcile parent
            original_parent_footprint = parent_node.footprint.copy()
            original_interior = parent_node.interior_footprint.copy()
            original_x, original_y = parent_node.current_x, parent_node.current_y

            if self._reconcile_parent_for_interior_placement(parent_node, feature_data, all_branches):
                # Parent grew, retry partitioning
                if self._repartition_interior_space(parent_node):
                    new_node.is_interior = True
                    return new_node

            # Reconciliation failed or second partitioning failed. Revert everything.
            parent_node.subfeatures.remove(new_node)
            parent_node.footprint = original_parent_footprint
            parent_node.interior_footprint = original_interior
            parent_node.current_x, parent_node.current_y = original_x, original_y
            parent_node.update_bounding_box_from_footprint()
            self._repartition_interior_space(parent_node)  # Restore previous children
            return None

    def place_new_root_branch(self, feature_data: Dict, all_branches: List[FeatureNode]) -> Optional[FeatureNode]:
        prop_w, prop_h = self._calculate_feature_size(feature_data, is_root=True)
        temp_node = FeatureNode("temp_root", feature_data['type'], prop_w, prop_h)
        temp_node.natures = feature_data.get('natures', [])

        map_grid = self._get_temp_grid(all_branches)
        collision_mask = map_grid != self.void_space_index
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
        new_root_node.natures = feature_data.get('natures', [])
        all_branches.append(new_root_node)
        return new_root_node

    def handle_connector_placement(self, feature_data: Dict, parent_branch: FeatureNode,
                                   all_branches: List[FeatureNode]) -> Optional[List[FeatureNode]]:
        """
        Handles the placement of a 'CONNECTOR' type feature, using the "probe, then ask"
        method to ensure geometric validity before calling the LLM.
        """
        # --- Path 1: Attempt to connect to an existing valid target ---
        valid_targets = []
        for target_branch in all_branches:
            if target_branch.get_root() is parent_branch.get_root():
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

        # --- Path 2: Create a new feature to connect to ---
        child_feature_data = self.llm.create_connector_child(grandparent_node=parent_branch,
                                                             connector_data=feature_data)
        if not child_feature_data:
            utils.log_message('debug',
                              f"  [Placement FAIL] LLM failed to create child for connector '{feature_data['name']}'.")
            return None

        feature_def = config.features.get(feature_data['type'], {})
        min_len = feature_def.get('min_length', 2)
        max_len = feature_def.get('max_length', 15)

        collision_mask = self._get_temp_grid(all_branches) != self.void_space_index
        parent_perimeter_mask = np.ones_like(collision_mask)
        for x, y in self.pathing._get_node_exterior_perimeter(parent_branch):
            if 0 <= y < self.map_height and 0 <= x < self.map_width:
                parent_perimeter_mask[y, x] = False

        distance_grid = distance_transform_edt(parent_perimeter_mask)
        search_zone_mask = (distance_grid > min_len) & (distance_grid <= max_len) & ~collision_mask
        valid_dest_coords = np.argwhere(search_zone_mask.T)
        if valid_dest_coords.size == 0:
            utils.log_message('debug', f"  [Placement FAIL] No valid destination zone for new connector child.")
            return None

        random.shuffle(valid_dest_coords)
        for dest_x, dest_y in valid_dest_coords:
            w, h = self._calculate_feature_size(child_feature_data, parent_node=parent_branch)
            origin_x, origin_y = dest_x - w // 2, dest_y - h // 2

            temp_child_node = FeatureNode("temp", child_feature_data['type'], w, h)
            child_footprint_abs = {(rx + origin_x, ry + origin_y) for rx, ry in temp_child_node.footprint}

            if self._is_placement_valid(child_footprint_abs, collision_mask):
                child_node = FeatureNode(child_feature_data['name'], child_feature_data['type'], w, h, origin_x,
                                         origin_y)
                child_node.natures = child_feature_data.get('natures', [])
                child_node.footprint = temp_child_node.footprint
                child_node.interior_footprint = temp_child_node.interior_footprint
                child_node.update_bounding_box_from_footprint()

                temp_branches_for_pathing = all_branches + [child_node]
                path_node = self.place_subfeature_path_between_branches(feature_data, parent_branch, child_node,
                                                                        temp_branches_for_pathing, dry_run=False)
                if path_node:
                    child_node.parent = parent_branch
                    parent_branch.subfeatures.append(child_node)
                    return [path_node, child_node]
                else:
                    utils.log_message('debug',
                                      f"  [Placement] Found spot for '{child_node.name}' but pathing failed. Trying another spot.")
                    continue

        utils.log_message('debug', f"  [Placement FAIL] Exhausted all valid destination points for connector child.")
        return None

    def _create_path_node_from_centerline(self, centerline: List[Tuple[int, int]], feature_data: Dict,
                                          clearance: int) -> FeatureNode:
        brush = {(dx, dy) for dx in range(-clearance, clearance + 1) for dy in range(-clearance, clearance + 1) if
                 dx * dx + dy * dy <= clearance * clearance}
        path_footprint = {(x + dx, y + dy) for x, y in centerline for dx, dy in brush}
        new_path_node = FeatureNode(name=feature_data['name'], feature_type=feature_data['type'], abs_w=0, abs_h=0,
                                    x=0, y=0)
        new_path_node.natures = feature_data.get('natures', [])
        new_path_node.path_coords = list(path_footprint)
        new_path_node.centerline_coords = centerline
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

        temp_path_node = FeatureNode(feature_data['name'], feature_data['type'], 0, 0)
        temp_path_node.natures = feature_data.get('natures', [])

        connection = self.pathing.find_first_valid_connection(start_points, end_points, clearance,
                                                              all_branches, path_feature_def=feature_def)
        if not connection or connection == (None, None):
            return None

        start_pos, end_pos = connection
        if start_pos and end_pos:
            if dry_run: return FeatureNode("probe", "PROBE", 1, 1)
            centerline = self.pathing.find_path_with_clearance(start_pos, end_pos, clearance, all_branches,
                                                               temp_path_node)
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

        temp_path_node = FeatureNode(feature_data['name'], feature_data['type'], 0, 0)
        temp_path_node.natures = feature_data.get('natures', [])

        connection = self.pathing.find_first_valid_connection(start_points, end_points, clearance,
                                                              all_branches, path_feature_def=feature_def)
        if not connection or connection == (None, None):
            return None

        start_pos, end_pos = connection
        if start_pos and end_pos:
            centerline = self.pathing.find_path_with_clearance(start_pos, end_pos, clearance, all_branches,
                                                               temp_path_node)
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

        temp_path_node = FeatureNode(feature_data['name'], feature_data['type'], 0, 0)
        temp_path_node.natures = feature_data.get('natures', [])

        connection = self.pathing.find_first_valid_connection(start_points, end_points, clearance,
                                                              all_branches, path_feature_def=feature_def)
        if not connection or connection == (None, None):
            return None

        start_pos, end_pos = connection
        if start_pos and end_pos:
            centerline = self.pathing.find_path_with_clearance(start_pos, end_pos, clearance, all_branches,
                                                               temp_path_node)
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

        temp_path_node = FeatureNode(feature_data['name'], feature_data['type'], 0, 0)
        temp_path_node.natures = feature_data.get('natures', [])

        connection = self.pathing.find_first_valid_connection(start_points, end_points, clearance,
                                                              all_branches, path_feature_def=feature_def)
        if not connection or connection == (None, None):
            return None

        start_pos, end_pos = connection
        if start_pos and end_pos:
            centerline = self.pathing.find_path_with_clearance(start_pos, end_pos, clearance, all_branches,
                                                               temp_path_node)
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

        temp_path_node = FeatureNode(feature_data['name'], feature_data['type'], 0, 0)
        temp_path_node.natures = feature_data.get('natures', [])

        connection = self.pathing.find_first_valid_connection(start_points, end_points, clearance,
                                                              all_branches, path_feature_def=feature_def)
        if not connection or connection == (None, None):
            return None

        start_pos, end_pos = connection
        if start_pos and end_pos:
            centerline = self.pathing.find_path_with_clearance(start_pos, end_pos, clearance, all_branches,
                                                               temp_path_node)
            if centerline:
                new_path_node = self._create_path_node_from_centerline(centerline, feature_data, clearance)
                all_branches.append(new_path_node)
                return new_path_node
        return None

    def place_region_around_branches(self, region_data: Dict, branches_to_group: List[FeatureNode],
                                     all_branches: List[FeatureNode]) -> Optional[FeatureNode]:
        """Creates a new region that encompasses a list of feature branches."""
        if not branches_to_group: return None

        all_points = np.array([p for branch in branches_to_group for node in branch.get_all_nodes_in_branch() for p in
                               node.get_absolute_footprint()])
        if len(all_points) < 3: return None  # ConvexHull needs at least 3 points

        hull = ConvexHull(all_points)
        hull_path = [tuple(p) for p in all_points[hull.vertices]]

        # Create a filled polygon from the hull path
        initial_footprint = set()
        if hull_path:
            min_x = min(p[0] for p in hull_path)
            max_x = max(p[0] for p in hull_path)
            min_y = min(p[1] for p in hull_path)
            max_y = max(p[1] for p in hull_path)

            for y in range(min_y, max_y + 1):
                for x in range(min_x, max_x + 1):
                    # Simple point-in-polygon test
                    if utils.point_in_poly(x, y, hull_path):
                        initial_footprint.add((x, y))

        if not initial_footprint: return None

        # Now expand this initial footprint outwards into empty space
        collision_mask = self._get_temp_grid(
            [b for b in all_branches if b not in branches_to_group]) != self.void_space_index
        final_footprint = set(initial_footprint)
        q = deque(list(initial_footprint))
        visited = set(initial_footprint)

        while q:
            x, y = q.popleft()
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                neighbor = (nx, ny)
                if neighbor in visited: continue
                if 0 <= ny < self.map_height and 0 <= nx < self.map_width:
                    visited.add(neighbor)
                    if not collision_mask[ny, nx]:
                        final_footprint.add(neighbor)
                        q.append(neighbor)

        # Create the new region node
        min_x = min(p[0] for p in final_footprint)
        min_y = min(p[1] for p in final_footprint)

        region_node = FeatureNode(name=region_data['name'], feature_type=region_data['type'], abs_w=1, abs_h=1, x=min_x,
                                  y=min_y)
        region_node.natures = region_data.get('natures', [])
        region_node.footprint = {(x - min_x, y - min_y) for x, y in final_footprint}
        region_node.interior_footprint = {(rx, ry) for rx, ry in region_node.footprint if any(
            (rx + dx, ry + dy) not in region_node.footprint for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)])}
        region_node.update_bounding_box_from_footprint()

        for branch in branches_to_group:
            branch.parent = region_node
            region_node.subfeatures.append(branch)

        return region_node

    def place_initial_region_seed_at_edge(self, feature_data: Dict, all_branches: List[FeatureNode]) -> Optional[
        FeatureNode]:
        """Places a small seed for a region at a random point along a map edge."""
        w, h = MIN_FEATURE_SIZE, MIN_FEATURE_SIZE
        collision_mask = self._get_temp_grid(all_branches) != self.void_space_index

        edge_points = []
        edge_points.extend([(x, 1) for x in range(1, self.map_width - w - 1)])  # North
        edge_points.extend([(x, self.map_height - h - 2) for x in range(1, self.map_width - w - 1)])  # South
        edge_points.extend([(1, y) for y in range(1, self.map_height - h - 1)])  # West
        edge_points.extend([(self.map_width - w - 2, y) for y in range(1, self.map_height - h - 1)])  # East

        if not edge_points: return None
        random.shuffle(edge_points)

        for x, y in edge_points:
            footprint = {(px + x, py + y) for px in range(w) for py in range(h)}
            if self._is_placement_valid(footprint, collision_mask):
                new_region_seed = FeatureNode(feature_data['name'], feature_data['type'], w, h, x, y, parent=None)
                new_region_seed.natures = feature_data.get('natures', [])
                return new_region_seed

        utils.log_message('debug',
                          f"  [Placement FAIL] Could not find space for initial region seed '{feature_data['name']}'.")
        return None