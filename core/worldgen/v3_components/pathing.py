# /core/worldgen/v3_components/pathing.py

import math
import random
from typing import List, Dict, Tuple, Optional

import numpy as np
import tcod

from core.common import utils
from core.common.game_state import GameMap
from core.worldgen.procgen_utils import UnionFind, find_contiguous_regions
from .feature_node import FeatureNode
from ...common.config_loader import config
from .placement import Placement


class Pathing:
    """
    Handles creating connections (paths, doors) between features, both within a branch
    and between separate branches.
    """

    def __init__(self, game_map: GameMap):
        self.game_map = game_map
        self.map_height = game_map.height
        self.map_width = game_map.width
        self.placement_utils = Placement(self.map_width, self.map_height)
        self.void_space_index = config.tile_type_map.get("VOID_SPACE", -1)

    def find_path_with_clearance(self, start_coords: Tuple[int, int], end_coords: Tuple[int, int], clearance: int,
                                 all_branches: List[FeatureNode], path_feature_def: dict) -> Optional[
        List[Tuple[int, int]]]:
        """
        Finds a path using a cost grid built from terrain intersection rules, then validates its clearance.
        """
        terrain_grid = self.placement_utils._get_temp_grid(all_branches)
        cost = np.full((self.map_height, self.map_width), np.inf, dtype=np.float32)
        cost[terrain_grid == self.void_space_index] = 1.0

        for rule in path_feature_def.get('intersects_ok', []):
            intersect_type_name = rule.get('type')
            cost_mod = rule.get('cost_mod', 1.0)
            if intersect_type_name in config.tile_type_map:
                intersect_type_index = config.tile_type_map[intersect_type_name]
                base_cost = config.tile_types.get(intersect_type_name, {}).get('movement_cost', 1.0)
                cost[terrain_grid == intersect_type_index] = base_cost * cost_mod

        astar = tcod.path.AStar(cost=cost.T, diagonal=0)
        start_x, start_y = start_coords
        end_x, end_y = end_coords
        path_xy = astar.get_path(start_x, start_y, end_x, end_y)

        if not path_xy:
            return None

        path_is_valid = True
        path_radius = clearance
        path_dim = clearance * 2 + 1
        obstacle_grid = (cost == np.inf)

        for x_center, y_center in path_xy:
            px, py = x_center - path_radius, y_center - path_radius
            if not (px >= 0 and py >= 0 and (px + path_dim) <= self.map_width and (py + path_dim) <= self.map_height):
                path_is_valid = False
                break
            if np.any(obstacle_grid[py:py + path_dim, px:px + path_dim]):
                path_is_valid = False
                break

        if not path_is_valid:
            return None

        return path_xy

    def _get_connection_points_for_rectangle(self, node: FeatureNode, clearance: int,
                                             all_branches: List[FeatureNode]) -> List[Tuple[int, int]]:
        """Finds valid connection points on the exterior faces of a rectangular feature."""
        valid_points = []
        x, y, w, h = node.get_rect()
        temp_grid = self.placement_utils._get_temp_grid(all_branches, exclude_node=node)
        path_feature_def = config.features.get("GENERIC_PATH", {})

        path_radius = clearance
        path_dim = clearance * 2 + 1
        offset = clearance + 1

        for i in range(w):
            center_x, center_y = x + i, y - offset
            px, py = center_x - path_radius, center_y - path_radius
            if self.placement_utils._is_placement_valid((px, py, path_dim, path_dim), temp_grid, path_feature_def):
                valid_points.append((center_x, center_y))
        for i in range(w):
            center_x, center_y = x + i, y + h - 1 + offset
            px, py = center_x - path_radius, center_y - path_radius
            if self.placement_utils._is_placement_valid((px, py, path_dim, path_dim), temp_grid, path_feature_def):
                valid_points.append((center_x, center_y))
        for i in range(h):
            center_x, center_y = x - offset, y + i
            px, py = center_x - path_radius, center_y - path_radius
            if self.placement_utils._is_placement_valid((px, py, path_dim, path_dim), temp_grid, path_feature_def):
                valid_points.append((center_x, center_y))
        for i in range(h):
            center_x, center_y = x + w - 1 + offset, y + i
            px, py = center_x - path_radius, center_y - path_radius
            if self.placement_utils._is_placement_valid((px, py, path_dim, path_dim), temp_grid, path_feature_def):
                valid_points.append((center_x, center_y))
        return valid_points

    def _get_connection_points_for_path(self, node: FeatureNode, clearance: int, all_branches: List[FeatureNode]) -> \
            List[Tuple[int, int]]:
        """Finds valid connection points by checking the perimeter of a path feature."""
        if not node.path_coords: return []
        valid_points_set = set()
        path_tiles_set = set(map(tuple, node.path_coords))
        temp_grid = self.placement_utils._get_temp_grid(all_branches, exclude_node=node)
        path_feature_def = config.features.get("GENERIC_PATH", {})
        path_radius = clearance
        path_dim = clearance * 2 + 1
        offset = clearance + 1
        for x, y in path_tiles_set:
            for nx, ny in [(x, y - 1), (x, y + 1), (x - 1, y), (x + 1, y)]:
                if (nx, ny) in path_tiles_set: continue
                if ny < y:
                    center_x, center_y = nx, ny - offset
                elif ny > y:
                    center_x, center_y = nx, ny + offset
                elif nx < x:
                    center_x, center_y = nx - offset, ny
                else:
                    center_x, center_y = nx + offset, ny
                px, py = center_x - path_radius, center_y - path_radius
                if self.placement_utils._is_placement_valid((px, py, path_dim, path_dim), temp_grid, path_feature_def):
                    valid_points_set.add((center_x, center_y))
        return list(valid_points_set)

    def get_valid_connection_points(self, node: FeatureNode, clearance: int, all_branches: List[FeatureNode]) -> List[
        Tuple[int, int]]:
        """
        Finds valid connection points on a feature's exterior, dispatching to the
        correct algorithm based on the feature's defined shape.
        """
        feature_def = config.features.get(node.feature_type, {})
        shape = feature_def.get('default_shape', 'rectangle')
        if shape == 'path':
            return self._get_connection_points_for_path(node, clearance, all_branches)
        else:
            return self._get_connection_points_for_rectangle(node, clearance, all_branches)

    def _get_contiguous_border_segments(self, border_points: set) -> List[List[Tuple[int, int]]]:
        """Groups a set of border points into contiguous line segments."""
        if not border_points: return []
        segments = []
        visited = set()
        for point in border_points:
            if point in visited: continue
            segment = []
            q = [point]
            visited.add(point)
            while q:
                current = q.pop(0)
                segment.append(current)
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx == 0 and dy == 0: continue
                        neighbor = (current[0] + dx, current[1] + dy)
                        if neighbor in border_points and neighbor not in visited:
                            visited.add(neighbor)
                            q.append(neighbor)
            segments.append(segment)
        return segments

    def _create_path_intersection_openings(self, all_branches: List[FeatureNode]) -> List[Dict]:
        """Finds where paths cross the borders of other features and creates door pairs for each contiguous segment."""
        openings = []
        all_nodes = [node for branch in all_branches for node in branch.get_all_nodes_in_branch()]
        path_nodes = [node for node in all_nodes if node.path_coords and config.features.get(node.feature_type, {}).get(
            'placement_strategy') == 'PATHING']
        other_nodes = [node for node in all_nodes if node not in path_nodes]
        other_borders = {node: self._get_node_wall_perimeter(node) for node in other_nodes}

        for path_node in path_nodes:
            path_coords_set = set(map(tuple, path_node.path_coords))
            path_border_set = self._get_node_wall_perimeter(path_node)
            for other_node, other_border_set in other_borders.items():
                intersection_points = path_coords_set.intersection(other_border_set)
                if not intersection_points: continue

                intersection_segments = self._get_contiguous_border_segments(intersection_points)
                for segment in intersection_segments:
                    mid_index = len(segment) // 2
                    crossing_point = segment[mid_index]
                    path_door_pos = min(path_border_set,
                                        key=lambda p: math.hypot(p[0] - crossing_point[0], p[1] - crossing_point[1]))

                    other_def = config.features.get(other_node.feature_type, {})
                    if other_door_tile := other_def.get('door_tile_type'):
                        openings.append({'pos': crossing_point, 'type': other_door_tile})
                    path_def = config.features.get(path_node.feature_type, {})
                    if path_door_tile := path_def.get('door_tile_type'):
                        openings.append({'pos': path_door_pos, 'type': path_door_tile})
        return openings

    def create_all_connections(self, initial_feature_branches: List[FeatureNode]) -> Tuple[
        List[Dict], List[Dict], List[Dict]]:
        """
        Creates all connections and returns the physics joints and a list of door placements.
        """
        displaced_conns, new_hallways = self._connect_displaced_subfeatures(initial_feature_branches)
        internal_connections = self._connect_internal_branch_doors(initial_feature_branches)
        external_connections = self._connect_external_branches(initial_feature_branches)
        path_intersection_openings = self._create_path_intersection_openings(initial_feature_branches)
        door_placements = []
        all_conns_for_processing = internal_connections + external_connections + displaced_conns
        for conn in all_conns_for_processing:
            node_a, node_b = conn.get('node_a'), conn.get('node_b')
            coords = conn.get('door_coords')
            if not node_a or not node_b or not coords or len(coords) < 2: continue
            node_a_def = config.features.get(node_a.feature_type, {})
            if door_tile := node_a_def.get('door_tile_type'):
                door_placements.append({'pos': coords[0], 'type': door_tile})
            node_b_def = config.features.get(node_b.feature_type, {})
            if door_tile := node_b_def.get('door_tile_type'):
                door_placements.append({'pos': coords[1], 'type': door_tile})
        door_placements.extend(path_intersection_openings)
        final_connections_for_physics = internal_connections + external_connections + displaced_conns
        return final_connections_for_physics, door_placements, new_hallways

    def _reconnect_detached_paths(self, all_branches: List[FeatureNode], existing_connections: List[Dict]) -> Tuple[
        List[Dict], List[Dict]]:
        """Finds paths that are no longer connected to their intended start/end points and bridges the gap."""
        reconnections = []
        new_hallways = []
        path_connections = [c for c in existing_connections if c['node_a'].path_coords or c['node_b'].path_coords]

        for conn in path_connections:
            node_a, node_b = conn['node_a'], conn['node_b']
            door_a, door_b = conn['door_coords'][0], conn['door_coords'][1]

            # Check if door_a is adjacent to node_b's perimeter, and vice-versa
            is_a_connected = any(
                math.hypot(door_a[0] - x, door_a[1] - y) < 1.5 for x, y in self._get_node_wall_perimeter(node_b))
            is_b_connected = any(
                math.hypot(door_b[0] - x, door_b[1] - y) < 1.5 for x, y in self._get_node_wall_perimeter(node_a))

            if is_a_connected and is_b_connected: continue  # Path is fine

            # Bridge the gap
            start_pos = door_a
            end_pos = min(list(self._get_node_exterior_perimeter(node_b)),
                          key=lambda p: math.hypot(p[0] - start_pos[0], p[1] - start_pos[1]))

            path_centerline = self.find_path_with_clearance(start_pos, end_pos, 1, all_branches, {})
            if path_centerline:
                path_coords = set()
                for x, y in path_centerline:
                    for i in range(-1, 2):
                        for j in range(-1, 2):
                            path_coords.add((x + i, y + j))
                new_hallways.append({
                    "path_coords": list(path_coords),
                    "source_a_type": node_a.feature_type, "source_b_type": node_b.feature_type
                })
                # We can reuse the original door positions as they are on the correct features
                reconnections.append(
                    {'node_a': node_a, 'node_b': node_b, 'position': door_a, 'door_coords': [door_a, door_b]})
        return reconnections, new_hallways

    def _connect_displaced_subfeatures(self, all_branches: List[FeatureNode]) -> Tuple[List[Dict], List[Dict]]:
        connections, new_hallway_data = [], []
        all_nodes = [node for branch in all_branches for node in branch.get_all_nodes_in_branch()]
        for parent in all_nodes:
            for child in parent.subfeatures:
                px, py, pw, ph = parent.get_rect()
                cx, cy, cw, ch = child.get_rect()
                is_adjacent = ((px + pw == cx or cx + cw == px) and (py < cy + ch and py + ph > cy)) or \
                              ((py + ph == cy or cy + ch == py) and (px < cx + cw and px + pw > cx))
                if is_adjacent: continue
                parent_ext_perimeter = list(self._get_node_exterior_perimeter(parent))
                child_ext_perimeter = list(self._get_node_exterior_perimeter(child))
                if not parent_ext_perimeter or not child_ext_perimeter: continue

                point_pairs = sorted(
                    ((p1, p2) for p1 in parent_ext_perimeter for p2 in child_ext_perimeter),
                    key=lambda points: math.hypot(points[0][0] - points[1][0], points[0][1] - points[1][1])
                )
                path_found = False
                for start_pos, end_pos in point_pairs[:10]:
                    path_centerline = self.find_path_with_clearance(start_pos, end_pos, 1, all_branches, {})
                    if path_centerline:
                        path_coords = set()
                        for x, y in path_centerline:
                            for i in range(-1, 2):
                                for j in range(-1, 2):
                                    path_coords.add((x + i, y + j))
                        new_hallway_data.append({
                            "path_coords": list(path_coords),
                            "source_a_type": parent.feature_type, "source_b_type": child.feature_type
                        })
                        parent_wall_points, child_wall_points = list(self._get_node_wall_perimeter(parent)), list(
                            self._get_node_wall_perimeter(child))
                        if not parent_wall_points or not child_wall_points: continue
                        parent_door = min(parent_wall_points, key=lambda p: math.hypot(p[0] - path_centerline[0][0],
                                                                                       p[1] - path_centerline[0][1]))
                        child_door = min(child_wall_points, key=lambda p: math.hypot(p[0] - path_centerline[-1][0],
                                                                                     p[1] - path_centerline[-1][1]))
                        connections.append(
                            {'node_a': parent, 'node_b': child, 'position': parent_door,
                             'door_coords': [parent_door, child_door]}
                        )
                        path_found = True
                        break
                if not path_found:
                    utils.log_message('debug',
                                      f"  [HALLWAY FAIL] Could not connect displaced subfeature '{child.name}'.")
        return connections, new_hallway_data

    def _connect_internal_branch_doors(self, initial_feature_branches: List[FeatureNode]) -> List[Dict]:
        connections = []
        all_nodes = [node for branch in initial_feature_branches for node in branch.get_all_nodes_in_branch()]
        for parent in all_nodes:
            for child in parent.subfeatures:
                if config.features.get(parent.feature_type, {}).get('placement_strategy') == 'BLOCKING' or \
                        config.features.get(child.feature_type, {}).get('placement_strategy') == 'BLOCKING' or \
                        not child.anchor_to_parent_face:
                    continue
                px, py, pw, ph = parent.get_rect()
                cx, cy, cw, ch = child.get_rect()
                face = child.anchor_to_parent_face
                shared_border = []
                if face == 'N' and (py == cy + ch):
                    overlap_start, overlap_end = max(px, cx), min(px + pw, cx + cw)
                    shared_border = [(x, py) for x in range(overlap_start, overlap_end)]
                elif face == 'S' and (py + ph == cy):
                    overlap_start, overlap_end = max(px, cx), min(px + pw, cx + cw)
                    shared_border = [(x, py + ph - 1) for x in range(overlap_start, overlap_end)]
                elif face == 'W' and (px == cx + cw):
                    overlap_start, overlap_end = max(py, cy), min(py + ph, cy + ch)
                    shared_border = [(px, y) for y in range(overlap_start, overlap_end)]
                elif face == 'E' and (px + pw == cx):
                    overlap_start, overlap_end = max(py, cy), min(py + ph, cy + ch)
                    shared_border = [(px + pw - 1, y) for y in range(overlap_start, overlap_end)]

                if len(shared_border) > 2:
                    door_pos_parent = random.choice(shared_border[1:-1])
                    if door_pos_parent:
                        if face == 'N':
                            dp_child = (door_pos_parent[0], door_pos_parent[1] - 1)
                        elif face == 'S':
                            dp_child = (door_pos_parent[0], door_pos_parent[1] + 1)
                        elif face == 'W':
                            dp_child = (door_pos_parent[0] - 1, door_pos_parent[1])
                        else:
                            dp_child = (door_pos_parent[0] + 1, door_pos_parent[1])
                        connections.append({'node_a': parent, 'node_b': child, 'position': door_pos_parent,
                                            'door_coords': [door_pos_parent, dp_child]})
                elif shared_border:
                    door_pos_parent = random.choice(shared_border)
                    if face == 'N':
                        dp_child = (door_pos_parent[0], door_pos_parent[1] - 1)
                    elif face == 'S':
                        dp_child = (door_pos_parent[0], door_pos_parent[1] + 1)
                    elif face == 'W':
                        dp_child = (door_pos_parent[0] - 1, door_pos_parent[1])
                    else:
                        dp_child = (door_pos_parent[0] + 1, door_pos_parent[1])
                    connections.append({'node_a': parent, 'node_b': child, 'position': door_pos_parent,
                                        'door_coords': [door_pos_parent, dp_child]})
        return connections

    def _connect_external_branches(self, initial_feature_branches: List[FeatureNode]) -> List[Dict]:
        utils.log_message('debug', "[PEGv3 Pathing] Creating external branch connections (A*)...")
        connectable_roots = [
            b for b in initial_feature_branches
            if config.features.get(b.feature_type, {}).get('placement_strategy') != 'BLOCKING'
        ]
        if len(connectable_roots) < 2: return []
        connections = []
        uf = UnionFind(connectable_roots)

        for _ in range(len(connectable_roots) - 1):
            if uf.num_sets <= 1: break
            source_root, dest_root = None, None
            shuffled_roots = random.sample(connectable_roots, len(connectable_roots))
            for i in range(len(shuffled_roots)):
                for j in range(i + 1, len(shuffled_roots)):
                    if not uf.connected(shuffled_roots[i], shuffled_roots[j]):
                        source_root, dest_root = shuffled_roots[i], shuffled_roots[j]
                        break
                if source_root: break
            if not source_root: break

            source_nodes = source_root.get_all_nodes_in_branch()
            dest_nodes = dest_root.get_all_nodes_in_branch()
            random.shuffle(source_nodes)
            random.shuffle(dest_nodes)

            path_found = False
            for s_node in source_nodes:
                for d_node in dest_nodes:
                    s_perimeter = list(self._get_node_exterior_perimeter(s_node))
                    d_perimeter = list(self._get_node_exterior_perimeter(d_node))
                    if not s_perimeter or not d_perimeter: continue

                    start_pos, end_pos = min(
                        ((p1, p2) for p1 in s_perimeter for p2 in d_perimeter),
                        key=lambda points: math.hypot(points[0][0] - points[1][0], points[0][1] - points[1][1])
                    )

                    path = self.find_path_with_clearance(start_pos, end_pos, 0, initial_feature_branches, {})
                    if path:
                        start_path_pos, end_path_pos = path[0], path[-1]
                        s_wall_points, d_wall_points = list(self._get_node_wall_perimeter(s_node)), list(
                            self._get_node_wall_perimeter(d_node))
                        if not s_wall_points or not d_wall_points: continue

                        start_door = min(s_wall_points,
                                         key=lambda p: math.hypot(p[0] - start_path_pos[0], p[1] - start_path_pos[1]))
                        end_door = min(d_wall_points,
                                       key=lambda p: math.hypot(p[0] - end_path_pos[0], p[1] - end_path_pos[1]))

                        connections.append({'node_a': s_node, 'node_b': d_node, 'position': start_pos,
                                            'door_coords': [start_door, end_door]})
                        uf.union(source_root, dest_root)
                        path_found = True
                        break
                if path_found: break
        return connections

    def _get_node_wall_perimeter(self, node: FeatureNode) -> set:
        """Helper to get all wall coordinates for a node or its entire branch."""
        wall_points = set()
        for sub_node in node.get_all_nodes_in_branch():
            if sub_node.path_coords:
                path_set = set(map(tuple, sub_node.path_coords))
                for x, y in path_set:
                    is_border = False
                    for nx in range(x - 1, x + 2):
                        for ny in range(y - 1, y + 2):
                            if (nx, ny) != (x, y) and (nx, ny) not in path_set:
                                is_border = True
                                break
                        if is_border: break
                    if is_border:
                        wall_points.add((x, y))
            else:
                x, y, w, h = sub_node.get_rect()
                for i in range(x, x + w):
                    wall_points.add((i, y))
                    wall_points.add((i, y + h - 1))
                for i in range(y + 1, y + h - 1):
                    wall_points.add((x, i))
                    wall_points.add((x + w - 1, i))
        return wall_points

    def _get_node_exterior_perimeter(self, node: FeatureNode) -> set:
        """Helper to get all coordinates just outside a given node."""
        perimeter = set()
        if node.path_coords:
            path_set = set(map(tuple, node.path_coords))
            for x, y in path_set:
                for nx, ny in [(x, y - 1), (x, y + 1), (x - 1, y), (x + 1, y)]:
                    if (nx, ny) not in path_set and 0 <= nx < self.map_width and 0 <= ny < self.map_height:
                        perimeter.add((nx, ny))
        else:
            x, y, w, h = node.get_rect()
            for i in range(w):
                perimeter.add((x + i, y - 1))
                perimeter.add((x + i, y + h))
            for i in range(h):
                perimeter.add((x - 1, y + i))
                perimeter.add((x + w, y + i))
        return perimeter