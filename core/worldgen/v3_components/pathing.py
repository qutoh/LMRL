# /core/worldgen/v3_components/pathing.py

import math
import random
from typing import List, Dict, Tuple, Optional, Set, Callable
from collections import deque

import numpy as np
import tcod
from scipy.ndimage import distance_transform_edt, gaussian_filter, binary_dilation

from core.common import utils
from core.common.game_state import GameMap
from . import geometry_probes
from .feature_node import FeatureNode
from ...common.config_loader import config


class Pathing:
    def __init__(self, game_map: GameMap, get_temp_grid_func: Callable, get_door_mask_func: Callable):
        self.game_map = game_map
        self.map_height = game_map.height
        self.map_width = game_map.width
        self.void_space_index = config.tile_type_map.get("VOID_SPACE", -1)
        self._get_temp_grid = get_temp_grid_func
        # This parameter is no longer used, but kept for signature consistency until all refs are updated.
        self._get_door_placement_mask_legacy = get_door_mask_func

    def find_first_valid_connection(self, start_points: List[Tuple[int, int]], end_points: List[Tuple[int, int]],
                                    clearance: int, all_branches: List[FeatureNode],
                                    path_feature_def: Optional[Dict] = None) -> Optional[
        Tuple[Tuple[int, int], Tuple[int, int]]]:
        if not start_points or not end_points: return None

        start_points_set = set(start_points)
        end_points_set = set(end_points)
        if start_points_set == end_points_set: return None

        clearance_mask = self._create_clearance_mask(clearance, all_branches, path_feature_def)

        shuffled_starts = list(start_points)
        random.shuffle(shuffled_starts)
        culled_starts = set()

        for start_pos in shuffled_starts:
            if start_pos in culled_starts: continue

            q = deque([start_pos])
            visited_this_flood = {start_pos}

            found_target = None
            while q:
                cx, cy = q.popleft()

                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nx, ny = cx + dx, cy + dy
                    neighbor = (nx, ny)

                    if neighbor in end_points_set:
                        found_target = neighbor
                        break

                    if 0 <= ny < self.map_height and 0 <= nx < self.map_width and neighbor not in visited_this_flood and not \
                            clearance_mask[ny, nx]:
                        visited_this_flood.add(neighbor)
                        q.append(neighbor)
                if found_target:
                    break

            if found_target:
                return start_pos, found_target
            else:
                for p in visited_this_flood:
                    if p in start_points_set:
                        culled_starts.add(p)
        return None, None

    def reconcile_connection(self, parent: FeatureNode, child: FeatureNode, all_branches: List[FeatureNode],
                             proposed_footprints: Dict[FeatureNode, Set[Tuple[int, int]]]) -> Optional[Dict]:
        parent_def = config.features.get(parent.feature_type, {})
        child_def = config.features.get(child.feature_type, {})
        if parent_def.get('feature_type') == 'BARRIER' or child_def.get('feature_type') == 'BARRIER': return None

        parent_abs_footprint = proposed_footprints[parent]
        child_abs_footprint = proposed_footprints[child]

        parent_rel_footprint = {(x - parent.current_x, y - parent.current_y) for x, y in parent_abs_footprint}
        child_rel_footprint = {(x - child.current_x, y - child.current_y) for x, y in child_abs_footprint}

        parent_points = geometry_probes.find_potential_connection_points(parent_rel_footprint)
        child_points = geometry_probes.find_potential_connection_points(child_rel_footprint)

        if not parent_points or not child_points: return None

        # Check for simple adjacency first
        for (px, py), _ in parent_points.items():
            for (cx, cy), _ in child_points.items():
                abs_px, abs_py = px + parent.current_x, py + parent.current_y
                abs_cx, abs_cy = cx + child.current_x, cy + child.current_y
                if abs(abs_px - abs_cx) + abs(abs_py - abs_cy) == 1:
                    return {'status': 'connected'}

        # If no simple adjacency, try to form a short hallway
        obstacle_grid = self._get_temp_grid(all_branches) != self.void_space_index

        # Add the features themselves to the obstacle grid for pathfinding between them
        for x, y in parent_abs_footprint:
            if 0 <= y < self.map_height and 0 <= x < self.map_width:
                obstacle_grid[y, x] = True
        for x, y in child_abs_footprint:
            if 0 <= y < self.map_height and 0 <= x < self.map_width:
                obstacle_grid[y, x] = True

        parent_coords = list(parent_points.keys())
        child_coords = list(child_points.keys())

        for _ in range(20):  # Try a few random pairs
            px_rel, py_rel = random.choice(parent_coords)
            cx_rel, cy_rel = random.choice(child_coords)

            start = (px_rel + parent.current_x, py_rel + parent.current_y)
            end = (cx_rel + child.current_x, cy_rel + child.current_y)

            path = tcod.path.AStar(obstacle_grid.T).get_path(start[0], start[1], end[0], end[1])

            if path and 1 < len(path) <= 4:  # Found a short path (2-3 tiles long)
                hallway_tiles = [tuple(p) for p in path[1:-1]]  # Exclude start/end points
                return {'status': 'connected', 'hallway': hallway_tiles}

        return None

    def get_border_coordinates_for_direction(self, direction: str) -> List[Tuple[int, int]]:
        direction = direction.upper()
        wedges = {'EAST': (315, 45), 'NORTHEAST': (45, 90), 'NORTH': (90, 135), 'NORTHWEST': (135, 180),
                  'WEST': (180, 225), 'SOUTHWEST': (225, 270), 'SOUTH': (270, 315), 'SOUTHEAST': (315, 360)}
        if direction not in wedges: return []
        center_x, center_y = self.map_width / 2.0, self.map_height / 2.0
        min_angle, max_angle = wedges[direction]
        border_coords = []
        for x in range(self.map_width):
            border_coords.append((x, 0))
            border_coords.append((x, self.map_height - 1))
        for y in range(1, self.map_height - 1):
            border_coords.append((0, y))
            border_coords.append((self.map_width - 1, y))
        valid_points = []
        for x, y in set(border_coords):
            angle = (math.degrees(math.atan2(y - center_y, x - center_x)) + 360) % 360
            if min_angle > max_angle:
                if angle >= min_angle or angle < max_angle: valid_points.append((x, y))
            elif min_angle <= angle < max_angle:
                valid_points.append((x, y))
        return valid_points

    def _build_organic_modifiers(self, all_branches: List[FeatureNode], path_feature_def: dict) -> np.ndarray:
        modifier_grid = np.ones((self.map_height, self.map_width), dtype=np.float32)
        if modifiers := path_feature_def.get('pathfinding_modifiers'):
            feature_type_coords = {}
            all_nodes = [node for branch in all_branches for node in branch.get_all_nodes_in_branch()]
            for node in all_nodes:
                if node.feature_type not in feature_type_coords:
                    feature_type_coords[node.feature_type] = set()
                abs_footprint = node.get_absolute_footprint()
                for i, j in abs_footprint:
                    if 0 <= i < self.map_width and 0 <= j < self.map_height:
                        feature_type_coords[node.feature_type].add((i, j))
            for rule in modifiers:
                target_type = rule.get('type')
                influence = rule.get('influence', 1.0)
                decay = rule.get('decay', 0.1)
                if not target_type or influence == 1.0: continue
                source_map = np.ones((self.map_height, self.map_width), dtype=bool)
                if target_coords := feature_type_coords.get(target_type):
                    if not target_coords: continue
                    rows, cols = zip(*target_coords)
                    source_map[list(cols), list(rows)] = False
                distance_grid = distance_transform_edt(source_map)
                influence_grid = 1 + (influence - 1) * np.exp(-decay * distance_grid)
                modifier_grid *= influence_grid
            modifier_grid = gaussian_filter(modifier_grid, sigma=1.5)
        turbulence = path_feature_def.get('turbulence', 0.0)
        if turbulence > 0.0:
            noise = tcod.noise.Noise(dimensions=2, algorithm=tcod.noise.Algorithm.PERLIN,
                                     implementation=tcod.noise.Implementation.TURBULENCE, hurst=1.0,
                                     lacunarity=min(max(0.0, turbulence), 10.0), octaves=2, seed=None)
            scale = path_feature_def.get('turbulence_scale', 0.1)
            grid_y, grid_x = np.ogrid[0:self.map_height, 0:self.map_width]
            samples = noise.sample_ogrid([grid_x * scale, grid_y * scale])
            turbulence_modifier = 1.0 + (samples.T * turbulence)
            modifier_grid *= turbulence_modifier
        return gaussian_filter(modifier_grid, sigma=0.5)

    def find_path_with_clearance(self, start_coords: Tuple[int, int], end_coords: Tuple[int, int], clearance: int,
                                 all_branches: List[FeatureNode], path_feature_def: dict) -> Optional[
        List[Tuple[int, int]]]:
        clearance_mask = self._create_clearance_mask(clearance, all_branches, path_feature_def)
        terrain_grid = self._get_temp_grid(all_branches)

        cost = np.full((self.map_height, self.map_width), 1.0, dtype=np.float32)
        for rule in path_feature_def.get('intersects_ok', []):
            intersect_type_name = rule.get('type')
            cost_mod = rule.get('cost_mod', 1.0)
            if intersect_type_name in config.tile_type_map:
                intersect_type_index = config.tile_type_map[intersect_type_name]
                base_cost = config.tile_types.get(intersect_type_name, {}).get('movement_cost', 1.0)
                cost[terrain_grid == intersect_type_index] = base_cost * cost_mod

        organic_modifiers = self._build_organic_modifiers(all_branches, path_feature_def)
        cost *= organic_modifiers
        cost[clearance_mask] = np.inf

        astar = tcod.path.AStar(cost=cost.T, diagonal=0)
        path_xy = astar.get_path(start_coords[0], start_coords[1], end_coords[0], end_coords[1])
        return path_xy if path_xy else None

    def _create_clearance_mask(self, clearance: int, all_branches: List[FeatureNode],
                               path_feature_def: Optional[Dict] = None) -> np.ndarray:
        terrain_grid = self._get_temp_grid(all_branches)

        if path_feature_def:
            allowed_indices = {self.void_space_index}
            for rule in path_feature_def.get('intersects_ok', []):
                if intersect_type_name := rule.get('type'):
                    if intersect_type_name in config.tile_type_map:
                        allowed_indices.add(config.tile_type_map[intersect_type_name])
            obstacle_grid = ~np.isin(terrain_grid, list(allowed_indices))
        else:
            obstacle_grid = terrain_grid != self.void_space_index

        if clearance > 0:
            structure = np.ones((clearance * 2 + 1, clearance * 2 + 1), dtype=bool)
            return binary_dilation(obstacle_grid, structure=structure)
        return obstacle_grid

    def get_valid_connection_points(self, node: FeatureNode, clearance: int, all_branches: List[FeatureNode]) -> List[
        Tuple[int, int]]:
        temp_grid = self._get_temp_grid(all_branches, exclude_node=node)
        collision_mask = temp_grid != self.void_space_index
        node_perimeter = self._get_node_exterior_perimeter(node)

        valid_points = [p for p in node_perimeter if not collision_mask[p[1], p[0]]]
        return valid_points

    def create_all_connections(self, initial_feature_branches: List[FeatureNode]) -> Tuple[
        List[Dict], List[Dict], List[Dict]]:
        internal_connections = self._connect_internal_branch_doors(initial_feature_branches)
        door_placements = []
        hallways = []

        for conn in internal_connections:
            node_a, node_b = conn.get('node_a'), conn.get('node_b')
            coords = conn.get('door_coords')

            if conn.get('hallway'):
                hallways.append({
                    'tiles': conn['hallway'],
                    'type_a': node_a.feature_type,
                    'type_b': node_b.feature_type
                })

            if not node_a or not node_b or not coords or len(coords) < 2: continue

            node_a_def = config.features.get(node_a.feature_type, {})
            node_b_def = config.features.get(node_b.feature_type, {})

            if any(d.get('feature_type') == 'BARRIER' for d in [node_a_def, node_b_def]): continue

            is_a_portal = node_a_def.get('feature_type') == 'PORTAL'
            is_b_portal = node_b_def.get('feature_type') == 'PORTAL'

            if not is_a_portal and not is_b_portal:
                if door_tile := node_a_def.get('door_tile_type'): door_placements.append(
                    {'pos': coords[0], 'type': door_tile})
                if door_tile := node_b_def.get('door_tile_type'): door_placements.append(
                    {'pos': coords[1], 'type': door_tile})

        return internal_connections, door_placements, hallways

    def _connect_internal_branch_doors(self, initial_feature_branches: List[FeatureNode]) -> List[Dict]:
        connections = []
        all_nodes = [node for branch in initial_feature_branches for node in branch.get_all_nodes_in_branch()]
        for parent in all_nodes:
            for child in parent.subfeatures:
                reconciliation = self.reconcile_connection(parent, child, all_nodes, {
                    parent: parent.get_absolute_footprint(),
                    child: child.get_absolute_footprint()
                })
                if reconciliation:
                    # Find closest points for door placement even if hallway was made
                    parent_points = list(
                        geometry_probes.find_potential_connection_points(parent.footprint).keys())
                    child_points = list(geometry_probes.find_potential_connection_points(child.footprint).keys())
                    if not parent_points or not child_points: continue

                    min_dist = float('inf')
                    best_pair = (None, None)
                    for px_rel, py_rel in parent_points:
                        for cx_rel, cy_rel in child_points:
                            px_abs, py_abs = px_rel + parent.current_x, py_rel + parent.current_y
                            cx_abs, cy_abs = cx_rel + child.current_x, cy_rel + child.current_y
                            dist = math.hypot(px_abs - cx_abs, py_abs - cy_abs)
                            if dist < min_dist:
                                min_dist = dist
                                best_pair = ((px_abs, py_abs), (cx_abs, cy_abs))

                    if best_pair[0]:
                        conn_data = {'node_a': parent, 'node_b': child, 'position': best_pair[0],
                                     'door_coords': best_pair}
                        if 'hallway' in reconciliation:
                            conn_data['hallway'] = reconciliation['hallway']
                        connections.append(conn_data)
        return connections

    def _get_node_exterior_perimeter(self, node: FeatureNode) -> set:
        perimeter = set()
        footprint = node.get_absolute_footprint()
        if not footprint: return perimeter
        for x, y in footprint:
            for nx, ny in [(x, y - 1), (x, y + 1), (x - 1, y), (x + 1, y)]:
                p = (nx, ny)
                if p not in footprint and 0 <= p[1] < self.map_height and 0 <= p[0] < self.map_width:
                    perimeter.add(p)
        return perimeter