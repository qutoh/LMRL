# /core/worldgen/v3_components/pathing.py

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
        # 1. Generate the terrain grid representing the current map layout.
        terrain_grid = self.placement_utils._get_temp_grid(all_branches)

        # 2. Build the A* cost grid using vectorized operations.
        cost = np.full((self.map_height, self.map_width), np.inf, dtype=np.float32)

        # Allow pathing through empty space.
        cost[terrain_grid == self.void_space_index] = 1.0

        # Allow pathing through intersectable tiles with a cost penalty.
        for rule in path_feature_def.get('intersects_ok', []):
            intersect_type_name = rule.get('type')
            cost_mod = rule.get('cost_mod', 1.0)
            if intersect_type_name in config.tile_type_map:
                intersect_type_index = config.tile_type_map[intersect_type_name]
                base_cost = config.tile_types.get(intersect_type_name, {}).get('movement_cost', 1.0)
                cost[terrain_grid == intersect_type_index] = base_cost * cost_mod

        # 3. Find a simple, 1-tile wide centerline path using the new cost grid.
        astar = tcod.path.AStar(cost=cost.T, diagonal=0)  # tcod expects (width, height)
        start_x, start_y = start_coords
        end_x, end_y = end_coords
        path_xy = astar.get_path(start_x, start_y, end_x, end_y)

        if not path_xy:
            return None

        # 4. Validate the found path for the required clearance.
        path_is_valid = True
        path_radius = clearance
        path_dim = clearance * 2 + 1
        obstacle_grid = (cost == np.inf)

        for x_center, y_center in path_xy:
            px, py = x_center - path_radius, y_center - path_radius

            if not (px >= 0 and py >= 0 and (px + path_dim) <= self.map_width and (py + path_dim) <= self.map_height):
                path_is_valid = False
                break
            # Check the obstacle grid (where cost is infinity) for collisions.
            if np.any(obstacle_grid[py:py + path_dim, px:px + path_dim]):
                path_is_valid = False
                break

        if not path_is_valid:
            #if config.settings.get('LOG_LEVEL', 'debug') == 'full':
                #view_title = f"Path Validation Failure (Clearance: {clearance})"
                #ascii_view = utils.print_ascii_cost_grid(view_title, cost.T, start_coords, end_coords)
                #utils.log_message('full', f"\n{ascii_view}")
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

        # Check North face
        for i in range(w):
            center_x, center_y = x + i, y - offset
            px, py = center_x - path_radius, center_y - path_radius
            if self.placement_utils._is_placement_valid((px, py, path_dim, path_dim), temp_grid, path_feature_def):
                valid_points.append((center_x, center_y))

        # Check South face
        for i in range(w):
            center_x, center_y = x + i, y + h - 1 + offset
            px, py = center_x - path_radius, center_y - path_radius
            if self.placement_utils._is_placement_valid((px, py, path_dim, path_dim), temp_grid, path_feature_def):
                valid_points.append((center_x, center_y))

        # Check West face
        for i in range(h):
            center_x, center_y = x - offset, y + i
            px, py = center_x - path_radius, center_y - path_radius
            if self.placement_utils._is_placement_valid((px, py, path_dim, path_dim), temp_grid, path_feature_def):
                valid_points.append((center_x, center_y))

        # Check East face
        for i in range(h):
            center_x, center_y = x + w - 1 + offset, y + i
            px, py = center_x - path_radius, center_y - path_radius
            if self.placement_utils._is_placement_valid((px, py, path_dim, path_dim), temp_grid, path_feature_def):
                valid_points.append((center_x, center_y))

        return valid_points

    def _get_connection_points_for_path(self, node: FeatureNode, clearance: int, all_branches: List[FeatureNode]) -> \
    List[Tuple[int, int]]:
        """Finds valid connection points by checking the perimeter of a path feature."""
        if not node.path_coords:
            return []

        valid_points_set = set()
        path_tiles_set = set(map(tuple, node.path_coords))  # Ensure coords are tuples
        temp_grid = self.placement_utils._get_temp_grid(all_branches, exclude_node=node)
        path_feature_def = config.features.get("GENERIC_PATH", {})

        path_radius = clearance
        path_dim = clearance * 2 + 1
        offset = clearance + 1

        # Iterate over each tile in the path to find its border.
        for x, y in path_tiles_set:
            # Check the four cardinal neighbors of the path tile.
            for nx, ny in [(x, y - 1), (x, y + 1), (x - 1, y), (x + 1, y)]:
                if (nx, ny) in path_tiles_set:
                    continue  # Neighbor is part of the path, not a border.

                if ny < y:
                    center_x, center_y = nx, ny - offset  # North border
                elif ny > y:
                    center_x, center_y = nx, ny + offset  # South border
                elif nx < x:
                    center_x, center_y = nx - offset, ny  # West border
                else:
                    center_x, center_y = nx + offset, ny  # East border

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
        else:  # Default to rectangle logic for 'rectangle' and any other shape.
            return self._get_connection_points_for_rectangle(node, clearance, all_branches)

    def create_all_connections(self, initial_feature_branches: List[FeatureNode]) -> List[Dict]:
        internal = self._connect_internal_branch_doors(initial_feature_branches)
        external = self._connect_external_branches(initial_feature_branches)
        final_internal = []
        for conn in internal:
            parent, child, pos = conn['node_a'], conn['node_b'], conn['position']
            final_internal.append({'node_a': parent, 'node_b': child, 'position': pos, 'door_coords': [pos]})
            final_internal.append({'node_a': child, 'node_b': parent, 'position': pos, 'door_coords': [pos]})
        return final_internal + external

    def _connect_internal_branch_doors(self, initial_feature_branches: List[FeatureNode]) -> List[Dict]:
        utils.log_message('debug', "[PEGv3 Pathing] Creating internal branch connections...")
        connections = []
        all_nodes = [node for branch in initial_feature_branches for node in branch.get_all_nodes_in_branch()]

        for parent in all_nodes:
            for child in parent.subfeatures:
                parent_def = config.features.get(parent.feature_type, {})
                child_def = config.features.get(child.feature_type, {})

                if parent_def.get('placement_strategy') == 'BLOCKING' or \
                        child_def.get('placement_strategy') == 'BLOCKING' or \
                        not child.anchor_to_parent_face:
                    continue

                px, py, pw, ph = parent.get_rect()
                cx, cy, cw, ch = child.get_rect()
                face = child.anchor_to_parent_face

                door_pos = None

                try:
                    if face in ('N', 'S'):
                        overlap_start, overlap_end = max(px, cx), min(px + pw, cx + cw)
                        if overlap_start < overlap_end:
                            door_x = random.randint(overlap_start, overlap_end - 1)
                            door_y = py if face == 'N' else py + ph - 1
                            door_pos = (door_x, door_y)

                    elif face in ('E', 'W'):
                        overlap_start, overlap_end = max(py, cy), min(py + ph, cy + ch)
                        if overlap_start < overlap_end:
                            door_y = random.randint(overlap_start, overlap_end - 1)
                            door_x = px if face == 'W' else px + pw - 1
                            door_pos = (door_x, door_y)

                    if door_pos:
                        connections.append({
                            'node_a': parent,
                            'node_b': child,
                            'position': door_pos,
                            'door_coords': [door_pos]
                        })

                except (ValueError, IndexError):
                    utils.log_message('debug', f"  Skipping door for '{child.name}': No valid border overlap found.")

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
        path_feature_def = config.features.get("GENERIC_PATH", {})

        for _ in range(len(connectable_roots) * 2):
            if uf.num_sets <= 1: break
            source_root, dest_root = None, None
            shuffled = random.sample(connectable_roots, len(connectable_roots))
            for i in range(len(shuffled)):
                for j in range(i + 1, len(shuffled)):
                    if not uf.connected(shuffled[i], shuffled[j]):
                        source_root, dest_root = shuffled[i], shuffled[j]
                        break
                if source_root: break
            if not source_root: break

            # --- Flood Fill Pre-computation ---
            # Define what's traversable for a path.
            traversable_indices = {self.void_space_index}
            for rule in path_feature_def.get('intersects_ok', []):
                if t_type := rule.get('type'):
                    traversable_indices.add(config.tile_type_map.get(t_type))

            # Create a grid of obstacles, excluding the two branches we want to connect.
            obstacle_branches = [b for b in initial_feature_branches if b not in (source_root, dest_root)]
            terrain_grid = self.placement_utils._get_temp_grid(obstacle_branches)

            # Find all contiguous pockets of traversable space.
            regions = find_contiguous_regions(terrain_grid, traversable_indices)

            # Find the perimeters of our source and destination branches.
            source_perimeter = {p for node in source_root.get_all_nodes_in_branch() for p in
                                self._get_node_perimeter(node)}
            dest_perimeter = {p for node in dest_root.get_all_nodes_in_branch() for p in self._get_node_perimeter(node)}

            # Find the "bridging" regions that touch both perimeters.
            bridging_regions = [r for r in regions if
                                r.intersection(source_perimeter) and r.intersection(dest_perimeter)]

            if not bridging_regions:
                continue  # No path through open/allowed space between these two branches.

            # We only care about the border points that are within these valid bridging regions.
            source_borders = list(source_perimeter.intersection(*bridging_regions))
            dest_borders = list(dest_perimeter.intersection(*bridging_regions))

            if not source_borders or not dest_borders: continue

            start_door_pos = random.choice(source_borders)
            path_found = False
            for end_door_pos in random.sample(dest_borders, len(dest_borders)):
                path = self.find_path_with_clearance(start_door_pos, end_door_pos, 0, initial_feature_branches,
                                                     path_feature_def)
                if path:
                    # Choose a random node from each branch to be the logical connection owner
                    source_node = random.choice(source_root.get_all_nodes_in_branch())
                    dest_node = random.choice(dest_root.get_all_nodes_in_branch())

                    door_coords = [path[0], path[-1]]
                    connections.append({'node_a': source_node, 'node_b': dest_node, 'position': start_door_pos,
                                        'door_coords': door_coords})
                    uf.union(source_root, dest_root)
                    path_found = True
                    break

            if not path_found:
                # If no path is found, we might want to log this, but we don't create a dead-end connection.
                pass

        return connections

    def _get_node_perimeter(self, node: FeatureNode) -> set:
        """Helper to get all perimeter coordinates for a given node."""
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