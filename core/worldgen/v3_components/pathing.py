# /core/worldgen/v3_components/pathing.py

import random
from typing import List, Dict, Tuple

import numpy as np
import tcod

from .feature_node import FeatureNode
from core.common import utils
from core.common.game_state import GameMap
from core.worldgen.procgen_utils import UnionFind
from ...common.config_loader import config
from ...components import game_functions


class Pathing:
    """
    Handles creating connections (paths, doors) between features, both within a branch
    and between separate branches.
    """

    def __init__(self, game_map: GameMap):
        self.game_map = game_map
        self.map_height = game_map.height
        self.map_width = game_map.width

    def _get_feature_borders(self, node: FeatureNode) -> List[Tuple[int, int]]:
        """Identifies the coordinates of a feature's border tiles."""
        feature_def = config.features.get(node.feature_type, {})
        border_thickness = feature_def.get('border_thickness', 1)
        if border_thickness == 0:
            return []

        x, y, w, h = node.get_rect()
        border_coords = []
        # Top and Bottom borders
        for i in range(w):
            border_coords.append((x + i, y))
            border_coords.append((x + i, y + h - 1))
        # Left and Right borders
        for i in range(1, h - 1):
            border_coords.append((x, y + i))
            border_coords.append((x + w - 1, y + i))
        return border_coords

    def create_all_connections(self, initial_feature_branches: List[FeatureNode]) -> List[Dict]:
        """Orchestrates internal and external connection creation."""
        internal = self._connect_internal_branch_doors(initial_feature_branches)
        external = self._connect_external_branches(initial_feature_branches)

        # Post-process internal connections to be reciprocal
        final_internal = []
        for conn in internal:
            parent, child, pos = conn['node_a'], conn['node_b'], conn['position']
            final_internal.append({'node_a': parent, 'node_b': child, 'position': pos, 'door_coords': [pos]})
            final_internal.append({'node_a': child, 'node_b': parent, 'position': pos, 'door_coords': [pos]})

        return final_internal + external

    def _connect_internal_branch_doors(self, initial_feature_branches: List[FeatureNode]) -> List[Dict]:
        """Connects parent features to anchored children along their shared face."""
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
                        overlap_start, overlap_end = max(px, cx) + 1, min(px + pw, cx + cw) - 1
                        if overlap_start < overlap_end:
                            door_x = random.randint(overlap_start, overlap_end - 1)
                            door_y = py - 1 if face == 'N' else py + ph
                            # We place the door on the child's wall for consistency
                            door_pos = (door_x, cy + ch - 1 if face == 'N' else cy)

                    elif face in ('E', 'W'):
                        overlap_start, overlap_end = max(py, cy) + 1, min(py + ph, cy + ch) - 1
                        if overlap_start < overlap_end:
                            door_y = random.randint(overlap_start, overlap_end - 1)
                            door_x = px - 1 if face == 'W' else px + pw
                            # We place the door on the child's wall
                            door_pos = (cx + cw - 1 if face == 'W' else cx, door_y)

                    if door_pos:
                        connections.append({
                            'node_a': parent,
                            'node_b': child,
                            'position': door_pos,
                            'door_coords': [door_pos]  # This will be made reciprocal later
                        })

                except (ValueError, IndexError):
                    utils.log_message('debug', f"  Skipping door for '{child.name}': No valid border overlap found.")

        return connections

    def _connect_external_branches(self, initial_feature_branches: List[FeatureNode]) -> List[Dict]:
        """Connects disconnected branches using the centralized A* pathfinder."""
        utils.log_message('debug', "[PEGv3 Pathing] Creating external branch connections (A*)...")
        connectable_roots = [
            b for b in initial_feature_branches
            if config.features.get(b.feature_type, {}).get('placement_strategy') != 'BLOCKING'
        ]

        if len(connectable_roots) < 2: return []

        connections = []
        uf = UnionFind(connectable_roots)

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

            source_node = random.choice([n for n in source_root.get_all_nodes_in_branch() if
                                         config.features.get(n.feature_type, {}).get('feature_type') != 'BARRIER'])
            dest_node = random.choice([n for n in dest_root.get_all_nodes_in_branch() if
                                       config.features.get(n.feature_type, {}).get('feature_type') != 'BARRIER'])
            if not source_node or not dest_node: continue

            source_borders = self._get_feature_borders(source_node)
            dest_borders = self._get_feature_borders(dest_node)
            if not source_borders or not dest_borders: continue

            start_door_pos = random.choice(source_borders)
            path_found = False

            for end_door_pos in random.sample(dest_borders, len(dest_borders)):
                path = game_functions.find_path(self.game_map, start_door_pos, end_door_pos)
                if path:
                    door_coords = [start_door_pos, end_door_pos]
                    connections.append({'node_a': source_node, 'node_b': dest_node, 'position': start_door_pos,
                                        'door_coords': door_coords})
                    uf.union(source_root, dest_root)
                    path_found = True
                    break

            if not path_found:
                connections.append({'node_a': source_node, 'node_b': None, 'position': start_door_pos,
                                    'door_coords': [start_door_pos]})

        return connections