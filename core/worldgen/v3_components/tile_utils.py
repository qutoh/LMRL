# /core/worldgen/v3_components/tile_utils.py

from typing import List, Callable
import numpy as np

from .feature_node import FeatureNode
from ...common.config_loader import config


def calculate_and_apply_tile_overrides(all_branches: List[FeatureNode], get_temp_grid_func: Callable[[List[FeatureNode]], np.ndarray]):
    """
    Iterates through all pathing features after final placement and calculates necessary tile replacements
    based on the 'intersects_ok' rules. This should be called after all geometric operations (like jitter) are complete.

    Args:
        all_branches: The final list of all root FeatureNode branches.
        get_temp_grid_func: A function that can generate a grid representing the current state of the map.
    """
    all_nodes = [node for branch in all_branches for node in branch.get_all_nodes_in_branch()]
    path_nodes = [node for node in all_nodes if config.features.get(node.feature_type, {}).get('placement_strategy') == 'PATHING']

    if not path_nodes:
        return

    # Create a single grid representing the final state of all non-path features
    non_path_branches = [branch for branch in all_branches if branch not in path_nodes]
    base_grid = get_temp_grid_func(non_path_branches)
    reverse_tile_map = config.tile_type_map_reverse

    for node in path_nodes:
        feature_def = config.features.get(node.feature_type, {})
        if not node.path_coords:
            continue

        tile_overrides = {}
        for x, y in node.path_coords:
            if not (0 <= y < base_grid.shape[0] and 0 <= x < base_grid.shape[1]):
                continue

            existing_tile_index = base_grid[y, x]
            existing_tile_name = reverse_tile_map.get(existing_tile_index)
            if not existing_tile_name or existing_tile_name == "VOID_SPACE":
                continue

            for rule in feature_def.get('intersects_ok', []):
                if rule.get('type') == existing_tile_name and 'replaces_with_floor' in rule:
                    tile_overrides[(x, y)] = rule['replaces_with_floor']
                    break  # First matching rule wins

        if tile_overrides:
            node.tile_overrides = tile_overrides