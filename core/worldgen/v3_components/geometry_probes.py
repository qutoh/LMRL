# /core/worldgen/v3_components/geometry_probes.py

from typing import Set, Tuple, Dict, Callable, List, Optional
import random

from .feature_node import FeatureNode
from ...common.config_loader import config

MIN_FEATURE_SIZE = 3


def find_potential_connection_points(footprint: Set[Tuple[int, int]]) -> Dict[Tuple[int, int], Dict]:
    """
    Analyzes a feature's footprint to find all potential connection points, including "perfect"
    flat edges and "reconcilable" near-misses like bracketed corners.

    Args:
        footprint: A set of relative (x, y) coordinates for the feature's shape.

    Returns:
        A dictionary where keys are the (x, y) coordinates of potential connection points,
        and values are dictionaries describing the connection type and any required
        reconciliation action.
    """
    if not footprint:
        return {}

    connection_points = {}
    border = {(x, y) for x, y in footprint if
              any((x + dx, y + dy) not in footprint for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)])}

    for x, y in border:
        # --- Kernel 1: Perfect Flat Edge ---
        # Horizontal check: Is there a border tile to the left and right?
        if (x - 1, y) in border and (x + 1, y) in border:
            # Is the space 'in front' (north or south) empty?
            if (x, y - 1) not in footprint or (x, y + 1) not in footprint:
                connection_points[(x, y)] = {'type': 'PERFECT', 'reconciliation': None}
                continue  # A perfect point can't also be a reconcilable one
        # Vertical check: Is there a border tile above and below?
        if (x, y - 1) in border and (x, y + 1) in border:
            # Is the space 'in front' (east or west) empty?
            if (x - 1, y) not in footprint or (x + 1, y) not in footprint:
                connection_points[(x, y)] = {'type': 'PERFECT', 'reconciliation': None}
                continue

        # --- Kernel 2: Recessed Point / Bracket Reconciliation ---
        # Check for a 'U' shape pointing north
        if (x - 1, y) in border and (x + 1, y) in border and (x - 1, y - 1) in footprint and (x + 1,
                                                                                              y - 1) in footprint and (
                x, y - 1) not in footprint:
            connection_points[(x, y)] = {'type': 'RECESSED', 'reconciliation': {'extrude': (x, y - 1)}}
            continue
        # Check for a 'U' shape pointing south
        if (x - 1, y) in border and (x + 1, y) in border and (x - 1, y + 1) in footprint and (x + 1,
                                                                                              y + 1) in footprint and (
                x, y + 1) not in footprint:
            connection_points[(x, y)] = {'type': 'RECESSED', 'reconciliation': {'extrude': (x, y + 1)}}
            continue
        # Check for a 'C' shape pointing west
        if (x, y - 1) in border and (x, y + 1) in border and (x - 1, y - 1) in footprint and (x - 1,
                                                                                              y + 1) in footprint and (
                x - 1, y) not in footprint:
            connection_points[(x, y)] = {'type': 'RECESSED', 'reconciliation': {'extrude': (x - 1, y)}}
            continue
        # Check for a 'C' shape pointing east
        if (x, y - 1) in border and (x, y + 1) in border and (x + 1, y - 1) in footprint and (x + 1,
                                                                                              y + 1) in footprint and (
                x + 1, y) not in footprint:
            connection_points[(x, y)] = {'type': 'RECESSED', 'reconciliation': {'extrude': (x + 1, y)}}
            continue

    return connection_points


def probe_parent_for_any_placement(parent_node: FeatureNode, all_branches: List[FeatureNode],
                                   is_placement_valid_func: Callable, get_temp_grid_func: Callable) -> bool:
    """
    A minimal, fast check to see if *any* subfeature could conceivably be placed on a parent.
    It uses a generic, minimal path feature as the probe.
    """
    # Use a generic path feature as the "can anything at all attach here?" probe
    clearance = 1
    start_points = [p for n in parent_node.get_all_nodes_in_branch() for p in
                    get_valid_connection_points_for_probe(n, clearance, all_branches, get_temp_grid_func)]

    if not start_points:
        return False

    # Check for connection to ANY other branch. If none, check for connection to map border.
    other_branches = [b for b in all_branches if b.get_root() is not parent_node.get_root()]
    if other_branches:
        end_points = [p for b in other_branches for n in b.get_all_nodes_in_branch() for p in
                      get_valid_connection_points_for_probe(n, clearance, all_branches, get_temp_grid_func)]
        if start_points and end_points:
            # A simple adjacency check is enough for a probe
            start_set = set(start_points)
            for ex, ey in end_points:
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    if (ex + dx, ey + dy) in start_set:
                        return True
    return True  # If there are no other branches, it's not blocked from spawning a new one


def probe_general_placement_for_size_tier(size_tier: str, all_branches: List[FeatureNode],
                                          is_placement_valid_func: Callable, get_temp_grid_func: Callable) -> bool:
    """
    A fast, general probe to see if there is likely space anywhere on the map
    to place a new feature of a given size tier.
    """
    size_map = {'large': (8, 8), 'medium': (5, 5), 'small': (MIN_FEATURE_SIZE, MIN_FEATURE_SIZE)}
    probe_w, probe_h = size_map.get(size_tier, (MIN_FEATURE_SIZE, MIN_FEATURE_SIZE))

    probe_node = FeatureNode("size_probe", "PROBE", probe_w, probe_h)
    probe_connection_points = find_potential_connection_points(probe_node.footprint)
    if not probe_connection_points: return False  # Should not happen for a simple rectangle

    for parent_node in all_branches:
        if parent_node.is_blocked:
            continue

        parent_connection_points = find_potential_connection_points(parent_node.footprint)
        if not parent_connection_points:
            continue

        base_collision_mask = get_temp_grid_func(all_branches, exclude_node=parent_node) != -1

        for px, py in parent_connection_points.keys():
            for cx, cy in probe_connection_points.keys():
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    abs_px, abs_py = parent_node.current_x + px, parent_node.current_y + py
                    origin_x = abs_px + dx - cx
                    origin_y = abs_py + dy - cy

                    footprint_abs = {(x + origin_x, y + origin_y) for x, y in probe_node.footprint}

                    if is_placement_valid_func(footprint_abs, base_collision_mask):
                        if footprint_abs.isdisjoint(parent_node.get_absolute_footprint()):
                            return True
    return False


def get_valid_connection_points_for_probe(node: FeatureNode, clearance: int, all_branches: List[FeatureNode],
                                          get_temp_grid_func: Callable) -> List[Tuple[int, int]]:
    """Helper for probing, basically a slimmed down version of Pathing's function."""
    temp_grid = get_temp_grid_func(all_branches, exclude_node=node)
    collision_mask = temp_grid != -1  # Assuming -1 is void space index

    perimeter = set()
    footprint = node.get_absolute_footprint()
    if not footprint: return []
    for x, y in footprint:
        for nx, ny in [(x, y - 1), (x, y + 1), (x - 1, y), (x + 1, y)]:
            p = (nx, ny)
            if p not in footprint:
                if 0 <= p[1] < temp_grid.shape[0] and 0 <= p[0] < temp_grid.shape[1]:
                    perimeter.add(p)

    valid_points = [p for p in perimeter if not collision_mask[p[1], p[0]]]
    return valid_points