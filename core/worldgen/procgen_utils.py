# /core/worldgen/procgen_utils.py

import re
from collections import deque
from typing import List, Set, Tuple, Any, Optional, Callable, Generator

import numpy as np

from ..common import utils, config_loader
from ..common.localization import loc_raw
from .v3_components.feature_node import FeatureNode
from .v3_components.geometry_probes import find_potential_connection_points
from .semantic_search import SemanticSearch

_RELATIONSHIPS_DATA_CACHE = None


def get_relationships_data():
    """Loads and caches the relationships data from localization files."""
    global _RELATIONSHIPS_DATA_CACHE
    if _RELATIONSHIPS_DATA_CACHE is None:
        _RELATIONSHIPS_DATA_CACHE = loc_raw("relationships_data", default={'hierarchical': [], 'lattice': {}})
    return _RELATIONSHIPS_DATA_CACHE


def find_contiguous_regions(grid: np.ndarray, traversable_indices: set) -> List[Set[Tuple[int, int]]]:
    """
    Finds all contiguous regions of specific tile types in a grid using a flood fill (BFS) algorithm.
    """
    if not traversable_indices:
        return []
    height, width = grid.shape
    visited = np.zeros_like(grid, dtype=bool)
    regions = []
    for y in range(height):
        for x in range(width):
            if not visited[y, x] and grid[y, x] in traversable_indices:
                new_region = set()
                q = deque([(x, y)])
                visited[y, x] = True
                new_region.add((x, y))
                while q:
                    cx, cy = q.popleft()
                    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        nx, ny = cx + dx, cy + dy
                        if 0 <= nx < width and 0 <= ny < height and not visited[ny, nx] and grid[
                            ny, nx] in traversable_indices:
                            visited[ny, nx] = True
                            q.append((nx, ny))
                            new_region.add((nx, ny))
                regions.append(new_region)
    return regions


def _parse_single_dimension(val: Any) -> Tuple[int | str, int]:
    """Parses a single dimension value (str, int, etc.) into a usable value and a numeric tier for comparison."""
    s = str(val).lower()
    if 'large' in s: return 'large', 3
    if 'medium' in s: return 'medium', 2
    if 'small' in s: return 'small', 1

    numbers = [float(n) for n in re.findall(r"(\d*\.?\d+)", s)]
    if not numbers: return 'medium', 2  # Default fallback

    if "meter" in s or "m" in s.strip().split()[-1]:
        numbers = [n * 3.28084 for n in numbers]

    tiles = max(1, round(numbers[0] / 5))

    if tiles > 15: return tiles, 3
    if tiles > 7: return tiles, 2
    return tiles, 1


def parse_dimensions_from_text(text: Any) -> tuple[int | str, int | str, str] | None:
    """
    Parses natural language or structured dimensions into tile sizes or semantic tags.
    Returns (width, height, size_tier).
    """
    if not text: return None

    if isinstance(text, dict):
        width_str = text.get('width', 'medium')
        height_str = text.get('height', 'medium')

        width_val, width_tier = _parse_single_dimension(width_str)
        height_val, height_tier = _parse_single_dimension(height_str)

        final_tier_num = max(width_tier, height_tier)
        final_tier_str = 'large' if final_tier_num == 3 else 'medium' if final_tier_num == 2 else 'small'

        return width_val, height_val, final_tier_str

    if isinstance(text, list):
        text = " ".join(map(str, text))

    text_lower = str(text).lower()
    size_tier = 'medium'
    if 'small' in text_lower: size_tier = 'small'
    if 'large' in text_lower: size_tier = 'large'

    # Check for two-dimensional descriptions first
    match = re.search(r'(\S+)\s*(?:by|x)\s*(\S+)', text_lower)
    if match:
        width_str, height_str = match.groups()
        width_val, _ = _parse_single_dimension(width_str)
        height_val, _ = _parse_single_dimension(height_str)
        return width_val, height_val, size_tier

    # Fallback to single dimension parsing
    width_val, _ = _parse_single_dimension(text_lower)
    return width_val, width_val, size_tier


class UnionFind:
    """A data structure to track connectivity between feature branches."""

    def __init__(self, elements):
        self.parent = {element: element for element in elements}
        self.num_sets = len(elements)

    def find(self, i):
        if self.parent[i] == i:
            return i
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i, j):
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            self.parent[root_j] = root_i
            self.num_sets -= 1
            return True
        return False

    def connected(self, i, j):
        return self.find(i) == self.find(j)


def get_door_placement_mask_for_pathing(footprint: Set[Tuple[int, int]], collision_mask: np.ndarray) -> Set[
    Tuple[int, int]]:
    points = find_potential_connection_points(footprint)
    return {coord for coord, data in points.items() if data['type'] == 'PERFECT'}


def resolve_parent_node(chosen_parent_name: str, valid_parent_nodes: List[FeatureNode],
                        semantic_search: SemanticSearch) -> Optional[FeatureNode]:
    if not chosen_parent_name or 'none' in chosen_parent_name.lower(): return None
    resolved = next((n for n in valid_parent_nodes if n.name.lower() == chosen_parent_name.lower().strip()), None)
    if resolved: return resolved
    best_match = semantic_search.find_best_match(chosen_parent_name, [n.name for n in valid_parent_nodes])
    return next((n for n in valid_parent_nodes if n.name == best_match), None) if best_match else None


def get_combined_feature_rules(node: FeatureNode) -> dict:
    """
    Merges pathfinding and intersection rules from a feature's base definition
    and all of its natures.
    """
    combined_rules = {
        'pathfinding_modifiers': [],
        'intersects_ok': []
    }

    # First, get rules from the base feature definition
    feature_def = config_loader.config.features.get(node.feature_type, {})
    combined_rules['pathfinding_modifiers'].extend(feature_def.get('pathfinding_modifiers', []))
    combined_rules['intersects_ok'].extend(feature_def.get('intersects_ok', []))

    # Then, accumulate rules from each nature
    for nature_name in node.natures:
        nature_def = config_loader.config.natures.get(nature_name, {})
        combined_rules['pathfinding_modifiers'].extend(nature_def.get('pathfinding_modifiers', []))
        combined_rules['intersects_ok'].extend(nature_def.get('intersects_ok', []))

    return combined_rules