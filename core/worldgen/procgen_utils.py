# /core/worldgen/procgen_utils.py

import re
from collections import deque
from typing import List, Set, Tuple
import numpy as np


def find_contiguous_regions(grid: np.ndarray, traversable_indices: set) -> List[Set[Tuple[int, int]]]:
    """
    Finds all contiguous regions of specific tile types in a grid using a flood fill (BFS) algorithm.

    Args:
        grid: A 2D NumPy array of integer tile type indices.
        traversable_indices: A set of integer indices that the flood fill is allowed to traverse.

    Returns:
        A list of sets, where each set contains the (x, y) coordinates of a single contiguous region.
    """
    if not traversable_indices:
        return []

    height, width = grid.shape
    visited = np.zeros_like(grid, dtype=bool)
    regions = []

    for y in range(height):
        for x in range(width):
            if not visited[y, x] and grid[y, x] in traversable_indices:
                # Start a new flood fill for a new region
                new_region = set()
                q = deque([(x, y)])
                visited[y, x] = True
                new_region.add((x, y))

                while q:
                    cx, cy = q.popleft()

                    # Check neighbors
                    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        nx, ny = cx + dx, cy + dy

                        if 0 <= nx < width and 0 <= ny < height and not visited[ny, nx] and grid[ny, nx] in traversable_indices:
                            visited[ny, nx] = True
                            q.append((nx, ny))
                            new_region.add((nx, ny))
                regions.append(new_region)

    return regions


def get_perimeter_faces(bounding_box: tuple) -> dict:
    """Calculates the coordinates of the four faces just outside a bounding box."""
    x1, y1, x2, y2 = bounding_box
    return {
        'north': [(x, y1 - 1) for x in range(x1, x2)],
        'south': [(x, y2) for x in range(x1, x2)],
        'west': [(x1 - 1, y) for y in range(y1, y2)],
        'east': [(x2, y) for y in range(y1, y2)],
    }


def check_for_overlap(new_box: tuple, placed_features: dict, ignore_tag: str = None) -> bool:
    """Checks if a new bounding box intersects with any existing ones."""
    nx1, ny1, nx2, ny2 = new_box
    for tag, feature in placed_features.items():
        if tag == ignore_tag:
            continue
        px1, py1, px2, py2 = feature['bounding_box']
        # Standard AABB collision detection logic
        if not (nx2 <= px1 or nx1 >= px2 or ny2 <= py1 or ny1 >= py2):
            return True
    return False


def parse_dimensions_from_text(text: str) -> tuple[int | str, int | str, str] | None:
    """
    Parses natural language dimensions into tile sizes or semantic tags.
    Returns (width, height, size_tier), where width/height can be int or str.
    """
    if not text: return None
    text_lower = text.lower()

    size_tier = 'medium'
    if 'small' in text_lower: size_tier = 'small'
    if 'large' in text_lower: size_tier = 'large'

    # Handle explicit tile dimensions for interactables
    if 'tile' in text_lower:
        numbers = [int(n) for n in re.findall(r"(\d+)", text)]
        if len(numbers) == 2:
            return numbers[0], numbers[1], 'small'  # Treat explicit tiles as small tier
        elif len(numbers) == 1:
            return numbers[0], numbers[0], 'small'

    if 'small' in text_lower: return 'small', 'small', size_tier
    if 'medium' in text_lower: return 'medium', 'medium', size_tier
    if 'large' in text_lower: return 'large', 'large', size_tier

    shape = "rectangle"
    if "diameter" in text_lower or "circle" in text_lower:
        shape = "circle"

    numbers = [float(n) for n in re.findall(r"(\d*\.?\d+)", text)]
    if not numbers: return None

    if "meter" in text_lower or "m" in text.strip().split()[-1]:
        numbers = [n * 3.28084 for n in numbers]

    tiles = [max(1, round(n / 5)) for n in numbers]

    if shape == "circle": return tiles[0], tiles[0], size_tier
    if len(tiles) == 1: return tiles[0], tiles[0], size_tier
    return tiles[0], tiles[1], size_tier


def parse_value_and_reason(raw_text: str) -> tuple[str, str]:
    """
    Splits raw LLM output into a value and a justification based on a keyword.
    Returns (value, reason).
    """
    if not raw_text:
        return "", ""

    # Use a case-insensitive search for the reason keyword
    match = re.search(r'\s*REASON:\s*(.*)', raw_text, re.IGNORECASE)

    if match:
        reason = match.group(1).strip()
        # The value is everything before the match
        value = raw_text[:match.start()].strip()
        return value, reason
    else:
        # If no reason keyword is found, the whole text is the value
        return raw_text.strip(), ""


def format_feature_for_validation(data: dict, reasons: dict) -> str:
    """Formats feature data and reasons into a human-readable string for prompts."""
    lines = []
    # Define the order and formatting for each key
    key_order = ['name', 'description', 'type', 'dimensions_tiles', 'passable', 'flooring']

    for key in key_order:
        if key in data:
            value = data[key]
            reason = reasons.get(key, "No reason provided.")

            # Special formatting for dimensions tuple
            if key == 'dimensions_tiles' and isinstance(value, tuple):
                value_str = f"({value[0]}, {value[1]})"
            else:
                value_str = str(value)

            lines.append(f"- {key}: {value_str} (REASON: {reason})")

    return "\n".join(lines)


def get_tangent_and_normal_dims(is_horizontal_face: bool, dimensions: tuple[int, int]) -> tuple[int, int]:
    """Given a face orientation, returns the tangent (parallel) and normal (perpendicular) dimensions."""
    width, height = dimensions
    if is_horizontal_face:
        return width, height
    else:
        return height, width


class LayoutSolver:
    """A stateless utility for solving the 1D packing problem of feature placement."""

    def _solve_single_face(self, face_length: int, children_to_fit: list[dict], is_horizontal: bool) -> list[
                                                                                                            dict] | None:
        if not children_to_fit:
            return []

        dim_index = 0 if is_horizontal else 1
        children_copies = [c.copy() for c in children_to_fit]

        total_required_space = sum(c['dimensions_tiles'][dim_index] for c in children_copies)

        if total_required_space <= face_length:
            return children_copies

        oversize_amount = total_required_space - face_length
        shrinkable_children = sorted(
            [c for c in children_copies if
             c['dimensions_tiles'][dim_index] > c.get('min_dimensions_tiles', [1, 1])[dim_index]],
            key=lambda c: c['dimensions_tiles'][dim_index],
            reverse=True
        )

        while oversize_amount > 0 and shrinkable_children:
            child_to_shrink = shrinkable_children.pop(0)

            new_dims = list(child_to_shrink['dimensions_tiles'])
            new_dims[dim_index] -= 1
            child_to_shrink['dimensions_tiles'] = tuple(new_dims)
            oversize_amount -= 1

            if new_dims[dim_index] > child_to_shrink.get('min_dimensions_tiles', [1, 1])[dim_index]:
                shrinkable_children.append(child_to_shrink)
                shrinkable_children.sort(key=lambda c: c['dimensions_tiles'][dim_index], reverse=True)

        return children_copies if oversize_amount <= 0 else None

    def solve_for_face(self, face_length: int, children_on_face: list[dict], is_horizontal_face: bool) -> list[
                                                                                                              dict] | None:
        dim_index = 0 if is_horizontal_face else 1

        processed_children = []
        for child in children_on_face:
            child_copy = child.copy()
            if child_copy['type'] == 'GENERIC_PORTAL':
                tangent, normal = get_tangent_and_normal_dims(is_horizontal_face, child_copy['dimensions_tiles'])
                if normal > tangent:
                    normal = tangent
                    if is_horizontal_face:
                        child_copy['dimensions_tiles'] = (tangent, normal)
                    else:
                        child_copy['dimensions_tiles'] = (normal, tangent)
            processed_children.append(child_copy)

        sorted_children = sorted(
            processed_children,
            key=lambda c: c['dimensions_tiles'][dim_index],
            reverse=True
        )

        for i in range(len(sorted_children), 0, -1):
            subset_to_try = sorted_children[:i]
            solution = self._solve_single_face(face_length, subset_to_try, is_horizontal_face)
            if solution:
                return solution

        return None


class UnionFind:
    """A data structure to track connectivity between feature branches."""

    def __init__(self, elements):
        self.parent = {element: element for element in elements}
        self.num_sets = len(elements)

    def find(self, i):
        if self.parent[i] == i:
            return i
        # Path compression
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