# /core/worldgen/v3_components/shape_utils.py

from typing import Set, Tuple
import numpy as np


def generate_ellipse_footprint(width: int, height: int) -> Set[Tuple[int, int]]:
    """Generates a set of relative (x, y) coordinates for a filled ellipse, corrected for a discrete grid."""
    if width <= 0 or height <= 0:
        return set()

    footprint = set()
    # Correct center calculation for a 0-indexed discrete grid
    center_x = (width - 1) / 2.0
    center_y = (height - 1) / 2.0
    # Radii should still be based on the full dimension
    radius_x = width / 2.0
    radius_y = height / 2.0

    # To avoid division by zero for 1x1 features
    if radius_x == 0: radius_x = 0.5
    if radius_y == 0: radius_y = 0.5

    for y in range(height):
        for x in range(width):
            # Equation for a point within an ellipse
            if ((x - center_x) / radius_x) ** 2 + ((y - center_y) / radius_y) ** 2 <= 1:
                footprint.add((x, y))

    return footprint


def generate_rectangle_footprint(width: int, height: int) -> Set[Tuple[int, int]]:
    """Generates a set of relative (x, y) coordinates for a filled rectangle."""
    return {(x, y) for x in range(width) for y in range(height)}