# /core/worldgen/v3_components/feature_node.py

from typing import Optional, List, Tuple, Set, Dict
from . import shape_utils
from ...common.config_loader import config


class FeatureNode:
    def __init__(self, name: str, feature_type: str, abs_w: int, abs_h: int, x: int = 0, y: int = 0,
                 parent: Optional['FeatureNode'] = None):
        self.name = name
        self.feature_type = feature_type

        self.current_x = x
        self.current_y = y

        self.interior_footprint: Set[Tuple[int, int]] = self._generate_initial_footprint(abs_w, abs_h)
        self.footprint: Set[Tuple[int, int]] = self.interior_footprint.copy()

        feature_def = config.features.get(self.feature_type, {})
        border_thickness = feature_def.get('border_thickness', 0)
        if border_thickness > 0:
            current_body = self.footprint
            for _ in range(border_thickness):
                new_perimeter = {(px + dx, py + dy) for px, py in current_body for dx in [-1, 0, 1] for dy in [-1, 0, 1]
                                 if (px + dx, py + dy) not in current_body}
                self.footprint.update(new_perimeter)
                current_body = self.footprint.copy()

        if self.footprint:
            min_rx = min(p[0] for p in self.footprint)
            min_ry = min(p[1] for p in self.footprint)
            self.footprint = {(rx - min_rx, ry - min_ry) for rx, ry in self.footprint}
            self.interior_footprint = {(rx - min_rx, ry - min_ry) for rx, ry in self.interior_footprint}

        self.current_abs_width = 0
        self.current_abs_height = 0
        self.update_bounding_box_from_footprint()

        self.parent = parent
        self.subfeatures: List['FeatureNode'] = []

        self.narrative_log: str = ""
        self.interior_features_to_place: List[dict] = []
        self.placed_interior_features: List[dict] = []
        self.sentence_count: int = 0

        self.target_area: float = 0.0
        self.size_tier: str = "medium"
        self.is_stuck: bool = False
        self.growth_multiplier: int = 1
        self.target_aspect_ratio: Tuple[int, int] = (1, 1)
        self.target_growth_rect: Optional[Tuple[int, int, int, int]] = None
        self.relocation_history: set = set()

        self.path_coords: Optional[List[Tuple[int, int]]] = None
        self.tile_overrides: Optional[Dict[Tuple[int, int], str]] = None

        self.jitter_budget: int = 0
        self.erosion_budget: int = 0
        self.organic_op_budget: int = 0
        self.is_blocked: bool = False

    def get_root(self) -> 'FeatureNode':
        """Traverses up the parent chain to find the root node of this branch."""
        node = self
        while node.parent:
            node = node.parent
        return node

    def _generate_initial_footprint(self, width: int, height: int) -> Set[Tuple[int, int]]:
        feature_def = config.features.get(self.feature_type, {})
        shape = feature_def.get('shape', 'rectangle')
        if shape == 'ellipse':
            return shape_utils.generate_ellipse_footprint(width, height)
        return shape_utils.generate_rectangle_footprint(width, height)

    def get_all_nodes_in_branch(self) -> List['FeatureNode']:
        """Recursively gets all nodes in this feature's branch, including itself."""
        nodes = [self]
        for sub in self.subfeatures:
            nodes.extend(sub.get_all_nodes_in_branch())
        return nodes

    def get_absolute_footprint(self) -> Set[Tuple[int, int]]:
        if self.path_coords: return set(self.path_coords)
        if not self.footprint: return set()
        return {(rx + self.current_x, ry + self.current_y) for rx, ry in self.footprint}

    def get_absolute_interior_footprint(self) -> Set[Tuple[int, int]]:
        if self.path_coords: return set(self.path_coords)
        if not self.interior_footprint: return set()
        return {(rx + self.current_x, ry + self.current_y) for rx, ry in self.interior_footprint}

    def get_absolute_border_footprint(self) -> Set[Tuple[int, int]]:
        return self.get_absolute_footprint() - self.get_absolute_interior_footprint()

    def update_bounding_box_from_footprint(self):
        if not self.footprint:
            self.current_abs_width = 0
            self.current_abs_height = 0
            return
        min_rx = min(p[0] for p in self.footprint)
        max_rx = max(p[0] for p in self.footprint)
        min_ry = min(p[1] for p in self.footprint)
        max_ry = max(p[1] for p in self.footprint)
        self.current_abs_width = (max_rx - min_rx) + 1
        self.current_abs_height = (max_ry - min_ry) + 1

    def get_rect(self) -> Tuple[int, int, int, int]:
        """
        Returns the feature's bounding box. For standard features, it's the stored
        rectangle. For path features, it's calculated from the path coordinates.
        """
        if self.path_coords:
            if not self.path_coords: return 0, 0, 0, 0
            min_x = min(c[0] for c in self.path_coords)
            max_x = max(c[0] for c in self.path_coords)
            min_y = min(c[1] for c in self.path_coords)
            max_y = max(c[1] for c in self.path_coords)
            return min_x, min_y, (max_x - min_x) + 1, (max_y - min_y) + 1
        if not self.footprint: return self.current_x, self.current_y, 0, 0
        return self.current_x, self.current_y, self.current_abs_width, self.current_abs_height