# /core/worldgen/v3_components/feature_node.py

from typing import Optional, List, Tuple


class FeatureNode:
    """A class to represent a feature during the v3 generation process."""

    def __init__(self, name: str, feature_type: str, rel_w: float, rel_h: float, abs_w: int, abs_h: int, x: int = 0,
                 y: int = 0, parent: Optional['FeatureNode'] = None, anchor_face: Optional[str] = None):
        self.name = name
        self.feature_type = feature_type
        self.relative_width = rel_w
        self.relative_height = rel_h
        self.current_x = x
        self.current_y = y
        self.current_abs_width = abs_w
        self.current_abs_height = abs_h
        self.parent = parent
        self.anchor_to_parent_face = anchor_face
        self.subfeatures: List['FeatureNode'] = []
        self.narrative_log: str = ""  # Only used for root nodes of a branch
        self.interior_features_to_place: List[dict] = []
        self.placed_interior_features: List[dict] = []

        # --- Growth Algorithm Attributes ---
        self.target_area: float = 0.0
        self.size_tier: str = "medium"
        self.is_stuck: bool = False
        self.growth_multiplier: int = 1
        self.target_aspect_ratio: Tuple[int, int] = (1, 1)


    def get_all_nodes_in_branch(self) -> List['FeatureNode']:
        """Recursively gets all nodes in this feature's branch, including itself."""
        nodes = [self]
        for sub in self.subfeatures:
            nodes.extend(sub.get_all_nodes_in_branch())
        return nodes

    def get_rect(self) -> Tuple[int, int, int, int]:
        """Returns the current absolute position and size as a tuple (x, y, w, h)."""
        return self.current_x, self.current_y, self.current_abs_width, self.current_abs_height