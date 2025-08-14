# /core/worldgen/v3_components/map_ops.py

import random
import numpy as np
from typing import Callable, List, Dict, Tuple

from .feature_node import FeatureNode
from core.common import utils
from core.common.config_loader import config

# --- Jitter Constants ---
MIN_FEATURE_SIZE = 3
JITTER_ITERATIONS = 50
JITTER_MAX_TRANSLATION = 2
JITTER_MAX_SCALE_DELTA = 0.1


class MapOps:
    """
    Handles organic map operations like layout relaxation ("jittering").
    """

    def __init__(self, map_width: int, map_height: int):
        self.map_width = map_width
        self.map_height = map_height

    def _propose_transform_for_branch(self, feature: FeatureNode, dx: int, dy: int, scale: float,
                                      parent_proposal: Tuple[int, int, int, int] | None) -> Dict[
                                                                                                FeatureNode, Tuple[
                                                                                                    int, int, int, int]] | None:
        """
        Recursively calculates a new proposed state for a feature and its entire sub-branch
        based on a single transformation (dx, dy, scale).
        """
        prop_w = max(MIN_FEATURE_SIZE, int(feature.current_abs_width * scale))
        prop_h = max(MIN_FEATURE_SIZE, int(feature.current_abs_height * scale))

        if parent_proposal is None:
            prop_x = feature.current_x + dx
            prop_y = feature.current_y + dy
        else:
            ppx, ppy, ppw, pph = parent_proposal
            orig_parent_rect = feature.parent.get_rect()
            parent_scale_w = ppw / orig_parent_rect[2] if orig_parent_rect[2] > 0 else 1.0
            parent_scale_h = pph / orig_parent_rect[3] if orig_parent_rect[3] > 0 else 1.0
            y_offset = int((feature.current_y - orig_parent_rect[1]) * parent_scale_h)
            x_offset = int((feature.current_x - orig_parent_rect[0]) * parent_scale_w)

            face = feature.anchor_to_parent_face
            if face == 'E':
                prop_x, prop_y = ppx + ppw, ppy + y_offset
                if prop_y < ppy or (prop_y + prop_h) > (ppy + pph): return None
            elif face == 'W':
                prop_x, prop_y = ppx - prop_w, ppy + y_offset
                if prop_y < ppy or (prop_y + prop_h) > (ppy + pph): return None
            elif face == 'S':
                prop_x, prop_y = ppx + x_offset, ppy + pph
                if prop_x < ppx or (prop_x + prop_w) > (ppx + ppw): return None
            elif face == 'N':
                prop_x, prop_y = ppx + x_offset, ppy - prop_h
                if prop_x < ppx or (prop_x + prop_w) > (ppx + ppw): return None
            else:
                prop_x, prop_y = feature.current_x + dx, feature.current_y + dy

        my_proposal = (prop_x, prop_y, prop_w, prop_h)
        all_proposals = {feature: my_proposal}

        for sub in feature.subfeatures:
            child_proposals = self._propose_transform_for_branch(sub, dx, dy, scale, my_proposal)
            if not child_proposals: return None
            all_proposals.update(child_proposals)

        return all_proposals

    def _is_proposal_valid(self, proposal: dict, other_nodes: list) -> bool:
        """Checks if an entire branch proposal is valid against bounds, other branches, and itself."""
        temp_grid_others = np.zeros((self.map_height, self.map_width), dtype=int)
        for other_node in other_nodes:
            ox, oy, ow, oh = other_node.get_rect()
            temp_grid_others[oy:oy + oh, ox:ox + ow] = 1

        proposal_grid = np.zeros((self.map_height, self.map_width), dtype=int)

        for node, (px, py, pw, ph) in proposal.items():
            if not (px >= 1 and py >= 1 and (px + pw) <= (self.map_width - 1) and (py + ph) <= (
                    self.map_height - 1)):
                return False
            if np.any(temp_grid_others[py:py + ph, px:px + pw] == 1):
                return False
            if np.any(proposal_grid[py:py + ph, px:px + pw] == 1):
                return False
            proposal_grid[py:py + ph, px:px + pw] = 1

        return True

    def apply_jitter(self, initial_feature_branches: List[FeatureNode], on_iteration_end: Callable):
        """Randomly move and scale feature branches to create a more organic layout."""
        utils.log_message('debug', "[PEGv3 Jitter] Applying jitter...")

        num_features = len(
            [node for branch in initial_feature_branches for node in branch.get_all_nodes_in_branch()])
        base_bias = config.settings.get("PEG_V3_JITTER_SCALE_BIAS", 0.02)
        density_factor = config.settings.get("PEG_V3_JITTER_DENSITY_FACTOR", 0.005)
        total_bias = base_bias + (num_features * density_factor)

        for _ in range(JITTER_ITERATIONS):
            if not initial_feature_branches:
                break

            branch_to_jitter = random.choice(initial_feature_branches)
            other_nodes = []
            for branch in initial_feature_branches:
                if branch is not branch_to_jitter:
                    other_nodes.extend(branch.get_all_nodes_in_branch())

            dx = random.randint(-JITTER_MAX_TRANSLATION, JITTER_MAX_TRANSLATION)
            dy = random.randint(-JITTER_MAX_TRANSLATION, JITTER_MAX_TRANSLATION)

            random_component = random.uniform(-JITTER_MAX_SCALE_DELTA, JITTER_MAX_SCALE_DELTA)
            final_delta = random_component + total_bias
            clamped_delta = max(-JITTER_MAX_SCALE_DELTA * 2, min(JITTER_MAX_SCALE_DELTA * 2, final_delta))
            scale = 1.0 + clamped_delta

            proposal = self._propose_transform_for_branch(branch_to_jitter, dx, dy, scale, None)

            if not proposal: continue

            if self._is_proposal_valid(proposal, other_nodes):
                for node, (px, py, pw, ph) in proposal.items():
                    node.current_x, node.current_y = px, py
                    node.current_abs_width, node.current_abs_height = pw, ph
                on_iteration_end()