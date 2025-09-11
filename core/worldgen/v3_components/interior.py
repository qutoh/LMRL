# /core/worldgen/v3_components/interior.py

import random
from typing import Generator, Tuple, Callable

import numpy as np

from core.common import utils
from core.common.game_state import GenerationState

from .feature_node import FeatureNode
from .placement import Placement
from ...common.config_loader import config

MIN_FEATURE_SIZE = 3


class Interior:
    def __init__(self, placement: Placement, map_width: int, map_height: int):
        self.placement = placement
        self.map_width = map_width
        self.map_height = map_height

    def finalize_placements(self, all_branches: list[FeatureNode], gen_state: GenerationState) -> Generator[
        None, None, None]:
        utils.log_message('debug', "[PEGv3 Interior] Finalizing interior feature placement...")
        all_nodes = [node for branch in all_branches for node in branch.get_all_nodes_in_branch()]
        for parent_node in all_nodes:
            if not parent_node.interior_features_to_place:
                continue
            features_to_place_this_node = list(parent_node.interior_features_to_place)
            for feature_data in features_to_place_this_node:
                self._place_or_adapt_feature(feature_data, parent_node, all_branches)
                yield

    def _place_or_adapt_feature(self, feature_data: dict, parent: FeatureNode, all_branches: list[FeatureNode]):
        size_map = {'large': (3, 3), 'medium': (2, 2), 'small': (1, 1)}
        size_tiers = ['large', 'medium', 'small']
        initial_tier = feature_data.get('size_tier', 'medium')
        start_index = size_tiers.index(initial_tier) if initial_tier in size_tiers else 1
        feature_def = config.features.get(feature_data.get('type'), {})

        for i in range(start_index, len(size_tiers)):
            tier = size_tiers[i]
            w, h = size_map[tier]

            if feature_def.get('feature_type') != 'INTERACTABLE':
                w = max(w, MIN_FEATURE_SIZE)
                h = max(h, MIN_FEATURE_SIZE)

            if placement := self._find_placement((w, h), parent, feature_def):
                x, y = placement
                feature_data['bounding_box'] = (x, y, x + w, y + h)
                parent.placed_interior_features.append(feature_data)
                return

    def _find_placement(self, size: Tuple[int, int], parent: FeatureNode, feature_def: dict) -> Tuple[int, int] | None:
        w, h = size
        px, py, pw, ph = parent.get_rect()

        interior_w, interior_h = pw - 2, ph - 2
        if w > interior_w or h > interior_h: return None

        possible_positions = []
        if interior_w >= w and interior_h >= h:
            for x_offset in range(interior_w - w + 1):
                for y_offset in range(interior_h - h + 1):
                    possible_positions.append((px + 1 + x_offset, py + 1 + y_offset))

        if not possible_positions: return None
        random.shuffle(possible_positions)

        temp_grid = self.placement._get_temp_grid(all_branches)
        collision_mask = temp_grid != self.placement.void_space_index

        for x, y in possible_positions:
            footprint = {(ix, iy) for ix in range(x, x + w) for iy in range(y, y + h)}
            if self.placement._is_placement_valid(footprint, collision_mask):
                return x, y
        return None