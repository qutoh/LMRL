# /core/worldgen/v3_components/map_ops.py

import random
from typing import Callable, List, Dict, Tuple, Set, Optional, Generator
from collections import deque

import numpy as np

from core.common import utils
from core.common.config_loader import config
from . import geometry_probes
from .feature_node import FeatureNode
from .pathing import Pathing
from .placement import Placement

JITTER_SCALING_FACTOR = 0.75
EROSION_SCALING_FACTOR = 0.3
ORGANIC_OP_SCALING_FACTOR = 0.4
JITTER_MAX_TRANSLATION = 1
MIN_FEATURE_SIZE = 3


class MapOps:
    """
    Handles atomic, organic map operations like erosion and layout relaxation ("jittering").
    Each operation is proposed and then validated against feature promises before being committed.
    """

    def __init__(self, map_width: int, map_height: int, pathing: 'Pathing', placement: 'Placement'):
        self.map_width = map_width
        self.map_height = map_height
        self.pathing = pathing
        self.placement = placement

    def _find_main_component(self, footprint: Set[Tuple[int, int]]) -> Set[Tuple[int, int]]:
        """Finds the largest contiguous component in a footprint, discarding any smaller islands."""
        if not footprint:
            return set()

        max_component = set()
        visited = set()
        for start_node in footprint:
            if start_node not in visited:
                component = set()
                q = deque([start_node])
                visited.add(start_node)
                component.add(start_node)
                while q:
                    x, y = q.popleft()
                    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        neighbor = (x + dx, y + dy)
                        if neighbor in footprint and neighbor not in visited:
                            visited.add(neighbor)
                            component.add(neighbor)
                            q.append(neighbor)
                if len(component) > len(max_component):
                    max_component = component
        return max_component

    def _is_proposal_valid(self, node: FeatureNode, proposal_x: int, proposal_y: int,
                           proposal_footprint: Set[Tuple[int, int]], all_branches: List[FeatureNode]) -> bool:
        """Validates a single proposed transformation for a single node."""
        if not proposal_footprint:
            return False

        temp_grid_others = self.pathing._get_temp_grid(all_branches, exclude_node=node)
        absolute_proposal_footprint = {(rx + proposal_x, ry + proposal_y) for rx, ry in proposal_footprint}

        for px, py in absolute_proposal_footprint:
            if not (0 <= px < self.map_width and 0 <= py < self.map_height) or \
                    temp_grid_others[py, px] != self.pathing.void_space_index:
                return False

        original_x, original_y, original_footprint = node.current_x, node.current_y, node.footprint
        node.current_x, node.current_y, node.footprint = proposal_x, proposal_y, proposal_footprint

        if node.parent:
            if not self.pathing.reconcile_connection(node.parent, node, all_branches,
                                                     {node: node.get_absolute_footprint(),
                                                      node.parent: node.parent.get_absolute_footprint()}):
                node.current_x, node.current_y, node.footprint = original_x, original_y, original_footprint
                return False

        for child in node.subfeatures:
            if not self.pathing.reconcile_connection(node, child, all_branches,
                                                     {node: node.get_absolute_footprint(),
                                                      child: child.get_absolute_footprint()}):
                node.current_x, node.current_y, node.footprint = original_x, original_y, original_footprint
                return False

        node.current_x, node.current_y, node.footprint = original_x, original_y, original_footprint
        return True

    def _propose_and_reconcile_branch_move(self, node: FeatureNode, dx: int, dy: int,
                                           branch_nodes: Set[FeatureNode], all_branches: List[FeatureNode],
                                           proposed_moves: Dict[FeatureNode, Tuple[int, int]]) -> bool:
        """
        Recursively attempts to move a node and its children. If a child move fails,
        it attempts a small secondary jitter to reconcile the position.
        This is an all-or-nothing operation managed by the proposed_moves dictionary.
        """
        # Base case: if we have already calculated a move for this node in this transaction, it's valid.
        if node in proposed_moves:
            return True

        # Create a collision mask that excludes the entire branch being moved.
        other_branches = [b for b in all_branches if b not in branch_nodes]
        collision_mask = self.pathing._get_temp_grid(other_branches) != self.pathing.void_space_index

        # --- Attempt 1: Direct Translation ---
        proposed_x, proposed_y = node.current_x + dx, node.current_y + dy
        proposed_footprint = {(rx + proposed_x, ry + proposed_y) for rx, ry in node.footprint}

        is_valid_move = self.placement._is_placement_valid(proposed_footprint, collision_mask)

        if is_valid_move:
            # Check connection to parent if it has one and is part of the move
            if node.parent and node.parent in branch_nodes:
                parent_new_x, parent_new_y = proposed_moves[node.parent]

                # Create temporary nodes to represent the proposed new state for reconciliation check
                temp_parent = FeatureNode("temp", node.parent.feature_type, 0, 0, x=parent_new_x, y=parent_new_y)
                temp_parent.footprint = node.parent.footprint
                temp_child = FeatureNode("temp", node.feature_type, 0, 0, x=proposed_x, y=proposed_y)
                temp_child.footprint = node.footprint

                if not self.pathing.reconcile_connection(temp_parent, temp_child, all_branches, {
                    temp_parent: temp_parent.get_absolute_footprint(),
                    temp_child: temp_child.get_absolute_footprint()
                }):
                    is_valid_move = False

        if is_valid_move:
            proposed_moves[node] = (proposed_x, proposed_y)
            if all(self._propose_and_reconcile_branch_move(child, dx, dy, branch_nodes, all_branches, proposed_moves)
                   for child in node.subfeatures):
                return True
            else:
                del proposed_moves[node]  # Backtrack on failure

        # --- Attempt 2: Reconciliation via Secondary Jitter ---
        for _ in range(5):
            rdx = random.randint(-JITTER_MAX_TRANSLATION, JITTER_MAX_TRANSLATION)
            rdy = random.randint(-JITTER_MAX_TRANSLATION, JITTER_MAX_TRANSLATION)
            if rdx == 0 and rdy == 0: continue

            recon_x, recon_y = node.current_x + dx + rdx, node.current_y + dy + rdy
            recon_footprint = {(rx + recon_x, ry + recon_y) for rx, ry in node.footprint}

            is_valid_recon = self.placement._is_placement_valid(recon_footprint, collision_mask)

            if is_valid_recon and node.parent and node.parent in branch_nodes:
                parent_new_x, parent_new_y = proposed_moves[node.parent]
                temp_parent = FeatureNode("temp", node.parent.feature_type, 0, 0, x=parent_new_x, y=parent_new_y)
                temp_parent.footprint = node.parent.footprint
                temp_child = FeatureNode("temp", node.feature_type, 0, 0, x=recon_x, y=recon_y)
                temp_child.footprint = node.footprint

                if not self.pathing.reconcile_connection(temp_parent, temp_child, all_branches, {
                    temp_parent: temp_parent.get_absolute_footprint(),
                    temp_child: temp_child.get_absolute_footprint()
                }):
                    is_valid_recon = False

            if is_valid_recon:
                proposed_moves[node] = (recon_x, recon_y)
                if all(self._propose_and_reconcile_branch_move(child, dx, dy, branch_nodes, all_branches,
                                                               proposed_moves) for child in node.subfeatures):
                    return True
                else:
                    del proposed_moves[node]

        return False

    def apply_jitter(self, node_to_move: FeatureNode, all_branches: List[FeatureNode]) -> bool:
        """
        Attempts to translate an entire branch of features starting from the given node.
        If any part of the branch cannot be moved or reconciled, the entire operation fails.
        On success, consumes the jitter budget for all moved nodes.
        """
        if node_to_move.jitter_budget <= 0:
            return False

        dx = random.randint(-JITTER_MAX_TRANSLATION, JITTER_MAX_TRANSLATION)
        dy = random.randint(-JITTER_MAX_TRANSLATION, JITTER_MAX_TRANSLATION)
        if dx == 0 and dy == 0:
            return False

        branch_nodes = set(node_to_move.get_all_nodes_in_branch())
        proposed_moves = {}

        if self._propose_and_reconcile_branch_move(node_to_move, dx, dy, branch_nodes, all_branches, proposed_moves):
            # If the entire chain of moves is valid, commit them
            for node, (new_x, new_y) in proposed_moves.items():
                if node.jitter_budget > 0:
                    node.current_x = new_x
                    node.current_y = new_y
                    node.update_bounding_box_from_footprint()
                    node.jitter_budget -= 1
            return True

        return False

    def apply_erosion(self, node_to_erode: FeatureNode, all_branches: List[FeatureNode]) -> bool:
        """Attempts to erode a single 'NATURAL' node, decrementing its budget on success."""
        if node_to_erode.erosion_budget <= 0:
            return False

        feature_def = config.features.get(node_to_erode.feature_type, {})
        if "NATURAL" not in feature_def.get("natures", []) or len(node_to_erode.footprint) <= MIN_FEATURE_SIZE ** 2:
            return False

        border = {(x, y) for x, y in node_to_erode.footprint if any(
            (x + dx, y + dy) not in node_to_erode.footprint for dx, dy in
            [(0, 1), (0, -1), (1, 0), (-1, 0)])}
        if not border:
            return False

        tile_to_remove = random.choice(list(border))
        eroded_footprint = node_to_erode.footprint.copy()
        eroded_footprint.remove(tile_to_remove)
        if not eroded_footprint:
            return False

        final_footprint = self._find_main_component(eroded_footprint)

        if self._is_proposal_valid(node_to_erode, node_to_erode.current_x, node_to_erode.current_y,
                                   final_footprint, all_branches):
            node_to_erode.footprint = final_footprint
            node_to_erode.interior_footprint = {
                (x, y) for x, y in node_to_erode.footprint
                if (x + 1, y) in node_to_erode.footprint and
                   (x - 1, y) in node_to_erode.footprint and
                   (x, y + 1,) in node_to_erode.footprint and
                   (x, y - 1) in node_to_erode.footprint
            }
            node_to_erode.update_bounding_box_from_footprint()
            node_to_erode.erosion_budget -= 1
            return True
        return False

    def apply_organic_reshaping(self, node: FeatureNode, all_branches: List[FeatureNode]) -> bool:
        """Swells or shrinks a feature's perimeter based on an organic cost map."""
        if node.organic_op_budget <= 0:
            return False

        feature_def = config.features.get(node.feature_type, {})
        if "ORGANIC" not in feature_def.get("natures", []):
            return False

        cost_grid = self.pathing._build_organic_modifiers(all_branches, feature_def)

        # Work with relative coordinates to avoid conversion errors.
        origin_x, origin_y = node.current_x, node.current_y
        relative_footprint = node.footprint

        # 1. Find potential tiles to ADD (swell) in relative coordinates
        swell_candidates = {}  # key: relative coord, value: weight
        for rx, ry in relative_footprint:
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                neighbor_rel = (rx + dx, ry + dy)
                if neighbor_rel not in relative_footprint:
                    neighbor_abs_x, neighbor_abs_y = neighbor_rel[0] + origin_x, neighbor_rel[1] + origin_y
                    if 0 <= neighbor_abs_y < self.map_height and 0 <= neighbor_abs_x < self.map_width:
                        swell_candidates[neighbor_rel] = 1 / max(0.1, cost_grid[neighbor_abs_y, neighbor_abs_x])

        # 2. Find potential tiles to REMOVE (shrink) in relative coordinates
        relative_border = {(rx, ry) for rx, ry in relative_footprint if
                           any((rx + dx, ry + dy) not in relative_footprint for dx, dy in
                               [(-1, 0), (1, 0), (0, -1), (0, 1)])}
        shrink_candidates = {}  # key: relative coord, value: weight
        for rx, ry in relative_border:
            abs_x, abs_y = rx + origin_x, ry + origin_y
            if 0 <= abs_y < self.map_height and 0 <= abs_x < self.map_width:
                shrink_candidates[(rx, ry)] = cost_grid[abs_y, abs_x]

        # 3. Choose an operation
        all_candidates = {**swell_candidates, **shrink_candidates}
        if not all_candidates:
            return False

        coords_list = list(all_candidates.keys())
        weights = list(all_candidates.values())
        chosen_relative_coord = random.choices(coords_list, weights=weights, k=1)[0]

        # 4. Apply the operation
        new_footprint = relative_footprint.copy()
        if chosen_relative_coord in shrink_candidates:  # This is a shrink operation
            if len(new_footprint) <= MIN_FEATURE_SIZE ** 2: return False
            new_footprint.remove(chosen_relative_coord)
        else:  # This is a swell operation
            new_footprint.add(chosen_relative_coord)

        # 5. Validate and commit
        final_footprint = self._find_main_component(new_footprint)
        if self._is_proposal_valid(node, node.current_x, node.current_y, final_footprint, all_branches):
            node.footprint = final_footprint
            node.interior_footprint = {
                (x, y) for x, y in node.footprint
                if (x + 1, y) in node.footprint and
                   (x - 1, y) in node.footprint and
                   (x, y + 1,) in node.footprint and
                   (x, y - 1) in node.footprint
            }
            node.update_bounding_box_from_footprint()
            node.organic_op_budget -= 1
            return True

        return False

    def run_refinement_phase(self, all_branches: List[FeatureNode], on_iteration_end: Callable) -> Generator:
        """Runs the final jitter and erosion passes, spending all remaining budgets."""
        all_nodes = [n for b in all_branches for n in b.get_all_nodes_in_branch()]

        jitter_ops_remaining = sum(n.jitter_budget for n in all_nodes)
        for i in range(jitter_ops_remaining):
            candidates = [n for n in all_nodes if n.jitter_budget > 0]
            if not candidates: break
            if self.apply_jitter(random.choice(candidates), all_branches):
                if i % 20 == 0:
                    on_iteration_end()
                    yield

        erosion_ops_remaining = sum(n.erosion_budget for n in all_nodes)
        for i in range(erosion_ops_remaining):
            candidates = [n for n in all_nodes if n.erosion_budget > 0]
            if not candidates: break
            if self.apply_erosion(random.choice(candidates), all_branches):
                if i % 10 == 0:
                    on_iteration_end()
                    yield

        organic_ops_remaining = sum(n.organic_op_budget for n in all_nodes)
        for i in range(organic_ops_remaining):
            candidates = [n for n in all_nodes if n.organic_op_budget > 0]
            if not candidates: break
            if self.apply_organic_reshaping(random.choice(candidates), all_branches):
                if i % 10 == 0:
                    on_iteration_end()
                    yield

        on_iteration_end()
        yield