# /core/worldgen/map_architect_v3.py

import random
import numpy as np
from typing import Callable, Tuple, Generator, Optional, List

from .v3_components.converter import Converter
from .v3_components.feature_node import FeatureNode
from .v3_components.interior import Interior
from .v3_components.map_ops import MapOps
from .v3_components.pathing import Pathing
from .v3_components.placement import Placement, MIN_FEATURE_SIZE
from .v3_components.v3_llm import V3_LLM
from ..common import utils
from ..common.config_loader import config
from ..common.game_state import GenerationState, MapArtist

# Set extremely high because we expect to reach a natural geometric limit before then
# only here to catch short infinite loops
MAX_SUBFEATURES_TO_PLACE = 500


class MapArchitectV3:
    """
    Implements a map generation strategy based on the 'peg3_prototype.py' logic:
    iterative shrink-and-place, followed by jitter, and robust pathfinding.
    """

    def __init__(self, engine, game_map, world_theme, scene_prompt):
        self.engine = engine
        self.game_map = game_map
        self.map_width = game_map.width
        self.map_height = game_map.height
        self.scene_prompt = scene_prompt

        self.llm = V3_LLM(engine)
        self.placement = Placement(self.map_width, self.map_height)
        self.map_ops = MapOps(self.map_width, self.map_height)
        self.interior = Interior(self.placement, self.map_width, self.map_height)
        self.pathing = Pathing(self.game_map)
        self.converter = Converter()

        self.initial_feature_branches: List[FeatureNode] = []

    def _grow_subfeature_coroutine(self, node: FeatureNode, draw_callback: Callable):
        if not node.target_growth_rect or not node.anchor_to_parent_face:
            return
        target_x, target_y, target_w, target_h = node.target_growth_rect
        anchor_face = node.anchor_to_parent_face
        while (node.current_abs_width < target_w) or (node.current_abs_height < target_h):
            node.current_abs_width = min(target_w, node.current_abs_width + 2)
            node.current_abs_height = min(target_h, node.current_abs_height + 2)
            if anchor_face == 'N':
                node.current_x = target_x + (target_w - node.current_abs_width) // 2
                node.current_y = (target_y + target_h) - node.current_abs_height
            elif anchor_face == 'S':
                node.current_x = target_x + (target_w - node.current_abs_width) // 2
                node.current_y = target_y
            elif anchor_face == 'W':
                node.current_y = target_y + (target_h - node.current_abs_height) // 2
                node.current_x = (target_x + target_w) - node.current_abs_width
            else:  # E
                node.current_y = target_y + (target_h - node.current_abs_height) // 2
                node.current_x = target_x
            draw_callback()
            yield
        node.current_x, node.current_y = target_x, target_y
        node.current_abs_width, node.current_abs_height = target_w, target_h
        draw_callback()
        yield

    def _draw_path_coroutine(self, node: FeatureNode, draw_callback: Callable):
        if not node.path_coords:
            return
        full_path = list(node.path_coords)
        node.path_coords.clear()
        chunk_size = 5
        for i in range(0, len(full_path), chunk_size):
            chunk = full_path[i:i + chunk_size]
            node.path_coords.extend(chunk)
            draw_callback()
            yield
        node.path_coords = full_path
        draw_callback()
        yield

    def _handle_pathing_placement(self, feature_data: dict, parent_branch: FeatureNode, narrative_log: str,
                                  narrative_beat: str) -> Optional[FeatureNode]:
        """Orchestrates finding and creating a feature via the LLM-guided, clearance-aware pathfinder."""
        size_tier = feature_data.get('size_tier', 'medium')
        clearance_map = {'small': 0, 'medium': 1, 'large': 2}
        clearance = clearance_map.get(size_tier, 1)
        utils.log_message('debug',
                          f"[PEGv3 Pathing] Placing '{feature_data['name']}' (clearance: {clearance}) from '{parent_branch.name}'...")

        other_branches = [b for b in self.initial_feature_branches if b is not parent_branch]
        if not other_branches:
            utils.log_message('debug', "  FAIL: No other branches to path to.")
            return None
        target_options_str = "\n".join(f"- {b.name}" for b in other_branches)

        chosen_target_name = self.llm.choose_path_target_feature(narrative_log, narrative_beat, target_options_str)
        if not chosen_target_name or 'none' in chosen_target_name.lower():
            utils.log_message('debug', "  LLM decided not to place a path.")
            return None

        target_branch = next((b for b in other_branches if b.name.lower() in chosen_target_name.lower()), None)
        if not target_branch:
            utils.log_message('debug', f"  FAIL: LLM chose an invalid target '{chosen_target_name}'.")
            return None
        utils.log_message('debug', f"  LLM chose target branch: '{target_branch.name}'")

        valid_start_points = []
        for node in parent_branch.get_all_nodes_in_branch():
            valid_start_points.extend(
                self.pathing.get_valid_connection_points(node, clearance, self.initial_feature_branches))

        valid_end_points = []
        for node in target_branch.get_all_nodes_in_branch():
            valid_end_points.extend(
                self.pathing.get_valid_connection_points(node, clearance, self.initial_feature_branches))

        if not valid_start_points or not valid_end_points:
            utils.log_message('debug',
                              f"  FAIL: No valid connection points found on parent or target for clearance {clearance}.")
            return None

        random.shuffle(valid_start_points)
        random.shuffle(valid_end_points)
        utils.log_message('full',
                          f"  Found {len(valid_start_points)} start points and {len(valid_end_points)} end points.")

        path_feature_def = config.features.get(feature_data.get('type'), {})

        for start_pos in valid_start_points:
            for end_pos in valid_end_points:
                path = self.pathing.find_path_with_clearance(start_pos, end_pos, clearance,
                                                             self.initial_feature_branches, path_feature_def)
                if path:
                    utils.log_message('debug', f"  SUCCESS: Pathfinder connected from {start_pos} to {end_pos}.")
                    path_tile_coords = set()
                    for x, y in path:
                        for i in range(-clearance, clearance + 1):
                            for j in range(-clearance, clearance + 1):
                                path_tile_coords.add((x + i, y + j))

                    new_path_node = FeatureNode(name=feature_data['name'], feature_type=feature_data['type'],
                                                rel_w=0, rel_h=0, abs_w=0, abs_h=0, x=0, y=0, parent=parent_branch)
                    new_path_node.path_coords = list(path_tile_coords)
                    parent_branch.subfeatures.append(new_path_node)
                    return new_path_node

        utils.log_message('debug', f"  FAIL: Could not connect '{parent_branch.name}' to '{target_branch.name}'.")
        return None

    def _place_new_root_branch(self, feature_data: dict, ui_callback: Callable, existing_branches: List[FeatureNode]) -> \
    Optional[FeatureNode]:
        """
        Attempts to place a new root feature branch, trying to attach it to an existing
        feature or placing it in an empty spot if no direct connection is chosen.
        """
        size_tier_map = {'large': 0.8, 'medium': 0.5, 'small': 0.25}
        # Calculate proposed initial dimensions based on size tier
        rel_dim = size_tier_map.get(feature_data.get('size_tier', 'medium'), 0.5)
        prop_w = max(MIN_FEATURE_SIZE, int((self.map_width - 2) * rel_dim))
        prop_h = max(MIN_FEATURE_SIZE, int((self.map_height - 2) * rel_dim))
        feature_def = config.features.get(feature_data['type'], {})

        all_existing_nodes = [node for branch in existing_branches for node in branch.get_all_nodes_in_branch()]
        # Get all nodes names for parent selection
        global_parent_options_str = "\n".join(
            f"- {name}" for name in sorted(list(set(n.name for n in all_existing_nodes))))

        # Ask LLM to choose a parent from all existing features
        chosen_global_parent_name = self.llm.choose_parent_feature(
            self.scene_prompt,  # Using scene_prompt as general narrative context
            feature_data.get('description_sentence', feature_data.get('description', feature_data['name'])),
            global_parent_options_str
        )

        chosen_global_parent_node = None
        if chosen_global_parent_name and 'none' not in chosen_global_parent_name.lower():
            for branch in existing_branches:
                for node in branch.get_all_nodes_in_branch():
                    if node.name == chosen_global_parent_name:
                        chosen_global_parent_node = node
                        break
                if chosen_global_parent_node:
                    break

        placement_found = False
        new_x, new_y = 0, 0
        new_w, new_h = prop_w, prop_h
        anchor_face = 'any'

        temp_grid = self.placement._get_temp_grid(existing_branches)

        if chosen_global_parent_node:
            utils.log_message('debug',
                              f"[PEGv3 Architect] LLM chose '{chosen_global_parent_node.name}' as parent for new area '{feature_data['name']}'.")
            # Try to place adjacent to the chosen parent
            valid_placements = self.placement._find_valid_placements(prop_w, prop_h, temp_grid,
                                                                     [chosen_global_parent_node], feature_def)
            if valid_placements:
                px, py, parent_node, face = random.choice(valid_placements)
                new_x, new_y, new_w, new_h = px, py, prop_w, prop_h  # Initial placement is a seed, will grow
                anchor_face = face
                placement_found = True
                utils.log_message('debug',
                                  f"[PEGv3 Architect] Placed new branch '{feature_data['name']}' next to '{parent_node.name}' on face '{face}'.")
            else:
                utils.log_message('debug',
                                  f"[PEGv3 Architect] No valid placement found adjacent to chosen parent '{chosen_global_parent_node.name}'. Falling back to general placement.")

        if not placement_found:
            utils.log_message('debug',
                              f"[PEGv3 Architect] Attempting to place new branch '{feature_data['name']}' in any valid open space.")
            # Fallback: Find any open space adjacent to any existing feature
            all_valid_placements = self.placement._find_valid_placements(prop_w, prop_h, temp_grid, existing_branches,
                                                                         feature_def)
            if all_valid_placements:
                px, py, parent_node, face = random.choice(all_valid_placements)
                new_x, new_y, new_w, new_h = px, py, prop_w, prop_h
                anchor_face = face
                placement_found = True
                utils.log_message('debug',
                                  f"[PEGv3 Architect] Placed new branch '{feature_data['name']}' next to '{parent_node.name}' on face '{face}'.")
            else:
                utils.log_message('debug',
                                  f"[PEGv3 Architect] No valid open space found for new branch '{feature_data['name']}'.")
                return None  # No placement found, cannot place this feature

        # Create as a seed 1x1 first, then set its target growth rect
        root_node = FeatureNode(feature_data['name'], feature_data['type'], rel_dim, rel_dim, 1, 1, new_x, new_y)
        root_node.narrative_log = feature_data.get('description_sentence', feature_data.get('description', ''))
        root_node.target_growth_rect = (new_x, new_y, new_w, new_h)  # Set target dimensions
        root_node.anchor_to_parent_face = anchor_face  # Record for potential visual connection
        root_node.sentence_count = 0  # Initialize for new branch

        self.initial_feature_branches.append(root_node)
        # Immediately animate its growth after placement
        growth_anim = self._grow_subfeature_coroutine(root_node, ui_callback)
        for _ in growth_anim:
            ui_callback(None)  # Pass None to indicate animation step, not full state update
            # This yield ensures the animation steps are drawn.
        utils.log_message('debug',
                          f"[PEGv3 Architect] Grown new branch '{root_node.name}' to final size {new_w}x{new_h}.")

        return root_node

    def generate_layout_in_steps(self, ui_callback: Callable) -> Generator[Tuple[str, GenerationState], None, None]:
        gen_state = GenerationState(self.game_map)
        artist = MapArtist()

        def update_and_draw_with_delay():
            self.converter.populate_generation_state(gen_state, self.initial_feature_branches)
            artist.draw_map(self.game_map, gen_state, config.features)
            ui_callback(gen_state)
            import time
            time.sleep(0.05)

        def update_and_draw_no_delay():
            self.converter.populate_generation_state(gen_state, self.initial_feature_branches)
            artist.draw_map(self.game_map, gen_state, config.features)
            ui_callback(gen_state)

        initial_specs = self.llm.get_initial_features()
        if not initial_specs:
            update_and_draw_no_delay()
            gen_state.layout_graph = self.converter.serialize_feature_tree_to_graph(self.initial_feature_branches)
            gen_state.physics_layout = self.converter.convert_to_vertex_representation(gen_state.layout_graph, [])
            yield "FINAL", gen_state
            return

        growth_generator = self.placement.place_and_grow_initial_features(initial_specs)
        for growing_branches in growth_generator:
            self.initial_feature_branches = growing_branches
            update_and_draw_with_delay()
            yield "INITIAL_GROWTH_STEP", gen_state

        # Initialize sentence count for root branches
        for branch in self.initial_feature_branches:
            branch.sentence_count = 0  # Initialize to 0, will increment when first beat is added to it.

        active_parenting_branches = list(
            self.initial_feature_branches)  # Only contains branches that LLM can add sub-features to.

        # This new yield creates a clear state transition point.
        update_and_draw_no_delay()
        yield "POST_GROWTH", gen_state

        shrink_factor = 0.1
        main_loop_iteration_count = 0  # Track iterations to prevent infinite loops if LLM gets stuck
        while main_loop_iteration_count < MAX_SUBFEATURES_TO_PLACE:
            main_loop_iteration_count += 1

            if not active_parenting_branches:
                utils.log_message('debug',
                                  "[PEGv3 Architect] No more active branches for parenting. Attempting to create new nearby areas.")

                # --- Attempt to create a new nearby area ---
                all_areas_narrative = "\n".join(
                    f"- {b.name}: {b.narrative_log}" for b in self.initial_feature_branches if b.narrative_log)
                new_nearby_sentence = self.llm.get_nearby_feature_sentence(all_areas_narrative)

                if not new_nearby_sentence or 'none' in new_nearby_sentence.lower():
                    utils.log_message('debug',
                                      "[PEGv3 Architect] LLM decided the scene is complete. Generation finished.")
                    break  # No new areas, end generation

                new_feature_data = self.llm.define_feature_from_sentence(new_nearby_sentence)
                if not new_feature_data or new_feature_data.get('type') == 'CHARACTER':
                    utils.log_message('debug',
                                      "[PEGv3 Architect] LLM failed to define new nearby feature or defined a character. Generation complete.")
                    break  # Failed to define a valid new area, end generation

                # --- Try to place the new root branch ---
                new_root_branch = self._place_new_root_branch(new_feature_data, update_and_draw_with_delay,
                                                              self.initial_feature_branches)

                if new_root_branch:
                    active_parenting_branches.append(new_root_branch)
                    utils.log_message('debug',
                                      f"[PEGv3 Architect] Successfully placed new root branch: '{new_root_branch.name}'.")
                    yield "NEW_BRANCH_PLACEMENT", gen_state  # Yield after placing new root branch
                    # Continue loop, now with a new active branch
                else:
                    utils.log_message('debug',
                                      "[PEGv3 Architect] Could not find valid placement for new root branch. Generation complete.")
                    break  # No placement found, end generation
                continue  # Skip to the next iteration of the main loop

            # Select a branch to work on, cycling through active ones
            parent_branch = active_parenting_branches[
                main_loop_iteration_count % len(active_parenting_branches)]  # Using iteration count to cycle

            if not parent_branch.narrative_log:
                parent_branch.narrative_log = self.llm.get_narrative_seed(parent_branch.name) or ""

            other_features_context_list = [
                f"({branch.name} - \"{branch.narrative_log}\")"
                for branch in self.initial_feature_branches  # Use all initial branches for context
                if branch is not parent_branch
            ]
            other_features_context = "\n".join(other_features_context_list) if other_features_context_list else "None."

            narrative_beat = self.llm.get_next_narrative_beat(parent_branch.narrative_log, other_features_context)
            if not narrative_beat: continue  # If no beat, skip and try another branch or end.
            parent_branch.narrative_log += " " + narrative_beat
            utils.log_message('story', f"-> {narrative_beat}")

            feature_data = self.llm.define_feature_from_sentence(narrative_beat)
            if not feature_data or feature_data['type'] == 'CHARACTER': continue

            # For choosing a parent for sub-features, we only consider nodes within the *active* parenting branches
            all_current_branch_nodes_for_parenting = [node for branch in active_parenting_branches for node in
                                                      branch.get_all_nodes_in_branch()]
            parent_options_str = "\n".join(
                f"- {name}" for name in sorted(list(set(n.name for n in all_current_branch_nodes_for_parenting))))
            chosen_parent_name = self.llm.choose_parent_feature(parent_branch.narrative_log, narrative_beat,
                                                                parent_options_str)

            # Resolve the chosen parent to a FeatureNode.
            resolved_parent_node = None
            if chosen_parent_name and 'none' not in chosen_parent_name.lower():
                for branch in active_parenting_branches:
                    for node_in_branch in branch.get_all_nodes_in_branch():
                        if node_in_branch.name == chosen_parent_name:
                            resolved_parent_node = node_in_branch
                            break
                    if resolved_parent_node:
                        break

            # Increment sentence count only if a valid parent was selected or if it's an interactable.
            # This counts sentences that contribute to the current branch.
            if resolved_parent_node or feature_data.get('placement_strategy') == 'INTERIOR':
                parent_branch.sentence_count += 1
            else:
                utils.log_message('debug',
                                  f"[PEGv3 Architect] LLM chose 'NONE' or an invalid parent for new feature '{feature_data['name']}'.")

            is_interactable = feature_data.get('placement_strategy') == 'INTERIOR'
            is_branch_retirement_triggered = False

            # Check for retirement conditions only if a valid parent was found for the current beat,
            # or if an interactable was created (which can lead to retirement even without a specific parent node).
            if resolved_parent_node or is_interactable:
                if parent_branch.sentence_count >= 3:
                    if is_interactable or not resolved_parent_node:  # Second condition: no valid parent for a *non-interactable* feature.
                        is_branch_retirement_triggered = True

            if is_interactable:
                # If it's an interactable, we add it to the chosen node's interior features.
                # If no specific node was resolved, it defaults to the current parent_branch's root.
                target_interior_node = resolved_parent_node if resolved_parent_node else parent_branch
                target_interior_node.interior_features_to_place.append(feature_data)
                utils.log_message('debug',
                                  f"[PEGv3 Architect] Queued interior feature '{feature_data['name']}' for '{target_interior_node.name}'.")
                yield "SUBFEATURE_STEP", gen_state  # Yield even for queued features to show activity

                if is_branch_retirement_triggered:
                    if parent_branch in active_parenting_branches:
                        utils.log_message('debug',
                                          f"[PEGv3 Architect] Branch '{parent_branch.name}' retired (interactable placed at sentence limit).")
                        active_parenting_branches.remove(parent_branch)
                    continue  # Skip to the next iteration of the main loop

            elif feature_data.get('placement_strategy') == 'PATHING':
                new_path_node = self._handle_pathing_placement(feature_data, parent_branch, parent_branch.narrative_log,
                                                               narrative_beat)
                if new_path_node:
                    path_anim = self._draw_path_coroutine(new_path_node, update_and_draw_with_delay)
                    for _ in path_anim:
                        yield "PATH_DRAW_STEP", gen_state
                continue

            elif resolved_parent_node:  # Standard subfeature, not interior or pathing
                new_subfeature = self.placement.find_and_place_subfeature(
                    feature_data, resolved_parent_node, self.initial_feature_branches, chosen_parent_name, shrink_factor
                )
                if new_subfeature:
                    growth_anim = self._grow_subfeature_coroutine(new_subfeature, update_and_draw_with_delay)
                    for _ in growth_anim:
                        yield "SUBFEATURE_GROWTH_STEP", gen_state
                    update_and_draw_no_delay()
                    yield "SUBFEATURE_STEP", gen_state
                else:
                    utils.log_message('debug',
                                      f"[PEGv3 Architect] Failed to place subfeature '{feature_data['name']}' under '{resolved_parent_node.name}'.")

            if is_branch_retirement_triggered:
                if parent_branch in active_parenting_branches:
                    utils.log_message('debug',
                                      f"[PEGv3 Architect] Branch '{parent_branch.name}' retired (no valid parent for new feature at sentence limit).")
                    active_parenting_branches.remove(parent_branch)
                continue

        update_and_draw_no_delay()
        yield "PRE_JITTER", gen_state

        self.map_ops.apply_jitter(self.initial_feature_branches, on_iteration_end=update_and_draw_with_delay)
        update_and_draw_no_delay()
        yield "PRE_INTERIOR_PLACEMENT", gen_state

        interior_generator = self.interior.finalize_placements(self.initial_feature_branches, gen_state)
        for _ in interior_generator:
            update_and_draw_no_delay()
            yield "INTERIOR_PLACEMENT_STEP", gen_state

        update_and_draw_no_delay()
        yield "PRE_CONNECT", gen_state

        all_connections = self.pathing.create_all_connections(self.initial_feature_branches)
        all_door_coords = [coord for conn in all_connections for coord in conn['door_coords']]
        gen_state.door_locations = all_door_coords

        artist.draw_map(self.game_map, gen_state, config.features)
        ui_callback(gen_state)
        yield "POST_CONNECT", gen_state

        gen_state.layout_graph = self.converter.serialize_feature_tree_to_graph(self.initial_feature_branches)
        self.converter.populate_generation_state(gen_state, self.initial_feature_branches)
        gen_state.physics_layout = self.converter.convert_to_vertex_representation(gen_state.layout_graph,
                                                                                   all_connections)
        gen_state.narrative_log = ""  # Placeholder for full narrative log
        yield "VERTEX_DATA", gen_state