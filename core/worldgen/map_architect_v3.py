# /core/worldgen/map_architect_v3.py

import math
import random

from typing import Callable, Tuple, Generator, Optional, List

from .v3_components.converter import Converter
from .v3_components.feature_node import FeatureNode
from .v3_components.interior import Interior
from .v3_components.map_ops import MapOps
from .v3_components.pathing import Pathing
from .v3_components.placement import Placement, MIN_FEATURE_SIZE
from .semantic_search import SemanticSearch
from .v3_components.v3_llm import V3_LLM
from ..common import utils
from ..common.config_loader import config
from ..common.game_state import GenerationState, MapArtist
from ..worldgen import procgen_utils

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
        self.semantic_search = SemanticSearch(engine.embedding_model)

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
                                  narrative_beat: str, gen_state: GenerationState) -> Generator[
        Tuple[str, GenerationState], None, Optional[FeatureNode]]:
        """Orchestrates finding and creating a feature via the LLM-guided, clearance-aware pathfinder."""
        size_tier = feature_data.get('size_tier', 'medium')
        clearance_map = {'small': 0, 'medium': 1, 'large': 2}
        clearance = clearance_map.get(size_tier, 1)
        utils.log_message('debug',
                          f"[PEGv3 Pathing] Placing '{feature_data['name']}' (clearance: {clearance}) from '{parent_branch.name}'...")

        valid_target_branches = []
        for b in self.initial_feature_branches:
            if b is not parent_branch:
                for node in b.get_all_nodes_in_branch():
                    if self.pathing.get_valid_connection_points(node, clearance, self.initial_feature_branches):
                        valid_target_branches.append(b)
                        break

        target_options = [b.name for b in valid_target_branches]

        # Add map border options
        vertical_rels = ['UP', 'DOWN', 'ABOVE', 'BELOW']
        relationships_data = procgen_utils.get_relationships_data()
        lattice_rels = relationships_data.get("lattice", {})
        border_options = [f"{direction}_BORDER" for direction in lattice_rels if direction not in vertical_rels]
        target_options.extend(border_options)

        target_options_str = "\n".join(f"- {name}" for name in target_options)
        chosen_target_name = self.llm.choose_path_target_feature(narrative_log, narrative_beat, target_options_str)

        if not chosen_target_name or 'none' in chosen_target_name.lower():
            utils.log_message('debug', "  LLM decided not to place a path. Discarding feature.")
            return None

        target_branch = None
        valid_end_points = []

        if "_BORDER" in chosen_target_name.upper():
            direction = chosen_target_name.upper().replace("_BORDER", "")
            valid_end_points = self.pathing.get_border_coordinates_for_direction(direction)
            if not valid_end_points:
                utils.log_message('debug', f"  FAIL: No valid border coordinates found for direction '{direction}'.")
                return None
        else:
            target_branch = next(
                (b for b in valid_target_branches if b.name.lower() == chosen_target_name.lower().strip()), None)
            if not target_branch:
                best_match_name = self.semantic_search.find_best_match(chosen_target_name,
                                                                       [b.name for b in valid_target_branches])
                if best_match_name:
                    target_branch = next((b for b in valid_target_branches if b.name == best_match_name), None)
                if not target_branch:
                    utils.log_message('debug', f"  FAIL: Could not find a match for '{chosen_target_name}'.")
                    return None

            for node in target_branch.get_all_nodes_in_branch():
                valid_end_points.extend(
                    self.pathing.get_valid_connection_points(node, clearance, self.initial_feature_branches))

        valid_start_points = []
        for node in parent_branch.get_all_nodes_in_branch():
            valid_start_points.extend(
                self.pathing.get_valid_connection_points(node, clearance, self.initial_feature_branches))

        if not valid_start_points or not valid_end_points:
            utils.log_message('debug', f"  FAIL: No valid start or end points found for path.")
            return None

        path_feature_def = config.features.get(feature_data.get('type'), {})
        start_pos, end_pos = self.pathing.find_first_valid_connection(valid_start_points, valid_end_points, clearance,
                                                                      self.initial_feature_branches)

        if start_pos and end_pos:
            clearance_mask = self.pathing._create_clearance_mask(clearance, self.initial_feature_branches)
            gen_state.clearance_mask = clearance_mask
            yield "PRE_PATH_DRAW", gen_state
            gen_state.clearance_mask = None

            path = self.pathing.find_path_with_clearance(start_pos, end_pos, clearance, self.initial_feature_branches,
                                                         path_feature_def)
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

        utils.log_message('debug', f"  FAIL: Could not connect '{parent_branch.name}' to '{chosen_target_name}'.")
        return None

    def _resolve_parent_node(self, chosen_parent_name: str, valid_parent_nodes: List[FeatureNode]) -> Optional[
        FeatureNode]:
        """Resolves a parent node from an LLM-chosen name, with semantic fallback."""
        if not chosen_parent_name or 'none' in chosen_parent_name.lower():
            return None

        valid_parent_names = [n.name for n in valid_parent_nodes]
        resolved_parent_node = next(
            (n for n in valid_parent_nodes if n.name.lower() == chosen_parent_name.lower().strip()), None)
        if resolved_parent_node:
            return resolved_parent_node

        utils.log_message('debug',
                          f"  LLM chose an invalid parent '{chosen_parent_name}'. Attempting semantic fallback.")
        best_match_name = self.semantic_search.find_best_match(chosen_parent_name, valid_parent_names)
        if best_match_name:
            resolved_parent_node = next((n for n in valid_parent_nodes if n.name == best_match_name), None)
            if resolved_parent_node:
                utils.log_message('debug',
                                  f"  Semantic fallback succeeded. Using parent '{resolved_parent_node.name}'.")
                return resolved_parent_node

        utils.log_message('debug', f"  Could not resolve a valid parent for '{chosen_parent_name}'.")
        return None

    def _place_new_root_branch(self, feature_data: dict, ui_callback: Callable, existing_branches: List[FeatureNode]) -> \
            Optional[FeatureNode]:
        """
        Attempts to place a new root feature branch, trying to attach it to an existing
        feature or placing it in an empty spot if no direct connection is chosen.
        """
        size_tier_map = {'large': 0.8, 'medium': 0.5, 'small': 0.25}
        rel_dim = size_tier_map.get(feature_data.get('size_tier', 'medium'), 0.5)
        prop_w = max(MIN_FEATURE_SIZE, int((self.map_width - 2) * rel_dim))
        prop_h = max(MIN_FEATURE_SIZE, int((self.map_height - 2) * rel_dim))
        feature_def = config.features.get(feature_data['type'], {})
        temp_grid = self.placement._get_temp_grid(existing_branches)

        all_existing_nodes = [node for branch in existing_branches for node in branch.get_all_nodes_in_branch()]
        physically_valid_parents = [
            node for node in all_existing_nodes
            if self.placement._find_valid_placements(prop_w, prop_h, temp_grid, [node], feature_def)
        ]
        if not physically_valid_parents:
            utils.log_message('debug', "[PEGv3 Architect] No physically valid spot found to place a new root branch.")
            return None

        global_parent_options_str = "\n".join(
            f"- {name}" for name in sorted(list(set(n.name for n in physically_valid_parents)))
        )

        chosen_global_parent_name = self.llm.choose_parent_feature(
            self.scene_prompt,
            feature_data.get('description_sentence', feature_data.get('description', feature_data['name'])),
            global_parent_options_str
        )

        chosen_global_parent_node = self._resolve_parent_node(chosen_global_parent_name, physically_valid_parents)
        if not chosen_global_parent_node:
            utils.log_message('debug', "[PEGv3 Architect] LLM chose no valid parent for new root branch. Discarding.")
            return None

        utils.log_message('debug',
                          f"[PEGv3 Architect] LLM chose '{chosen_global_parent_node.name}' as parent for new area '{feature_data['name']}'.")
        valid_placements = self.placement._find_valid_placements(prop_w, prop_h, temp_grid, [chosen_global_parent_node],
                                                                 feature_def)
        if not valid_placements:
            utils.log_message('debug',
                              f"[PEGv3 Architect] FATAL: No valid placement found adjacent to pre-filtered parent '{chosen_global_parent_node.name}'. This indicates a logic error.")
            return None

        px, py, parent_node, face = random.choice(valid_placements)
        new_x, new_y, new_w, new_h = px, py, prop_w, prop_h
        anchor_face = face
        utils.log_message('debug',
                          f"[PEGv3 Architect] Placed new branch '{feature_data['name']}' next to '{parent_node.name}' on face '{face}'.")

        root_node = FeatureNode(feature_data['name'], feature_data['type'], rel_dim, rel_dim, 1, 1, new_x, new_y)
        root_node.narrative_log = feature_data.get('description_sentence', feature_data.get('description', ''))
        root_node.target_growth_rect = (new_x, new_y, new_w, new_h)
        root_node.anchor_to_parent_face = anchor_face
        root_node.sentence_count = 0

        self.initial_feature_branches.append(root_node)
        growth_anim = self._grow_subfeature_coroutine(root_node, ui_callback)
        for _ in growth_anim:
            pass
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

        for branch in self.initial_feature_branches:
            branch.sentence_count = 0

        active_parenting_branches = list(self.initial_feature_branches)
        update_and_draw_no_delay()
        yield "POST_GROWTH", gen_state

        shrink_factor = 0.1
        main_loop_iteration_count = 0
        while main_loop_iteration_count < MAX_SUBFEATURES_TO_PLACE:
            main_loop_iteration_count += 1
            if not active_parenting_branches:
                utils.log_message('debug',
                                  "[PEGv3 Architect] No more active branches for parenting. Attempting to create new nearby areas.")
                all_areas_narrative = "\n".join(
                    f"- {b.name}: {b.narrative_log}" for b in self.initial_feature_branches if b.narrative_log)
                new_nearby_sentence = self.llm.get_nearby_feature_sentence(all_areas_narrative)

                if not new_nearby_sentence or 'none' in new_nearby_sentence.lower():
                    utils.log_message('debug',
                                      "[PEGv3 Architect] LLM decided the scene is complete. Generation finished.")
                    break
                new_feature_data = self.llm.define_feature_from_sentence(new_nearby_sentence)
                if not new_feature_data or new_feature_data.get('type') == 'CHARACTER':
                    utils.log_message('debug',
                                      "[PEGv3 Architect] LLM failed to define new nearby feature or defined a character. Generation complete.")
                    break
                new_root_branch = self._place_new_root_branch(new_feature_data, update_and_draw_with_delay,
                                                              self.initial_feature_branches)
                if new_root_branch:
                    active_parenting_branches.append(new_root_branch)
                    utils.log_message('debug',
                                      f"[PEGv3 Architect] Successfully placed new root branch: '{new_root_branch.name}'.")
                    yield "NEW_BRANCH_PLACEMENT", gen_state
                else:
                    utils.log_message('debug',
                                      "[PEGv3 Architect] Could not find valid placement for new root branch. Generation complete.")
                    break
                continue
            parent_branch = active_parenting_branches[main_loop_iteration_count % len(active_parenting_branches)]
            if not parent_branch.narrative_log:
                parent_branch.narrative_log = self.llm.get_narrative_seed(parent_branch.name) or ""

            other_features_context_list = [f"({branch.name} - \"{branch.narrative_log}\")" for branch in
                                           self.initial_feature_branches if branch is not parent_branch]
            other_features_context = "\n".join(other_features_context_list) if other_features_context_list else "None."
            narrative_beat = self.llm.get_next_narrative_beat(parent_branch.narrative_log, other_features_context)
            if not narrative_beat or 'none' in narrative_beat.lower():
                parent_branch.sentence_count += 1
                if parent_branch.sentence_count >= 5 and parent_branch in active_parenting_branches:
                    utils.log_message('debug',
                                      f"[PEGv3 Architect] Branch '{parent_branch.name}' retired (no new beats generated).")
                    active_parenting_branches.remove(parent_branch)
                continue

            feature_data = self.llm.define_feature_from_sentence(narrative_beat)
            if not feature_data or feature_data.get('type') == 'CHARACTER':
                continue

            was_placed_successfully = False
            placement_strategy = feature_data.get('placement_strategy')

            if placement_strategy == 'CONNECTOR':
                new_nodes = self.placement.handle_connector_placement(
                    feature_data, parent_branch, self.initial_feature_branches, self.llm, self.pathing,
                    self.semantic_search,
                    self._grow_subfeature_coroutine, update_and_draw_with_delay
                )
                if new_nodes:
                    was_placed_successfully = True
                    for new_node in new_nodes:
                        if new_node.path_coords:
                            path_anim = self._draw_path_coroutine(new_node, update_and_draw_with_delay)
                            for _ in path_anim: yield "PATH_DRAW_STEP", gen_state
                        else:
                            if new_node.target_growth_rect:
                                growth_anim = self._grow_subfeature_coroutine(new_node, update_and_draw_with_delay)
                                for _ in growth_anim: yield "SUBFEATURE_GROWTH_STEP", gen_state

            elif placement_strategy == 'PATHING':
                pathing_generator = self._handle_pathing_placement(feature_data, parent_branch,
                                                                   parent_branch.narrative_log,
                                                                   narrative_beat, gen_state)
                new_node = yield from pathing_generator
                if new_node:
                    was_placed_successfully = True
                    path_anim = self._draw_path_coroutine(new_node, update_and_draw_with_delay)
                    for _ in path_anim:
                        yield "PATH_DRAW_STEP", gen_state
            else:  # INTERIOR or EXTERIOR/BLOCKING (subfeature)
                size_tier_map = {'large': (12, 12), 'medium': (8, 8), 'small': (4, 4)}
                prop_w, prop_h = size_tier_map.get(feature_data.get('size_tier', 'medium'))
                temp_grid = self.placement._get_temp_grid(self.initial_feature_branches)

                physically_valid_parents = [
                    node for branch in active_parenting_branches for node in branch.get_all_nodes_in_branch()
                    if self.placement._find_valid_placements(prop_w, prop_h, temp_grid, [node], feature_data)
                ]

                if not physically_valid_parents:
                    utils.log_message('debug',
                                      f"No physically valid parent found for '{feature_data['name']}'. Discarding.")
                else:
                    parent_options_str = "\n".join(
                        f"- {name}" for name in sorted(list(set(n.name for n in physically_valid_parents))))
                    chosen_parent_name = self.llm.choose_parent_feature(parent_branch.narrative_log, narrative_beat,
                                                                        parent_options_str)
                    resolved_parent_node = self._resolve_parent_node(chosen_parent_name, physically_valid_parents)

                    if placement_strategy == 'INTERIOR':
                        target_interior_node = resolved_parent_node if resolved_parent_node else parent_branch
                        target_interior_node.interior_features_to_place.append(feature_data)
                        utils.log_message('debug',
                                          f"[PEGv3 Architect] Queued interior feature '{feature_data['name']}' for '{target_interior_node.name}'.")
                        was_placed_successfully = True
                        yield "SUBFEATURE_STEP", gen_state
                    elif resolved_parent_node:
                        new_subfeature = self.placement.find_and_place_subfeature(
                            feature_data, resolved_parent_node, self.initial_feature_branches,
                            resolved_parent_node.name,
                            shrink_factor
                        )
                        if new_subfeature:
                            was_placed_successfully = True
                            growth_anim = self._grow_subfeature_coroutine(new_subfeature, update_and_draw_with_delay)
                            for _ in growth_anim:
                                yield "SUBFEATURE_GROWTH_STEP", gen_state
                            update_and_draw_no_delay()
                            yield "SUBFEATURE_STEP", gen_state
                        else:
                            utils.log_message('debug',
                                              f"[PEGv3 Architect] Failed physical placement for subfeature '{feature_data['name']}' under '{resolved_parent_node.name}'.")
                    else:
                        utils.log_message('debug',
                                          f"  LLM chose no valid parent for '{feature_data['name']}'. Discarding feature.")
            if was_placed_successfully:
                parent_branch.narrative_log += " " + narrative_beat
                utils.log_message('story', f"-> {narrative_beat}")
                parent_branch.sentence_count += 1
                if parent_branch.sentence_count >= 3 and parent_branch in active_parenting_branches:
                    utils.log_message('debug',
                                      f"[PEGv3 Architect] Branch '{parent_branch.name}' retired (reached sentence limit).")
                    active_parenting_branches.remove(parent_branch)
            else:
                utils.log_message('debug', f"  Discarding narrative beat due to placement failure: '{narrative_beat}'")
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
        all_connections, all_door_placements, all_hallways = self.pathing.create_all_connections(
            self.initial_feature_branches)
        reconns, new_hallways = self.pathing._reconnect_detached_paths(self.initial_feature_branches, all_connections)
        all_connections.extend(reconns)
        all_hallways.extend(new_hallways)
        gen_state.door_locations = all_door_placements
        gen_state.blended_hallways = all_hallways
        artist.draw_map(self.game_map, gen_state, config.features)
        ui_callback(gen_state)
        yield "POST_CONNECT", gen_state
        gen_state.layout_graph = self.converter.serialize_feature_tree_to_graph(self.initial_feature_branches)
        self.converter.populate_generation_state(gen_state, self.initial_feature_branches)
        gen_state.physics_layout = self.converter.convert_to_vertex_representation(gen_state.layout_graph,
                                                                                   all_connections)
        if self.semantic_search.model:
            utils.log_message('debug', "[PEGv3 Architect] Generating feature embeddings for semantic search...")
            for tag, feature_data in gen_state.placed_features.items():
                desc = f"{feature_data.get('name', tag)}: {feature_data.get('description', '')}"
                embedding = self.semantic_search.model.encode(desc)
                gen_state.feature_embeddings[tag] = embedding.tolist()
        gen_state.narrative_log = ""
        yield "VERTEX_DATA", gen_state