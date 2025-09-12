# /core/worldgen/map_architect_v3.py

import numpy as np
import random
from collections import deque, defaultdict
from typing import Callable, Tuple, Generator, Optional, List, Set, Dict

from .v3_components.converter import Converter
from .v3_components.feature_node import FeatureNode
from .v3_components.interior import Interior
from .v3_components.map_ops import MapOps
from .v3_components.pathing import Pathing
from .v3_components.placement import Placement
from .procgen_utils import UnionFind, resolve_parent_node
from .semantic_search import SemanticSearch
from .v3_components.v3_llm import V3_LLM
from .v3_components import tile_utils, geometry_probes
from ..common import utils
from ..common.config_loader import config
from ..common.game_state import GenerationState, MapArtist

MAX_SUBFEATURES_TO_PLACE = 500
MIN_FEATURE_SIZE = 3
GROWTH_COVERAGE_THRESHOLD = 0.40
MAX_GROWTH_ITERATIONS = 200


class MapArchitectV3:
    def __init__(self, manager, game_map, world_theme, scene_prompt):
        self.manager = manager
        self.engine = manager.engine
        self.game_map = game_map
        self.map_width = game_map.width
        self.map_height = game_map.height
        self.scene_prompt = scene_prompt

        # --- Components are now passed from the manager ---
        self.llm = manager.llm
        self.pathing = manager.pathing
        self.semantic_search = manager.semantic_search
        self.placement = manager.placement
        self.map_ops = manager.map_ops
        self.interior = manager.interior
        self.converter = manager.converter

        self.initial_feature_branches: List[FeatureNode] = []

    def _handle_pathing_placement(self, feature_data: dict, parent_branch: FeatureNode, narrative_log: str,
                                  narrative_beat: str) -> Optional[FeatureNode]:
        """Orchestrates the LLM choice and subsequent placement of a path feature."""
        utils.log_message('debug', f"[PEGv3 Pathing] Placing '{feature_data['name']}' from '{parent_branch.name}'...")

        # Build a list of valid targets for the LLM
        valid_targets = []
        for b in self.initial_feature_branches:
            if b.get_root() is not parent_branch.get_root():
                valid_targets.append(b)

        target_options = [b.name for b in valid_targets] + ["NORTH_BORDER", "SOUTH_BORDER", "EAST_BORDER",
                                                            "WEST_BORDER"]

        # Ask the LLM to choose a target
        chosen_target_name = self.llm.choose_path_target_feature(narrative_log, narrative_beat,
                                                                 "\n".join(f"- {n}" for n in target_options))

        # Delegate the actual placement to the generic utility function in the manager
        return self.manager.place_path_feature(feature_data, parent_branch,
                                               chosen_target_name, self.initial_feature_branches)

    def _place_initial_non_path_seeds(self, specs_to_place: List[Dict],
                                      existing_branches: List[FeatureNode]) -> Tuple[List[FeatureNode], bool]:
        """Tries to place a list of feature specs as small seeds in random locations."""
        placed_nodes = []
        MAX_INDIVIDUAL_PLACEMENT_ATTEMPTS = 100
        for spec in specs_to_place:
            node_placed = False
            for _ in range(MAX_INDIVIDUAL_PLACEMENT_ATTEMPTS):
                w, h = MIN_FEATURE_SIZE, MIN_FEATURE_SIZE
                if self.map_width <= w + 2 or self.map_height <= h + 2: continue
                x = random.randint(1, self.map_width - w - 2)
                y = random.randint(1, self.map_height - h - 2)
                all_current_branches = existing_branches + placed_nodes
                temp_grid = self.manager._get_temp_grid(all_current_branches)
                collision_mask = temp_grid != self.placement.void_space_index
                temp_node = FeatureNode(spec['name'], spec['type'], w, h, x, y)
                if self.placement._is_placement_valid(temp_node.get_absolute_footprint(), collision_mask):
                    self.manager._assign_op_budgets(temp_node)
                    placed_nodes.append(temp_node)
                    node_placed = True
                    break
            if not node_placed:
                utils.log_message('debug', f"[PEGv3 Setup] Failed to place initial seed for '{spec['name']}'.")
                return [], False
        return placed_nodes, True

    def _place_initial_layout(self, initial_specs: List[Dict], update_and_draw: Callable) -> Generator[
        Tuple[str, None], None, bool]:
        """Generator to handle the entire initial placement and growth phase."""
        path_specs = [s for s in initial_specs if
                      config.features.get(s.get('type'), {}).get('placement_strategy') == 'PATHING']
        non_path_specs = [s for s in initial_specs if
                          config.features.get(s.get('type'), {}).get('placement_strategy') != 'PATHING']

        anchor_names = {p['source'] for p in path_specs if "_BORDER" not in p['source']} | \
                       {p['destination'] for p in path_specs if "_BORDER" not in p['destination']}
        anchor_specs = [s for s in non_path_specs if s['name'] in anchor_names]
        other_specs = [s for s in non_path_specs if s['name'] not in anchor_names]

        MAX_DISTRIBUTION_ATTEMPTS = 5
        for attempt in range(MAX_DISTRIBUTION_ATTEMPTS):
            self.initial_feature_branches.clear()
            anchor_nodes, success = self._place_initial_non_path_seeds(anchor_specs, [])
            if not success:
                utils.log_message('debug', f"Anchor placement failed, retrying distribution ({attempt + 1})")
                continue

            self.initial_feature_branches.extend(anchor_nodes)
            update_and_draw()
            yield "ANCHOR_PLACEMENT", None

            node_map = {node.name: node for node in self.initial_feature_branches}
            paths_placed_successfully = True
            for path_spec in path_specs:
                source_name, dest_name = path_spec['source'], path_spec['destination']
                source_is_border, dest_is_border = "_BORDER" in source_name.upper(), "_BORDER" in dest_name.upper()
                newly_placed_path = None

                if not source_is_border and not dest_is_border:
                    source_node, dest_node = node_map.get(source_name), node_map.get(dest_name)
                    if source_node and dest_node: newly_placed_path = self.placement.place_initial_path_between_branches(
                        path_spec, source_node, dest_node, self.initial_feature_branches)
                elif source_is_border and not dest_is_border:
                    dest_node = node_map.get(dest_name)
                    if dest_node:
                        coords = self.pathing.get_border_coordinates_for_direction(source_name.replace("_BORDER", ""))
                        newly_placed_path = self.placement.place_initial_path_to_border(path_spec, dest_node, coords,
                                                                                        self.initial_feature_branches)
                elif not source_is_border and dest_is_border:
                    source_node = node_map.get(source_name)
                    if source_node:
                        coords = self.pathing.get_border_coordinates_for_direction(dest_name.replace("_BORDER", ""))
                        newly_placed_path = self.placement.place_initial_path_to_border(path_spec, source_node, coords,
                                                                                        self.initial_feature_branches)
                elif source_is_border and dest_is_border:
                    s_coords = self.pathing.get_border_coordinates_for_direction(source_name.replace("_BORDER", ""))
                    d_coords = self.pathing.get_border_coordinates_for_direction(dest_name.replace("_BORDER", ""))
                    newly_placed_path = self.placement.place_initial_path_between_borders(path_spec, s_coords, d_coords,
                                                                                          self.initial_feature_branches)

                if newly_placed_path:
                    self.manager._assign_op_budgets(newly_placed_path)
                    yield from self.manager._draw_path_coroutine(newly_placed_path, update_and_draw)
                else:
                    utils.log_message('debug',
                                      f"Failed to place path '{path_spec['name']}' between '{source_name}' and '{dest_name}'.")
                    paths_placed_successfully = False
                    break

            if not paths_placed_successfully:
                utils.log_message('debug', f"Path placement failed, retrying distribution ({attempt + 1})")
                continue

            other_nodes, success = self._place_initial_non_path_seeds(other_specs, self.initial_feature_branches)
            if not success:
                utils.log_message('debug', f"Other feature placement failed, retrying distribution ({attempt + 1})")
                continue

            self.initial_feature_branches.extend(other_nodes)
            update_and_draw()
            yield "REMAINING_PLACEMENT", None
            return True

        return False

    def _grow_initial_features(self, update_and_draw: Callable) -> Generator[Tuple[str, None], None, None]:
        """Generator to handle the growth phase of initial non-path features."""
        non_path_nodes = [b for b in self.initial_feature_branches if
                          config.features.get(b.feature_type, {}).get('placement_strategy') != 'PATHING']
        if not non_path_nodes:
            yield "POST_GROWTH", None
            return

        total_target_area = self.map_width * self.map_height * GROWTH_COVERAGE_THRESHOLD
        target_area_per_node = total_target_area / len(non_path_nodes)
        for node in non_path_nodes:
            node.target_area = target_area_per_node
            node.target_aspect_ratio = (random.randint(1, 16), random.randint(1, 16))

        for i in range(MAX_GROWTH_ITERATIONS):
            current_area = sum(len(n.footprint) for n in self.initial_feature_branches)
            if current_area >= total_target_area: break
            growable_nodes = [n for n in non_path_nodes if not n.is_stuck and len(n.footprint) < n.target_area]
            if not growable_nodes: break

            tick_changed = False
            random.shuffle(growable_nodes)
            for feature in growable_nodes:
                if self.placement._attempt_aspect_aware_growth(feature, self.initial_feature_branches):
                    tick_changed = True

            if tick_changed:
                update_and_draw()
                yield "INITIAL_GROWTH_STEP", None
            else:
                break

        yield "POST_GROWTH", None

    def _place_subfeatures_conversationally(self, gen_state: GenerationState, update_and_draw: Callable) -> Generator[
        Tuple[str, None], None, None]:
        """Generator for the main conversational sub-feature placement loop."""
        active_parenting_branches = list(self.initial_feature_branches)
        branch_exhaustion_counters = defaultdict(int)

        for i in range(MAX_SUBFEATURES_TO_PLACE):
            if not active_parenting_branches:
                utils.log_message('debug', "[PEGv3] All branches exhausted. Trying to start a new branch.")
                all_areas_narrative = "\n".join(
                    f"- {b.name}: {b.narrative_log}" for b in self.initial_feature_branches if b.narrative_log)
                new_nearby_sentence = self.llm.get_nearby_feature_sentence(all_areas_narrative)
                if not new_nearby_sentence or 'none' in new_nearby_sentence.lower():
                    utils.log_message('debug', "[PEGv3] LLM has no ideas for new branches. Finalizing map.")
                    break

                new_feature_data = self.llm.define_feature_from_sentence(new_nearby_sentence)
                if not new_feature_data or new_feature_data.get('type') == 'CHARACTER':
                    utils.log_message('debug', "[PEGv3] LLM failed to define a new branch. Finalizing map.")
                    break

                newly_placed_node = self.placement.place_new_root_branch(new_feature_data,
                                                                         self.initial_feature_branches)
                if newly_placed_node:
                    self.manager._assign_op_budgets(newly_placed_node)
                    active_parenting_branches.append(newly_placed_node)
                    update_and_draw()
                    yield "NEW_BRANCH_PLACEMENT", None
                    continue
                else:
                    utils.log_message('debug', "[PEGv3] Failed to place new root branch. Finalizing map.")
                    break

            parent_branch = random.choice(active_parenting_branches)
            all_placeable_nodes_in_branch = [n for n in parent_branch.get_all_nodes_in_branch() if not n.is_blocked]

            if not all_placeable_nodes_in_branch:
                active_parenting_branches.remove(parent_branch)
                continue

            parent_node = random.choice(all_placeable_nodes_in_branch)

            if not parent_branch.narrative_log:
                parent_branch.narrative_log = self.llm.get_narrative_seed(parent_branch.name) or ""

            other_features_context = "\n".join(
                [f'({b.name} - "{b.narrative_log}")' for b in self.initial_feature_branches if b is not parent_branch])
            narrative_beat = self.llm.get_next_narrative_beat(parent_branch.narrative_log, other_features_context)
            feature_data = self.llm.define_feature_from_sentence(
                narrative_beat) if narrative_beat and 'none' not in narrative_beat.lower() else None

            if not feature_data or feature_data.get('type') in ['CHARACTER', 'GENERIC_INTERACTABLE']:
                branch_exhaustion_counters[parent_branch.name] += 1
                if branch_exhaustion_counters[parent_branch.name] >= 3:
                    active_parenting_branches.remove(parent_branch)
                if feature_data and feature_data.get('type') == 'CHARACTER':
                    gen_state.character_creation_queue.append(feature_data)
                continue

            if feature_data.get('placement_strategy') == 'INTERIOR':
                parent_node.interior_features_to_place.append(feature_data)
                parent_branch.sentence_count += 1
                continue

            newly_placed_nodes = []
            placement_strategy = config.features.get(feature_data['type'], {}).get('placement_strategy')

            if placement_strategy == 'PATHING':
                node = self._handle_pathing_placement(feature_data, parent_branch, parent_branch.narrative_log,
                                                      narrative_beat)
                if node: newly_placed_nodes.append(node)
            elif placement_strategy == 'CONNECTOR':
                nodes = self.placement.handle_connector_placement(feature_data, parent_branch,
                                                                  self.initial_feature_branches)
                if nodes: newly_placed_nodes.extend(nodes)
            elif placement_strategy == 'BRANCHING':
                all_nodes = [n for b in self.initial_feature_branches for n in b.get_all_nodes_in_branch()]
                possible_parents = [node for node in all_nodes if
                                    self.placement.can_place_subfeature(feature_data, node,
                                                                        self.initial_feature_branches)]
                if possible_parents:
                    parent_options_str = "\n".join(
                        f"- {n.name}" for n in sorted(list(set(possible_parents)), key=lambda x: x.name))
                    chosen_parent_name = self.llm.choose_parent_feature(parent_branch.narrative_log, narrative_beat,
                                                                        parent_options_str)
                    chosen_parent = resolve_parent_node(chosen_parent_name, possible_parents,
                                                                      self.semantic_search)
                    if chosen_parent:
                        node = self.placement.find_and_place_subfeature(feature_data, chosen_parent,
                                                                        self.initial_feature_branches)
                        if node: newly_placed_nodes.append(node)

            if newly_placed_nodes:
                parent_branch.narrative_log += " " + narrative_beat
                parent_branch.sentence_count += 1
                branch_exhaustion_counters[parent_branch.name] = 0  # Reset counter on success
                yield from self.manager._process_newly_placed_nodes(newly_placed_nodes, self.initial_feature_branches,
                                                                    update_and_draw)
            else:
                utils.log_message('debug',
                                  f"Placement failed for '{feature_data.get('name')}'.")
                branch_exhaustion_counters[parent_branch.name] += 1
                if branch_exhaustion_counters[parent_branch.name] >= 3:
                    if parent_branch in active_parenting_branches:
                        active_parenting_branches.remove(parent_branch)

        # Final Interior Detailing Phase
        utils.log_message('debug', "Entering final interior detailing phase...")
        for branch in self.initial_feature_branches:
            for node in branch.get_all_nodes_in_branch():
                if not node.is_blocked and node.sentence_count < 3:
                    utils.log_message('debug', f"Adding interior details to '{node.name}'...")
                    other_features_context = "\n".join(
                        [f'({b.name} - "{b.narrative_log}")' for b in self.initial_feature_branches if
                         b.get_root() is not branch.get_root()])

                    # Generate a few interior features to fill out the narrative
                    for _ in range(3 - node.sentence_count):
                        narrative_beat = self.llm.get_next_narrative_beat(node.narrative_log, other_features_context)
                        if not narrative_beat or 'none' in narrative_beat.lower():
                            break

                        feature_data = self.llm.define_feature_from_sentence(narrative_beat)
                        if feature_data and feature_data.get('placement_strategy') == 'INTERIOR':
                            node.narrative_log += " " + narrative_beat
                            node.interior_features_to_place.append(feature_data)
                            node.sentence_count += 1
                        elif feature_data and feature_data.get('type') == 'CHARACTER':
                            gen_state.character_creation_queue.append(feature_data)
                        else:
                            break  # Stop if we get a non-interior feature

    def _ensure_all_branches_are_connected(self, all_internal_connections: List[Dict], update_and_draw: Callable) -> \
    Generator[Tuple[str, None], None, None]:
        """
        Uses a Union-Find data structure to identify disconnected "islands" of features
        and strategically creates new exterior paths to connect them until the entire
        map layout is a single contiguous graph.
        """
        if len(self.initial_feature_branches) < 2:
            return

        # Initialize Union-Find with all current root branches.
        # This creates initial sets for each independent root branch.
        uf = UnionFind(self.initial_feature_branches)

        # Process existing internal connections to unite initial branches
        for conn in all_internal_connections:
            if conn['node_a'] and conn['node_b']:
                root_a = conn['node_a'].get_root()
                root_b = conn['node_b'].get_root()
                # Ensure roots are actual roots of the initial feature branches
                if root_a in self.initial_feature_branches and root_b in self.initial_feature_branches:
                    uf.union(root_a, root_b)

        # 1. Find all suitable pathing features from config that are walkable on the ground
        available_path_types = []
        for f_type, f_def in config.features.items():
            if f_def.get('placement_strategy') == 'PATHING':
                tile_type_name = f_def.get('tile_type')
                if tile_type_name and tile_type_name in config.tile_types:
                    tile_def = config.tile_types[tile_type_name]
                    if "GROUND" in tile_def.get("pass_methods", []):
                        available_path_types.append(f_type)

        if not available_path_types:
            utils.log_message('debug',
                              "[WARNING] No suitable ground-based PATHING features found in config. Cannot create exterior connections.")
            return

        # 2. Determine the dominant nature of the existing map features
        nature_counts = defaultdict(int)
        all_nodes_in_map = [n for b in self.initial_feature_branches for n in b.get_all_nodes_in_branch()]
        for node in all_nodes_in_map:
            feature_def = config.features.get(node.feature_type, {})
            for nature in feature_def.get('natures', []):
                nature_counts[nature] += 1

        dominant_nature = max(nature_counts, key=nature_counts.get) if nature_counts else None

        # 3. Select the best path type based on nature matching
        best_path_type = None
        if dominant_nature:
            matching_paths = [
                p_type for p_type in available_path_types
                if dominant_nature in config.features.get(p_type, {}).get('natures', [])
            ]
            if matching_paths:
                best_path_type = random.choice(matching_paths)

        if not best_path_type:
            best_path_type = random.choice(available_path_types)

        utils.log_message('debug',
                          f"Selected '{best_path_type}' for exterior connections based on dominant nature: '{dominant_nature}'.")

        MAX_ATTEMPTS = 50
        attempts = 0
        failed_pairs = set()

        while uf.num_sets > 1 and attempts < MAX_ATTEMPTS:
            attempts += 1

            # Re-collect roots and branches for current state of UnionFind
            current_islands = defaultdict(list)
            for branch in self.initial_feature_branches:
                # Ensure the branch is actually in the UnionFind's parent map.
                # If a new path was added as a root branch and not yet united,
                # it might not have an entry. Initialize it if missing.
                if branch not in uf.parent:
                    uf.parent[branch] = branch
                    uf.num_sets += 1  # Temporarily increase to reflect new root, will decrease upon union

                root = uf.find(branch)
                current_islands[root].append(branch)

            if len(current_islands) <= 1: break

            # Pick two different islands to connect
            island_roots = random.sample(list(current_islands.keys()), 2)
            source_island_root, dest_island_root = island_roots[0], island_roots[1]

            source_branch = random.choice(current_islands[source_island_root])
            dest_branch = random.choice(current_islands[dest_island_root])

            # Ensure we're not trying to connect branches already in the same set
            if uf.connected(source_branch, dest_branch):
                continue

            pair_key = frozenset([source_branch, dest_branch])
            if pair_key in failed_pairs:
                continue

            feature_data = {
                'name': f"Path between {source_branch.name} and {dest_branch.name}",
                'type': best_path_type,
                'size_tier': 'small'
            }

            # Attempt to place the path
            newly_placed_path = self.placement.place_initial_path_between_branches(
                feature_data, source_branch, dest_branch, self.initial_feature_branches
            )

            if newly_placed_path:
                utils.log_message('debug',
                                  f"Successfully created exterior path between '{source_branch.name}' and '{dest_branch.name}'.")

                # Assign budgets and draw the new path
                self.manager._assign_op_budgets(newly_placed_path)
                yield from self.manager._draw_path_coroutine(newly_placed_path, update_and_draw)

                # Add the new path to UnionFind and unite the islands
                # The newly_placed_path is a new root-level branch.
                if newly_placed_path not in uf.parent:
                    uf.parent[newly_placed_path] = newly_placed_path
                uf.union(source_branch, newly_placed_path)
                uf.union(dest_branch, newly_placed_path)

            else:
                failed_pairs.add(pair_key)
                utils.log_message('debug',
                                  f"Failed to create exterior path between '{source_branch.name}' and '{dest_branch.name}'.")

        if uf.num_sets > 1:
            utils.log_message('debug', f"[WARNING] Could not connect all feature islands. {uf.num_sets} remain.")

    def generate_layout_in_steps(self, ui_callback: Callable) -> Generator[Tuple[str, GenerationState], None, None]:
        gen_state = GenerationState(self.game_map)
        artist = MapArtist()

        def update_and_draw():
            self.converter.populate_generation_state(gen_state, self.initial_feature_branches)
            artist.draw_map(self.game_map, gen_state, config.features)
            ui_callback(gen_state)

        initial_specs = self.llm.get_initial_features()
        if not initial_specs:
            yield "FINAL", gen_state
            return

        placement_generator = self._place_initial_layout(initial_specs, update_and_draw)
        placement_ok = False
        for phase, _ in placement_generator:
            yield phase, gen_state
            placement_ok = True

        if not placement_ok:
            utils.log_message('debug', "[PEGv3 FATAL] Failed to find a valid initial layout.")
            yield "FINAL", gen_state
            return

        growth_generator = self._grow_initial_features(update_and_draw)
        for phase, _ in growth_generator:
            yield phase, gen_state

        subfeature_generator = self._place_subfeatures_conversationally(gen_state, update_and_draw)
        for phase, _ in subfeature_generator:
            yield phase, gen_state

        yield "FINAL_REFINEMENT", gen_state
        refinement_generator = self.map_ops.run_refinement_phase(self.initial_feature_branches, update_and_draw)
        for _ in refinement_generator:
            yield "REFINEMENT_STEP", gen_state

        yield "TILE_RECONCILIATION", gen_state
        tile_utils.calculate_and_apply_tile_overrides(self.initial_feature_branches, self.manager._get_temp_grid)
        update_and_draw()

        yield "PRE_CONNECT", gen_state
        all_connections, doors, hallways = self.pathing.create_all_connections(self.initial_feature_branches)
        gen_state.door_locations = doors
        gen_state.blended_hallways = hallways
        update_and_draw()

        yield "EXTERIOR_CONNECT", gen_state
        exterior_connection_gen = self._ensure_all_branches_are_connected(all_connections, update_and_draw)
        for phase, _ in exterior_connection_gen:
            yield phase, gen_state

        yield "FINAL", gen_state