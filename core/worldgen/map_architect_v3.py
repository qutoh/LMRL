import re
import random
from collections import deque, defaultdict
from datetime import datetime
from typing import Callable, Tuple, Generator, Optional, List, Set, Dict

import math
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
from ..common.game_state import GenerationState, MapArtist, GameState, Entity

MAX_SUBFEATURES_TO_PLACE = 500
MIN_FEATURE_SIZE = 3
GROWTH_COVERAGE_THRESHOLD = 0.40
MAX_GROWTH_ITERATIONS = 200


class MapArchitectV3:
    def __init__(self, manager, game_map, game_state: GameState, world_theme, scene_prompt):
        self.manager = manager
        self.engine = manager.engine
        self.game_map = game_map
        self.game_state = game_state
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
        self.manual_stop_generation = False

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
                                      existing_seeds: List[FeatureNode],
                                      confining_footprint: Optional[Set[Tuple[int, int]]] = None) -> Tuple[
        List[FeatureNode], bool]:
        """Tries to place a list of feature specs as small seeds in random locations."""
        placed_nodes = []
        placement_area = list(confining_footprint) if confining_footprint else []

        for spec in specs_to_place:
            node_placed = False
            for _ in range(200):  # More attempts for constrained placement
                w, h = MIN_FEATURE_SIZE, MIN_FEATURE_SIZE

                if confining_footprint:
                    if not placement_area: return [], False  # No space left
                    x, y = random.choice(placement_area)
                else:
                    if self.map_width <= w + 2 or self.map_height <= h + 2: continue
                    x = random.randint(1, self.map_width - w - 2)
                    y = random.randint(1, self.map_height - h - 2)

                all_current_seeds = existing_seeds + placed_nodes
                # Create a collision grid ONLY from other seeds, not the container.
                temp_grid = self.manager._get_temp_grid(all_current_seeds)
                collision_mask = temp_grid != self.placement.void_space_index

                temp_node = FeatureNode(spec['name'], spec['type'], w, h, x, y)

                absolute_fp = temp_node.get_absolute_footprint()
                if confining_footprint and not absolute_fp.issubset(confining_footprint):
                    continue

                if self.placement._is_placement_valid(absolute_fp, collision_mask):
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

        def get_strat(spec):
            return config.features.get(spec.get('type'), {}).get('placement_strategy')

        self.initial_feature_branches.clear()
        region_specs = [s for s in initial_specs if get_strat(s) == 'REGION']
        path_specs = [s for s in initial_specs if get_strat(s) == 'PATHING']
        seed_specs = [s for s in initial_specs if s not in region_specs and s not in path_specs]

        # --- Phase 1: Place Seeds and Paths first ---
        placed_seeds, success = self._place_initial_non_path_seeds(seed_specs, [])
        if not success and seed_specs:
            return False  # Only fail if there were seeds to place and they failed.

        self.initial_feature_branches.extend(placed_seeds)

        node_map = {node.name: node for node in placed_seeds}
        for path_spec in path_specs:
            source_name, dest_name = path_spec.get('source'), path_spec.get('destination')
            if not source_name or not dest_name:
                utils.log_message('debug', f"Path '{path_spec['name']}' missing source/dest, cannot be placed initially.")
                continue

            source_is_border, dest_is_border = "_BORDER" in source_name.upper(), "_BORDER" in dest_name.upper()
            newly_placed_path = None

            source_node = node_map.get(source_name) if not source_is_border else None
            dest_node = node_map.get(dest_name) if not dest_is_border else None

            if source_node and dest_node:
                newly_placed_path = self.placement.place_initial_path_between_branches(path_spec, source_node,
                                                                                       dest_node,
                                                                                       self.initial_feature_branches)
            elif source_is_border and dest_node:
                coords = self.pathing.get_border_coordinates_for_direction(source_name.replace("_BORDER", ""))
                newly_placed_path = self.placement.place_initial_path_to_border(path_spec, dest_node, coords,
                                                                                self.initial_feature_branches)
            elif source_node and dest_is_border:
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

        update_and_draw()
        yield "SEED_PLACEMENT", None

        # --- Phase 2: Create Regions around existing features, or place them standalone ---
        if region_specs:
            non_region_branches = [b for b in self.initial_feature_branches if b.feature_type != 'REGION']
            if non_region_branches:
                # Group existing branches into regions
                non_region_specs = [s for s in initial_specs if get_strat(s) != 'REGION']
                assignments = self.llm.assign_features_to_regions(region_specs, non_region_specs)
                features_by_region = defaultdict(list)
                for feature_name, region_name in assignments.items():
                    features_by_region[region_name].append(feature_name)

                for region_spec in region_specs:
                    feature_names = features_by_region.get(region_spec['name'], [])
                    branches_to_group = [b for b in non_region_branches if b.name in feature_names]
                    if not branches_to_group: continue

                    new_region_node = self.placement.place_region_around_branches(region_spec, branches_to_group,
                                                                                  self.initial_feature_branches)
                    if new_region_node:
                        self.manager._assign_op_budgets(new_region_node)
                        for child in branches_to_group:
                            if child in self.initial_feature_branches: self.initial_feature_branches.remove(child)
                        self.initial_feature_branches.append(new_region_node)
            else:
                # Place regions as standalone entities by seeding them at the edges
                for region_spec in region_specs:
                    if new_region := self.placement.place_initial_region_seed_at_edge(region_spec,
                                                                                      self.initial_feature_branches):
                        self.manager._assign_op_budgets(new_region)
                        self.initial_feature_branches.append(new_region)

            update_and_draw()
            yield "REGION_PLACEMENT", None

        return True

    def _grow_initial_features(self, update_and_draw: Callable) -> Generator[Tuple[str, None], None, None]:
        """Generator to handle the growth phase of initial non-path features."""
        all_nodes = [n for b in self.initial_feature_branches for n in b.get_all_nodes_in_branch()]
        growable_nodes = [
            n for n in all_nodes if
            config.features.get(n.feature_type, {}).get('placement_strategy') not in ['PATHING']
        ]

        if not growable_nodes:
            yield "POST_GROWTH", None
            return

        total_target_area = self.map_width * self.map_height * GROWTH_COVERAGE_THRESHOLD
        target_area_per_node = total_target_area / len(growable_nodes) if growable_nodes else 0
        for node in growable_nodes:
            node.target_area = target_area_per_node
            node.target_aspect_ratio = (random.randint(1, 16), random.randint(1, 16))

        for i in range(MAX_GROWTH_ITERATIONS):
            # Calculate current area based on only growable features to respect map density
            current_area = sum(len(n.get_absolute_footprint()) for n in growable_nodes)
            if current_area >= total_target_area: break

            still_growable_nodes = [n for n in growable_nodes if not n.is_stuck and len(n.footprint) < n.target_area]
            if not still_growable_nodes: break

            tick_changed = False
            random.shuffle(still_growable_nodes)
            for feature in still_growable_nodes:
                if feature.organic_op_budget > 0:
                    if self.map_ops.apply_organic_reshaping(feature, self.initial_feature_branches):
                        tick_changed = True
                elif self.placement._attempt_aspect_aware_growth(feature, self.initial_feature_branches):
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
        branch_exhaustion_counters = defaultdict(int)

        max_depth = config.settings.get("PEG_V3_MAX_DEPTH", 0)
        max_time_minutes = config.settings.get("PEG_V3_MAX_TIME", 0)
        max_narrative_len = config.settings.get("PEG_V3_MAX_NARRATIVE_LENGTH", 0)
        start_time = datetime.now()

        for i in range(MAX_SUBFEATURES_TO_PLACE):
            if self.manual_stop_generation:
                utils.log_message('debug', "[PEGv3] Manual stop requested. Finalizing.")
                break

            if max_time_minutes > 0:
                elapsed_minutes = (datetime.now() - start_time).total_seconds() / 60
                if elapsed_minutes > max_time_minutes:
                    utils.log_message('debug',
                                      f"[PEGv3] Max generation time of {max_time_minutes} min reached. Finalizing.")
                    break

            # A branch is active if it's not a region, OR it's a region with children. Empty regions are excluded.
            active_parenting_branches = [
                b for b in self.initial_feature_branches
                if not b.is_blocked and (b.feature_type != 'REGION' or b.subfeatures)
            ]

            if not active_parenting_branches:
                empty_regions = [b for b in self.initial_feature_branches if
                                 b.feature_type == 'REGION' and not b.subfeatures and not b.is_blocked]

                if empty_regions:
                    utils.log_message('debug', "[PEGv3] No active branches. Seeding an empty region.")
                    region_to_seed = random.choice(empty_regions)
                    if not region_to_seed.narrative_log:
                        region_to_seed.narrative_log = self.llm.get_narrative_seed(region_to_seed.name) or ""

                    new_feature_sentence = self.llm.add_feature_to_empty_region(
                        region_to_seed.name, region_to_seed.narrative_log
                    )
                    if not new_feature_sentence or 'none' in new_feature_sentence.lower():
                        utils.log_message('debug',
                                          f"[PEGv3] LLM had no ideas for region '{region_to_seed.name}'. Blocking it.")
                        region_to_seed.is_blocked = True
                        continue

                    new_feature_data = self.llm.define_feature_from_sentence(
                        new_feature_sentence, exclude_strategies=['REGION']
                    )
                    if not new_feature_data or new_feature_data.get('type') == 'CHARACTER' or config.features.get(
                            new_feature_data.get('type'), {}).get('placement_strategy') == 'REGION':
                        utils.log_message('debug',
                                          f"[PEGv3] LLM failed to define a valid, non-region seed for '{region_to_seed.name}'.")
                        continue

                    newly_placed_node = self.placement.place_inside(new_feature_data, region_to_seed,
                                                                    self.initial_feature_branches)

                    if newly_placed_node:
                        utils.log_message('debug',
                                          f"[PEGv3] Seeded '{newly_placed_node.name}' inside region '{region_to_seed.name}'.")
                        self.manager._assign_op_budgets(newly_placed_node)
                        region_to_seed.narrative_log += " " + new_feature_sentence
                        yield from self.manager._process_newly_placed_nodes([newly_placed_node],
                                                                            self.initial_feature_branches,
                                                                            update_and_draw)
                    else:
                        utils.log_message('debug',
                                          f"[PEGv3] Failed to place seed '{new_feature_data.get('name')}' inside region '{region_to_seed.name}'.")
                    continue

                # If no empty regions are left, then all branches are truly exhausted.
                utils.log_message('debug', "[PEGv3] All branches and regions exhausted. Trying to start a new branch.")
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
                    update_and_draw()
                    yield "NEW_BRANCH_PLACEMENT", None
                else:
                    utils.log_message('debug', "[PEGv3] Failed to place new root branch. Finalizing map.")
                    break
                continue

            parent_branch = random.choice(active_parenting_branches)
            if max_narrative_len > 0 and utils.count_tokens(parent_branch.narrative_log) > max_narrative_len:
                utils.log_message('debug',
                                  f"[PEGv3] Branch '{parent_branch.name}' exceeded narrative length limit ({max_narrative_len} tokens). Finalizing branch.")
                parent_branch.is_blocked = True
                continue

            all_placeable_nodes_in_branch = [n for n in parent_branch.get_all_nodes_in_branch() if not n.is_blocked]

            if not all_placeable_nodes_in_branch:
                parent_branch.is_blocked = True
                continue

            parent_node = random.choice(all_placeable_nodes_in_branch)

            if not parent_branch.narrative_log:
                parent_branch.narrative_log = self.llm.get_narrative_seed(parent_branch.name) or ""

            other_features_context = "\n".join(
                [f'({b.name} - "{b.narrative_log}")' for b in self.initial_feature_branches if b is not parent_branch])

            narrative_beat = self.llm.get_next_narrative_beat(parent_branch.narrative_log, other_features_context)

            if len(list(filter(None, re.split(r'[.?!]\s', narrative_beat)))) > 1:
                utils.log_message('debug', "[PEGv3] Chatty model detected. Retrying narrative beat...")
                narrative_beat = self.llm.get_next_narrative_beat(parent_branch.narrative_log, other_features_context)
                if len(list(filter(None, re.split(r'[.?!]\s', narrative_beat)))) > 1:
                    utils.log_message('debug', "[PEGv3] Chatty model persisted. Taking first sentence.")
                    narrative_beat = re.split(r'(?<=[.?!])\s', narrative_beat)[0]

            is_in_region = parent_branch.feature_type == 'REGION' or (
                        parent_branch.parent and parent_branch.parent.feature_type == 'REGION')
            exclude_strategies = ['REGION'] if is_in_region else []
            if max_depth > 0 and parent_node.sentence_count >= max_depth:
                exclude_strategies.append('INTERIOR')

            feature_data = self.llm.define_feature_from_sentence(
                narrative_beat, exclude_strategies=exclude_strategies
            ) if narrative_beat and 'none' not in narrative_beat.lower() else None

            if not feature_data or not feature_data.get('name') or 'Unnamed' in feature_data.get('name', ''):
                utils.log_message('debug',
                                  f"[PEGv3 FAIL] Failed to define a valid feature from narrative beat: '{narrative_beat}'")
                branch_exhaustion_counters[parent_branch.name] += 1
                if branch_exhaustion_counters[parent_branch.name] >= 3:
                    parent_branch.is_blocked = True
                continue

            newly_placed_nodes = []
            placement_strategy = config.features.get(feature_data['type'], {}).get('placement_strategy')
            is_deferred = False

            if placement_strategy in ['PATHING', 'CONNECTOR']:
                if placement_strategy == 'PATHING':
                    node = self._handle_pathing_placement(feature_data, parent_branch, parent_branch.narrative_log,
                                                          narrative_beat)
                    if node: newly_placed_nodes.append(node)
                else:
                    nodes = self.placement.handle_connector_placement(feature_data, parent_branch,
                                                                      self.initial_feature_branches)
                    if nodes: newly_placed_nodes.extend(nodes)
            elif placement_strategy == 'REGION':
                all_branches = self.initial_feature_branches
                unregioned_branches = [b for b in all_branches if b.feature_type != 'REGION' and not b.parent]
                other_branch_options = [b.name for b in unregioned_branches if b is not parent_branch]
                chosen_branch_names = self.llm.choose_branches_for_region(feature_data['name'], parent_branch.name,
                                                                          other_branch_options)
                branches_to_group = [parent_branch] + [b for b in unregioned_branches if b.name in chosen_branch_names]
                new_region_node = self.placement.place_region_around_branches(feature_data, branches_to_group,
                                                                              all_branches)
                if new_region_node:
                    newly_placed_nodes.append(new_region_node)
                    for b in branches_to_group:
                        if b in self.initial_feature_branches: self.initial_feature_branches.remove(b)
                    self.initial_feature_branches.append(new_region_node)
            elif placement_strategy in ['INTERIOR', 'CHARACTER']:
                is_deferred = True
                if placement_strategy == 'INTERIOR':
                    gen_state.interior_feature_queue.append(
                        {'feature_data': feature_data, 'narrative_context': parent_node})
                else:
                    gen_state.character_creation_queue.append(
                        {'feature_data': feature_data, 'narrative_context': parent_node})
            else:
                can_be_region = len(
                    [b for b in self.initial_feature_branches if not b.parent and b.feature_type != 'REGION']) > 1
                strategy = self.llm.choose_placement_keyword(parent_branch.narrative_log, narrative_beat, can_be_region)

                if strategy not in ['INSIDE', 'OUTSIDE', 'NEARBY']:
                    strategy = 'NEARBY'

                placement_functions = {'NEARBY': self.placement.place_nearby, 'OUTSIDE': self.placement.place_outside,
                                       'INSIDE': self.placement.place_inside}
                node = placement_functions[strategy](feature_data, parent_node, self.initial_feature_branches)

                if not node:
                    other_nodes_in_branch = [n for n in parent_branch.get_all_nodes_in_branch() if n is not parent_node]
                    random.shuffle(other_nodes_in_branch)
                    for fallback_node in other_nodes_in_branch:
                        node = placement_functions[strategy](feature_data, fallback_node, self.initial_feature_branches)
                        if node: break

                if node: newly_placed_nodes.append(node)

            if newly_placed_nodes:
                parent_branch.narrative_log += " " + narrative_beat
                parent_node.sentence_count += 1
                if max_depth > 0 and parent_node.sentence_count >= max_depth:
                    parent_node.is_blocked = True
                    utils.log_message('debug',
                                      f"[PEGv3] Node '{parent_node.name}' reached max depth of {max_depth}. Blocking node.")
                branch_exhaustion_counters[parent_branch.name] = 0
                yield from self.manager._process_newly_placed_nodes(newly_placed_nodes, self.initial_feature_branches,
                                                                    update_and_draw)
            elif is_deferred:
                parent_branch.narrative_log += " " + narrative_beat
                parent_node.sentence_count += 1
                if max_depth > 0 and parent_node.sentence_count >= max_depth:
                    parent_node.is_blocked = True
                    utils.log_message('debug',
                                      f"[PEGv3] Node '{parent_node.name}' reached max depth of {max_depth} (deferred). Blocking node.")
                branch_exhaustion_counters[parent_branch.name] = 0
            else:
                utils.log_message('debug', f"Placement failed for '{feature_data.get('name')}'.")
                branch_exhaustion_counters[parent_branch.name] += 1
                if branch_exhaustion_counters[parent_branch.name] >= 3:
                    parent_branch.is_blocked = True

        yield "PRE_INTERIOR_DETAILING", None

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

    def _flesh_out_sparse_areas(self, gen_state: GenerationState, update_and_draw: Callable) -> Generator[
        Tuple[str, None], None, None]:
        """Generates new interior features for narratively sparse areas."""
        all_nodes = [n for b in self.initial_feature_branches for n in b.get_all_nodes_in_branch()]
        for node in all_nodes:
            if not node.is_blocked and not node.is_interior and node.sentence_count < 3:
                utils.log_message('debug', f"Fleshing out sparsely detailed area: '{node.name}'")
                other_features_context = "\n".join(
                    [f'({b.name} - "{b.narrative_log}")' for b in self.initial_feature_branches if
                     b.get_root() is not node.get_root()])

                for _ in range(3 - node.sentence_count):
                    narrative_beat = self.llm.get_next_narrative_beat(node.narrative_log, other_features_context)
                    if not narrative_beat or 'none' in narrative_beat.lower():
                        break

                    feature_data = self.llm.define_interior_feature_from_sentence(narrative_beat)
                    if feature_data and feature_data.get('placement_strategy') == 'INTERIOR':
                        node.narrative_log += " " + narrative_beat
                        node.interior_features_to_place.append(feature_data)
                        node.sentence_count += 1
                        if node.sentence_count >= 3:
                            node.is_blocked = True
                        yield "FLESH_OUT_STEP", None
                    elif feature_data and feature_data.get('type') == 'CHARACTER':
                        gen_state.character_creation_queue.append(
                            {'feature_data': feature_data, 'narrative_context': node})
                    else:
                        break
        yield "FLESH_OUT_COMPLETE", None

    def _interior_detailing_phase(self, gen_state: GenerationState, update_and_draw: Callable) -> Generator[
        Tuple[str, None], None, None]:
        """Handles the deferred placement of all interior features and characters."""
        utils.log_message('debug', "Entering final interior detailing phase...")

        all_nodes = [n for b in self.initial_feature_branches for n in b.get_all_nodes_in_branch()]

        # 1. Place queued interior features
        for item in gen_state.interior_feature_queue:
            feature_data = item['feature_data']
            narrative_node = item['narrative_context']

            container_options = [n.name for n in all_nodes if
                                 config.features.get(n.feature_type, {}).get('feature_type') in ['CONTAINER',
                                                                                                 'REGION'] and not n.is_interior]
            if not container_options:
                utils.log_message('debug',
                                  f"No valid containers to place interior item '{feature_data['name']}'. Skipping.")
                continue

            chosen_parent_name = self.llm.choose_interior_location(feature_data, container_options)
            parent_node = resolve_parent_node(chosen_parent_name, all_nodes, self.semantic_search) or narrative_node

            if parent_node:
                parent_node.interior_features_to_place.append(feature_data)

        # After assigning all queued features, run the final placement for all nodes
        for _ in self.interior.finalize_placements(self.initial_feature_branches, gen_state):
            update_and_draw()
            yield "INTERIOR_PLACEMENT_STEP", None

        # 2. Place queued characters
        for item in gen_state.character_creation_queue:
            char_data = item['feature_data']
            container_options = [n.name for n in all_nodes if not n.is_interior]
            if not container_options:
                utils.log_message('debug', f"No valid locations to place character '{char_data['name']}'. Skipping.")
                continue

            chosen_parent_name = self.llm.choose_interior_location(char_data, container_options)
            parent_node = resolve_parent_node(chosen_parent_name, all_nodes, self.semantic_search)

            if parent_node:
                interior_tiles = list(parent_node.get_absolute_interior_footprint())
                if interior_tiles:
                    walkable_tiles = [p for p in interior_tiles if self.game_map.is_walkable(p[0], p[1])]
                    if walkable_tiles:
                        x, y = random.choice(walkable_tiles)
                        new_entity = Entity(name=char_data['name'], x=x, y=y, char='C', color=(255, 255, 0))
                        self.game_state.add_entity(new_entity)
                        update_and_draw()
                        yield "CHARACTER_PLACEMENT", None

        # Rerun finalize_placements to place any newly generated items
        for _ in self.interior.finalize_placements(self.initial_feature_branches, gen_state):
            update_and_draw()
            yield "INTERIOR_PLACEMENT_STEP", None

        yield "INTERIOR_DETAILING_COMPLETE", None

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
        try:
            while True:
                phase, _ = next(placement_generator)
                yield phase, gen_state
        except StopIteration as e:
            placement_ok = e.value if e.value is not None else True

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

        yield "PRE_CONNECT", gen_state
        all_connections, doors, hallways = self.pathing.create_all_connections(self.initial_feature_branches)
        gen_state.door_locations = doors
        gen_state.blended_hallways = hallways
        update_and_draw()

        yield "EXTERIOR_CONNECT", gen_state
        exterior_connection_gen = self._ensure_all_branches_are_connected(all_connections, update_and_draw)
        for phase, _ in exterior_connection_gen:
            yield phase, gen_state

        fleshing_out_generator = self._flesh_out_sparse_areas(gen_state, update_and_draw)
        for phase, _ in fleshing_out_generator:
            yield phase, gen_state

        interior_detailing_generator = self._interior_detailing_phase(gen_state, update_and_draw)
        for phase, _ in interior_detailing_generator:
            yield phase, gen_state

        yield "TILE_RECONCILIATION", gen_state
        tile_utils.calculate_and_apply_tile_overrides(self.initial_feature_branches, self.manager._get_temp_grid)
        update_and_draw()

        yield "FINAL", gen_state