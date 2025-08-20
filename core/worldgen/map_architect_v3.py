# /core/worldgen/map_architect_v3.py

import random
from typing import Callable, Tuple, Generator

from .v3_components.converter import Converter
from .v3_components.interior import Interior
from .v3_components.map_ops import MapOps
from .v3_components.pathing import Pathing
from .v3_components.placement import Placement
from .v3_components.v3_llm import V3_LLM
from ..common.config_loader import config
from ..common.game_state import GenerationState, MapArtist

# --- Algorithm Constants ---
MAX_SUBFEATURES_TO_PLACE = 15


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

        self.llm = V3_LLM(engine)
        self.placement = Placement(self.map_width, self.map_height)
        self.map_ops = MapOps(self.map_width, self.map_height)
        self.interior = Interior(self.placement, self.map_width, self.map_height)
        self.pathing = Pathing(self.game_map)
        self.converter = Converter()

        self.initial_feature_branches = []

    def generate_layout_in_steps(self, ui_callback: Callable) -> Generator[Tuple[str, GenerationState], None, None]:
        """
        The main orchestration method for the V3 architect. Yields tuples of (phase_name, state)
        to allow the UI to pause and display step-by-step progress.
        """
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

        self.initial_feature_branches = self.placement.place_initial_features(initial_specs)
        update_and_draw_no_delay()
        yield "INITIAL_PLACEMENT", gen_state

        shrink_factor = 0.1
        for i in range(MAX_SUBFEATURES_TO_PLACE):
            if not self.initial_feature_branches: break
            parent_branch = self.initial_feature_branches[i % len(self.initial_feature_branches)]

            if not parent_branch.narrative_log: parent_branch.narrative_log = self.llm.get_narrative_seed(
                parent_branch.name) or ""

            other_features_context_list = [
                f"({branch.feature_type} - \"{branch.narrative_log}...\")"
                for branch in self.initial_feature_branches if branch is not parent_branch
            ]
            other_features_context = "\n".join(other_features_context_list) if other_features_context_list else "None."

            narrative_beat = self.llm.get_next_narrative_beat(parent_branch.narrative_log, other_features_context)
            if not narrative_beat: continue
            parent_branch.narrative_log += " " + narrative_beat
            feature_data = self.llm.define_feature_from_sentence(narrative_beat)
            if not feature_data or feature_data['type'] == 'CHARACTER': continue

            if feature_data.get('placement_strategy') == 'INTERIOR':
                parent_options = parent_branch.get_all_nodes_in_branch()
                if parent_options:
                    chosen_parent_node = random.choice(parent_options)
                    chosen_parent_node.interior_features_to_place.append(feature_data)
                yield "SUBFEATURE_STEP", gen_state
                continue

            parent_options_str = "\n".join(
                f"- {name}" for name in sorted(list(set(n.name for n in parent_branch.get_all_nodes_in_branch()))))
            chosen_parent_name = self.llm.choose_parent_feature(
                parent_branch.narrative_log, narrative_beat, parent_options_str
            )

            if not chosen_parent_name or 'none' in chosen_parent_name.lower(): continue
            if self.placement.find_and_place_subfeature(
                    feature_data, parent_branch, self.initial_feature_branches, chosen_parent_name, shrink_factor
            ):
                update_and_draw_no_delay()
                yield "SUBFEATURE_STEP", gen_state

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
        yield "VERTEX_DATA", gen_state