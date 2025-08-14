# /core/worldgen/map_architect.py

import random
from typing import Callable

from .v3_components.v3_llm import V3_LLM
from .v3_components.placement import Placement
from .v3_components.map_ops import MapOps
from .v3_components.interior import Interior
from .v3_components.pathing import Pathing
from .v3_components.converter import Converter
from .v3_components.feature_node import FeatureNode

from ..common.game_state import GenerationState, MapArtist
from ..common.config_loader import config
from ..common import utils


class MapArchitect:
    """
    Handles map generation using a conversational, narrative-first approach (V1),
    but refactored to use the robust V3 components for placement, jittering, and pathing.
    """

    def __init__(self, engine, game_map, world_theme, scene_prompt):
        self.engine = engine
        self.game_map = game_map
        self.map_width = game_map.width
        self.map_height = game_map.height
        self.scene_prompt = scene_prompt

        # Shared V3 Components
        self.llm = V3_LLM(engine)
        self.placement = Placement(self.map_width, self.map_height)
        self.map_ops = MapOps(self.map_width, self.map_height)
        self.interior = Interior(self.placement, self.map_width, self.map_height)
        self.pathing = Pathing(self.game_map)
        self.converter = Converter()

        self.initial_feature_branches = []
        self.narrative_log = ""

    def generate_layout(self, ui_callback: Callable) -> GenerationState:
        """The main orchestrator for the refactored V1 algorithm."""
        gen_state = GenerationState(self.game_map)
        artist = MapArtist()

        def update_and_draw():
            self.converter.populate_generation_state(gen_state, self.initial_feature_branches)
            artist.draw_map(self.game_map, gen_state, config.features)
            ui_callback(gen_state)

        # --- Phase 1: Conversational Generation and Initial Placement ---
        self.narrative_log = f"The story begins like this: {self.scene_prompt}"
        narrative_beat = self.llm.get_next_narrative_beat(self.narrative_log)
        if not narrative_beat: return gen_state
        self.narrative_log += " " + narrative_beat
        utils.log_message('story', f"-> {narrative_beat}")

        root_feature_data = self.llm.define_feature_from_sentence(narrative_beat)
        if not root_feature_data or root_feature_data.get('type') == 'CHARACTER':
            utils.log_message('debug',
                              "[PEG Architect] The first narrative beat must be a place, not a character. Aborting.")
            return gen_state

        size_tier_map = {'large': 0.8, 'medium': 0.5, 'small': 0.25}
        rel_dim = size_tier_map.get(root_feature_data.get('size_tier', 'medium'), 0.5)
        w = int((self.map_width - 2) * rel_dim)
        h = int((self.map_height - 2) * rel_dim)
        x = (self.map_width - w) // 2
        y = (self.map_height - h) // 2
        root_node = FeatureNode(root_feature_data['name'], root_feature_data['type'], rel_dim, rel_dim, w, h, x, y)
        self.initial_feature_branches.append(root_node)
        update_and_draw()

        # Loop to generate subsequent features
        features_placed = 1
        max_features = config.settings.get('MAX_PROCGEN_FEATURES', 5)
        shrink_factor = 0.1

        while features_placed < max_features:
            narrative_beat = self.llm.get_next_narrative_beat(self.narrative_log)
            if not narrative_beat: break
            self.narrative_log += " " + narrative_beat
            utils.log_message('story', f"-> {narrative_beat}")

            feature_data = self.llm.define_feature_from_sentence(narrative_beat)
            if not feature_data: continue

            if feature_data.get('type') == 'CHARACTER':
                gen_state.character_creation_queue.append(feature_data)
                utils.log_message('debug',
                                  f"[PEG] Identified character '{feature_data.get('name')}' and queued for creation.")
                continue

            all_nodes_in_map = [node for branch in self.initial_feature_branches for node in
                                branch.get_all_nodes_in_branch()]
            parent_options_str = "\n".join(f"- {name}" for name in sorted(list(set(n.name for n in all_nodes_in_map))))
            chosen_parent_name = self.llm.choose_parent_feature(self.narrative_log, narrative_beat, parent_options_str)
            if not chosen_parent_name or 'none' in chosen_parent_name.lower(): continue

            parent_branch = next((branch for branch in self.initial_feature_branches if
                                  chosen_parent_name in [n.name for n in branch.get_all_nodes_in_branch()]), None)
            if not parent_branch: parent_branch = self.initial_feature_branches[0]

            if feature_data.get('placement_strategy') == 'INTERIOR':
                chosen_parent_node = next(
                    (n for n in parent_branch.get_all_nodes_in_branch() if n.name == chosen_parent_name), parent_branch)
                chosen_parent_node.interior_features_to_place.append(feature_data)
            else:
                if self.placement.find_and_place_subfeature(feature_data, parent_branch, self.initial_feature_branches,
                                                            chosen_parent_name, shrink_factor):
                    features_placed += 1
                    update_and_draw()

        # --- Phase 2 & 3: Jitter and Interior Placement ---
        self.map_ops.apply_jitter(self.initial_feature_branches, on_iteration_end=update_and_draw)
        interior_generator = self.interior.finalize_placements(self.initial_feature_branches, gen_state)
        for _ in interior_generator:
            update_and_draw()

        # --- Phase 4: Pathing ---
        all_connections = self.pathing.create_all_connections(self.initial_feature_branches)
        all_door_coords = [coord for conn in all_connections for coord in conn.get('door_coords', [])]
        gen_state.door_locations = all_door_coords

        # --- Phase 5: Final Conversion ---
        gen_state.layout_graph = self.converter.serialize_feature_tree_to_graph(self.initial_feature_branches)
        self.converter.populate_generation_state(gen_state, self.initial_feature_branches)
        gen_state.physics_layout = self.converter.convert_to_vertex_representation(gen_state.layout_graph,
                                                                                    all_connections)
        gen_state.narrative_log = self.narrative_log
        artist.draw_map(self.game_map, gen_state, config.features)
        ui_callback(gen_state)
        return gen_state