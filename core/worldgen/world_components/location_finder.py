# /core/worldgen/world_components/location_finder.py

import random

from sentence_transformers.util import cos_sim

from core.common import utils, file_io, command_parser
from core.common.config_loader import config
from core.llm.llm_api import execute_task
from .atlas_logic import AtlasLogic
from .content_generator import ContentGenerator
from .world_actions import WorldActions
from .world_graph_navigator import WorldGraphNavigator


class LocationFinder:
    """
    A service class responsible for the entire scene placement process, from finding
    a starting point via search to navigating and creating new locations to find
    a home for a scene prompt.
    """

    def __init__(self, engine, navigator: WorldGraphNavigator, content_generator: ContentGenerator,
                 atlas_logic: AtlasLogic, world_actions: WorldActions):
        self.engine = engine
        self.navigator = navigator
        self.content_generator = content_generator
        self.atlas_logic = atlas_logic
        self.world_actions = world_actions
        self.atlas_agent = config.agents.get('ATLAS')

    def get_all_locations_with_breadcrumbs(self) -> list[tuple[dict, list]]:
        """Recursively traverses the world dict and returns a flat list of (location_data, breadcrumb) tuples."""
        locations = []

        def recurse_nodes(node_dict, current_path):
            for name, data in node_dict.items():
                if isinstance(data, dict):
                    new_path = current_path + [name]
                    locations.append((data, new_path))
                    if "Children" in data and data["Children"]:
                        recurse_nodes(data["Children"], new_path)

        recurse_nodes(config.world, [])
        return locations

    def find_best_location_by_semantic_search(self, scene_prompt: str) -> list:
        """Uses vector embeddings to find the most thematically similar location for a scene."""
        if not self.engine.embedding_model:
            utils.log_message('debug', "[ATLAS] Embedding model not available for search. Falling back to random.")
            return self.get_random_starting_breadcrumb()

        all_locations = self.get_all_locations_with_breadcrumbs()
        if not all_locations:
            return [next(iter(config.world))]  # Fallback to root if no locations found

        # Create a text document for each location
        location_docs = [f"{loc.get('Name', '')}: {loc.get('Description', '')}" for loc, _ in all_locations]

        # Generate embeddings
        scene_embedding = self.engine.embedding_model.encode(scene_prompt, convert_to_tensor=True)
        location_embeddings = self.engine.embedding_model.encode(location_docs, convert_to_tensor=True)

        # Find the best match
        similarities = cos_sim(scene_embedding, location_embeddings)
        best_match_index = similarities.argmax().item()

        best_location_node, best_breadcrumb = all_locations[best_match_index]
        utils.log_message('debug',
                          f"[ATLAS] Semantic search best match: '{best_location_node.get('Name')}' (Score: {similarities[0][best_match_index]:.4f})")
        return best_breadcrumb

    def get_random_starting_breadcrumb(self) -> list:
        """Traverses the entire world structure and returns the breadcrumb of a random node."""
        all_locations = self.get_all_locations_with_breadcrumbs()
        if not all_locations:
            return [next(iter(config.world))]

        _, random_breadcrumb = random.choice(all_locations)
        return random_breadcrumb

    def find_or_create_location_for_scene(self, world_name: str, world_theme: str, scene_prompt: str) -> tuple[
                                                                                                                dict, list] | tuple[
                                                                                                                None, None]:
        utils.log_message('game', "[ATLAS] Starting to place the scene...")

        # Step 1: Create the target location concept in a vacuum.
        raw_target_response = execute_task(self.engine, self.atlas_agent, 'WORLDGEN_CREATE_LOCATION_FROM_SCENE', [],
                                           task_prompt_kwargs={'scene_prompt': scene_prompt,
                                                               'world_theme': world_theme})
        target_location_concept = command_parser.parse_structured_command(self.engine, raw_target_response, 'ATLAS',
                                                                          'CH_FIX_ATLAS')
        if not target_location_concept or not all(
                k in target_location_concept for k in ['Name', 'Description', 'Type']):
            utils.log_message('debug',
                              "[ATLAS ERROR] Could not create a valid target location concept from the scene. Aborting.")
            return None, None

        utils.log_message('game',
                          f"[ATLAS] Conceived target location: '{target_location_concept['Name']}' ({target_location_concept['Type']})")

        # Step 2: Find the best starting point for navigation.
        strategy = config.settings.get("ATLAS_SCENE_PLACEMENT_STRATEGY", "SEARCH").upper()
        if strategy == 'SEARCH':
            breadcrumb = self.find_best_location_by_semantic_search(scene_prompt)
        else:  # RANDOM or ROOT
            breadcrumb = self.get_random_starting_breadcrumb()

        visited_breadcrumbs = [breadcrumb]
        max_depth = 10

        # Step 3: Navigate/build towards the target.
        for i in range(max_depth):
            current_node = file_io._find_node_by_breadcrumb(config.world, breadcrumb)
            if not current_node:
                utils.log_message('debug',
                                  f"[ATLAS WARNING] Breadcrumb {breadcrumb} was invalid. Resetting to world root.")
                breadcrumb = [next(iter(config.world))]
                current_node = file_io._find_node_by_breadcrumb(config.world, breadcrumb)

            utils.log_message('game',
                              f"[ATLAS] Step {i + 1}/{max_depth}: Currently at {self.navigator.build_descriptive_breadcrumb_trail(breadcrumb)}")
            context_string, connections = self.atlas_logic.build_context_for_decision(breadcrumb, current_node,
                                                                                      visited_breadcrumbs)

            action = self.atlas_logic.decide_next_action_for_scene_placement(world_theme, scene_prompt,
                                                                             context_string,
                                                                             target_location_concept).upper()
            utils.log_message('game', f"[ATLAS] Decided action: {action}")

            if action == "ACCEPT":
                break

            if action == "NAVIGATE":
                if not connections:
                    utils.log_message('debug', "[ATLAS] Decided to NAVIGATE but no connections exist. Breaking loop.")
                    break

                breadcrumb, current_node, _ = self.world_actions.navigate(current_node, breadcrumb, world_theme,
                                                                          scene_prompt, connections,
                                                                          target_location_concept)
                visited_breadcrumbs.append(breadcrumb)


            elif action == "CREATE":
                breadcrumb, current_node, _ = self.world_actions.create_and_place_location(world_name, world_theme,
                                                                                           breadcrumb,
                                                                                           current_node, None,
                                                                                           scene_prompt,
                                                                                           target_location_concept)
                visited_breadcrumbs.append(breadcrumb)

        # Step 4: Finalize placement.
        current_node = file_io._find_node_by_breadcrumb(config.world, breadcrumb)
        final_relationship = self.atlas_logic.determine_final_relationship(current_node, target_location_concept,
                                                                           breadcrumb)
        utils.log_message('game',
                          f"[ATLAS] Finalizing placement. Connecting '{target_location_concept['Name']}' to '{current_node['Name']}' with relationship: {final_relationship}")

        target_location_concept["Relationship"] = final_relationship
        rel = target_location_concept.pop("Relationship").upper()
        new_name = target_location_concept["Name"]
        final_breadcrumb = breadcrumb

        if rel in self.navigator.HIERARCHICAL_RELATIONSHIPS:
            target_location_concept.setdefault("relationships", {})["PARENT_RELATION"] = rel
            file_io.add_child_to_world(world_name, breadcrumb, target_location_concept)
            final_breadcrumb = breadcrumb + [new_name]
        elif rel in self.navigator.LATTICE_RELATIONSHIPS:
            parent_breadcrumb = breadcrumb[:-1]
            file_io.add_child_to_world(world_name, parent_breadcrumb, target_location_concept)
            reciprocal = self.navigator.LATTICE_RELATIONSHIPS.get(rel)
            final_breadcrumb = parent_breadcrumb + [new_name]
            file_io.add_relationship_to_node(world_name, breadcrumb, rel, new_name)
            if reciprocal:
                file_io.add_relationship_to_node(world_name, final_breadcrumb, reciprocal, current_node["Name"])

        config.load_world_data(world_name)
        final_node = file_io._find_node_by_breadcrumb(config.world, final_breadcrumb)
        return final_node, final_breadcrumb