# /core/worldgen/world_components/world_actions.py

from core.common import file_io, utils
from core.common.config_loader import config
from core.llm.llm_api import execute_task
from .content_generator import ContentGenerator
from .world_graph_navigator import WorldGraphNavigator


class WorldActions:
    """
    A service class that handles the execution of world-building actions,
    such as navigating the graph or creating new locations. This logic is
    shared between different high-level processes like scene placement and
    autonomous exploration.
    """

    def __init__(self, engine, navigator: WorldGraphNavigator, content_generator: ContentGenerator):
        self.engine = engine
        self.navigator = navigator
        self.content_generator = content_generator
        self.atlas_agent = config.agents.get('ATLAS')

    def navigate(self, current_node, breadcrumb, world_theme, scene_prompt, connections,
                 target_location_concept: dict | None = None):
        """Handles the NAVIGATE action by asking the LLM to choose a destination."""
        if not connections: return breadcrumb, current_node, "ACCEPT"

        dest_str = "\n".join(f"- {name} ({rel})" for rel, name in connections.items())

        task_key = 'WORLDGEN_CHOOSE_NAVIGATION_TARGET_SCENEPLACE' if scene_prompt else 'WORLDGEN_CHOOSE_NAVIGATION_TARGET_EXPLORE'

        prompt_kwargs = {
            "world_theme": world_theme,
            "scene_prompt": scene_prompt or "N/A",  # Will be "N/A" for pure exploration
            "current_location_name": current_node.get("Name"),
            "available_destinations_string": dest_str
        }

        if target_location_concept:  # For scene placement navigation
            prompt_kwargs["target_location_name"] = target_location_concept.get("Name")
            prompt_kwargs["target_location_description"] = target_location_concept.get("Description")
        else:  # For pure exploration
            prompt_kwargs["target_location_name"] = "N/A"
            prompt_kwargs["target_location_description"] = "N/A"

        target_name_from_llm = execute_task(self.engine, self.atlas_agent, task_key, [],
                                            task_prompt_kwargs=prompt_kwargs)

        if target_name_from_llm:
            for rel, name_str in connections.items():
                node_names = [n.strip() for n in name_str.split(',')]
                for individual_name in node_names:
                    if individual_name.lower() in target_name_from_llm.lower():
                        new_breadcrumb = self.navigator.calculate_new_breadcrumb(breadcrumb, rel, individual_name)
                        if new_breadcrumb:
                            return new_breadcrumb, file_io._find_node_by_breadcrumb(config.world,
                                                                                    new_breadcrumb), "CONTINUE"

        utils.log_message('debug',
                          f"[ATLAS] Navigation failed. LLM chose an invalid target: '{target_name_from_llm}'. Accepting current state.")
        return breadcrumb, current_node, "ACCEPT"

    def create_and_place_location(self, world_name, world_theme, breadcrumb, current_node,
                                  relationship_override, scene_prompt,
                                  target_location_concept: dict | None = None):
        """Handles the CREATE action, including content generation and file system updates."""
        new_loc_data = self.content_generator.create_location(
            world_theme, breadcrumb, current_node, scene_prompt,
            target_location_summary=target_location_concept.get('Name',
                                                                'N/A') if target_location_concept else None
        )
        if not new_loc_data:
            utils.log_message('debug',
                              f"[ATLAS] Location creation failed for parent '{current_node.get('Name')}'. Accepting current state.")
            return breadcrumb, current_node, "ACCEPT"

        if relationship_override:
            utils.log_message('debug',
                              f"[ATLAS] Overriding generated relationship with pre-decided action: '{relationship_override}'")
            new_loc_data["Relationship"] = relationship_override

        rel = new_loc_data.pop("Relationship").upper()
        new_name = new_loc_data["Name"]

        if rel == "OUTSIDE":
            if "PARENT" in current_node.get("relationships", {}):
                new_breadcrumb = breadcrumb[:-1]
                return new_breadcrumb, file_io._find_node_by_breadcrumb(config.world, new_breadcrumb), "CONTINUE"
        elif rel in self.navigator.HIERARCHICAL_RELATIONSHIPS:
            new_loc_data.setdefault("relationships", {})["PARENT_RELATION"] = rel
            file_io.add_child_to_world(world_name, breadcrumb, new_loc_data)
            config.load_world_data(world_name)
            new_node_breadcrumb = breadcrumb + [new_name]
            file_io.add_relationship_to_node(world_name, new_node_breadcrumb, "PARENT", current_node["Name"])
            config.load_world_data(world_name)
            return new_node_breadcrumb, file_io._find_node_by_breadcrumb(config.world,
                                                                         new_node_breadcrumb), "CONTINUE"
        elif rel in self.navigator.LATTICE_RELATIONSHIPS:
            parent_breadcrumb = breadcrumb[:-1]
            file_io.add_child_to_world(world_name, parent_breadcrumb, new_loc_data)
            config.load_world_data(world_name)
            reciprocal = self.navigator.LATTICE_RELATIONSHIPS[rel]
            new_node_breadcrumb = parent_breadcrumb + [new_name]
            file_io.add_relationship_to_node(world_name, breadcrumb, rel, new_name)
            file_io.add_relationship_to_node(world_name, new_node_breadcrumb, reciprocal, current_node["Name"])
            file_io.add_relationship_to_node(world_name, new_node_breadcrumb, "PARENT",
                                             current_node.get("relationships", {}).get("PARENT"))
            config.load_world_data(world_name)
            return new_node_breadcrumb, file_io._find_node_by_breadcrumb(config.world,
                                                                         new_node_breadcrumb), "CONTINUE"
        return breadcrumb, current_node, "ACCEPT"