# /core/worldgen/world_components/atlas_logic.py

from core.common import file_io, utils
from core.common.config_loader import config
from core.llm.llm_api import execute_task
from .content_generator import ContentGenerator
from .world_actions import WorldActions
from .world_graph_navigator import WorldGraphNavigator

GENERIC_ACTIONS = ["NAVIGATE", "CREATE", "ACCEPT", "POPULATE"]


class AtlasLogic:
    """
    A service class that encapsulates the "thinking" process of the Atlas agent.
    It builds context strings and uses the LLM to make strategic decisions.
    """

    def __init__(self, engine, navigator: WorldGraphNavigator, content_generator: ContentGenerator,
                 world_actions: WorldActions):
        self.engine = engine
        self.navigator = navigator
        self.content_generator = content_generator
        self.world_actions = world_actions
        self.atlas_agent = config.agents.get('ATLAS')

    def build_context_for_decision(self, breadcrumb_trail: list, current_location: dict,
                                   visited_breadcrumbs: list = None) -> tuple[str, dict]:
        """Builds a descriptive string of the current location and its connections for the LLM."""
        if visited_breadcrumbs is None:
            visited_breadcrumbs = []

        lines = [f"You are currently at: {self.navigator.build_descriptive_breadcrumb_trail(breadcrumb_trail)}"]
        lines.append(f"  - Description: {current_location.get('Description', 'N/A')}")

        inhabitants = current_location.get('inhabitants', [])
        inhabitant_details = []
        for i in inhabitants:
            name, desc = None, None
            if isinstance(i, str):
                parts = i.split(' - ', 1)
                name = parts[0].strip()
                if len(parts) > 1:
                    desc = parts[1].strip()
            elif isinstance(i, dict):
                name = i.get('name')
                desc = i.get('description')

            if name and desc:
                inhabitant_details.append(f"{name} ({desc})")
            elif name:
                inhabitant_details.append(name)

        lines.append(f"  - Inhabitants: {', '.join(inhabitant_details) or 'None'}")
        lines.append("\n  Available connections:")

        connections = {}
        has_connections = False

        # Get all explicit relationships
        for rel, name in current_location.get("relationships", {}).items():
            if rel not in ["PARENT", "PARENT_RELATION"]:
                connections[rel] = name
                has_connections = True

        # Get all hierarchical children
        for name, child_data in current_location.get("Children", {}).items():
            rel = child_data.get("relationships", {}).get("PARENT_RELATION", "INSIDE")
            if rel in connections:
                connections[rel] += f", {name}"
            else:
                connections[rel] = name
            has_connections = True

        # Add parent relationship as OUTSIDE
        if "PARENT" in current_location.get("relationships", {}):
            connections["OUTSIDE"] = current_location["relationships"]["PARENT"]
            has_connections = True

        if has_connections:
            for rel, name_str in sorted(connections.items()):
                node_names = [n.strip() for n in name_str.split(',')]
                for name in node_names:
                    # Find the node to get its description
                    breadcrumb_to_node = self.navigator.find_breadcrumb_for_node(name)
                    node_data = file_io._find_node_by_breadcrumb(config.world,
                                                                 breadcrumb_to_node) if breadcrumb_to_node else None
                    desc = f" ({node_data.get('Description')})" if node_data and node_data.get('Description') else ""
                    visited_annot = " (recently visited)" if breadcrumb_to_node in visited_breadcrumbs else ""
                    lines.append(f"  - {rel}: {name}{desc}{visited_annot}")
        else:
            lines.append("  - None")

        return "\n".join(lines), connections

    def decide_next_action_for_scene_placement(self, world_theme: str, scene_prompt: str, context_string: str,
                                               target_location: dict) -> str:
        prompt_kwargs = {
            "world_theme": world_theme,
            "scene_prompt": scene_prompt,
            "location_context_string": context_string,
            "target_location_name": target_location.get("Name"),
            "target_location_description": target_location.get("Description")
        }
        response = execute_task(self.engine, self.atlas_agent, 'WORLDGEN_DECIDE_ACTION_SCENEPLACE', [],
                                task_prompt_kwargs=prompt_kwargs)
        for keyword in GENERIC_ACTIONS:
            if keyword in response.upper():
                return keyword
        return "ACCEPT"

    def decide_explore_action(self, world_theme: str, context_string: str) -> str:
        """Decides the next autonomous action during world exploration."""
        goal_description = "build out the world."
        prompt_kwargs = {
            "world_theme": world_theme,
            "goal_description": goal_description,
            "scene_prompt": "N/A",
            "location_context_string": context_string
        }
        return execute_task(self.engine, self.atlas_agent, 'WORLDGEN_DECIDE_ACTION_EXPLORE', [],
                            task_prompt_kwargs=prompt_kwargs).upper()

    def determine_final_relationship(self, current_node: dict, target_node: dict, breadcrumb: list) -> str:
        available_rels = self.navigator.get_available_relationships(breadcrumb, current_node)
        prompt_kwargs = {
            "current_location_name": current_node.get("Name"),
            "current_location_description": current_node.get("Description"),
            "target_location_name": target_node.get("Name"),
            "target_location_description": target_node.get("Description"),
            "available_relationships_list": "\n- ".join(available_rels)
        }
        response = execute_task(self.engine, self.atlas_agent, 'WORLDGEN_DETERMINE_FINAL_RELATIONSHIP', [],
                                task_prompt_kwargs=prompt_kwargs)
        if response and response.strip().upper() in available_rels:
            return response.strip().upper()
        return "NEARBY"

    def run_exploration_step(self, world_name: str, world_theme: str, breadcrumb: list[str], current_node: dict,
                             visited_breadcrumbs: list):
        """Orchestrates a single, autonomous step of world exploration."""
        context_string, connections = self.build_context_for_decision(breadcrumb, current_node, visited_breadcrumbs)

        raw_action = self.decide_explore_action(world_theme, context_string)
        action = raw_action.strip().replace('`', '')
        utils.log_message('game', f"[ATLAS] Decided action: {action}")

        # --- Handle Action Short-circuiting ---
        if action in connections:
            target_name = connections[action]
            utils.log_message('game', f"[ATLAS] Action '{action}' moved to '{target_name}'.")
            new_breadcrumb = self.navigator.calculate_new_breadcrumb(breadcrumb, action, target_name)
            if new_breadcrumb:
                return new_breadcrumb, file_io._find_node_by_breadcrumb(config.world, new_breadcrumb), "CONTINUE"
        elif action in self.navigator.get_available_relationships(breadcrumb,
                                                                  current_node) and action not in connections:
            utils.log_message('game', f"[ATLAS] Action '{action}' pushed through the void.")
            return self.world_actions.create_and_place_location(world_name, world_theme, breadcrumb, current_node,
                                                                relationship_override=action, scene_prompt=None)

        # --- Handle Generic Actions ---
        if "NAVIGATE" in action:
            return self.world_actions.navigate(current_node, breadcrumb, world_theme, None, connections)
        elif "CREATE" in action:
            return self.world_actions.create_and_place_location(world_name, world_theme, breadcrumb, current_node,
                                                                relationship_override=None, scene_prompt=None)
        elif "POPULATE" in action:
            if new_npc := self.content_generator.generate_npc_concept_for_location(world_theme, current_node):
                file_io.add_inhabitant_to_location(world_name, breadcrumb, {"name": new_npc.split(' - ')[0],
                                                                            "description":
                                                                                new_npc.split(' - ', 1)[1]})
                config.load_world_data(world_name)
                return breadcrumb, file_io._find_node_by_breadcrumb(config.world, breadcrumb), "CONTINUE"

        return breadcrumb, current_node, "ACCEPT"