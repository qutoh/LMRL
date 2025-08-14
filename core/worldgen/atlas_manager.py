# /core/worldgen/atlas_manager.py

from core.common.config_loader import config
from core.llm.llm_api import execute_task
from core.common import file_io, utils, command_parser

HIERARCHICAL_RELATIONSHIPS = ["INSIDE", "NEARBY", "REGION"]
LATTICE_RELATIONSHIPS = {
    "NORTH": "SOUTH", "SOUTH": "NORTH",
    "EAST": "WEST", "WEST": "EAST",
    "UP": "DOWN", "DOWN": "UP",
    "ABOVE": "BELOW", "BELOW": "ABOVE",
    "FORWARD": "BACK", "BACK": "FORWARD",
    "LEFT": "RIGHT", "RIGHT": "LEFT"
}
VALID_RELATIONSHIPS = HIERARCHICAL_RELATIONSHIPS + list(LATTICE_RELATIONSHIPS.keys())
GENERIC_ACTIONS = ["NAVIGATE", "CREATE", "ACCEPT", "POPULATE"]


class AtlasManager:
    """
    Manages the procedural generation of a new world and can now also find
    or create a home for a pre-existing scene within that world.
    """

    def __init__(self, engine):
        self.engine = engine
        self.atlas_agent = config.agents.get('ATLAS')
        if not self.atlas_agent:
            raise ValueError("ATLAS agent not found in agents.json")

    def _get_valid_relationships(self, breadcrumb: list[str]) -> list[str]:
        """Returns a list of valid relationship types based on the current location's depth."""
        # A root node cannot have a lattice relationship (NORTH, SOUTH, etc.)
        if len(breadcrumb) <= 1:
            return HIERARCHICAL_RELATIONSHIPS
        return VALID_RELATIONSHIPS

    def _generate_npc_concept_for_location(self, world_theme: str, location: dict) -> str | None:
        """Orchestrates the generation of a single NPC concept string for a location."""
        utils.log_message('game',
                          f"[ATLAS] Decided to generate an NPC concept for {location.get('Name', 'this place')}.")
        loc_name = location.get("Name", "this place")
        loc_desc = location.get("Description", "")

        npc_concept_kwargs = {"world_theme": world_theme, "location_name": loc_name, "location_description": loc_desc}
        npc_concept = execute_task(
            self.engine,
            self.atlas_agent,
            'WORLDGEN_CREATE_NPC_CONCEPT',
            [],
            task_prompt_kwargs=npc_concept_kwargs
        )

        if npc_concept and " - " in npc_concept:
            utils.log_message('game', f"[ATLAS] ...generated concept: '{npc_concept}'.")
            return npc_concept.strip()
        return None
    def _build_context_for_decision(self, breadcrumb_trail: list, current_location: dict) -> tuple[str, dict]:
        """Builds a descriptive string of the current location and its connections for the LLM."""
        lines = [f"You are currently at: {self._build_descriptive_breadcrumb_trail(breadcrumb_trail)}"]
        lines.append(f"  - Description: {current_location.get('Description', 'N/A')}")

        inhabitants = current_location.get('inhabitants', [])
        inhabitant_names = [i.split(' - ')[0] if isinstance(i, str) else i.get('name', '') for i in inhabitants]
        lines.append(f"  - Inhabitants: {', '.join(filter(None, inhabitant_names)) or 'None'}")
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
                    breadcrumb_to_node = self._find_breadcrumb_for_node(name)
                    node_data = file_io._find_node_by_breadcrumb(config.world,
                                                                 breadcrumb_to_node) if breadcrumb_to_node else None
                    desc = f" ({node_data.get('Description')})" if node_data and node_data.get('Description') else ""
                    lines.append(f"  - {rel}: {name}{desc}")
        else:
            lines.append("  - None")

        return "\n".join(lines), connections

    def _find_breadcrumb_for_node(self, target_name: str, current_data=None, current_breadcrumb=None, visited=None) -> list | None:
        """Recursively searches the world data for a node by name and returns its breadcrumb, now with cycle detection."""
        if current_data is None: current_data = config.world
        if current_breadcrumb is None: current_breadcrumb = []
        if visited is None: visited = set()

        for key, value in current_data.items():
            if key in visited:
                continue

            if isinstance(value, dict):
                visited.add(key)
                new_breadcrumb = current_breadcrumb + [key]
                if value.get("Name") == target_name:
                    return new_breadcrumb

                found = self._find_breadcrumb_for_node(target_name, value.get("Children"), new_breadcrumb, visited)
                if found:
                    return found
        return None

    def _build_descriptive_breadcrumb_trail(self, breadcrumb_list: list) -> str:
        """Traverses the world data to build a descriptive path string using Type."""
        if not breadcrumb_list:
            return "Top Level"
        parts = []
        for i in range(len(breadcrumb_list)):
            node = file_io._find_node_by_breadcrumb(config.world, breadcrumb_list[:i + 1])
            if not node:
                parts.append(f"{breadcrumb_list[i]} (Unknown Location)")
                break
            type_str = node.get("Type", "Location")
            parts.append(f"{breadcrumb_list[i]} ({type_str})")
        return " -> ".join(parts)

    def _get_world_name(self, theme: str, ui_manager=None) -> str:
        while True:
            name = execute_task(self.engine, self.atlas_agent, 'WORLDGEN_NAME_WORLD', [],
                                task_prompt_kwargs={'world_theme': theme})
            if name and not any(p in name for p in [',', '.', '!']):
                break
            utils.log_message('debug', f"[ATLAS] Invalid world name generated: '{name}'.")
            if config.settings.get("ATLAS_PLAYER_FALLBACK_ON_NAME_FAIL", False) and ui_manager:
                reason = "The AI generated an invalid name (contains punctuation or is empty)."
                prompt_text = f"LLM FAILED: {reason}\n\nInvalid Name: '{name}'\nnEnter a new name, or leave blank for a random one:"
                player_name = ui_manager.get_player_input(prompt_text)
                name = player_name.strip() if player_name and player_name.strip() else file_io.get_random_world_name()
                break
            else:
                utils.log_message('debug', "[ATLAS] Retrying name generation...")
        return file_io.sanitize_filename(name.strip())

    def _create_location_stepwise_fallback(self, world_theme: str, parent_location: dict, context: str,
                                           relationship_override: str | None) -> dict | None:
        """Fallback: Creates a new location using a robust, multi-step process."""
        utils.log_message('debug', "[ATLAS] Single-shot failed. Falling back to step-wise generation.")
        name_kwargs = {"world_theme": world_theme, "parent_name": parent_location.get("Name", "the world"),
                       "parent_description": parent_location.get("Description", ""), "user_idea": "A new place."}
        new_name = execute_task(self.engine, self.atlas_agent, 'WORLDGEN_CREATE_LOCATION_NAME', [],
                                task_prompt_kwargs=name_kwargs)
        if not new_name or not new_name.strip(): new_name = file_io.get_random_name()

        desc_kwargs = {"world_theme": world_theme, "parent_name": parent_location.get("Name", "the world"),
                       "parent_description": parent_location.get("Description", ""), "new_name": new_name}
        new_desc = execute_task(self.engine, self.atlas_agent, 'WORLDGEN_CREATE_LOCATION_DESC', [],
                                task_prompt_kwargs=desc_kwargs)
        if not new_desc: return None

        type_kwargs = {"world_theme": world_theme, "new_name": new_name, "new_description": new_desc}
        new_type = execute_task(self.engine, self.atlas_agent, 'WORLDGEN_CREATE_LOCATION_TYPE', [],
                                task_prompt_kwargs=type_kwargs) or "Region"

        relationship = relationship_override
        if not relationship:
            rel_kwargs = {"new_location_name": new_name, "parent_location_name": parent_location.get("Name", ""),
                          "creation_context": context}
            relationship = execute_task(self.engine, self.atlas_agent, 'WORLDGEN_DETERMINE_RELATIONSHIP', [],
                                        task_prompt_kwargs=rel_kwargs) or "NEARBY"

        return {"Name": new_name.strip(), "Description": new_desc.strip(), "Type": new_type.strip(),
                "Relationship": relationship.strip().upper()}

    def _create_location_one_shot(self, world_theme: str, parent_breadcrumb: list[str], parent_location: dict,
                                  relationship_override: str | None) -> dict | None:
        """Primary Method: Creates a new location with relationship using a single JSON prompt."""
        if relationship_override:
            return self._create_location_stepwise_fallback(world_theme, parent_location, "", relationship_override)

        valid_rels = self._get_valid_relationships(parent_breadcrumb)
        rel_str = "\n".join([f"- `{rel}`" for rel in valid_rels])

        kwargs = {
            "world_theme": world_theme,
            "current_location_summary": f"Name: {parent_location.get('Name', 'the world')}\nDescription: {parent_location.get('Description', '')}",
            "valid_relationships": rel_str
        }
        raw_response = execute_task(self.engine, self.atlas_agent, 'WORLDGEN_CREATE_LOCATION_JSON_WITH_RELATIONSHIP',
                                    [], task_prompt_kwargs=kwargs)
        command = command_parser.parse_structured_command(self.engine, raw_response, 'ATLAS',
                                                          fallback_task_key='CH_FIX_ATLAS')

        if command and all(k in command for k in ['Name', 'Description', 'Type', 'Relationship']):
            rel = command['Relationship'].upper()
            if rel not in valid_rels:
                file_io.log_to_ideation_file("ATLAS_RELATIONSHIP", rel,
                                             context=f"Generated for parent: {parent_location.get('Name')}")
                command['Relationship'] = "NEARBY"
            return command
        return self._create_location_stepwise_fallback(world_theme, parent_location, "", None)

    def explore_one_step(self, world_name: str, world_theme: str, breadcrumb: list[str], current_node: dict,
                         scene_prompt: str | None = None):
        """Orchestrates a single, autonomous step of world exploration, potentially guided by a scene."""
        context_string, connections = self._build_context_for_decision(breadcrumb, current_node)
        action = self._decide_next_action(world_theme, context_string, scene_prompt).upper()
        utils.log_message('game', f"[ATLAS] Decided action: {action}")

        # --- Handle Action Short-circuiting ---
        if action in connections:
            target_name = connections[action]
            utils.log_message('game', f"[ATLAS] Action '{action}' is a direct navigation command to '{target_name}'.")
            new_breadcrumb = self._calculate_new_breadcrumb(breadcrumb, action, target_name)
            if new_breadcrumb:
                return new_breadcrumb, file_io._find_node_by_breadcrumb(config.world, new_breadcrumb), "CONTINUE"
        elif action in self._get_valid_relationships(breadcrumb) and action not in connections:
            utils.log_message('game', f"[ATLAS] Action '{action}' is a direct command to create a new location.")
            return self._handle_location_creation(world_name, world_theme, breadcrumb, current_node,
                                                  relationship_override=action)

        # --- Handle Generic Actions ---
        if action == "NAVIGATE":
            return self._handle_navigation(current_node, breadcrumb, world_theme, scene_prompt, connections)
        elif action == "CREATE":
            return self._handle_location_creation(world_name, world_theme, breadcrumb, current_node,
                                                  relationship_override=None)
        elif action == "POPULATE":
            if new_npc := self._generate_npc_concept_for_location(world_theme, current_node):
                file_io.add_inhabitant_to_location(world_name, breadcrumb, {"name": new_npc.split(' - ')[0],
                                                                            "description": new_npc.split(' - ', 1)[1]})
                config.load_world_data(world_name)
                return breadcrumb, file_io._find_node_by_breadcrumb(config.world, breadcrumb), "CONTINUE"

        return breadcrumb, current_node, "ACCEPT"

    def _handle_navigation(self, current_node, breadcrumb, world_theme, scene_prompt, connections):
        """Handles the NAVIGATE action when the AI doesn't short-circuit."""
        if not connections: return breadcrumb, current_node, "CONTINUE"

        dest_str = "\n".join(f"- {name} ({rel})" for rel, name in connections.items())
        target_name = self._choose_navigation_target(world_theme, current_node.get("Name"), dest_str, scene_prompt)

        if target_name:
            for rel, name in connections.items():
                if target_name.lower() in name.lower():
                    new_breadcrumb = self._calculate_new_breadcrumb(breadcrumb, rel, target_name)
                    if new_breadcrumb:
                        return new_breadcrumb, file_io._find_node_by_breadcrumb(config.world,
                                                                                new_breadcrumb), "CONTINUE"
        return breadcrumb, current_node, "CONTINUE"

    def _handle_location_creation(self, world_name, world_theme, breadcrumb, current_node,
                                  relationship_override):
        """Handles the CREATE action, including lattice logic."""
        new_loc_data = self._create_location_one_shot(world_theme, breadcrumb, current_node, relationship_override)
        if not new_loc_data: return breadcrumb, current_node, "CONTINUE"

        rel = new_loc_data.pop("Relationship").upper()
        new_name = new_loc_data["Name"]

        if rel == "OUTSIDE":
            if "PARENT" in current_node.get("relationships", {}):
                new_breadcrumb = breadcrumb[:-1]
                return new_breadcrumb, file_io._find_node_by_breadcrumb(config.world, new_breadcrumb), "CONTINUE"
        elif rel in HIERARCHICAL_RELATIONSHIPS:
            new_loc_data.setdefault("relationships", {})["PARENT_RELATION"] = rel
            file_io.add_child_to_world(world_name, breadcrumb, new_loc_data)
            config.load_world_data(world_name)
            new_node_breadcrumb = breadcrumb + [new_name]
            file_io.add_relationship_to_node(world_name, new_node_breadcrumb, "PARENT", current_node["Name"])
            config.load_world_data(world_name)
            return new_node_breadcrumb, file_io._find_node_by_breadcrumb(config.world, new_node_breadcrumb), "CONTINUE"
        elif rel in LATTICE_RELATIONSHIPS:
            parent_breadcrumb = breadcrumb[:-1]
            file_io.add_child_to_world(world_name, parent_breadcrumb, new_loc_data)
            config.load_world_data(world_name)
            reciprocal = LATTICE_RELATIONSHIPS[rel]
            new_node_breadcrumb = parent_breadcrumb + [new_name]
            file_io.add_relationship_to_node(world_name, breadcrumb, rel, new_name)
            file_io.add_relationship_to_node(world_name, new_node_breadcrumb, reciprocal, current_node["Name"])
            file_io.add_relationship_to_node(world_name, new_node_breadcrumb, "PARENT",
                                             current_node.get("relationships", {}).get("PARENT"))
            config.load_world_data(world_name)
            return new_node_breadcrumb, file_io._find_node_by_breadcrumb(config.world, new_node_breadcrumb), "CONTINUE"
        return breadcrumb, current_node, "CONTINUE"

    def _decide_next_action(self, world_theme: str, context_string: str, scene_prompt: str | None) -> str:
        prompt_kwargs = {
            "world_theme": world_theme,
            "goal_description": f"find the best location for the scene: '{scene_prompt}'" if scene_prompt else "build out the world.",
            "scene_prompt": scene_prompt or "N/A",
            "location_context_string": context_string
        }
        response = execute_task(self.engine, self.atlas_agent, 'WORLDGEN_DECIDE_ACTION', [],
                                task_prompt_kwargs=prompt_kwargs)
        # Check for direct relationship keywords first
        for keyword in (VALID_RELATIONSHIPS + ["OUTSIDE"]):
            if keyword in response.upper().split():
                return keyword
        # Check for generic action keywords
        for keyword in GENERIC_ACTIONS:
            if keyword in response.upper():
                return keyword
        return "ACCEPT"

    def _choose_navigation_target(self, world_theme: str, current_name: str, destinations_str: str,
                                  scene_prompt: str | None) -> str:
        prompt_kwargs = {
            "world_theme": world_theme,
            "scene_prompt": scene_prompt or "N/A",
            "current_location_name": current_name,
            "available_destinations_string": destinations_str
        }
        response = execute_task(self.engine, self.atlas_agent, 'WORLDGEN_CHOOSE_NAVIGATION_TARGET', [],
                                task_prompt_kwargs=prompt_kwargs)
        return response.strip()

    def _calculate_new_breadcrumb(self, old_breadcrumb: list, relationship: str, target_name: str) -> list | None:
        if relationship in HIERARCHICAL_RELATIONSHIPS:
            return old_breadcrumb + [target_name]
        if relationship in LATTICE_RELATIONSHIPS:
            return old_breadcrumb[:-1] + [target_name] if old_breadcrumb else [target_name]
        if relationship == "OUTSIDE":
            return old_breadcrumb[:-1] if len(old_breadcrumb) > 1 else old_breadcrumb
        return None

    def find_or_create_location_for_scene(self, world_name: str, world_theme: str, scene_prompt: str) -> tuple[
        dict, list]:
        utils.log_message('game', "[ATLAS] Starting autonomous scene placement...")
        world_data = config.world
        root_key = next(iter(world_data))
        breadcrumb = [root_key]

        max_depth = 10
        for i in range(max_depth):
            current_node = file_io._find_node_by_breadcrumb(config.world, breadcrumb)
            if not current_node:
                utils.log_message('debug', f"[ATLAS ERROR] Breadcrumb {breadcrumb} led to a dead end.")
                # Fallback: reset to root
                breadcrumb = [root_key]
                current_node = world_data[root_key]

            utils.log_message('game',
                              f"[ATLAS] Analyzing fit in: {self._build_descriptive_breadcrumb_trail(breadcrumb)} (Step {i + 1}/{max_depth})")

            breadcrumb, current_node, status = self.explore_one_step(
                world_name, world_theme, breadcrumb, current_node, scene_prompt
            )
            if status == "ACCEPT":
                utils.log_message('game', f"[ATLAS] Final location found: {current_node['Name']}")
                return current_node, breadcrumb
        utils.log_message('game', f"[ATLAS] Reached max depth. Accepting current location.")
        current_node = file_io._find_node_by_breadcrumb(config.world, breadcrumb)
        return current_node, breadcrumb

    def create_new_world(self, theme: str, ui_manager=None) -> str | None:
        utils.log_message('game', f"[ATLAS] Forging new world with theme: {theme}")
        world_name = self._get_world_name(theme, ui_manager=ui_manager)
        utils.log_message('game', f"[ATLAS] World named: {world_name}")

        origin_location = self._create_location_one_shot(theme, [], {"Name": "The Cosmos"}, None)
        if not origin_location:
            utils.log_message('game', "[ATLAS] ERROR: Failed to create origin location.")
            return None

        origin_location.pop("Relationship", None)  # Root has no relationship
        origin_location['theme'] = theme
        utils.log_message('game', f"[ATLAS] Origin location created: {origin_location.get('Name')}")

        world_data = {origin_location['Name']: origin_location}
        world_dir = file_io.join_path(file_io.PROJECT_ROOT, 'data', 'worlds', world_name)
        file_io.create_directory(world_dir)
        world_filepath = file_io.join_path(world_dir, 'world.json')
        file_io.write_json(world_filepath, world_data)
        file_io.write_json(file_io.join_path(world_dir, 'levels.json'), {})
        file_io.write_json(file_io.join_path(world_dir, 'generated_scenes.json'), [])
        file_io.write_json(file_io.join_path(world_dir, 'casting_npcs.json'), [])

        exploration_steps = config.settings.get("ATLAS_AUTONOMOUS_EXPLORATION_STEPS", 0)
        if exploration_steps > 0:
            utils.log_message('game',
                              f"\n--- [ATLAS] Starting {exploration_steps}-step autonomous world exploration... ---")
            config.load_world_data(world_name)
            current_breadcrumb = [origin_location['Name']]

            for i in range(exploration_steps):
                current_node = file_io._find_node_by_breadcrumb(config.world, current_breadcrumb)
                if not current_node:
                    utils.log_message('debug', f"[ATLAS ERROR] Breadcrumb {current_breadcrumb} broke mid-exploration.")
                    break
                utils.log_message('game', f"\n[ATLAS EXPLORATION] Step {i + 1}/{exploration_steps}")
                breadcrumb, node, status = self.explore_one_step(world_name, theme, current_breadcrumb, current_node)
                current_breadcrumb = breadcrumb
                if status == "ACCEPT":
                    utils.log_message('game', "[ATLAS] Autonomous exploration concluded early by accepting a location.")
                    break
            utils.log_message('game', "--- [ATLAS] Autonomous exploration complete. ---\n")
        return world_name