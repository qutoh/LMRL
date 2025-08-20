# /core/worldgen/world_components/content_generator.py

import random

from core.common import file_io, utils, command_parser
from core.common.config_loader import config
from core.llm.llm_api import execute_task
from .world_graph_navigator import WorldGraphNavigator


class ContentGenerator:
    """
    A service class responsible for all LLM calls related to generating new
    world content, such as names, locations, and NPC concepts.
    """

    def __init__(self, engine, navigator: WorldGraphNavigator):
        self.engine = engine
        self.atlas_agent = config.agents.get('ATLAS')
        self.navigator = navigator

    def get_world_name(self, theme: str, ui_manager=None) -> str:
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

    def generate_npc_concept_for_location(self, world_theme: str, location: dict) -> str | None:
        """Orchestrates the generation of a single NPC concept string for a location."""
        utils.log_message('game',
                          f"[ATLAS] Creating a character for {location.get('Name', 'this place')}.")
        loc_name = location.get("Name", "this place")
        loc_desc = location.get("Description", "")

        inhabitants = location.get('inhabitants', [])
        existing_inhabitants_details = []
        for i in inhabitants:
            if isinstance(i, str):
                existing_inhabitants_details.append(i)  # Already in "Name - Description" format
            elif isinstance(i, dict):
                if name := i.get('name'):
                    desc = i.get('description', 'No description.')
                    existing_inhabitants_details.append(f"{name} - {desc}")

        existing_inhabitants_str = "\n".join(
            existing_inhabitants_details) if existing_inhabitants_details else "None"

        npc_concept_kwargs = {
            "world_theme": world_theme,
            "location_name": loc_name,
            "location_description": loc_desc,
            "existing_inhabitants_list": existing_inhabitants_str
        }

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

    def create_location(self, world_theme: str, parent_breadcrumb: list[str], parent_location: dict,
                        scene_prompt: str | None = None, target_location_summary: str | None = None) -> dict | None:
        """Orchestrates creating a new location, using the correct prompt and fallback for the context."""
        available_rels = self.navigator.get_available_relationships(parent_breadcrumb, parent_location)

        if scene_prompt:  # We are creating an interstitial location to approach a target
            task_key = 'WORLDGEN_CREATE_LOCATION_JSON_INTERSTITIAL'
            fallback_func = self._create_interstitial_location_stepwise_fallback
            kwargs = {
                "world_theme": world_theme, "scene_prompt": scene_prompt,
                "current_location_summary": f"Name: {parent_location.get('Name', 'the world')}\nDescription: {parent_location.get('Description', '')}",
                "target_location_summary": target_location_summary,
                "valid_relationships": "\n".join([f"- `{rel}`" for rel in available_rels])
            }
        else:  # We are creating a location for pure world exploration
            task_key = 'WORLDGEN_CREATE_LOCATION_JSON_EXPLORE'
            fallback_func = self._create_explore_location_stepwise_fallback
            kwargs = {
                "world_theme": world_theme,
                "current_location_summary": f"Name: {parent_location.get('Name', 'the world')}\nDescription: {parent_location.get('Description', '')}",
                "valid_relationships": "\n".join([f"- `{rel}`" for rel in available_rels])
            }

        raw_response = execute_task(self.engine, self.atlas_agent, task_key, [], task_prompt_kwargs=kwargs)
        command = command_parser.parse_structured_command(self.engine, raw_response, 'ATLAS',
                                                          fallback_task_key='CH_FIX_ATLAS')

        if command and all(k in command for k in ['Name', 'Description', 'Type', 'Relationship']):
            return command

        return fallback_func(world_theme, parent_location, parent_breadcrumb, scene_prompt, target_location_summary)

    def _create_interstitial_location_stepwise_fallback(self, world_theme, parent_location, parent_breadcrumb,
                                                        scene_prompt, target_location_summary):
        """Stepwise fallback for creating a location that connects to a target scene."""
        utils.log_message('debug', "[ATLAS] Single-shot interstitial failed. Falling back to step-wise generation.")
        base_kwargs = {"world_theme": world_theme, "scene_prompt": scene_prompt,
                       "current_location_summary": f"Name: {parent_location.get('Name')}\nDesc: {parent_location.get('Description')}",
                       "target_location_summary": target_location_summary}
        new_name = execute_task(self.engine, self.atlas_agent, 'WORLDGEN_CREATE_INTERSTITIAL_NAME', [],
                                task_prompt_kwargs=base_kwargs) or "Path"
        new_desc = execute_task(self.engine, self.atlas_agent, 'WORLDGEN_CREATE_INTERSTITIAL_DESC', [],
                                task_prompt_kwargs={**base_kwargs, "new_name": new_name}) or "A path forward."
        new_type = execute_task(self.engine, self.atlas_agent, 'WORLDGEN_CREATE_INTERSTITIAL_TYPE', [],
                                task_prompt_kwargs={**base_kwargs, "new_name": new_name,
                                                    "new_description": new_desc}) or "Path"

        available_rels = self.navigator.get_available_relationships(parent_breadcrumb, parent_location)
        relationship = random.choice(available_rels)  # Simple random choice for fallback

        return {"Name": new_name.strip(), "Description": new_desc.strip(), "Type": new_type.strip(),
                "Relationship": relationship}

    def _create_explore_location_stepwise_fallback(self, world_theme, parent_location, parent_breadcrumb,
                                                   scene_prompt, target_location_summary):
        """Stepwise fallback for creating a location during pure exploration."""
        utils.log_message('debug', "[ATLAS] Single-shot exploration failed. Falling back to step-wise generation.")
        base_kwargs = {"world_theme": world_theme,
                       "current_location_summary": f"Name: {parent_location.get('Name')}\nDesc: {parent_location.get('Description')}"}
        new_name = execute_task(self.engine, self.atlas_agent, 'WORLDGEN_CREATE_EXPLORE_NAME', [],
                                task_prompt_kwargs=base_kwargs) or "New Area"
        new_desc = execute_task(self.engine, self.atlas_agent, 'WORLDGEN_CREATE_EXPLORE_DESC', [],
                                task_prompt_kwargs={**base_kwargs, "new_name": new_name}) or "An interesting new place."
        new_type = execute_task(self.engine, self.atlas_agent, 'WORLDGEN_CREATE_EXPLORE_TYPE', [],
                                task_prompt_kwargs={**base_kwargs, "new_name": new_name,
                                                    "new_description": new_desc}) or "Region"

        available_rels = self.navigator.get_available_relationships(parent_breadcrumb, parent_location)
        relationship = random.choice(available_rels)

        return {"Name": new_name.strip(), "Description": new_desc.strip(), "Type": new_type.strip(),
                "Relationship": relationship}