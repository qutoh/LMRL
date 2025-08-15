# /core/components/item_manager.py

import re
import json
from ..common.config_loader import config
from ..llm.llm_api import execute_task
from ..common import utils
from . import roster_manager


class ItemManager:
    """
    Manages all logic related to character equipment, including modification,
    description updates, and outfit caching.
    """

    def __init__(self, engine):
        self.engine = engine

    def modify_character_equipment(self, dialogue_entry: dict):
        """
        The main entry point for processing an equipment change. Orchestrates
        agent calls and updates the character's state.
        """
        character = roster_manager.find_character(self.engine, dialogue_entry['speaker'])
        if not character or 'equipment' not in character:
            return

        equipment_agent = config.agents['EQUIPMENT_MANAGER']
        equipped_items = character['equipment'].get('equipped', [])
        equipped_str = ", ".join([item['name'] for item in equipped_items]) if equipped_items else "None"

        mod_kwargs = {"recent_events": dialogue_entry.get('content', ''), "equipped_items_list": equipped_str}
        response = execute_task(self.engine, equipment_agent, 'EQUIPMENT_MODIFY', [], task_prompt_kwargs=mod_kwargs)

        items_to_add_names = [name.strip() for name in
                              re.findall(r'ADD:\s*(.*?)(?:;|\n|$)', response, re.IGNORECASE)]
        items_to_remove_names = [name.strip() for name in
                                 re.findall(r'REMOVE:\s*(.*?)(?:;|\n|$)', response, re.IGNORECASE)]

        if not items_to_add_names and not items_to_remove_names:
            return

        equipment_changed = False
        final_added_names = []
        final_removed_names = []

        # Process removals
        for name in items_to_remove_names:
            item_found = next((item for item in equipped_items if item['name'].lower() == name.lower()), None)
            if item_found:
                character['equipment']['equipped'].remove(item_found)
                character['equipment']['removed'].append(item_found)
                final_removed_names.append(item_found['name'])
                equipment_changed = True
                utils.log_message('debug', f"[EQUIPMENT] Removed '{item_found['name']}' from {character['name']}.")
            else:
                # Handle missed item case
                items_to_add_names.append(f"{name} (Lost)")

        # Process additions and describe new items
        if items_to_add_names:
            desc_kwargs = {"item_names_list": "; ".join(items_to_add_names),
                           "context": dialogue_entry.get('content', '')}
            raw_desc_response = execute_task(self.engine, equipment_agent, 'EQUIPMENT_DESCRIBE_ITEMS', [],
                                             task_prompt_kwargs=desc_kwargs)
            json_str = utils.clean_json_from_llm(raw_desc_response)
            if json_str:
                try:
                    new_item_data = json.loads(json_str)
                    for item_name, item_details in new_item_data.items():
                        if "(Lost)" in item_name:
                            clean_name = item_name.replace("(Lost)", "").strip()
                            character['equipment']['removed'].append({"name": clean_name, **item_details})
                            utils.log_message('debug',
                                              f"[EQUIPMENT] Created and added missed item '{clean_name}' to removed list for {character['name']}.")
                        else:
                            character['equipment']['equipped'].append({"name": item_name, **item_details})
                            final_added_names.append(item_name)
                    equipment_changed = True
                except json.JSONDecodeError:
                    pass

        if not equipment_changed:
            return

        current_equipped_names = sorted([item['name'] for item in character['equipment']['equipped']])

        # Check for existing outfit to short-circuit Director call
        for name, outfit_data in character['equipment']['outfits'].items():
            if sorted(outfit_data.get('items', [])) == current_equipped_names:
                if character['physical_description'] != outfit_data['description']:
                    character['physical_description'] = outfit_data['description']
                    utils.log_message('game',
                                f"{character['name']}'s appearance changes to: {character['physical_description']}")
                utils.log_message('debug',
                                  f"[EQUIPMENT] Matched existing outfit '{name}' for {character['name']}. Short-circuiting Director call.")
                return

        # If no match was found, generate a new description and save the outfit
        director_agent = config.agents['DIRECTOR']
        update_desc_kwargs = {
            "original_description": character.get('physical_description', ''),
            "items_added": ", ".join(final_added_names) or "None",
            "items_removed": ", ".join(final_removed_names) or "None"
        }
        new_physical_description = execute_task(self.engine, director_agent,
                                                'DIRECTOR_UPDATE_PHYSICAL_DESCRIPTION', [],
                                                task_prompt_kwargs=update_desc_kwargs)
        if new_physical_description:
            character['physical_description'] = new_physical_description.strip()
            utils.log_message('game', f"{character['name']}'s appearance changes to: {character['physical_description']}")
            new_outfit_name = f"OUTFIT_{len(character['equipment']['outfits'])}"
            character['equipment']['outfits'][new_outfit_name] = {
                "items": current_equipped_names,
                "description": character['physical_description']
            }
            utils.log_message('debug',
                              f"[DIRECTOR] Updated physical description for {character['name']} and saved new outfit '{new_outfit_name}'.")