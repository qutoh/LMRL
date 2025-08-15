# /core/components/item_manager.py

import re
import json
from ..common.config_loader import config
from ..llm.llm_api import execute_task
from ..common import utils, command_parser
from . import roster_manager


class ItemManager:
    """
    Manages all logic related to character equipment, including modification,
    description updates, and outfit caching.
    """

    def __init__(self, engine):
        self.engine = engine

    def _describe_items_stepwise(self, items_to_add_names: list, context: str) -> list[dict]:
        """
        Fallback to generate descriptions for a list of items one by one. This is
        slower but much more robust against LLM formatting errors.
        """
        utils.log_message('debug', "[EQUIPMENT] JSON generation failed. Falling back to stepwise item description.")
        equipment_agent = config.agents['EQUIPMENT_MANAGER']
        described_items = []

        for item_name in items_to_add_names:
            desc_kwargs = {"context": context, "item_name": item_name}
            description = execute_task(self.engine, equipment_agent, 'EQUIPMENT_DESCRIBE_ONE_ITEM', [],
                                       task_prompt_kwargs=desc_kwargs)

            # If the simple task fails, create a generic but specific description as a last resort.
            final_description = description.strip() if description and description.strip() else f"A {item_name}."
            described_items.append({"name": item_name, "description": final_description})

        return described_items

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

            command = command_parser.parse_structured_command(
                self.engine, raw_desc_response, equipment_agent.get('name'),
                fallback_task_key='CH_FIX_INITIAL_EQUIPMENT_JSON'
            )

            all_described_items = []
            described_item_data = command if isinstance(command, dict) else {}

            # Figure out which items were successfully described and which need the fallback
            processed_names_lower = set()
            if described_item_data:
                for item_name_key, details in described_item_data.items():
                    if isinstance(details, dict) and 'description' in details:
                        original_name = next(
                            (name for name in items_to_add_names if name.lower() == item_name_key.lower()), None)
                        if original_name:
                            all_described_items.append({"name": original_name, **details})
                            processed_names_lower.add(original_name.lower())

            items_that_need_fallback = [name for name in items_to_add_names if
                                        name.lower() not in processed_names_lower]

            if items_that_need_fallback:
                fallback_items = self._describe_items_stepwise(items_that_need_fallback,
                                                               dialogue_entry.get('content', ''))
                all_described_items.extend(fallback_items)

            # Process the final list of all described items
            for item_data in all_described_items:
                item_name = item_data['name']
                if "(Lost)" in item_name:
                    clean_name = item_name.replace("(Lost)", "").strip()
                    item_data['name'] = clean_name
                    character['equipment']['removed'].append(item_data)
                    utils.log_message('debug',
                                      f"[EQUIPMENT] Created and added missed item '{clean_name}' to removed list for {character['name']}.")
                else:
                    character['equipment']['equipped'].append(item_data)
                    final_added_names.append(item_name)
                equipment_changed = True

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
                                  f"[EQUIPMENT] Matched existing outfit '{name}'. Short-circuiting Director call.")
                return

        # If no match was found, generate a new description and save the outfit
        director_agent = config.agents['DIRECTOR']

        # Find the character's base description to ensure updates are clean.
        base_description = None
        for outfit_data in character['equipment']['outfits'].values():
            if not outfit_data.get('items'):  # Find the outfit with an empty item list
                base_description = outfit_data['description']
                break

        if base_description:  # Paper Doll mode with a valid base description
            update_desc_kwargs = {
                "original_description": base_description,
                "items_added": ", ".join(current_equipped_names) or "None",
                "items_removed": "None"  # Rebuilding from base, so nothing is "removed" from it
            }
            utils.log_message('debug', "[DIRECTOR] Rebuilding physical description from base.")
        else:  # Fallback for standard mode or corrupted data
            update_desc_kwargs = {
                "original_description": character.get('physical_description', ''),
                "items_added": ", ".join(final_added_names) or "None",
                "items_removed": ", ".join(final_removed_names) or "None"
            }
            utils.log_message('debug', "[DIRECTOR] Incrementally updating physical description.")

        new_physical_description = execute_task(self.engine, director_agent,
                                                'DIRECTOR_UPDATE_PHYSICAL_DESCRIPTION', [],
                                                task_prompt_kwargs=update_desc_kwargs)
        if new_physical_description:
            character['physical_description'] = new_physical_description.strip()
            utils.log_message('game',
                              f"{character['name']}'s appearance changes to: {character['physical_description']}")
            new_outfit_name = f"OUTFIT_{len(character['equipment']['outfits'])}"
            character['equipment']['outfits'][new_outfit_name] = {
                "items": current_equipped_names,
                "description": character['physical_description']
            }
            utils.log_message('debug',
                              f"[DIRECTOR] Updated physical description for {character['name']} and saved new outfit '{new_outfit_name}'.")