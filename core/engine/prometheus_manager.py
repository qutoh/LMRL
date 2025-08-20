# /core/engine/prometheus_manager.py

import re

from core.common import file_io, utils
from core.common.config_loader import config
from core.components import character_factory
from core.components import roster_manager
from core.components import utility_tasks
from core.llm.llm_api import execute_task


def parse_and_log_tool_decisions(text: str, valid_tools: list[str], context: str) -> tuple[dict, list[str], str]:
    """
    Parses a string for key-value pairs and a justification.

    Returns a tuple containing:
    - A dictionary of valid tool decisions (True/False).
    - A list of any novel tool keys that were found.
    - The justification string, if found.
    """
    decisions = {}
    novel_keys_found = []
    justification = ""

    # Specific pattern to capture the justification prose
    justification_match = re.search(r"^\s*justification_prose\s*:\s*(.*)", text, re.IGNORECASE | re.MULTILINE)
    if justification_match:
        justification = justification_match.group(1).strip()

    all_keys_pattern = re.compile(r"^\s*([a-zA-Z_]+)\s*[:=\-]", re.IGNORECASE | re.MULTILINE)
    found_keys = [key.lower() for key in all_keys_pattern.findall(text)]

    valid_tool_keys = [k.lower() for k in valid_tools]
    # Exclude the justification key from being treated as a novel tool
    keys_to_ignore = {'justification_prose', 'justification'}
    novel_keys = set(found_keys) - set(valid_tool_keys) - keys_to_ignore

    for key in novel_keys:
        utils.log_message('debug', f"[PROMETHEUS IDEATION] Found novel tool suggestion: '{key}'. Logging.")
        file_io.log_to_ideation_file("PROMETHEUS_TOOL", key, context=context)
        novel_keys_found.append(key)

    for tool_name in valid_tool_keys:
        pattern = re.compile(rf"^{re.escape(tool_name)}.*?(true|false)", re.IGNORECASE | re.MULTILINE)
        match = pattern.search(text)
        if match:
            decisions[tool_name] = match.group(1).lower() == 'true'

    return decisions, novel_keys_found, justification


class PrometheusManager:
    """
    A manager for the Prometheus agent, which analyzes narrative context
    and determines which specialized tools should be activated.
    """

    def __init__(self, engine):
        self.engine = engine
        self.prometheus_agent = config.agents.get('PROMETHEUS')
        if not self.prometheus_agent:
            raise ValueError("PROMETHEUS agent not found in agents.json")

        self.tool_dispatch_table = {
            "is_refusal": "_call_handle_refusal",
            "summarize_for_memory": "_call_summarize_for_memory",
            "remove_character": "_call_remove_character",
            "new_character": "_call_create_character",
            "select_next_actor": "_call_select_next_actor",
            "modify_equipment": "_call_modify_equipment"
        }

    def _call_modify_equipment(self, dialogue_entry: dict, **kwargs) -> list[str]:
        """Dispatches to the ItemManager and returns a list of affected character names."""
        utils.log_message('debug', "[PROMETHEUS] Activating tool: modify_equipment.")
        if not hasattr(self.engine, 'item_manager'):
            return []

        equipment_agent = config.agents['EQUIPMENT_MANAGER']
        
        # Filter for only positional characters BEFORE asking the LLM to identify targets.
        all_positional_chars = [c for c in roster_manager.get_all_characters(self.engine) if c.get('is_positional')]
        char_list_str = "; ".join([c['name'] for c in all_positional_chars])
        
        if not char_list_str:
            return []

        target_kwargs = {"character_list": char_list_str, "recent_events": dialogue_entry.get('content', '')}

        targets_response = execute_task(self.engine, equipment_agent, 'EQUIPMENT_IDENTIFY_TARGETS', [],
                                        task_prompt_kwargs=target_kwargs)
        if not targets_response or "none" in targets_response.lower():
            return []

        affected_names = [name.strip() for name in targets_response.split(';') if name.strip()]
        final_affected = []
        for name in affected_names:
            target_char = next((char['name'] for char in all_positional_chars if name.lower() in char['name'].lower()), None)
            if target_char:
                utils.log_message('debug', f"[PROMETHEUS] Dispatching equipment modification for '{target_char}'.")
                self.engine.item_manager.modify_character_equipment(target_char, dialogue_entry['content'])
                final_affected.append(target_char)
        return final_affected

    def _call_handle_refusal(self, dialogue_entry: dict, original_messages: list, **kwargs) -> dict:
        """Dispatches to the utility task that handles re-prompting after a refusal."""
        utils.log_message('debug', "[PROMETHEUS] Activating tool: is_refusal.")
        original_actor = roster_manager.find_character(self.engine, dialogue_entry['speaker'])
        if not original_actor: return {'status': 'FAILED', 'reason': 'Original actor not found.'}

        new_prose = utility_tasks.reprompt_after_refusal(self.engine, original_actor, 'GENERIC_TURN',
                                                         original_messages)
        if new_prose:
            new_dialogue_entry = {'speaker': original_actor['name'], 'content': new_prose}
            next_actor_name, events = self.analyze_and_dispatch(
                new_prose, new_dialogue_entry, kwargs.get('unacted_roles', []), original_messages, is_reprompt=True
            )
            return {'status': 'REPROMPTED', 'new_prose': new_prose, 'next_actor_name': next_actor_name, 'events': events}
        return {'status': 'REPROMPT_FAILED'}

    def _call_summarize_for_memory(self, dialogue_entry: dict, **kwargs):
        """Calls the MemoryManager to summarize and store a single dialogue entry."""
        if hasattr(self.engine, 'memory_manager') and self.engine.memory_manager:
            utils.log_message('debug', "[PROMETHEUS] Activating tool: summarize_for_memory.")
            self.engine.memory_manager.save_memory(self.engine, dialogue_entry)

    def _call_remove_character(self, dialogue_entry: dict, **kwargs) -> str | None:
        """Initiates the character removal process and returns the name of the removed character."""
        utils.log_message('debug', "[PROMETHEUS] Activating tool: remove_character.")
        director_agent = config.agents.get('DIRECTOR')
        active_chars = roster_manager.get_all_characters(self.engine)
        char_list_str = ", ".join([f"'{c['name']}'" for c in active_chars])
        prompt_kwargs = {"recent_events": dialogue_entry.get('content', ''), "character_list": char_list_str}

        response = execute_task(self.engine, director_agent, 'DIRECTOR_CHOOSE_CHARACTER_TO_REMOVE', [],
                                task_prompt_kwargs=prompt_kwargs)
        if response and response.strip().upper() != 'NONE':
            name_to_remove = response.split(',')[0].strip()
            self.engine.director_manager.remove_character_from_story(name_to_remove)
            return name_to_remove
        return None

    def _call_create_character(self, dialogue_entry: dict, **kwargs) -> str | None:
        """Initiates the new character creation process and returns the new character's name."""
        utils.log_message('debug', "[PROMETHEUS] Activating tool: new_character.")
        director_agent = config.agents.get('DIRECTOR')
        active_chars = roster_manager.get_all_characters(self.engine)
        char_list_str = ", ".join([f"'{c['name']}'" for c in active_chars])
        
        # Step 1: Identify the name of the new character from the recent dialogue.
        prompt_kwargs_identify = {"recent_events": dialogue_entry.get('content', ''), "character_list": char_list_str}
        new_char_name = execute_task(self.engine, director_agent, 'DIRECTOR_IDENTIFY_NEW_CHARACTER', [],
                                     task_prompt_kwargs=prompt_kwargs_identify)

        if new_char_name and new_char_name.strip().upper() != 'NONE':
            # Step 2: Use the identified name and correct context to create the full profile.
            if new_npc := character_factory.create_temporary_npc(self.engine, director_agent, new_char_name, self.engine.dialogue_log):
                roster_manager.decorate_and_add_character(self.engine, new_npc, 'npc')
                return new_npc.get('name')
        return None

    def _call_select_next_actor(self, dialogue_entry: dict, **kwargs) -> str | None:
        """Asks the Director to choose the next actor based on the prose."""
        unacted_roles = kwargs.get('unacted_roles', [])
        if not unacted_roles: return None

        utils.log_message('debug', "[PROMETHEUS] Activating tool: select_next_actor.")
        unacted_roles_str = "\n".join([f"- {r['name']}" for r in unacted_roles])
        prompt_kwargs = {"most_recent_prose": dialogue_entry.get('content', ''),
                         "unacted_roles_list": unacted_roles_str}
        director_agent = config.agents.get('DIRECTOR')
        choice_response = execute_task(self.engine, director_agent, 'DIRECTOR_CHOOSE_NEXT_ACTOR', [],
                                      task_prompt_kwargs=prompt_kwargs)
        if choice_response and choice_response.strip().upper() != 'NONE':
            cleaned_response = choice_response.strip().lower()
            for character in unacted_roles:
                char_name_lower = character['name'].lower()
                if char_name_lower in cleaned_response:
                    utils.log_message('debug', f"[PROMETHEUS] Director chose next actor: {character['name']}.")
                    return character['name']
                first_name = char_name_lower.split(',')[0].strip().split(' ')[0]
                if first_name in cleaned_response.split(' '):
                    utils.log_message('debug',
                                      f"[PROMETHEUS] Director chose next actor by first name: {character['name']}.")
                    return character['name']
        return None

    def analyze_and_dispatch(self, recent_prose: str, dialogue_entry: dict, unacted_roles: list,
                             original_messages: list, is_reprompt: bool = False) -> tuple[str | None, dict]:
        """
        Asks Prometheus what to do, calls the appropriate functions, and returns
        the name of a selected next actor and a dictionary of events that occurred.
        """
        if not recent_prose: return None, {}

        non_dm_chars = [c['name'] for c in roster_manager.get_all_characters(self.engine) if c.get('role_type') != 'dm']
        char_list_str = "; ".join(non_dm_chars)
        prompt_kwargs = {"recent_events": recent_prose, "character_list": char_list_str}

        character = roster_manager.find_character(self.engine, dialogue_entry['speaker'])
        prompt_kwargs["current_equipment"] = ", ".join(
            [item['name'] for item in character.get('equipment', {}).get('equipped', [])]) or "None"

        raw_response = execute_task(self.engine, self.prometheus_agent, 'PROMETHEUS_DETERMINE_TOOL_USE', [],
                                    task_prompt_kwargs=prompt_kwargs)
        valid_tool_names = list(self.tool_dispatch_table.keys())
        tool_decisions, _, _ = parse_and_log_tool_decisions(raw_response, valid_tool_names, context=recent_prose)

        if not tool_decisions and not is_reprompt:
            # Handle potential refusal with a single re-prompt
            return None, {'refusal': True}

        if not raw_response: return None, {}
        utils.log_message('debug', f"[PROMETHEUS RESPONSE]\n{raw_response}\n")

        # --- Prioritized Tool Execution and Event Logging ---
        # NOTE FOR FUTURE DEVS: If you add a new tool, consider if its execution order matters
        # or if it affects the context for the character_state update. Add its logic here.
        events = {}
        next_actor_name = None

        if tool_decisions.get('is_refusal') and not is_reprompt:
            return None, {'refusal': True}

        if tool_decisions.get('modify_equipment'):
            affected_chars = self._call_modify_equipment(dialogue_entry)
            if affected_chars: events['equipment_changed'] = affected_chars

        if tool_decisions.get('summarize_for_memory'):
            self._call_summarize_for_memory(dialogue_entry)
            # This action implies time has passed, which is handled by TurnManager.
            # We just need to ensure the event is logged for the state update.
            duration = utility_tasks.get_duration_for_action(self.engine, dialogue_entry['content'])
            if duration > 0: events['time_passed_seconds'] = duration

        if tool_decisions.get('remove_character'):
            removed_name = self._call_remove_character(dialogue_entry)
            if removed_name: events['character_removed'] = removed_name

        if tool_decisions.get('new_character'):
            new_name = self._call_create_character(dialogue_entry)
            if new_name: events['character_added'] = new_name

        if tool_decisions.get('select_next_actor'):
            next_actor_name = self._call_select_next_actor(dialogue_entry, unacted_roles=unacted_roles)

        return next_actor_name, events