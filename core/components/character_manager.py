# /core/components/character_manager.py

from ..common import utils
from ..llm import llm_api
from . import position_manager


class CharacterManager:
    """
    A utility class to handle the creation and updating of a character's
    internal state (`character_state`).
    """
    def __init__(self):
        pass

    def _get_physical_context_for_state_update(self, engine, character: dict, events: dict) -> str:
        """Builds the physical context string, prioritizing an updated description if equipment changed."""
        if character['name'] in events.get('equipment_changed', []):
            return f"Your appearance or equipment has just changed. You now look like this:\n{character.get('physical_description', 'No description.')}"

        # Default physical context
        phys_desc = character.get('physical_description', 'No description.')
        equipment_lines = [f"- {item['name']}" for item in character.get('equipment', {}).get('equipped', [])]
        equipment_str = "\nYou are carrying/wearing:\n" + "\n".join(equipment_lines) if equipment_lines else ""
        return f"{phys_desc}{equipment_str}"

    def _get_events_context_for_state_update(self, engine, character_to_update: dict, dialogue_entry: dict, events: dict) -> str:
        """Builds the recent events summary string for the state update prompt."""
        event_lines = [f"The last thing that happened was: \"{dialogue_entry['speaker']}: {dialogue_entry['content']}\""]

        if character_to_update['name'] == dialogue_entry['speaker']:
            if time_seconds := events.get('time_passed_seconds'):
                if time_seconds > 60:
                    event_lines.append(f"This took about {time_seconds // 60} minute(s).")
                else:
                    event_lines.append("This took a few moments.")

        if new_char_name := events.get('character_added'):
            if new_char_name == character_to_update['name']:
                event_lines.append("You have just arrived in this scene.")
            else:
                event_lines.append(f"A new character, {new_char_name}, has just appeared.")

        if removed_char_name := events.get('character_removed'):
            event_lines.append(f"The character {removed_char_name} has just been removed from the story.")

        return "\n".join(event_lines)

    def _execute_state_update(self, engine, character: dict, surroundings: str, physical: str, events: str):
        """Executes the LLM call to update a character's state."""
        if not character:
            return

        kwargs = {
            "character_name": character.get('name', 'N/A'),
            "surroundings_context": surroundings,
            "physical_context": physical,
            "events_context": events
        }
        new_state = llm_api.execute_task(
            engine,
            character,
            'CHARACTER_UPDATE_STATE',
            [],
            task_prompt_kwargs=kwargs
        )
        if new_state:
            character['character_state'] = new_state.strip()
            utils.log_message('full',
                              f"[STATE UPDATE] {character['name']}'s new state: '{character['character_state']}'")

    def initialize_all_character_states(self, engine):
        """
        Sets the initial character_state for all characters at the start of a run.
        """
        utils.log_message('debug', "[SYSTEM] Initializing all character states...")
        for char in engine.characters:
            if not char.get('is_positional'):
                char['character_state'] = "I am observing the scene."
                continue

            surroundings = position_manager.get_local_context_for_character(engine, engine.game_state, char['name'])
            physical = self._get_physical_context_for_state_update(engine, char, {})
            events = "The story is just beginning."

            self._execute_state_update(engine, char, surroundings, physical, events)

    def update_character_state(self, engine, character_to_update: dict, dialogue_entry: dict, events: dict):
        """
        Builds a tailored context and calls the LLM to update a single character's
        internal state after their turn.
        """
        if not character_to_update:
            return

        # Handle the special case of a newly added character first.
        if character_to_update['name'] == events.get('character_added'):
            surroundings = position_manager.get_local_context_for_character(engine, engine.game_state, character_to_update['name'])
            physical = self._get_physical_context_for_state_update(engine, character_to_update, events)
            events_str = self._get_events_context_for_state_update(engine, character_to_update, dialogue_entry, events)
            self._execute_state_update(engine, character_to_update, surroundings, physical, events_str)
            return

        # Standard update for the character who just took their turn.
        surroundings = position_manager.get_local_context_for_character(engine, engine.game_state,
                                                                        character_to_update['name'])
        physical = self._get_physical_context_for_state_update(engine, character_to_update, events)
        events_str = self._get_events_context_for_state_update(engine, character_to_update, dialogue_entry, events)

        self._execute_state_update(engine, character_to_update, surroundings, physical, events_str)