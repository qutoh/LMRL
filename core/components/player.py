# /core/components/player.py

from core.common import file_io, utils
from core.common.config_loader import config
from core.common.localization import loc
from core.components import character_factory, roster_manager, position_manager


class PlayerInterface:
    """
    A stateful manager that handles all direct interaction with a human player,
    including menus and turn execution.
    """

    def __init__(self, engine, game_state):
        self.engine = engine
        self.game_state = game_state

    def _get_player_input(self, prompt: str, initial_text: str = "") -> str:
        """
        A centralized method for requesting and receiving free-text input.
        """
        try:
            self.engine.player_input_active = True
            self.engine.render_queue.put(('INPUT_REQUEST', prompt, self.engine.token_context_limit, initial_text))
            response = self.engine.input_queue.get()
            return response
        finally:
            self.engine.player_input_active = False

    def _get_player_menu_choice(self, title: str, options: list[str]) -> str | None:
        """
        A centralized method for requesting and receiving a choice from a menu.
        """
        try:
            self.engine.player_input_active = True
            self.engine.render_queue.put(('MENU_REQUEST', title, options))
            response = self.engine.input_queue.get()
            return response
        finally:
            self.engine.player_input_active = False

    def _prompt_for_choice(self, character_list, prompt_message, show_desc=True):
        """Generic helper to prompt the user to select a character from a list."""
        if not character_list:
            self.engine.render_queue.put(('ADD_EVENT_LOG', loc('player_no_characters_option'), (255, 255, 100)))
            return None

        options = []
        for char in character_list:
            name = char.get('name') if isinstance(char, dict) else char.name
            description = char.get('description') if isinstance(char, dict) else ""
            desc_str = f" - {description}" if show_desc and description else ""
            options.append(f"{name}{desc_str}")

        choice_str = self._get_player_menu_choice(prompt_message, options)
        if choice_str is None: return None  # Menu was cancelled

        return next((c for c in character_list if (c.get('name') if isinstance(c, dict) else c.name) in choice_str),
                    None)

    def get_character_via_menu(self) -> dict | None:
        """
        Presents the full character selection/creation menu and returns the resulting
        character data dictionary without adding it to the roster.
        """
        chosen_character = None
        while not chosen_character:
            menu_title = "Select or Create a Character"
            options = [
                "Control an existing Lead Character",
                "Control an existing NPC (will be promoted to a Lead)",
                "Load a character from the Casting files to play",
                "Create a new Lead Character",
                "Cancel"
            ]
            main_choice = self._get_player_menu_choice(menu_title, options)

            if main_choice == "Control an existing Lead Character":
                lead_list = [c for c in self.engine.characters if
                             c.get('role_type') == 'lead' and not c.get('controlled_by')]
                chosen_character = self._prompt_for_choice(lead_list, loc('player_menu_prompt_select_lead'))

            elif main_choice == "Control an existing NPC (will be promoted to a Lead)":
                npc_list = [c for c in self.engine.characters if
                            c.get('role_type') == 'npc' and c.get('is_controllable')]
                if npc_to_promote := self._prompt_for_choice(npc_list, loc('player_menu_prompt_select_npc')):
                    chosen_character = roster_manager.promote_npc_to_lead(self.engine, npc_to_promote['name'])

            elif main_choice == "Load a character from the Casting files to play":
                casting_list = roster_manager.get_available_casting_characters(self.engine)
                if char_to_load := self._prompt_for_choice(casting_list, loc('player_menu_prompt_select_casting')):
                    chosen_character = char_to_load

            elif main_choice == "Create a new Lead Character":
                self.engine.render_queue.put(('ADD_EVENT_LOG', loc('player_menu_creating_character')))
                if new_lead_data := character_factory.create_lead_stepwise(self.engine, self.engine.dialogue_log):
                    chosen_character = new_lead_data
                else:
                    self.engine.render_queue.put(('ADD_EVENT_LOG', loc('player_creation_failed'), (255, 100, 100)))

            elif main_choice == "Cancel" or main_choice is None:
                return None
            else:
                self.engine.render_queue.put(('ADD_EVENT_LOG', loc('player_invalid_choice'), (255, 100, 100)))

        return chosen_character

    def initiate_takeover_menu(self):
        """Handles the UI for a player to take control of a character."""
        self.engine.render_queue.put(('ADD_EVENT_LOG', loc('player_takeover_header'), (100, 255, 100)))
        for char in self.engine.characters:
            char.pop('controlled_by', None)

        character_data = self.get_character_via_menu()

        if not character_data:
            self.engine.render_queue.put(('ADD_EVENT_LOG', loc('player_takeover_canceled')))
            return

        final_character = roster_manager.find_character(self.engine, character_data['name'])
        if not final_character:
            role_type = 'npc'
            if roster_manager.find_character_in_list(character_data['name'], config.casting_leads):
                role_type = 'lead'
            elif roster_manager.find_character_in_list(character_data['name'], config.casting_dms):
                role_type = 'dm'

            roster_manager.decorate_and_add_character(self.engine, character_data, role_type)
            final_character = roster_manager.find_character(self.engine, character_data['name'])

        if final_character:
            if final_character.get('role_type') == 'npc':
                final_character = roster_manager.promote_npc_to_lead(self.engine, final_character['name'])

            final_character['controlled_by'] = 'human'
            file_io.save_active_character_files(self.engine)
            self.engine.render_queue.put(
                ('ADD_EVENT_LOG', loc('player_control_confirmed', character_name=final_character['name']),
                 (100, 255, 100)))
        else:
            self.engine.render_queue.put(
                ('ADD_EVENT_LOG', "Error: Could not finalize character selection.", (255, 100, 100)))

    def _handle_human_movement(self, current_actor):
        """Manages the movement sub-phase for a human player."""
        self.engine.render_queue.put(('ADD_EVENT_LOG', loc('player_movement_header')))
        other_entities = [e for e in self.game_state.entities if e.name.lower() != current_actor['name'].lower()]
        if not other_entities:
            self.engine.render_queue.put(('ADD_EVENT_LOG', loc('player_movement_no_targets')))
            return
        if not (
                target := self._prompt_for_choice(other_entities, loc('player_movement_select_target'),
                                                  show_desc=False)):
            self.engine.render_queue.put(('ADD_EVENT_LOG', loc('player_movement_canceled')))
            return

        movement_description = self._get_player_input(loc('player_movement_describe', target_name=target.name)).strip()

        if movement_description:
            position_manager.process_movement_intent(self.engine, self.game_state, current_actor['name'],
                                                     movement_description, "WALK")
        else:
            self.engine.render_queue.put(('ADD_EVENT_LOG', loc('player_movement_skipped')))

    def execute_turn(self, current_actor) -> str:
        """
        Executes a turn for a human player. This method's responsibility is now
        limited to gathering input. It returns the prose for the TurnManager to process.
        """
        utils.log_message('debug', loc('log_turn_header', character_name=f"{current_actor['name']} (Human)"))

        self.game_state.reset_entity_turn_stats(current_actor['name'])

        action_prompt = loc('player_turn_prompt_action', character_name=current_actor['name'])
        prose = self._get_player_input(action_prompt).strip()

        if prose:
            utils.log_message('story', f"({current_actor['name']}) {prose}")

        move_choice = self._get_player_menu_choice("Movement Phase", ["Move", "Skip"]).strip().lower()

        if move_choice.startswith('m'):
            self._handle_human_movement(current_actor)

        return prose

    def _handle_default_takeover(self, **handler_kwargs) -> str | None:
        """Handles the standard text input for a player task takeover."""
        self.engine.render_queue.put(('PLAYER_TASK_TAKEOVER_REQUEST', handler_kwargs))
        player_response = self.engine.input_queue.get()
        return None if player_response is None or not player_response.strip() else player_response.strip()

    def _handle_replacement_menu_takeover(self, **kwargs) -> str | None:
        """Handles the 'REPLACEMENT_MENU' handler by showing the character menu."""
        character_data = self.get_character_via_menu()
        if not character_data:
            return None  # User cancelled

        final_character = roster_manager.find_character(self.engine, character_data['name'])
        if not final_character:
            role_type = 'lead'  # Default for this handler
            if roster_manager.find_character_in_list(character_data['name'], config.casting_npcs):
                role_type = 'npc'

            roster_manager.decorate_and_add_character(self.engine, character_data, role_type)
            final_character = roster_manager.find_character(self.engine, character_data['name'])

        return final_character.get('name') if final_character else None

    def _handle_prometheus_menu_takeover(self, **handler_kwargs) -> str | None:
        """Handles the specialized menu for the Prometheus tool selection task."""
        self.engine.render_queue.put(('PROMETHEUS_MENU_REQUEST', handler_kwargs))
        player_choices = self.engine.input_queue.get()

        if player_choices is None:
            return None  # Player cancelled

        response_lines = [f"{tool}: {str(value).upper()}" for tool, value in player_choices.items()]
        return "\n".join(response_lines)

    def handle_task_takeover(self, **kwargs) -> str | None:
        """
        Dispatcher for different player takeover handlers based on task_key.
        Returns the player's response string, or None to let the AI handle it.
        """
        task_key = kwargs.get("task_key")

        # --- Redundant safety check to prevent deadlocks ---
        task_params = config.task_parameters.get(task_key, {})
        if not task_params.get("player_takeover_enabled"):
            utils.log_message('debug',
                              f"[PLAYER] handle_task_takeover called for disabled task '{task_key}'. Ignoring.")
            return None

        # --- Task-to-Handler Mapping ---
        if task_key == "PROMETHEUS_DETERMINE_TOOL_USE":
            return self._handle_prometheus_menu_takeover(**kwargs)
        elif task_key == "DIRECTOR_CAST_REPLACEMENT":
            return self._handle_replacement_menu_takeover(**kwargs)

        # --- Default Handler ---
        # Any other task with player_takeover_enabled: true will use this.
        return self._handle_default_takeover(**kwargs)

    def _handle_add_character_menu(self, dms_added_this_session: list):
        """Handles the sub-menu for adding characters."""
        while True:
            choice = self._get_player_menu_choice("Add Character", ["Load from Casting", "Create New Lead", "Back"])
            if not choice or "Back" in choice:
                break

            if "Load" in choice:
                available = roster_manager.get_available_casting_characters(self.engine)
                if char_data := self._prompt_for_choice(available, "Select a character to load"):
                    role_type = 'npc'
                    if roster_manager.find_character_in_list(char_data['name'], config.casting_leads):
                        role_type = 'lead'
                    elif roster_manager.find_character_in_list(char_data['name'], config.casting_dms):
                        role_type = 'dm'

                    if role_type == 'dm' and not config.settings.get("enable_multiple_dms", False):
                        dms_added_this_session.append(char_data)
                        self.engine.render_queue.put(
                            ('ADD_EVENT_LOG', f"Queued '{char_data['name']}' for DM fusion.", (150, 255, 150)))
                    else:
                        roster_manager.decorate_and_add_character(self.engine, char_data, role_type)
                        self.engine.render_queue.put(
                            ('ADD_EVENT_LOG', f"Added '{char_data['name']}' to the story.", (150, 255, 150)))

            elif "Create" in choice:
                role_prompt = "Enter a role for the new Lead (e.g., 'a grizzled veteran'), or leave blank for a generic one."
                role_archetype = self._get_player_input(role_prompt).strip() or "A new adventurer joining the story."

                context_str = "\n".join(
                    [f"{entry['speaker']}: {entry['content']}" for entry in self.engine.dialogue_log[-15:]])
                director_agent = config.agents['DIRECTOR']

                if new_lead := character_factory.create_lead_from_role_and_scene(self.engine, director_agent,
                                                                                 context_str, role_archetype):
                    roster_manager.decorate_and_add_character(self.engine, new_lead, 'lead')
                    self.engine.render_queue.put(
                        ('ADD_EVENT_LOG', f"Created and added new lead '{new_lead['name']}'.", (150, 255, 150)))

    def _handle_remove_character_menu(self):
        """Handles the menu for removing characters using the Director's logic."""
        all_chars = roster_manager.get_all_characters(self.engine)
        if char_to_remove := self._prompt_for_choice(all_chars, "Select a character to remove", show_desc=False):
            self.engine.director_manager.remove_character_from_story(char_to_remove['name'])
            # The director's method will print its own success/failure message.

    def initiate_cast_management_menu(self):
        """Handles the main player-facing menu for managing the story's cast."""
        self.engine.render_queue.put(('ADD_EVENT_LOG', "\n--- Cast & Crew Management ---", (100, 255, 255)))
        dms_added_this_session = []

        while True:
            choice = self._get_player_menu_choice("Cast & Crew",
                                                  ["Add Character or DM", "Remove Character or DM", "Done"])
            if not choice or "Done" in choice:
                break

            if "Add" in choice:
                self._handle_add_character_menu(dms_added_this_session)
            elif "Remove" in choice:
                self._handle_remove_character_menu()

        # Post-menu processing for DMs
        if dms_added_this_session and not config.settings.get("enable_multiple_dms", False):
            self.engine.render_queue.put(('ADD_EVENT_LOG', "Processing newly added DMs...", (200, 200, 255)))
            for dm_profile in dms_added_this_session:
                self.engine.dm_manager.fuse_dm_into_meta(dm_profile)

        file_io.save_active_character_files(self.engine)
        self.engine.render_queue.put(('ADD_EVENT_LOG', "--- Cast Management Closed ---", (100, 255, 255)))