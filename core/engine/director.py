# /core/director.py

import random
from core.common.config_loader import config
from core.common.localization import loc
from core.components import roster_manager
from core.components import character_factory
from core.common import command_parser, file_io, utils
from core.llm.llm_api import execute_task


class DirectorManager:
    """
    A stateful manager that handles the end-of-cycle review of characters,
    manages the cast, and prunes NPCs.
    """

    def __init__(self, engine):
        self.engine = engine
        self.director_agent = config.agents['DIRECTOR']

    def remove_character_from_story(self, character_name: str):
        """
        Handles the logic of permanently removing a character from the story
        and saving them to the appropriate casting file if they are an NPC.
        """
        char_to_remove = roster_manager.find_character(self.engine, character_name)
        if not char_to_remove:
            utils.log_message('debug', f"[DIRECTOR] REMOVE failed: Could not find character '{character_name}'.")
            return

        utils.log_message('debug', f"[DIRECTOR] Initiating removal process for: {char_to_remove['name']}")

        role_type = char_to_remove.get('role_type')
        should_save_to_casting = False
        casting_filename = None
        log_key = ''  # Initialize log_key

        if role_type == 'lead':
            casting_filename = 'casting_leads.json'
            should_save_to_casting = True
            log_key = 'system_lead_removed_and_saved'
        elif role_type == 'npc':
            # For NPCs, ask the Director if they should be saved.
            prompt_kwargs = {'npc_name': char_to_remove['name'],
                             'npc_description': char_to_remove.get('description', '')}
            response = execute_task(self.engine, self.director_agent, 'DIRECTOR_CONFIRM_NPC_SAVE', [],
                                    task_prompt_kwargs=prompt_kwargs)
            if response and 'yes' in response.strip().lower():
                casting_filename = file_io.join_path(self.engine.config.data_dir, 'worlds', self.engine.world_name,
                                                     'casting_npcs.json')
                should_save_to_casting = True
                log_key = 'system_npc_removed_and_saved'
            else:
                log_key = 'system_npc_removed_not_saved'

        if should_save_to_casting and casting_filename:
            file_io.save_character_to_casting_file(self.engine.run_path, char_to_remove, casting_filename)

        removed_char_name = char_to_remove['name']
        roster_manager.remove_character(self.engine, removed_char_name)

        if log_key:  # Ensure log_key was set before trying to use it
            log_args = {'lead_name' if role_type == 'lead' else 'npc_name': removed_char_name}
            utils.log_message('game', loc(log_key, **log_args))

    def establish_initial_cast(self, scene_prompt: str, location_summary: str):
        """
        Dynamically determines lead roles, casts them from files or creates new ones,
        and then selects and tunes appropriate DMs for the scene.
        """
        utils.log_message('game', "[DIRECTOR] The director is establishing an initial cast for the scene.")

        # --- Lead Character Management (Role-Based) ---
        npcs = [c for c in self.engine.characters if c.get('role_type') == 'npc']
        npc_summary = "\n".join([f"- {n['name']}: {n['description']}" for n in npcs]) if npcs else "None"

        roles_kwargs = {
            "scene_prompt": scene_prompt,
            "location_summary": location_summary,
            "npc_list_summary": npc_summary
        }
        roles_str = execute_task(self.engine, self.director_agent, 'DIRECTOR_DEFINE_LEAD_ROLES_FOR_SCENE', [],
                                 task_prompt_kwargs=roles_kwargs)
        needed_roles = [role.strip() for role in roles_str.split(';') if role.strip()]

        utils.log_message('game', f"[DIRECTOR] The director has defined the lead roles needed: {needed_roles}")

        available_leads = [c for c in config.casting_leads if not roster_manager.find_character(self.engine, c['name'])]

        for role in needed_roles:
            if available_leads:
                casting_list_str = "\n".join([f"- {c['name']}: {c['description']}" for c in available_leads])
                cast_kwargs = {"role_archetype": role, "scene_prompt": scene_prompt,
                               "lead_casting_list": casting_list_str}
                chosen_name = execute_task(self.engine, self.director_agent, 'DIRECTOR_CAST_LEAD_FOR_ROLE', [],
                                           task_prompt_kwargs=cast_kwargs)

                if chosen_name and 'none' not in chosen_name.lower():
                    if lead_to_load := roster_manager.find_character_in_list(chosen_name, available_leads):
                        roster_manager.decorate_and_add_character(self.engine, lead_to_load, 'lead')
                        utils.log_message('game',
                                          f"[DIRECTOR] Cast '{lead_to_load['name']}' for the role of '{role}'.")
                        available_leads = [c for c in available_leads if c['name'] != lead_to_load['name']]
                        continue

            utils.log_message('debug', f"[DIRECTOR] No suitable lead in casting for role '{role}'. Creating a new one.")
            if new_lead_data := character_factory.create_lead_from_role_and_scene(self.engine, self.director_agent,
                                                                                  scene_prompt, role):
                roster_manager.decorate_and_add_character(self.engine, new_lead_data, 'lead')
            else:
                utils.log_message('debug', f"[DIRECTOR WARNING] Failed to generate a required lead for role '{role}'.")

        # --- DM Management ---
        all_chars = self.engine.characters
        char_roster_summary = "\n".join(
            [f"- {c['name']} ({c.get('role_type', 'char')}): {c['description']}" for c in all_chars])

        run_dm_path = file_io.join_path(self.engine.run_path, 'casting_dms.json')
        dm_casting_list = file_io.read_json(run_dm_path, default=config.casting_dms)
        dm_list_str = "\n".join([f"- {dm['name']}: {dm['description']}" for dm in dm_casting_list])

        dm_kwargs = {"scene_prompt": scene_prompt, "character_roster_summary": char_roster_summary,
                     "dm_list": dm_list_str}
        chosen_dms_str = execute_task(self.engine, self.director_agent, 'DIRECTOR_CHOOSE_DMS_FOR_SCENE', [],
                                      task_prompt_kwargs=dm_kwargs)

        if chosen_dms_str and 'none' not in chosen_dms_str.lower():
            chosen_dm_names = [name.strip() for name in chosen_dms_str.split(';') if name.strip()]
            for name in chosen_dm_names:
                if dm_to_load := roster_manager.find_character_in_list(name, dm_casting_list):

                    # --- DM Tuning ---
                    tuned_dm = dm_to_load.copy()
                    tune_kwargs = {
                        "world_theme": self.engine.world_theme,
                        "scene_prompt": scene_prompt,
                        "dm_name": tuned_dm['name'],
                        "dm_instructions": tuned_dm.get('instructions', '')
                    }
                    new_instructions_str = execute_task(self.engine, self.director_agent,
                                                        'DIRECTOR_TUNE_DM_INSTRUCTIONS', [],
                                                        task_prompt_kwargs=tune_kwargs)

                    if new_instructions_str and 'none' not in new_instructions_str.lower():
                        tuned_dm['instructions'] = new_instructions_str.strip()

                        # Get annotation for the new version
                        anno_kwargs = {
                            "original_instructions": dm_to_load.get('instructions', ''),
                            "new_instructions": tuned_dm['instructions']
                        }
                        annotation = execute_task(self.engine, self.director_agent, 'DIRECTOR_GET_DM_ANNOTATION', [],
                                                  task_prompt_kwargs=anno_kwargs)

                        if annotation and 'none' not in annotation.lower():
                            tuned_dm['name'] = f"{tuned_dm['name']}, {annotation.strip()}"

                        # Save the new variant to the world's casting file
                        file_io.save_character_to_world_casting(self.engine.world_name, tuned_dm, 'dm')
                        utils.log_message('debug',
                                          f"[DIRECTOR] Created and saved new DM variant: '{tuned_dm['name']}'.")

                    roster_manager.decorate_and_add_character(self.engine, tuned_dm, 'dm')

        roster_manager.inject_lead_summary_into_dms(self.engine)

    def _get_director_command(self, character):
        """Prepares prompt, gets a response from the Director, and parses it."""
        conversation_str = "\n".join(
            [f"- {entry['speaker']}: {entry['content'][:250]}..." for entry in self.engine.dialogue_log])
        available_chars = roster_manager.get_available_casting_characters(self.engine)
        casting_list_str = "\n".join([f"- **{c['name']}**: {c['description']}" for c in
                                      available_chars]) if available_chars else "No new characters are available to load."

        prompt_kwargs = {
            'character_name': character['name'],
            'character_instructions': character.get('instructions', ''),
            'conversation_str': conversation_str,
            'casting_list_str': casting_list_str
        }

        raw_response = execute_task(
            self.engine,
            self.director_agent,
            'DIRECTOR_GET_COMMAND',
            [],
            task_prompt_kwargs=prompt_kwargs
        )

        return command_parser.parse_structured_command(
            self.engine,
            raw_response,
            'DIRECTOR',
            fallback_task_key='CH_FIX_DIRECTOR',
            fallback_prompt_kwargs={'casting_list_str': casting_list_str}
        )

    def _handle_character_loading(self, command):
        """Loads a character from casting files and adds them to the roster."""
        target_name = command.get("target_name", "")
        if not target_name: return
        utils.log_message('debug', loc('system_director_load_attempt', target_name=target_name))
        if char_to_load := roster_manager.find_character_in_list(target_name,
                                                                 roster_manager.get_available_casting_characters(
                                                                         self.engine)):
            is_lead = roster_manager.find_character_in_list(target_name, config.casting_leads) is not None
            role_type = 'lead' if is_lead else 'npc'
            roster_manager.decorate_and_add_character(self.engine, char_to_load, role_type)
            log_key = 'system_director_load_lead_success' if role_type == 'lead' else 'system_director_load_npc_success'
            log_args = {'lead_name' if role_type == 'lead' else 'npc_name': char_to_load['name']}
            utils.log_message('debug', loc(log_key, **log_args))
        else:
            utils.log_message('debug', loc('system_director_load_fail', target_name=target_name))

    def _process_character_decision(self, character):
        """Handles the Director's review and resulting action for a single character."""
        command = self._get_director_command(character)
        action = command.get("action", "NONE").upper()

        roster_changed = False
        if action == "UPDATE":
            new_instructions = command.get("new_instructions", "")
            if new_instructions and new_instructions != character.get('instructions'):
                # The Director can also update the core description along with instructions
                new_description = command.get("new_description")
                if new_description and new_description != character.get('description'):
                    character['description'] = new_description
                    # NEW: Log the updated core description
                    utils.log_message('game', f"{character['name']} is now described as: {new_description}")

                character['instructions'] = new_instructions
                utils.log_message('debug', loc('log_director_rewrites', character_name=character['name']))
                roster_changed = True
        elif action == "LOAD":
            self._handle_character_loading(command)
            roster_changed = True

        return roster_changed

    def execute_phase(self):
        """
        Manages the end-of-cycle Director's phase. This now only reviews ONE
        character per cycle, making it less overwhelming. The large-scale NPC
        pruning has been moved to the event-driven Prometheus manager.
        """
        utils.log_message('debug', loc('log_director_phase_header'))
        roster_changed = False

        # Select one character to review for potential updates or cast changes.
        if roles_to_process := [c for c in self.engine.characters if c.get('is_director_managed')]:
            character_to_review = random.choice(roles_to_process)
            if self._process_character_decision(character_to_review):
                roster_changed = True

        if roster_changed:
            file_io.save_active_character_files(self.engine)