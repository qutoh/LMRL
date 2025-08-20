# /core/engine/director.py

import random

from core.common import command_parser, file_io, utils
from core.common.config_loader import config
from core.common.localization import loc
from core.components import character_factory
from core.components import roster_manager
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
        log_key = ''

        if role_type in ['lead', 'dm']:
            should_save_to_casting = True
            log_key = 'system_lead_removed_and_saved' if role_type == 'lead' else ''  # Add DM log key if needed
        elif role_type == 'npc':
            prompt_kwargs = {'npc_name': char_to_remove['name'],
                             'npc_description': char_to_remove.get('description', '')}
            response = execute_task(self.engine, self.director_agent, 'DIRECTOR_CONFIRM_NPC_SAVE', [],
                                    task_prompt_kwargs=prompt_kwargs)
            if response and 'yes' in response.strip().lower():
                should_save_to_casting = True
                log_key = 'system_npc_removed_and_saved'
            else:
                log_key = 'system_npc_removed_not_saved'

        if should_save_to_casting:
            file_io.save_character_to_world_casting(self.engine.world_name, char_to_remove, role_type)

        removed_char_name = char_to_remove['name']
        roster_manager.remove_character(self.engine, removed_char_name)

        if log_key:
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

        shared_context_str = loc('prompt_substring_world_scene_context',
                                 world_theme=self.engine.world_theme or "A generic world.",
                                 scene_prompt=scene_prompt or "A story begins.")

        roles_kwargs = {
            "prompt_substring_world_scene_context": shared_context_str,
            "location_summary": location_summary or "An undescribed location.",
            "npc_list_summary": npc_summary
        }
        raw_roles_response = execute_task(self.engine, self.director_agent, 'DIRECTOR_DEFINE_LEAD_ROLES_FOR_SCENE', [],
                                          task_prompt_kwargs=roles_kwargs)

        needed_roles = []
        command = command_parser.parse_structured_command(
            self.engine, raw_roles_response, 'DIRECTOR', fallback_task_key='CH_FIX_LEAD_ROLES_JSON'
        )

        if command and isinstance(command.get('roles'), list) and command['roles']:
            needed_roles = command['roles']
        else:
            utils.log_message('debug', "[DIRECTOR] JSON for lead roles failed. Falling back to semicolon-based task.")
            fallback_response = execute_task(self.engine, self.director_agent,
                                             'DIRECTOR_DEFINE_LEAD_ROLES_FOR_SCENE_FALLBACK', [],
                                             task_prompt_kwargs=roles_kwargs)
            needed_roles = [role.strip() for role in fallback_response.split(';') if role.strip()]

        utils.log_message('game', f"[DIRECTOR] The director has defined the lead roles needed: {needed_roles}")

        available_leads = [c for c in config.casting_leads if not roster_manager.find_character(self.engine, c['name'])]

        for role in needed_roles:
            if available_leads:
                casting_list_str = "\n".join([f"- {c['name']}: {c['description']}" for c in available_leads])
                cast_kwargs = {
                    "prompt_substring_world_scene_context": shared_context_str,
                    "role_archetype": role,
                    "lead_casting_list": casting_list_str
                }
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

        # --- DM Management (Performed ONCE after all other characters are set) ---
        all_chars = self.engine.characters
        char_roster_summary = "\n".join(
            [f"- {c['name']} ({c.get('role_type', 'char')}): {c['description']}" for c in all_chars])

        dm_casting_list = config.casting_dms
        dm_list_str = "\n".join([f"- {dm['name']}: {dm['description']}" for dm in dm_casting_list])

        dm_kwargs = {
            "prompt_substring_world_scene_context": shared_context_str,
            "character_roster_summary": char_roster_summary,
            "dm_list": dm_list_str
        }
        chosen_dms_str = execute_task(self.engine, self.director_agent, 'DIRECTOR_CHOOSE_DMS_FOR_SCENE', [],
                                      task_prompt_kwargs=dm_kwargs)

        chosen_dm_profiles = []
        if chosen_dms_str and 'none' not in chosen_dms_str.lower():
            chosen_dm_names = [name.strip() for name in chosen_dms_str.split(';') if name.strip()]
            for name in chosen_dm_names:
                if dm_to_load := roster_manager.find_character_in_list(name, dm_casting_list):
                    chosen_dm_profiles.append(self.engine.dm_manager._tailor_dm_for_scene(dm_to_load))

        # --- DM Synthesis or Individual Addition ---
        if not config.settings.get("enable_multiple_dms", False) and len(chosen_dm_profiles) > 0:
            self.engine.dm_manager.initialize_meta_dm(chosen_dm_profiles)
        else:
            for dm_profile in chosen_dm_profiles:
                roster_manager.decorate_and_add_character(self.engine, dm_profile, 'dm')


    def _get_director_command(self, character):
        """Prepares prompt, gets a response from the Director, and parses it."""
    def _get_director_command(self, character):
        """Prepares prompt, gets a response from the Director, and parses it."""
        conversation_str = "\n".join(
            [f"- {entry['speaker']}: {entry['content'][:250]}..." for entry in self.engine.dialogue_log])
        available_chars = roster_manager.get_available_casting_characters(self.engine)
        casting_list_str = "\n".join([f"- **{c['name']}**: {c['description']}" for c in
                                      available_chars]) if available_chars else "No new characters are available to load."

        shared_context_str = loc('prompt_substring_world_scene_context',
                                 world_theme=self.engine.world_theme,
                                 scene_prompt=self.engine.scene_prompt)

        prompt_kwargs = {
            'prompt_substring_world_scene_context': shared_context_str,
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
        char_to_load = roster_manager.find_character_in_list(
            target_name, roster_manager.get_available_casting_characters(self.engine)
        )

        if not char_to_load:
            utils.log_message('debug', loc('system_director_load_fail', target_name=target_name))
            return

        role_type = 'npc'  # Default
        if roster_manager.find_character_in_list(target_name, config.casting_leads):
            role_type = 'lead'
        elif roster_manager.find_character_in_list(target_name, config.casting_dms):
            role_type = 'dm'

        if role_type == 'dm' and not config.settings.get("enable_multiple_dms", False):
            self.engine.dm_manager.fuse_dm_into_meta(char_to_load)
        else:
            roster_manager.decorate_and_add_character(self.engine, char_to_load, role_type)
            log_key = 'system_director_load_lead_success' if role_type == 'lead' else 'system_director_load_npc_success'
            log_args = {'lead_name' if role_type == 'lead' else 'npc_name': char_to_load['name']}
            utils.log_message('debug', loc(log_key, **log_args))

    def _process_character_decision(self, character):
        """Handles the Director's review and resulting action for a single character."""
        command = self._get_director_command(character)
        action = command.get("action", "NONE").upper()

        roster_changed = False
        if action == "UPDATE":
            new_instructions = command.get("new_instructions", "")
            if new_instructions and new_instructions != character.get('instructions'):
                new_description = command.get("new_description")
                if new_description and new_description != character.get('description'):
                    character['description'] = new_description
                    utils.log_message('game', f"{character['name']} is now described as: {new_description}")

                character['instructions'] = new_instructions
                utils.log_message('debug', loc('log_director_rewrites', character_name=character['name']))
                roster_changed = True
        elif action == "LOAD":
            self._handle_character_loading(command)
            roster_changed = True
        elif action == "UNFUSE_DM":
            self.engine.dm_manager.unfuse_dm_from_meta()
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

        if roles_to_process := [c for c in self.engine.characters if c.get('is_director_managed')]:
            character_to_review = random.choice(roles_to_process)
            if self._process_character_decision(character_to_review):
                roster_changed = True

        if roster_changed:
            file_io.save_active_character_files(self.engine)