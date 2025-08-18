# /core/components/dm_manager.py

from ..common import utils, file_io, command_parser
from ..common.config_loader import config
from ..common.localization import loc
from ..llm.llm_api import execute_task
from . import roster_manager


class DMManager:
    """
    Manages the logic for tailoring, fusing, and un-fusing Dungeon Master
    personas when the 'enable_multiple_dms' setting is false.
    """

    def __init__(self, engine):
        self.engine = engine

    def _tailor_dm_for_scene(self, dm_profile: dict) -> dict:
        """Creates a scene-specific variant of a global DM."""
        tuned_dm = dm_profile.copy()

        shared_context_str = loc('prompt_substring_world_scene_context',
                                 world_theme=self.engine.world_theme,
                                 scene_prompt=self.engine.scene_prompt)

        tune_kwargs = {
            "prompt_substring_world_scene_context": shared_context_str,
            "dm_name": tuned_dm['name'],
            "dm_instructions": tuned_dm.get('instructions', '')
        }
        new_instructions_str = execute_task(self.engine, config.agents['DIRECTOR'],
                                            'DIRECTOR_TUNE_DM_INSTRUCTIONS', [],
                                            task_prompt_kwargs=tune_kwargs)

        if new_instructions_str and 'none' not in new_instructions_str.lower():
            tuned_dm['instructions'] = new_instructions_str.strip()
            anno_kwargs = {
                "prompt_substring_world_scene_context": shared_context_str,
                "original_instructions": dm_profile.get('instructions', ''),
                "new_instructions": tuned_dm['instructions']
            }
            annotation = execute_task(self.engine, config.agents['DIRECTOR'], 'DIRECTOR_GET_DM_ANNOTATION', [],
                                      task_prompt_kwargs=anno_kwargs)
            if annotation and 'none' not in annotation.lower():
                tuned_dm['name'] = f"{tuned_dm['name']}, {annotation.strip()}"
            file_io.save_character_to_world_casting(self.engine.world_name, tuned_dm, 'dm')
            utils.log_message('debug', f"[DM FUSION] Created and saved new DM variant: '{tuned_dm['name']}'.")
        return tuned_dm

    def initialize_meta_dm(self, initial_dm_profiles: list):
        """
        Performs the initial synthesis of DMs at the start of a run,
        creating the meta-DM and its persona cache.
        """
        if not initial_dm_profiles:
            return

        # The key should be based on the original, untailored names.
        base_component_names = tuple(sorted([dm['name'].split(',')[0].strip() for dm in initial_dm_profiles]))

        dm_profiles_str = "\n---\n".join(
            f"Name: {dm.get('name')}\nDescription: {dm.get('description')}\nInstructions: {dm.get('instructions')}"
            for dm in initial_dm_profiles
        )
        synth_kwargs = {"dm_profiles_str": dm_profiles_str}
        raw_synth_response = execute_task(self.engine, config.agents['SUMMARIZER'], 'SUMMARIZE_SYNTHESIZE_DMS', [],
                                          task_prompt_kwargs=synth_kwargs)
        command = command_parser.parse_structured_command(self.engine, raw_synth_response, 'SUMMARIZER',
                                                          'CH_FIX_SYNTHESIZED_DM_JSON')

        if command and all(k in command for k in ['name', 'description', 'instructions']):
            utils.log_message('game',
                              f"[DIRECTOR] ...synthesis complete. Creating new Game Master: '{command['name']}'.")

            # 'command' is the pure persona. Create the full meta_dm object.
            meta_dm_data = command.copy()

            # The cache should store a copy of the pure persona to prevent recursion.
            meta_dm_data['fused_personas'] = {base_component_names: command.copy()}
            meta_dm_data['component_dms'] = [dm['name'] for dm in initial_dm_profiles]

            roster_manager.decorate_and_add_character(self.engine, meta_dm_data, 'dm')
        else:
            utils.log_message('debug', "[DIRECTOR] DM synthesis failed. Falling back to using multiple DMs.")
            for dm_profile in initial_dm_profiles:
                roster_manager.decorate_and_add_character(self.engine, dm_profile, 'dm')

    def fuse_dm_into_meta(self, dm_to_fuse: dict):
        """
        Handles the logic for tailoring and fusing a new DM into the
        active meta-DM during gameplay.
        """
        meta_dm = roster_manager.get_active_dm(self.engine)
        if not meta_dm:
            utils.log_message('debug', "[DM FUSION] Fuse called but no active meta-DM found. Aborting.")
            return

        base_name_to_add = dm_to_fuse['name'].split(',')[0].strip()
        current_base_components = [name.split(',')[0].strip() for name in meta_dm.get('component_dms', [])]
        if base_name_to_add in current_base_components:
            utils.log_message('debug', f"[DM FUSION] DM '{base_name_to_add}' is already part of the meta-DM. Aborting.")
            return

        is_global = roster_manager.find_character_in_list(dm_to_fuse['name'], config.dm_roles) is not None or \
                    roster_manager.find_character_in_list(dm_to_fuse['name'], config.casting_dms) is not None

        dm_variant_to_add = self._tailor_dm_for_scene(dm_to_fuse) if is_global else dm_to_fuse

        new_components_full_names = meta_dm['component_dms'] + [dm_variant_to_add['name']]
        new_cache_key = tuple(sorted(current_base_components + [base_name_to_add]))

        if new_cache_key in meta_dm.get('fused_personas', {}):
            utils.log_message('debug', f"[DM FUSION] Found cached persona for components: {new_cache_key}. Applying.")
            cached_persona = meta_dm['fused_personas'][new_cache_key]
            meta_dm.update(cached_persona)
            meta_dm['component_dms'] = new_components_full_names
        else:
            utils.log_message('debug', f"[DM FUSION] No cached persona found for {new_cache_key}. Synthesizing new one.")

            all_dm_profiles_for_synthesis = []

            # 1. Get profiles for existing components from the roster
            for name in meta_dm['component_dms']:
                 if profile := roster_manager.find_character(self.engine, name):
                    all_dm_profiles_for_synthesis.append(profile)

            # 2. Add the newly tailored profile
            all_dm_profiles_for_synthesis.append(dm_variant_to_add)

            dm_profiles_str = "\n---\n".join(
                f"Name: {dm.get('name')}\nDescription: {dm.get('description')}\nInstructions: {dm.get('instructions')}"
                for dm in all_dm_profiles_for_synthesis
            )
            synth_kwargs = {"dm_profiles_str": dm_profiles_str}
            raw_synth_response = execute_task(self.engine, config.agents['SUMMARIZER'], 'SUMMARIZE_SYNTHESIZE_DMS', [],
                                              task_prompt_kwargs=synth_kwargs)
            command = command_parser.parse_structured_command(self.engine, raw_synth_response, 'SUMMARIZER',
                                                              'CH_FIX_SYNTHESIZED_DM_JSON')

            if command and all(k in command for k in ['name', 'description', 'instructions']):
                meta_dm.update(command)
                meta_dm['component_dms'] = new_components_full_names
                meta_dm['fused_personas'][new_cache_key] = command
                utils.log_message('game', f"[DM FUSION] Meta-DM has evolved into: '{command['name']}'.")
            else:
                utils.log_message('debug', "[DM FUSION] Mid-game synthesis failed. No changes made.")

    def unfuse_dm_from_meta(self):
        """
        Handles removing a component DM from the meta-DM and reverting
        to a previous, cached persona.
        """
        meta_dm = roster_manager.get_active_dm(self.engine)
        if not meta_dm or len(meta_dm.get('component_dms', [])) < 2:
            utils.log_message('debug', "[DM FUSION] Un-fuse called, but meta-DM has fewer than 2 components. Aborting.")
            return

        component_list_str = "\n".join([f"- {name}" for name in meta_dm['component_dms']])
        kwargs = {"component_dm_list": component_list_str}
        response = execute_task(self.engine, config.agents['DIRECTOR'], 'DIRECTOR_CHOOSE_DM_TO_UNFUSE', [], task_prompt_kwargs=kwargs)

        if not response or 'none' in response.lower():
            utils.log_message('debug', "[DM FUSION] Director chose not to un-fuse a DM.")
            return

        name_to_remove = response.strip()

        full_name_to_remove = next((comp for comp in meta_dm['component_dms'] if comp.startswith(name_to_remove)), None)

        if not full_name_to_remove:
            utils.log_message('debug', f"[DM FUSION] Could not find component '{name_to_remove}' to un-fuse.")
            return

        new_components_full_names = [comp for comp in meta_dm['component_dms'] if comp != full_name_to_remove]
        new_cache_key = tuple(sorted([name.split(',')[0].strip() for name in new_components_full_names]))

        if new_cache_key in meta_dm.get('fused_personas', {}):
            cached_persona = meta_dm['fused_personas'][new_cache_key]
            meta_dm.update(cached_persona)
            meta_dm['component_dms'] = new_components_full_names
            utils.log_message('game', f"[DM FUSION] Meta-DM has simplified to: '{cached_persona['name']}'.")
        else:
            utils.log_message('debug', f"[DM FUSION] [ERROR] Cache miss on un-fusion for key {new_cache_key}. This should not happen. Aborting.")