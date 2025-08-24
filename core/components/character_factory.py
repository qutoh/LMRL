# /core/components/character_factory.py

from ..common import command_parser
from ..common import file_io
from ..common import utils
from ..common.config_loader import config
from ..common.localization import loc
from ..common.utils import log_message
from ..llm.llm_api import execute_task


def _generate_persona_description(engine, name: str, description: str) -> str:
    """Generates the simple, first-person persona description for a character."""
    kwargs = {"character_name": name, "character_description": description}
    # This requires a new task 'DIRECTOR_CREATE_PERSONA_DESC' to be defined.
    persona_desc = execute_task(engine, config.agents['DIRECTOR'], 'DIRECTOR_CREATE_PERSONA_DESC', [],
                                task_prompt_kwargs=kwargs)
    return persona_desc or f"You are {name}."  # Robust fallback for LLM failure


def _prepare_physical_description_task(base_kwargs: dict, role_hint: str, scene_context: str) -> tuple[str, dict]:
    """
    Determines the correct task key and arguments for generating a physical description
    based on the ENABLE_PAPER_DOLL_MODE setting.
    """
    kwargs = base_kwargs.copy()

    if config.settings.get("ENABLE_PAPER_DOLL_MODE"):
        kwargs['equipment_instruction'] = loc('prompt_substring_paper_doll_on')
    else:
        kwargs['equipment_instruction'] = loc('prompt_substring_paper_doll_off')
    kwargs['role_archetype'] = role_hint
    # No need to add scene_prompt here as it's part of the shared context
    task_key = 'DIRECTOR_CREATE_LEAD_PHYS_DESC_FROM_ROLE'
    return task_key, kwargs


def _create_character_one_shot(engine, agent, task_key, task_kwargs) -> dict | None:
    """Primary Method: Attempts to generate all descriptive fields in one LLM call."""
    if config.settings.get("ENABLE_PAPER_DOLL_MODE"):
        task_kwargs['equipment_instruction'] = loc('prompt_substring_paper_doll_on')
    else:
        task_kwargs['equipment_instruction'] = loc('prompt_substring_paper_doll_off')
    task_kwargs['full_json'] = loc('prompt_substring_full_json_character', **task_kwargs)

    raw_response = execute_task(engine, agent, task_key, [], task_prompt_kwargs=task_kwargs)

    fix_it_key = 'CH_FIX_FULL_CHAR_JSON'
    if task_key == 'DIRECTOR_CREATE_FULL_LEAD_PROFILE_FROM_ROLE':
        fix_it_key = 'CH_FIX_FULL_LEAD_JSON'

    command = command_parser.parse_structured_command(
        engine, raw_response, agent.get('name', 'DIRECTOR'),
        fallback_task_key=fix_it_key
    )
    if command and all(k in command for k in
                       ['name', 'description', 'instructions', 'physical_description', 'persona_description']):
        return command
    return None


def _create_npc_from_generation_sentence_stepwise(engine, char_data: dict) -> dict | None:
    """Fallback: Creates a full NPC profile from a sentence using a step-by-step process."""
    utils.log_message('debug', "[PEG CREATE] Single-shot failed. Falling back to step-wise NPC creation.")
    context_sentence = char_data.get('source_sentence')
    npc_name = char_data.get('name', 'Unnamed Character')
    creator_agent = config.agents['DIRECTOR']

    shared_context_str = loc('prompt_substring_world_scene_context', world_theme=engine.world_theme,
                             scene_prompt=engine.scene_prompt)
    base_kwargs = {"prompt_substring_world_scene_context": shared_context_str}

    desc_kwargs = {**base_kwargs, "new_name": npc_name, "context_sentence": context_sentence}
    description = execute_task(engine, creator_agent, 'NPC_CREATE_DESC_FROM_PROCGEN', [],
                               task_prompt_kwargs=desc_kwargs) or "An individual."

    instr_kwargs = {**base_kwargs, "npc_name": npc_name, "description": description}
    instructions = execute_task(engine, creator_agent, 'NPC_CREATE_INSTR_FROM_PROCGEN', [],
                                task_prompt_kwargs=instr_kwargs) or "Behave as described."

    base_phys_desc_kwargs = {**base_kwargs, "new_name": npc_name, "new_description": description}
    phys_desc_task, final_phys_desc_kwargs = _prepare_physical_description_task(
        base_kwargs=base_phys_desc_kwargs,
        role_hint=description,
        scene_context=context_sentence
    )

    physical_description = execute_task(engine, creator_agent, phys_desc_task, [],
                                        task_prompt_kwargs=final_phys_desc_kwargs) or "An unremarkable individual."

    persona_description = _generate_persona_description(engine, npc_name, description)

    return {"name": npc_name, "description": description, "instructions": instructions,
            "physical_description": physical_description, "persona_description": persona_description}


def _generate_initial_equipment_stepwise(engine, equipment_agent, physical_description: str) -> list[dict]:
    """Fallback for equipment generation: get names, then describe each."""
    utils.log_message('debug', "[EQUIPMENT] JSON generation failed. Falling back to stepwise equipment creation.")
    names_kwargs = {"physical_description": physical_description}
    names_response = execute_task(engine, equipment_agent, 'EQUIPMENT_GET_ITEM_NAMES_FROM_DESC', [],
                                  task_prompt_kwargs=names_kwargs)

    item_names = [name.strip() for name in names_response.split(';') if name.strip()]
    if not item_names:
        return []

    desc_kwargs = {"item_names_list": "; ".join(item_names), "context": physical_description}
    raw_desc_response = execute_task(engine, equipment_agent, 'EQUIPMENT_DESCRIBE_ITEMS', [],
                                     task_prompt_kwargs=desc_kwargs)

    command = command_parser.parse_structured_command(
        engine, raw_desc_response, equipment_agent.get('name', 'EQUIPMENT_MANAGER'),
        fallback_task_key='CH_FIX_INITIAL_EQUIPMENT_JSON'
    )

    final_items = []
    if command:
        for name, details in command.items():
            if isinstance(details, dict) and 'description' in details:
                final_items.append({"name": name, "description": details["description"]})
    return final_items


def _initialize_character_equipment(engine, character_data: dict, scene_context: str):
    """Handles both standard and 'paper doll' equipment generation for a new character."""
    equipment_agent = config.agents['EQUIPMENT_MANAGER']
    phys_desc = character_data.get('physical_description', '')

    if config.settings.get("ENABLE_PAPER_DOLL_MODE"):
        base_phys_desc = phys_desc
        character_data['equipment'] = {
            "equipped": [], "removed": [],
            "outfits": {"DEFAULT": {"items": [], "description": base_phys_desc}}
        }

        gear_kwargs = {
            "character_name": character_data.get('name', 'N/A'),
            "character_description": character_data.get('description', 'N/A'),
            "character_instructions": character_data.get('instructions', 'N/A'),
            "scene_context": scene_context
        }
        raw_items_response = execute_task(engine, equipment_agent, 'EQUIPMENT_GENERATE_GEAR_FROM_PROFILE', [],
                                          task_prompt_kwargs=gear_kwargs)

        command = command_parser.parse_structured_command(engine, raw_items_response, equipment_agent.get('name'),
                                                          fallback_task_key='CH_FIX_INITIAL_EQUIPMENT_JSON')
        starting_items = command.get('items', []) if command else []

        if not starting_items:
            return character_data

        character_data['equipment']['equipped'] = starting_items

        director_agent = config.agents['DIRECTOR']
        update_desc_kwargs = {
            "original_description": base_phys_desc,
            "items_added": ", ".join([item['name'] for item in starting_items]), "items_removed": "None"
        }
        final_phys_desc = execute_task(engine, director_agent, 'DIRECTOR_UPDATE_PHYSICAL_DESCRIPTION', [],
                                       task_prompt_kwargs=update_desc_kwargs)

        if final_phys_desc:
            character_data['physical_description'] = final_phys_desc.strip()
            character_data['equipment']['outfits']['OUTFIT_0'] = {
                "items": sorted([item['name'] for item in starting_items]),
                "description": final_phys_desc.strip()
            }
    else:
        equip_kwargs = {"physical_description": phys_desc}
        raw_items_response = execute_task(engine, equipment_agent, 'EQUIPMENT_GENERATE_INITIAL', [],
                                          task_prompt_kwargs=equip_kwargs)

        command = command_parser.parse_structured_command(engine, raw_items_response, equipment_agent.get('name'),
                                                          fallback_task_key='CH_FIX_INITIAL_EQUIPMENT_JSON')

        initial_items = []
        if command and isinstance(command.get('items'), list):
            initial_items = command['items']
        else:
            initial_items = _generate_initial_equipment_stepwise(engine, equipment_agent, phys_desc)

        character_data['equipment'] = {
            "equipped": initial_items, "removed": [],
            "outfits": {
                "DEFAULT": {
                    "items": sorted([item['name'] for item in initial_items]),
                    "description": phys_desc
                }
            }
        }
    return character_data


def create_npc_from_generation_sentence(engine, char_data: dict) -> dict | None:
    """Orchestrator for creating a full NPC profile from a single sentence."""
    context_sentence = char_data.get('source_sentence', '')
    npc_name = char_data.get('name', 'Unnamed Character')
    if not context_sentence: return None
    utils.log_message('debug', f"[PEG CREATE] Creating character '{npc_name}' from sentence: '{context_sentence}'")

    creator_agent = config.agents['DIRECTOR']
    shared_context_str = loc('prompt_substring_world_scene_context', world_theme=engine.world_theme,
                             scene_prompt=engine.scene_prompt)
    kwargs = {
        "prompt_substring_world_scene_context": shared_context_str,
        "context_sentence": context_sentence,
        "npc_name": npc_name
    }
    new_npc = _create_character_one_shot(engine, creator_agent, 'DIRECTOR_CREATE_FULL_NPC_PROFILE', kwargs)

    if not new_npc:
        new_npc = _create_npc_from_generation_sentence_stepwise(engine, char_data)

    if new_npc:
        new_npc = _initialize_character_equipment(engine, new_npc, context_sentence)
        log_message('debug', loc('system_npc_create_success', npc_name=new_npc['name']))
        log_message('full', loc('system_npc_created_full', npc_details=new_npc))

    return new_npc


def create_lead_from_role_and_scene(engine, director_agent, scene_prompt: str, role_archetype: str) -> dict | None:
    """Creates a new lead character guided by a specific role archetype."""
    log_message('debug', f"[DIRECTOR] Creating new lead for role: '{role_archetype}'.")
    shared_context_str = loc('prompt_substring_world_scene_context', world_theme=engine.world_theme,
                             scene_prompt=scene_prompt)
    kwargs = {
        "prompt_substring_world_scene_context": shared_context_str,
        "role_archetype": role_archetype
    }
    new_lead = _create_character_one_shot(engine, director_agent, 'DIRECTOR_CREATE_FULL_LEAD_PROFILE_FROM_ROLE',
                                          kwargs)

    if not new_lead:
        log_message('debug', "[DIRECTOR] One-shot method failed. Falling back to stepwise creation.")
        new_lead = _create_lead_from_role_and_scene_stepwise(engine, director_agent, scene_prompt, role_archetype)

    if not new_lead:
        log_message('debug', loc('system_director_create_instr_fail', new_name=role_archetype))
        return None

    new_lead = _initialize_character_equipment(engine, new_lead, scene_prompt)
    log_message('debug', f"[DIRECTOR] Successfully created new lead '{new_lead['name']}'.")
    return new_lead


def _create_lead_from_role_and_scene_stepwise(engine, director_agent, scene_prompt: str,
                                              role_archetype: str) -> dict | None:
    """Fallback: Creates a full lead profile from a role using a step-by-step process."""
    log_message('debug', f"[DIRECTOR] Creating new lead for role '{role_archetype}' via stepwise method.")

    shared_context_str = loc('prompt_substring_world_scene_context', world_theme=engine.world_theme,
                             scene_prompt=scene_prompt)
    base_kwargs = {
        "prompt_substring_world_scene_context": shared_context_str,
        "role_archetype": role_archetype
    }

    name_kwargs = base_kwargs.copy()
    new_name = execute_task(engine, director_agent, 'DIRECTOR_CREATE_LEAD_NAME_FROM_ROLE', [],
                            task_prompt_kwargs=name_kwargs)
    if not new_name or not new_name.strip(): new_name = file_io.get_random_name()
    new_name = new_name.strip()

    desc_kwargs = {**base_kwargs, "new_name": new_name}
    new_description = execute_task(engine, director_agent, 'DIRECTOR_CREATE_LEAD_DESC_FROM_ROLE', [],
                                   task_prompt_kwargs=desc_kwargs)
    if not new_description: return None
    new_description = new_description.strip()

    instr_kwargs = {**desc_kwargs, "new_description": new_description}
    new_instructions = execute_task(engine, director_agent, 'DIRECTOR_CREATE_LEAD_INSTR_FROM_ROLE', [],
                                    task_prompt_kwargs=instr_kwargs)
    if not new_instructions: return None
    new_instructions = new_instructions.strip()

    base_phys_desc_kwargs = {**base_kwargs, "new_name": new_name, "new_description": new_description}
    phys_desc_task, final_phys_desc_kwargs = _prepare_physical_description_task(
        base_kwargs=base_phys_desc_kwargs,
        role_hint=role_archetype,
        scene_context=scene_prompt
    )

    physical_description = execute_task(engine, director_agent, phys_desc_task, [],
                                        task_prompt_kwargs=final_phys_desc_kwargs) or "An unremarkable individual."

    persona_description = _generate_persona_description(engine, new_name, new_description)

    return {"name": new_name, "description": new_description, "instructions": new_instructions,
            "physical_description": physical_description.strip(), "persona_description": persona_description}


def _create_temporary_npc_stepwise(engine, director_agent, npc_name, context) -> dict | None:
    """Fallback for creating a temporary NPC when the one-shot method fails."""
    utils.log_message('debug', f"[DIRECTOR] Stepwise fallback for temporary NPC '{npc_name}'.")
    shared_context_str = loc('prompt_substring_world_scene_context', world_theme=engine.world_theme,
                             scene_prompt=engine.scene_prompt)
    base_kwargs = {"prompt_substring_world_scene_context": shared_context_str}

    desc_kwargs = {**base_kwargs, "context": context, "npc_name": npc_name}
    description = execute_task(engine, director_agent, 'NPC_CREATE_DESC_FROM_CONTEXT', [],
                               task_prompt_kwargs=desc_kwargs) or "An individual."

    instr_kwargs = {**base_kwargs, "npc_name": npc_name, "description": description}
    instructions = execute_task(engine, director_agent, 'NPC_CREATE_INSTR_FROM_CONTEXT', [],
                                task_prompt_kwargs=instr_kwargs) or "Behave as described."

    base_phys_desc_kwargs = {**base_kwargs, "new_name": npc_name, "new_description": description}
    phys_desc_task, final_phys_desc_kwargs = _prepare_physical_description_task(
        base_kwargs=base_phys_desc_kwargs,
        role_hint=description,
        scene_context=context
    )

    physical_description = execute_task(engine, director_agent, phys_desc_task, [],
                                        task_prompt_kwargs=final_phys_desc_kwargs) or "An unremarkable individual."

    persona_description = _generate_persona_description(engine, npc_name, description)

    return {"name": npc_name, "description": description, "instructions": instructions,
            "physical_description": physical_description, "persona_description": persona_description}


def create_temporary_npc(engine, creator_dm, npc_name, dialogue_log):
    """Creates a new temporary NPC, now using the robust one-shot/fallback pattern."""
    log_message('debug', loc('system_npc_create_attempt', npc_name=npc_name))
    recent_dialogue = "\n".join([f"{entry['speaker']}: {entry['content']}" for entry in dialogue_log[-15:]])

    director_agent = config.agents['DIRECTOR']
    shared_context_str = loc('prompt_substring_world_scene_context', world_theme=engine.world_theme,
                             scene_prompt=engine.scene_prompt)
    kwargs = {
        "prompt_substring_world_scene_context": shared_context_str,
        "context_sentence": recent_dialogue,
        "npc_name": npc_name
    }
    new_npc = _create_character_one_shot(engine, director_agent, 'DIRECTOR_CREATE_FULL_NPC_PROFILE', kwargs)

    if not new_npc:
        new_npc = _create_temporary_npc_stepwise(engine, director_agent, npc_name, recent_dialogue)

    if new_npc:
        new_npc = _initialize_character_equipment(engine, new_npc, recent_dialogue)
        log_message('debug', loc('system_npc_create_success', npc_name=new_npc['name']))
        log_message('full', loc('system_npc_created_full', npc_details=new_npc))

    return new_npc


def create_lead_stepwise(engine, dialogue_log):
    """Creates a new lead character for player takeover through a robust, step-by-step process."""
    log_message('debug', loc('system_director_create_start'))
    director_agent = config.agents['DIRECTOR']
    context_str = "\n".join([f"{entry['speaker']}: {entry['content']}" for entry in dialogue_log[-15:]])

    role_archetype = "A new adventurer joining the story."

    new_lead = create_lead_from_role_and_scene(engine, director_agent, context_str, role_archetype)

    return new_lead