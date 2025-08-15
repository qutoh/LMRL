# /core/components/character_factory.py

from ..common.config_loader import config
from ..llm.llm_api import execute_task
from ..common.utils import log_message
from ..common.localization import loc
from ..common import utils
from ..common import file_io
from ..common import command_parser


def _create_character_one_shot(engine, agent, task_key, task_kwargs) -> dict | None:
    """Primary Method: Attempts to generate all descriptive fields in one LLM call."""
    raw_response = execute_task(engine, agent, task_key, [], task_prompt_kwargs=task_kwargs)

    fix_it_key = 'CH_FIX_FULL_CHAR_JSON'
    if task_key == 'DIRECTOR_CREATE_FULL_LEAD_PROFILE_FROM_ROLE':
        fix_it_key = 'CH_FIX_FULL_LEAD_JSON'

    command = command_parser.parse_structured_command(
        engine, raw_response, agent.get('name', 'DIRECTOR'),
        fallback_task_key=fix_it_key
    )
    if command and all(k in command for k in ['name', 'description', 'instructions', 'physical_description']):
        return command
    return None


def _create_npc_from_generation_sentence_stepwise(engine, char_data: dict) -> dict | None:
    """Fallback: Creates a full NPC profile from a sentence using a step-by-step process."""
    utils.log_message('debug', "[PEG CREATE] Single-shot failed. Falling back to step-wise NPC creation.")
    context_sentence = char_data.get('source_sentence')
    npc_name = char_data.get('name', 'Unnamed Character')

    creator_agent = config.agents['DIRECTOR']

    desc_kwargs = {"new_name": npc_name, "context_sentence": context_sentence}
    description = execute_task(engine, creator_agent, 'NPC_CREATE_DESC_FROM_PROCGEN', [],
                               task_prompt_kwargs=desc_kwargs)
    if not description:
        utils.log_message('debug', "[PEG CREATE WARNING] Could not generate a description.")
        description = "An individual described during the scene's creation."

    instr_kwargs = {"npc_name": npc_name, "description": description}
    instructions = execute_task(engine, creator_agent, 'NPC_CREATE_INSTR_FROM_PROCGEN', [],
                                task_prompt_kwargs=instr_kwargs)
    if not instructions:
        utils.log_message('debug', "[PEG CREATE WARNING] Could not generate instructions.")
        return None

    return {"name": npc_name, "model": "", "description": description.strip(), "instructions": instructions.strip()}


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


def create_npc_from_generation_sentence(engine, char_data: dict) -> dict | None:
    """
    Orchestrator for creating a full NPC profile from a single sentence provided by the
    procedural generation system. Tries a single-shot JSON method first, then falls back.
    """
    context_sentence = char_data.get('source_sentence')
    npc_name = char_data.get('name', 'Unnamed Character')
    if not context_sentence:
        return None

    utils.log_message('debug', f"[PEG CREATE] Creating character '{npc_name}' from sentence: '{context_sentence}'")

    creator_agent = config.agents['DIRECTOR']
    kwargs = {"context_sentence": context_sentence, "npc_name": npc_name}
    new_npc = _create_character_one_shot(engine, creator_agent, 'DIRECTOR_CREATE_FULL_NPC_PROFILE', kwargs)

    if not new_npc:
        new_npc = _create_npc_from_generation_sentence_stepwise(engine, char_data)

    if new_npc:
        equipment_agent = config.agents['EQUIPMENT_MANAGER']
        phys_desc = new_npc.get('physical_description', '')
        equip_kwargs = {"physical_description": phys_desc}
        raw_items_response = execute_task(engine, equipment_agent, 'EQUIPMENT_GENERATE_INITIAL', [],
                                          task_prompt_kwargs=equip_kwargs)

        command = command_parser.parse_structured_command(
            engine, raw_items_response, equipment_agent.get('name'),
            fallback_task_key='CH_FIX_INITIAL_EQUIPMENT_JSON'
        )

        initial_items = []
        if command and isinstance(command.get('items'), list):
            initial_items = command['items']
        else:
            initial_items = _generate_initial_equipment_stepwise(engine, equipment_agent, phys_desc)

        new_npc['equipment'] = {
            "equipped": initial_items, "removed": [],
            "outfits": {
                "DEFAULT": {
                    "items": sorted([item['name'] for item in initial_items]),
                    "description": phys_desc
                }
            }
        }

        log_message('debug', loc('system_npc_create_success', npc_name=new_npc['name']))
        log_message('full', loc('system_npc_created_full', npc_details=new_npc))

    return new_npc


def create_instructions_from_description(engine, director_agent, npc_name: str, description: str,
                                         location_context: dict = None) -> str | None:
    """Generates only the instructions for a character based on their name and description."""
    instr_kwargs = {
        "npc_name": npc_name,
        "description": description,
        "world_theme": location_context.get('world_theme', ''),
        "scene_prompt": location_context.get('scene_prompt', ''),
        "location_description": location_context.get('location_description', '')
    }
    instructions = execute_task(
        engine,
        director_agent,
        'DIRECTOR_CREATE_INSTR_FROM_DESC',
        [],
        task_prompt_kwargs=instr_kwargs
    )
    if not instructions:
        log_message('debug', loc('system_npc_create_fail_instr'))
        return None
    return instructions.strip()


def _create_lead_from_role_and_scene_stepwise(engine, director_agent, scene_prompt: str,
                                              role_archetype: str) -> dict | None:
    """Fallback: Creates a full lead profile from a role using a step-by-step process."""
    log_message('debug', f"[DIRECTOR] Creating new lead for role '{role_archetype}' via stepwise method.")

    name_kwargs = {"scene_prompt": scene_prompt, "role_archetype": role_archetype}
    new_name = execute_task(engine, director_agent, 'DIRECTOR_CREATE_LEAD_NAME_FROM_ROLE', [],
                            task_prompt_kwargs=name_kwargs)
    if not new_name or not new_name.strip():
        new_name = file_io.get_random_name()
    new_name = new_name.strip()

    desc_kwargs = {"scene_prompt": scene_prompt, "new_name": new_name, "role_archetype": role_archetype}
    new_description = execute_task(engine, director_agent, 'DIRECTOR_CREATE_LEAD_DESC_FROM_ROLE', [],
                                   task_prompt_kwargs=desc_kwargs)
    if not new_description: return None
    new_description = new_description.strip()

    instr_kwargs = {"scene_prompt": scene_prompt, "new_name": new_name, "new_description": new_description,
                    "role_archetype": role_archetype}
    new_instructions = execute_task(engine, director_agent, 'DIRECTOR_CREATE_LEAD_INSTR_FROM_ROLE', [],
                                    task_prompt_kwargs=instr_kwargs)
    if not new_instructions: return None
    new_instructions = new_instructions.strip()

    phys_desc_kwargs = {"scene_prompt": scene_prompt, "new_name": new_name, "new_description": new_description,
                        "role_archetype": role_archetype}
    physical_description = execute_task(engine, director_agent, 'DIRECTOR_CREATE_LEAD_PHYS_DESC_FROM_ROLE', [],
                                        task_prompt_kwargs=phys_desc_kwargs) or "An unremarkable individual."

    return {"name": new_name, "description": new_description, "instructions": new_instructions,
            "physical_description": physical_description.strip()}


def create_lead_from_role_and_scene(engine, director_agent, scene_prompt: str, role_archetype: str) -> dict | None:
    """Creates a new lead character guided by a specific role archetype, using a one-shot JSON method with a stepwise fallback."""
    log_message('debug', f"[DIRECTOR] Creating new lead for role: '{role_archetype}'.")

    kwargs = {"scene_prompt": scene_prompt, "role_archetype": role_archetype}
    new_lead = _create_character_one_shot(engine, director_agent, 'DIRECTOR_CREATE_FULL_LEAD_PROFILE_FROM_ROLE',
                                          kwargs)

    if not new_lead:
        log_message('debug', "[DIRECTOR] One-shot method failed. Falling back to stepwise creation.")
        new_lead = _create_lead_from_role_and_scene_stepwise(engine, director_agent, scene_prompt, role_archetype)

    if not new_lead:
        log_message('debug', loc('system_director_create_instr_fail', new_name=role_archetype))
        return None

    equipment_agent = config.agents['EQUIPMENT_MANAGER']
    phys_desc = new_lead.get('physical_description', '')
    equip_kwargs = {"physical_description": phys_desc}
    raw_items_response = execute_task(engine, equipment_agent, 'EQUIPMENT_GENERATE_INITIAL', [],
                                      task_prompt_kwargs=equip_kwargs)

    command = command_parser.parse_structured_command(
        engine, raw_items_response, equipment_agent.get('name'),
        fallback_task_key='CH_FIX_INITIAL_EQUIPMENT_JSON'
    )
    initial_items = []
    if command and isinstance(command.get('items'), list):
        initial_items = command['items']
    else:
        initial_items = _generate_initial_equipment_stepwise(engine, equipment_agent, phys_desc)

    new_lead['equipment'] = {
        "equipped": initial_items, "removed": [],
        "outfits": {
            "DEFAULT": {
                "items": sorted([item['name'] for item in initial_items]),
                "description": phys_desc
            }
        }
    }
    log_message('debug', f"[DIRECTOR] Successfully created new lead '{new_lead['name']}'.")
    return new_lead


def create_lead_from_scene_context(engine, director_agent, scene_prompt: str) -> dict | None:
    """Creates a new lead character via a robust, step-by-step process using the starting scene."""
    log_message('debug', "[DIRECTOR] Creating a new lead character suitable for the scene.")

    new_name = None
    if "lead" in config.settings.get("FORCE_RANDOM_NAMES_FOR_ROLES", []):
        new_name = file_io.get_random_name()
        utils.log_message('debug', f"[DIRECTOR] Overriding Lead name generation. Using random name: '{new_name}'")
    else:
        name_kwargs = {"scene_prompt": scene_prompt}
        new_name = execute_task(engine, director_agent, 'DIRECTOR_CREATE_LEAD_NAME_FROM_SCENE', [],
                                task_prompt_kwargs=name_kwargs)

    if not new_name or not new_name.strip():
        new_name = file_io.get_random_name()
        log_message('debug', f"[DIRECTOR] LLM failed to provide lead name. Using fallback: '{new_name}'")
    new_name = new_name.strip()
    log_message('debug', f"[DIRECTOR] Step 1/3 SUCCESS: Generated name '{new_name}'.")

    desc_kwargs = {"scene_prompt": scene_prompt, "new_name": new_name}
    new_description = execute_task(engine, director_agent, 'DIRECTOR_CREATE_LEAD_DESC_FROM_SCENE', [],
                                   task_prompt_kwargs=desc_kwargs)
    if not new_description:
        log_message('debug', loc('system_director_create_desc_fail', new_name=new_name))
        return None
    new_description = new_description.strip()
    log_message('debug', f"[DIRECTOR] Step 2/3 SUCCESS: Generated description for '{new_name}'.")

    instr_kwargs = {"scene_prompt": scene_prompt, "new_name": new_name, "new_description": new_description}
    new_instructions = execute_task(engine, director_agent, 'DIRECTOR_CREATE_LEAD_INSTR_FROM_SCENE', [],
                                    task_prompt_kwargs=instr_kwargs)
    if not new_instructions:
        log_message('debug', loc('system_director_create_instr_fail', new_name=new_name))
        return None
    new_instructions = new_instructions.strip()
    log_message('debug', f"[DIRECTOR] Step 3/3 SUCCESS: Generated instructions for '{new_name}'.")

    return {
        "name": new_name,
        "description": new_description,
        "instructions": new_instructions
    }


def create_temporary_npc(engine, creator_dm, npc_name, dialogue_log):
    """Creates a new temporary NPC using the creator's LLM, with a simple name prompt."""
    log_message('debug', loc('system_npc_create_attempt', npc_name=npc_name))
    recent_dialogue = "\n".join([f"{entry['speaker']}: {entry['content']}" for entry in dialogue_log[-15:]])
    context = loc('prompt_npc_context', recent_dialogue=recent_dialogue)

    desc_kwargs = {"context": context, "npc_name": npc_name}
    description = execute_task(
        engine,
        creator_dm,
        'NPC_CREATE_DESC_FROM_CONTEXT',
        [],
        task_prompt_kwargs=desc_kwargs
    )
    if not description:
        log_message('debug', loc('system_npc_create_fail_desc'))
        return None

    instr_kwargs = {"npc_name": npc_name, "description": description}
    instructions = execute_task(
        engine,
        creator_dm,
        'NPC_CREATE_INSTR_FROM_CONTEXT',
        [],
        task_prompt_kwargs=instr_kwargs
    )
    if not instructions:
        log_message('debug', loc('system_npc_create_fail_instr'))
        return None

    new_npc = {"name": npc_name, "model": "", "description": description.strip(), "instructions": instructions.strip()}
    log_message('debug', loc('system_npc_create_success', npc_name=npc_name))
    log_message('full', loc('system_npc_created_full', npc_details=new_npc))
    return new_npc


def create_lead_stepwise(engine, dialogue_log):
    """
    Creates a new lead character through a robust, step-by-step process.
    MODIFIED: This function now only returns the character data dictionary. It no longer adds it to the roster.
    """
    log_message('debug', loc('system_director_create_start'))
    director_agent = config.agents['DIRECTOR']
    context_str = "\n".join([f"{entry['speaker']}: {entry['content']}" for entry in dialogue_log[-15:]])
    base_context = loc('prompt_npc_context', recent_dialogue=context_str)

    new_name = None
    if "lead" in config.settings.get("FORCE_RANDOM_NAMES_FOR_ROLES", []):
        new_name = file_io.get_random_name()
        utils.log_message('debug', f"[DIRECTOR] Overriding Lead name generation. Using random name: '{new_name}'")
    else:
        lead_kwargs = {"context": base_context}
        new_name = execute_task(engine, director_agent, 'LEAD_CREATE_NAME', [], task_prompt_kwargs=lead_kwargs)

    if not new_name or not new_name.strip():
        new_name = file_io.get_random_name()
        log_message('debug',
                    f"[DIRECTOR] LLM failed to provide lead name for replacement. Using fallback: '{new_name}'")
    new_name = new_name.strip()
    log_message('debug', loc('system_director_create_name_ok', new_name=new_name))

    desc_kwargs = {"context": base_context, "new_name": new_name}
    new_description = execute_task(engine, director_agent, 'LEAD_CREATE_DESC', [], task_prompt_kwargs=desc_kwargs)
    if not new_description:
        log_message('debug', loc('system_director_create_desc_fail', new_name=new_name))
        return None
    new_description = new_description.strip()
    log_message('debug', loc('system_director_create_desc_ok', new_name=new_name))

    instr_kwargs = {"new_name": new_name, "new_description": new_description}
    new_instructions = execute_task(engine, director_agent, 'LEAD_CREATE_INSTR', [], task_prompt_kwargs=instr_kwargs)
    if not new_instructions:
        log_message('debug', loc('system_director_create_instr_fail', new_name=new_name))
        return None
    new_instructions = new_instructions.strip()
    log_message('debug', loc('system_director_create_instr_ok', new_name=new_name))

    new_lead = {"name": new_name, "model": "", "description": new_description, "instructions": new_instructions}
    # The call to roster_manager has been removed from here.
    return new_lead