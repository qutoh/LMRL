# /core/components/roster_manager.py

import random
from ..common.config_loader import config
from ..common import utils
from ..common import file_io
from ..common.game_state import Entity
from ..common.localization import loc
from ..llm.llm_api import execute_task
from . import character_factory


def get_or_create_npc_from_data(engine, npc_data: dict | str, location_context: dict) -> dict | None:
    """
    Takes high-level NPC data, finds a full profile in casting,
    or creates one using the Director with full context if it doesn't exist.
    The newly created NPC is then cached to the world's casting file.
    """
    npc_concept_dict = {}
    if isinstance(npc_data, str):
        parts = npc_data.split(' - ', 1)
        npc_concept_dict['name'] = parts[0].strip()
        npc_concept_dict['description'] = parts[1].strip() if len(parts) > 1 else "An individual."
    elif isinstance(npc_data, dict):
        npc_concept_dict = npc_data
    else:
        return None

    npc_name = npc_concept_dict.get('name')
    if not npc_name: return None

    # Check world-specific casting file first
    world_casting_path = file_io.join_path(engine.config.data_dir, 'worlds', engine.world_name, 'casting_npcs.json')
    world_casting_list = file_io.read_json(world_casting_path, default=[])
    if found_char := find_character_in_list(npc_name, world_casting_list):
        utils.log_message('debug', f"[ROSTER] Found pre-existing NPC '{npc_name}' in world casting file.")
        return found_char

    # Check global casting file
    if found_char := find_character_in_list(npc_name, config.casting_npcs):
        utils.log_message('debug', f"[ROSTER] Found pre-existing NPC '{npc_name}' in global casting file.")
        return found_char

    utils.log_message('debug', f"[ROSTER] NPC '{npc_name}' not in casting. Generating full profile from concept.")
    director_agent = config.agents['DIRECTOR']

    # Pass full context to the instruction creation step
    instructions = character_factory.create_instructions_from_description(
        engine, director_agent, npc_name, npc_concept_dict.get('description', 'An ordinary person.'),
        location_context
    )
    if not instructions:
        return None

    full_npc_profile = {
        "name": npc_name,
        "description": npc_concept_dict.get('description'),
        "instructions": instructions
    }

    # Save the newly generated full profile back to the world's casting file for future use
    world_casting_list.append(full_npc_profile)
    file_io.write_json(world_casting_path, world_casting_list)
    utils.log_message('debug', f"[ROSTER] Saved newly generated NPC '{npc_name}' to world casting file.")

    return full_npc_profile


def resolve_character_from_description(engine, description: str, existing_characters: list[str]) -> str | None:
    """
    Uses the CHARACTER_RESOLVER agent to match a description to a list of names.
    """
    resolver_agent = config.agents.get('CHARACTER_RESOLVER')
    if not resolver_agent:
        utils.log_message('debug', "[SYSTEM WARNING] CHARACTER_RESOLVER agent not found.")
        return None

    if not existing_characters:
        return None

    user_prompt = (
        f"Description: '{description}'\n"
        f"Character List: {existing_characters}"
    )

    response = execute_task(
        engine,
        resolver_agent,
        'RESOLVE_CHARACTER_DESCRIPTION',
        [{"role": "user", "content": user_prompt}]
    )

    if response and response.strip().upper() != 'NONE':
        for char_name in existing_characters:
            if char_name.lower() == response.strip().lower():
                utils.log_message('debug',
                                  f"[RESOLVER] Matched description '{description}' to existing character '{char_name}'.")
                return char_name

    utils.log_message('debug', f"[RESOLVER] No existing character matched description '{description}'.")
    return None


def find_character_in_list(name: str, char_list: list) -> dict | None:
    """Finds a character by name in a given list of character dictionaries."""
    if not name or not char_list:
        return None
    for char in char_list:
        if char.get('name', '').lower() == name.lower():
            return char
    return None


def decorate_and_add_character(engine, character_data, role_type):
    """
    Adds internal attributes, default settings, adds character to the engine's
    roster, and saves them to persistent world files.
    """
    if find_character(engine, character_data['name']):
        return

    # --- Persist the new character to world files ---
    if engine.world_name:
        # 1. Save the full profile to the world's casting file.
        file_io.save_character_to_world_casting(engine.world_name, character_data, role_type)

        # 2. Save a simple concept to the location's inhabitant list.
        if role_type in ['lead', 'npc'] and engine.location_breadcrumb:
            concept = {
                "name": character_data.get("name"),
                "description": character_data.get("description")
            }
            file_io.add_inhabitant_to_location(engine.world_name, engine.location_breadcrumb, concept)

    # --- Decorate with in-memory run-time attributes ---
    char = character_data.copy()
    char['role_type'] = role_type

    if role_type == 'lead':
        char.update({'is_positional': True, 'is_director_managed': True, 'is_essential': True, 'is_controllable': True})
    elif role_type == 'npc':
        char.update(
            {'is_positional': True, 'is_director_managed': False, 'is_essential': False, 'is_controllable': True})
    elif role_type == 'dm':
        char.update(
            {'is_positional': False, 'is_director_managed': True, 'is_essential': False, 'is_controllable': False})

    if not char.get("model"):
        model_key = f"DEFAULT_{role_type.upper()}_MODEL"
        if default_model := config.settings.get(model_key):
            char["model"] = default_model

    temp_key = f"DEFAULT_{role_type.upper()}_TEMPERATURE"
    scale_key = f"DEFAULT_{role_type.upper()}_SCALING_FACTOR"

    default_temp = config.settings.get(temp_key, 0.75)
    default_scale = config.settings.get(scale_key, 1.0)

    char['temperature'] = char.get('temperature', default_temp)
    char['scaling_factor'] = char.get('scaling_factor', default_scale)

    if 'physical_description' not in char:
        char['physical_description'] = "No specific description available."
    if 'equipment' not in char:
        char['equipment'] = {"equipped": [], "removed": [], "outfits": {}}

    add_character(engine, char)
    utils.log_message('game', f"{char.get('name', 'A new character')} has joined the story.")
    if phys_desc := char.get('physical_description'):
        utils.log_message('game', f"-> {phys_desc}")
    utils.log_message('debug', loc('system_roster_loaded', role_type=role_type.upper(), char_name=char['name'],
                                   model_name=char.get('model')))


def load_initial_roster(engine):
    """Loads all characters from their respective files for a given run."""
    run_path = engine.run_path
    initial_leads = file_io.read_json(file_io.join_path(run_path, 'leads.json'), default=[])
    initial_dms = file_io.read_json(file_io.join_path(run_path, 'dm_roles.json'), default=[])
    initial_npcs = file_io.read_json(file_io.join_path(run_path, 'temporary_npcs.json'), default=[])
    for char in initial_leads: decorate_and_add_character(engine, char, 'lead')
    for char in initial_dms: decorate_and_add_character(engine, char, 'dm')
    for char in initial_npcs: decorate_and_add_character(engine, char, 'npc')


def load_characters_from_scene(engine, scene_data):
    """Loads characters specified in a scene object and from location inhabitants."""
    if not isinstance(scene_data, dict): return

    run_lead_casting_path = file_io.join_path(engine.run_path, 'casting_leads.json')
    all_casting_leads = config.casting_leads + file_io.read_json(run_lead_casting_path, default=[])
    all_casting_npcs = config.casting_npcs
    all_casting_dms = config.casting_dms

    def load_from_casting(names_to_load, role_type, source_list):
        for name in names_to_load:
            if char_to_load := find_character_in_list(name, source_list):
                decorate_and_add_character(engine, char_to_load, role_type)

    load_from_casting(scene_data.get('load_leads', []), 'lead', all_casting_leads)
    load_from_casting(scene_data.get('load_dms', []), 'dm', all_casting_dms)

    # --- Contextual and Scoped NPC Loading ---
    location_data = scene_data.get('source_location', {})
    location_context = {
        "world_theme": engine.world_theme,
        "scene_prompt": scene_data.get('scene_prompt'),
        "location_description": location_data.get('Description', '')
    }

    # 1. Load NPCs explicitly listed in the scene file
    for name in scene_data.get('load_npcs', []):
        if npc_profile := find_character_in_list(name, all_casting_npcs):
            decorate_and_add_character(engine, npc_profile, 'npc')

    # 2. Load/Generate NPCs from the location's "inhabitants" list
    if inhabitants := location_data.get('inhabitants'):
        for npc_concept in inhabitants:
            if full_npc_profile := get_or_create_npc_from_data(engine, npc_concept, location_context):
                decorate_and_add_character(engine, full_npc_profile, 'npc')


# # NEW: Wrapper function to centralize the create->add flow
def create_and_add_lead(engine, dialogue_log):
    """Calls the factory to create a new lead, then adds the result to the roster."""
    if new_lead_data := character_factory.create_lead_stepwise(engine, dialogue_log):
        decorate_and_add_character(engine, new_lead_data, 'lead')
        return new_lead_data
    return None


def inject_lead_summary_into_dms(engine):
    """Adds a summary of the current lead characters to the instructions of all DMs."""
    leads = [c for c in engine.characters if c.get('role_type') == 'lead']
    if not leads: return
    lead_summaries = [f"- **{lead['name']}**: {lead['description']}" for lead in leads]
    full_summary = f"\n\n--- CURRENT PARTY ROSTER ---\nThe current adventuring party you are overseeing consists of the following individuals:\n" + "\n".join(
        lead_summaries)
    utils.log_message('debug', loc('system_lead_summary_header'))
    utils.log_message('debug', full_summary)
    for char in engine.characters:
        if char.get('role_type') == 'dm':
            char['instructions'] += full_summary


def spawn_entities_from_roster(engine, game_state):
    """Creates Entity objects in the game_state for all positional characters."""
    if not game_state: return
    colors = [(255, 255, 255), (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255),
              (128, 0, 128)]
    color_index = 0
    for character in engine.characters:
        if not character.get('is_positional'): continue
        if game_state.get_entity(character['name']): continue
        x, y = 0, 0
        for _ in range(100):
            x = random.randint(1, game_state.game_map.width - 2)
            y = random.randint(1, game_state.game_map.height - 2)
            if game_state.game_map.is_walkable(x, y) and not any(e.x == x and e.y == y for e in game_state.entities):
                break
        char_visual = character['name'][0].upper()
        char_color = colors[color_index % len(colors)]
        color_index += 1
        entity = Entity(name=character['name'], x=x, y=y, char=char_visual, color=char_color)
        game_state.add_entity(entity)
        utils.log_message('debug', loc('system_roster_spawned_entity', entity_name=entity.name, x=entity.x, y=entity.y))


def place_entity_by_instruction(game_state: 'GameState', entity_name: str, instruction: dict, placed_features: dict):
    """
    Finds a specific entity and moves it to a position relative to a placed feature.
    """
    entity = game_state.get_entity(entity_name)
    if not entity:
        utils.log_message('debug', loc('warning_placement_nonexistent_entity', entity_name=entity_name))
        return

    placement = instruction.get("placement")
    if not placement or not placement.get("target_tag"):
        utils.log_message('debug', loc('warning_placement_no_valid_placement', entity_name=entity_name))
        return

    target_box = placed_features.get(placement["target_tag"])
    if not target_box:
        utils.log_message('debug', loc('warning_placement_target_not_found', target_tag=placement['target_tag'],
                                       entity_name=entity_name))
        return

    tx1, ty1, tx2, ty2 = target_box

    for _ in range(100):
        side = random.choice(['top', 'bottom', 'left', 'right'])

        try:
            if side in ['left', 'right']:
                x = tx1 - 1 if side == 'left' else tx2
                y = random.randint(ty1, ty2 - 1)
            else:
                x = random.randint(tx1, tx2 - 1)
                y = ty1 - 1 if side == 'top' else ty2
        except ValueError:
            x = tx1 - 1 if random.choice([True, False]) else tx1
            y = ty1 if x == tx1 - 1 else ty1 - 1

        if game_state.game_map.is_walkable(x, y) and not any(
                e.x == x and e.y == y for e in game_state.entities if e.name != entity.name):
            entity.x = x
            entity.y = y
            utils.log_message('debug',
                              f"[PLACEMENT] Placed entity '{entity_name}' at ({x},{y}) next to '{placement['target_tag']}'.")
            return

    utils.log_message('debug',
                      loc('warning_placement_no_walkable', entity_name=entity_name, target_tag=placement['target_tag']))


def find_character(engine, name):
    if not name or not engine.characters: return None
    return find_character_in_list(name, engine.characters)


def get_all_characters(engine):
    return engine.characters


def get_lead_names(engine):
    return [c['name'] for c in engine.characters if c.get('role_type') == 'lead']


def get_available_casting_characters(engine):
    active_names = {c['name'].lower() for c in get_all_characters(engine)}
    available_chars = []
    seen_names = set()
    run_casting_leads = file_io.read_json(file_io.join_path(engine.run_path, 'casting_leads.json'), default=[])

    # config.casting_npcs now contains the merged list of global + world NPCs
    casting_sources = [
        config.casting_leads, config.casting_npcs, config.casting_dms,
        run_casting_leads
    ]
    for char_list in casting_sources:
        for char in char_list:
            name_lower = char['name'].lower()
            if name_lower not in active_names and name_lower not in seen_names:
                available_chars.append(char)
                seen_names.add(name_lower)
    return available_chars


def add_character(engine, character_data):
    if not find_character(engine, character_data['name']):
        engine.characters.append(character_data)


def remove_character(engine, character_name):
    if character_to_remove := find_character(engine, character_name):
        engine.characters = [c for c in engine.characters if c['name'].lower() != character_name.lower()]
        return character_to_remove
    return None


def promote_npc_to_lead(engine, npc_name):
    if char_to_promote := find_character(engine, npc_name):
        if char_to_promote.get('role_type') == 'npc':
            char_to_promote['role_type'] = 'lead'
            char_to_promote['is_essential'] = True
            char_to_promote['is_director_managed'] = True

            utils.log_message('debug', loc('system_director_lead_promote', npc_name=npc_name,
                                           character_name='(N/A)'))  # Context may vary
            return char_to_promote
    return None


def update_character_instructions(character, new_instructions):
    if new_instructions and new_instructions != character.get('instructions'):
        character['instructions'] = new_instructions
        return True
    return False