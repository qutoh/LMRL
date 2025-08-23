# /core/components/roster_manager.py

import ast
import random

from . import character_factory
from ..common import file_io
from ..common import utils
from ..common.config_loader import config
from ..common.game_state import Entity
from ..common.localization import loc
from ..common.utils import log_message
from ..llm.llm_api import execute_task


def get_or_create_character_from_concept(engine, char_concept: dict | str, location_context: dict) -> tuple[
    dict | None, str | None]:
    """
    Takes a high-level character concept, finds a full profile in any casting list (inhabitant, lead, or NPC),
    or creates one if it doesn't exist. The newly created inhabitant is then cached.
    """
    concept_dict = {}
    if isinstance(char_concept, str):
        parts = char_concept.split(' - ', 1)
        concept_dict['name'] = parts[0].strip()
        concept_dict['description'] = parts[1].strip() if len(parts) > 1 else "An individual."
    elif isinstance(char_concept, dict):
        concept_dict = char_concept
    else:
        return None, None

    char_name = concept_dict.get('name')
    if not char_name: return None, None

    # --- SEARCH PHASE: Check all casting files first, with inhabitant cache as top priority ---
    inhabitant_path = file_io.join_path(engine.config.data_dir, 'worlds', engine.world_name,
                                        'inhabitants.json')
    inhabitant_list = file_io.read_json(inhabitant_path, default=[])

    # Priority 1: Check the dedicated inhabitant cache for this world.
    if found_char := find_character_in_list(char_name, inhabitant_list):
        utils.log_message('debug', f"[ROSTER] Found pre-existing inhabitant '{char_name}' in world's inhabitant cache.")
        return found_char, 'npc'  # Inhabitants are always treated as NPCs

    # Priority 2: Check standard world/global casting files
    world_leads_path = file_io.join_path(engine.config.data_dir, 'worlds', engine.world_name, 'casting_leads.json')
    world_leads_list = file_io.read_json(world_leads_path, default=[])
    world_npcs_path = file_io.join_path(engine.config.data_dir, 'worlds', engine.world_name, 'casting_npcs.json')
    world_npcs_list = file_io.read_json(world_npcs_path, default=[])

    if found_char := find_character_in_list(char_name, world_leads_list): return found_char, 'lead'
    if found_char := find_character_in_list(char_name, world_npcs_list): return found_char, 'npc'
    if found_char := find_character_in_list(char_name, config.casting_leads): return found_char, 'lead'
    if found_char := find_character_in_list(char_name, config.casting_npcs): return found_char, 'npc'

    # --- CREATION PHASE: Only if not found anywhere ---
    utils.log_message('debug', f"[ROSTER] Inhabitant concept '{char_name}' not found. Generating full profile.")

    # Check if the concept came from proc-gen, which has richer data
    if 'source_sentence' in concept_dict:
        full_char_profile = character_factory.create_npc_from_generation_sentence(engine, concept_dict)
    else: # It's a simple inhabitant from the world file
        char_data_for_creation = {
            "name": char_name,
            "source_sentence": concept_dict.get('description', 'An individual encountered in the scene.')
        }
        full_char_profile = character_factory.create_npc_from_generation_sentence(engine, char_data_for_creation)


    if not full_char_profile:
        return None, None

    # Save the newly generated profile to the inhabitant cache for persistence.
    inhabitant_list.append(full_char_profile)
    file_io.write_json(inhabitant_path, inhabitant_list)
    utils.log_message('debug', f"[ROSTER] Saved newly generated inhabitant '{char_name}' to world's inhabitant cache.")

    return full_char_profile, 'npc'


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
    roster, and saves NPCs as persistent inhabitants of their location.
    """
    if find_character(engine, character_data['name']):
        return

    # --- Persist the new character to world files ---
    if engine.world_name and role_type == 'npc' and engine.location_breadcrumb:
        # 1. Save a simple concept to the location's inhabitant list within world.json
        concept = {
            "name": character_data.get("name"),
            "description": character_data.get("description")
        }
        file_io.add_inhabitant_to_location(engine.world_name, engine.location_breadcrumb, concept)

        # 2. Save the full character profile to the world's dedicated inhabitant cache.
        inhabitant_path = file_io.join_path(engine.config.data_dir, 'worlds', engine.world_name, 'inhabitants.json')
        inhabitant_list = file_io.read_json(inhabitant_path, default=[])
        char_name_lower = character_data.get('name', '').lower()
        if not any(c.get('name', '').lower() == char_name_lower for c in inhabitant_list):
            inhabitant_list.append(character_data)
            file_io.write_json(inhabitant_path, inhabitant_list)

    # --- Decorate with in-memory run-time attributes ---
    char = character_data.copy()

    # Convert stringified tuple keys back to tuples upon loading a saved meta-DM.
    if 'fused_personas' in char and isinstance(char['fused_personas'], dict):
        tuple_keyed_personas = {}
        for key, value in char['fused_personas'].items():
            # Use ast.literal_eval to safely convert string back to tuple
            try:
                tuple_key = ast.literal_eval(key)
                if isinstance(tuple_key, tuple):
                    tuple_keyed_personas[tuple_key] = value
                else:  # Fallback if key isn't a tuple string (shouldn't happen with our save logic)
                    tuple_keyed_personas[key] = value
            except (ValueError, SyntaxError):
                # Key was not a string representation of a tuple, keep as is.
                tuple_keyed_personas[key] = value
        char['fused_personas'] = tuple_keyed_personas

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

    # Only add positional attributes if the character is meant to be in the scene.
    if char.get('is_positional'):
        char['character_state'] = ""
        if 'physical_description' not in char:
            char['physical_description'] = "No specific description available."
        if 'equipment' not in char:
            char['equipment'] = {"equipped": [], "removed": [], "outfits": {}}

    add_character(engine, char)
    log_message('game', f"{char.get('name', 'A new character')} has joined the story.")
    if char.get('is_positional'):
        if phys_desc := char.get('physical_description'):
            log_message('game', f"-> {phys_desc}")

    utils.log_message('debug', loc('system_roster_loaded', role_type=role_type.upper(), char_name=char['name'],
                                   model_name=char.get('model')))


def load_initial_roster(engine):
    """Loads all characters from their respective files for a given run."""
    run_path = engine.run_path

    # --- Global Characters (loaded into every game) ---
    global_leads = file_io.read_json(file_io.join_path(config.data_dir, 'leads.json'), default=[])
    global_npcs = file_io.read_json(file_io.join_path(config.data_dir, 'casting_npcs.json'), default=[])
    global_dms = file_io.read_json(file_io.join_path(config.data_dir, 'dm_roles.json'), default=[])

    for char in global_leads: decorate_and_add_character(engine, char, 'lead')
    for char in global_npcs: decorate_and_add_character(engine, char, 'npc')
    for char in global_dms: decorate_and_add_character(engine, char, 'dm')

    # --- Run-Specific Characters (from a saved game or scene) ---
    initial_leads = file_io.read_json(file_io.join_path(run_path, 'leads.json'), default=[])
    initial_dms = file_io.read_json(file_io.join_path(run_path, 'dm_roles.json'), default=[])
    initial_npcs = file_io.read_json(file_io.join_path(run_path, 'temporary_npcs.json'), default=[])
    for char in initial_leads: decorate_and_add_character(engine, char, 'lead')
    for char in initial_dms: decorate_and_add_character(engine, char, 'dm')
    for char in initial_npcs: decorate_and_add_character(engine, char, 'npc')


def load_characters_from_scene(engine, scene_data, hydrate_inhabitants: bool = True):
    """
    Loads characters specified in a scene object. Inhabitants from the location
    can be deferred by setting hydrate_inhabitants to False.
    """
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

    # 2. Load/Generate characters from the location's "inhabitants" list
    if inhabitants := location_data.get('inhabitants'):
        if hydrate_inhabitants:
            for char_concept in inhabitants:
                full_char_profile, role_type = get_or_create_character_from_concept(engine, char_concept, location_context)
                if full_char_profile and role_type:
                    decorate_and_add_character(engine, full_char_profile, role_type)
        else:
            # If not hydrating, add them to the engine's dehydrated list for later.
            engine.dehydrated_npcs.extend(inhabitants)


# # NEW: Wrapper function to centralize the create->add flow
def create_and_add_lead(engine, dialogue_log):
    """Calls the factory to create a new lead, then adds the result to the roster."""
    if new_lead_data := character_factory.create_lead_stepwise(engine, dialogue_log):
        decorate_and_add_character(engine, new_lead_data, 'lead')
        return new_lead_data
    return None


def spawn_single_entity(engine, game_state, character_name: str, target_feature_name: str | None = None):
    """Creates an Entity for a single character and places it, preferably in a specific feature."""
    if not game_state or game_state.get_entity(character_name): return

    character = find_character(engine, character_name)
    if not character: return

    # Simplified color logic for single spawn
    colors = [(255, 255, 255), (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255), (128, 0, 128)]
    color = colors[len(game_state.entities) % len(colors)]
    char_visual = character['name'][0].upper()

    x, y = -1, -1
    # Attempt to place in the target feature
    if target_feature_name and engine.generation_state:
        target_tag = next((tag for tag, data in engine.generation_state.placed_features.items() if data.get('name') == target_feature_name), None)
        if target_tag:
            bounds = engine.generation_state.placed_features[target_tag].get('bounding_box')
            if bounds:
                x1, y1, w, h = [int(c) for c in bounds]
                for _ in range(100): # 100 attempts to find a free spot
                    px, py = random.randint(x1, x1 + w - 1), random.randint(y1, y1 + h - 1)
                    if game_state.game_map.is_walkable(px, py) and not any(e.x == px and e.y == py for e in game_state.entities):
                        x, y = px, py
                        break

    # Fallback to random placement if specific placement fails or is not requested
    if x == -1 and y == -1:
        for _ in range(200):
            x = random.randint(1, game_state.game_map.width - 2)
            y = random.randint(1, game_state.game_map.height - 2)
            if game_state.game_map.is_walkable(x, y) and not any(e.x == x and e.y == y for e in game_state.entities):
                break

    entity = Entity(name=character['name'], x=x, y=y, char=char_visual, color=color)
    game_state.add_entity(entity)
    utils.log_message('debug', loc('system_roster_spawned_entity', entity_name=entity.name, x=entity.x, y=entity.y))


def spawn_entities_from_roster(engine, game_state):
    """Creates Entity objects in the game_state for all positional characters."""
    if not game_state: return
    for character in engine.characters:
        if character.get('is_positional'):
            spawn_single_entity(engine, game_state, character['name'])


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


def get_active_dm(engine):
    """Finds the single active meta-DM if single DM mode is enabled."""
    if not config.settings.get("enable_multiple_dms", False):
        for char in engine.characters:
            if char.get('role_type') == 'dm':
                return char
    return None