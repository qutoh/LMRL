# /core/components/position_manager.py

import math
import random
import re

import numpy as np
import tcod

from . import game_functions
from . import roster_manager
from ..common import command_parser
from ..common import utils
from ..common.config_loader import config
from ..common.game_state import GameState
from ..llm import llm_api

FIRST_PERSON_PRONOUNS = {"i", "me", "myself", "my"}

movement_classifiers = {
    "DASH": r"\b(dash|dashes|dashed|rush|rushes|rushed|charge|charges|charged|sprint|sprints|sprinted|bolt|bolts|bolted)\b",
    "SNEAK": r"\b(sneak|sneaks|snuck|creep|creeps|crept|stalk|stalks|stalked|sidle|sidles|sidled|tiptoe|tiptoes|tiptoed)\b",
    "CRAWL": r"\b(crawl|crawls|crawled)\b",
    "CLIMB": r"\b(climb|climbs|climbed|ascend|ascends|ascended|scrambles up)\b",
    "SWIM": r"\b(swim|swims|swam|wade|wades|waded)\b",
    "JUMP": r"\b(jump|jumps|jumped|leap|leaps|leapt)\b",
    "FALLBACK": r"\b(retreat|retreats|retreated|withdraw|withdraws|withdrew|backs away|backed away|fall back|falls back|fell back)\b",
    "RUN": r"\b(run|runs|ran)\b",
    "WALK": r"\b(walk|walks|walked|go|goes|went|move|moves|moved|proceed|proceeds|proceeded|advance|advances|advanced|step|steps|stepped|sets off|set off|position|positions|positioned)\b",
}


def classify_movement_mode(text: str) -> str | None:
    """Iterates through classifiers to find the dominant movement mode."""
    for mode, pattern in movement_classifiers.items():
        if re.search(pattern, text, re.IGNORECASE):
            return mode
    return None


def handle_turn_movement(engine, game_state: GameState, character_name: str, prose: str):
    """
    Classifies movement type from prose and calls the intent processor.
    This is the main entry point for handling movement during a character's turn.
    """
    if movement_mode := classify_movement_mode(prose):
        process_movement_intent(engine, game_state, character_name, prose, movement_mode)


def _get_nearby_feature_tags_for_leads(engine, game_state: GameState) -> set:
    """Helper to get the set of all feature tags currently visible to any lead character."""
    leads = [c for c in engine.characters if c.get('role_type') == 'lead' and c.get('is_positional')]
    visible_tags = set()
    if not engine.generation_state: return visible_tags

    for lead in leads:
        entity = game_state.get_entity(lead['name'])
        if not entity: continue

        current_tag = get_narrative_location_for_position(entity.x, entity.y, engine.generation_state)
        if current_tag:
            visible_tags.add(current_tag)
            # Add adjacent tags
            if engine.layout_graph:
                for p, c, _ in engine.layout_graph.edges:
                    if p == current_tag: visible_tags.add(c)
                    if c == current_tag: visible_tags.add(p)

    return visible_tags


def get_npcs_to_cull(engine, game_state: GameState) -> list[str]:
    """
    Returns a list of NPC names that are not in proximity to any lead characters
    and should be removed from the current turn order.
    """
    npcs = [c for c in engine.characters if c.get('role_type') == 'npc' and c.get('is_positional')]
    if not npcs: return []

    visible_tags = _get_nearby_feature_tags_for_leads(engine, game_state)
    if not visible_tags: # If no leads are in a defined feature, we can't determine proximity.
        return []

    cull_list = []
    for npc in npcs:
        entity = game_state.get_entity(npc['name'])
        if not entity: continue

        npc_tag = get_narrative_location_for_position(entity.x, entity.y, engine.generation_state)
        if npc_tag not in visible_tags:
            cull_list.append(npc['name'])

    return cull_list


def log_character_perspective(engine, game_state: GameState, character_name: str):
    """
    Builds a narrative description of a lead character's surroundings and echoes it to the game log.
    """
    character = roster_manager.find_character(engine, character_name)
    if not character:
        return

    # --- Log character name and description ---
    utils.log_message('game', f"\n--- {character['name']} ---")
    if phys_desc := character.get('physical_description'):
        utils.log_message('game', phys_desc)

    entity = game_state.get_entity(character_name)
    if not entity or not engine.generation_state:
        return

    # --- Determine visible features ---
    features_to_scan = set()
    current_feature_tag = get_narrative_location_for_position(entity.x, entity.y,
                                                              engine.generation_state)

    if current_feature_tag:
        features_to_scan.add(current_feature_tag)
        if engine.layout_graph:
            for parent, child, _ in engine.layout_graph.edges:
                if parent == current_feature_tag:
                    features_to_scan.add(child)
                elif child == current_feature_tag:
                    features_to_scan.add(parent)
    else:
        # Character is in an open area, find closest features
        all_features = engine.generation_state.placed_features
        if all_features:
            features_with_dist = []
            for tag, details in all_features.items():
                if bounds := details.get('bounding_box'):
                    x1, y1, w, h = [int(c) for c in bounds]
                    center_x, center_y = x1 + w / 2, y1 + h / 2
                    distance = math.dist((entity.x, entity.y), (center_x, center_y))
                    features_with_dist.append((distance, tag))
            features_with_dist.sort()
            for _, tag in features_with_dist[:4]:
                features_to_scan.add(tag)

    # --- Format and log location description ---
    if features_to_scan:
        location_descriptions = []
        for tag in sorted(list(features_to_scan)):
            feature_data = engine.generation_state.placed_features.get(tag)
            if feature_data and feature_data.get('description'):
                location_descriptions.append(feature_data['description'].strip())

        if location_descriptions:
            full_description = " ".join(location_descriptions)
            formatted_location_desc = utils.format_text_with_paragraph_breaks(full_description)
            utils.log_message('game', formatted_location_desc)

    # --- Find, group, and log nearby characters ---
    other_entities = [e for e in game_state.entities if e.name != character_name]
    characters_by_location = {}

    for other_entity in other_entities:
        occupant_loc_tag = get_narrative_location_for_position(other_entity.x,
                                                              other_entity.y,
                                                              engine.generation_state)
        if occupant_loc_tag and occupant_loc_tag in features_to_scan:
            loc_data = engine.generation_state.placed_features.get(occupant_loc_tag, {})
            loc_name = loc_data.get('name', occupant_loc_tag)
            if loc_name not in characters_by_location:
                characters_by_location[loc_name] = []
            characters_by_location[loc_name].append(other_entity.name)

    if characters_by_location:
        utils.log_message('game', "")  # Newline for spacing
        for loc_name, char_names in characters_by_location.items():
            if len(char_names) == 1:
                char_list_str = char_names[0]
            elif len(char_names) == 2:
                char_list_str = f"{char_names[0]} and {char_names[1]}"
            else:
                char_list_str = ", ".join(char_names[:-1]) + f" and {char_names[-1]}"

            utils.log_message('game', f"{loc_name}:")
            utils.log_message('game', f"You see {char_list_str} here.")


def get_narrative_location_for_position(x: int, y: int, generation_state) -> str | None:
    """
    Finds the narrative tag of the feature containing the given (x, y) coordinates.
    """
    if not generation_state or not generation_state.placed_features:
        return None
    for tag, details in generation_state.placed_features.items():
        if bounds := details.get('bounding_box'):
            x1, y1, w, h = [int(c) for c in bounds]
            x2, y2 = x1 + w, y1 + h
            if x1 <= x < x2 and y1 <= y < y2:
                return tag
    return None


def get_local_context_for_character(engine, game_state: GameState, character_name: str) -> str:
    """
    Generates a string describing the character's current location, adjacent locations,
    and any other characters present in those locations, with enhanced descriptions.
    """
    entity = game_state.get_entity(character_name)
    if not entity or not engine.generation_state or not engine.layout_graph:
        return ""

    # --- 1. Determine Character's Location and Nearby Features ---
    current_feature_tag = get_narrative_location_for_position(entity.x, entity.y, engine.generation_state)
    nearby_tags = set()
    context_lines = []

    if current_feature_tag:
        # Character is inside a feature. Find adjacent features.
        current_feature_data = engine.generation_state.placed_features.get(current_feature_tag, {})
        feature_name = current_feature_data.get('name', current_feature_tag)
        context_lines.append(f"You are in **{feature_name}**.")

        for parent, child, _ in engine.layout_graph.edges:
            if parent == current_feature_tag:
                nearby_tags.add(child)
            elif child == current_feature_tag:
                nearby_tags.add(parent)
    else:
        # Character is in an open area. Find the closest features.
        context_lines.append("You are standing in an open area.")
        all_features = engine.generation_state.placed_features
        if all_features:
            features_with_dist = []
            for tag, details in all_features.items():
                if bounds := details.get('bounding_box'):
                    x1, y1, w, h = [int(c) for c in bounds]
                    center_x, center_y = x1 + w / 2, y1 + h / 2
                    distance = math.dist((entity.x, entity.y), (center_x, center_y))
                    features_with_dist.append((distance, tag))

            # Get the 4 closest features
            features_with_dist.sort()
            for _, tag in features_with_dist[:4]:
                nearby_tags.add(tag)

    # --- 2. Build Combined Description of Nearby Features ---
    descriptive_sentences = []
    for tag in sorted(list(nearby_tags)):
        feature_data = engine.generation_state.placed_features.get(tag)
        if not feature_data: continue

        feature_name = feature_data.get('name', tag)
        feature_desc = feature_data.get('description', '')
        if feature_desc:
            descriptive_sentences.append(f"Nearby is the {feature_name}: {feature_desc}")

    if descriptive_sentences:
        full_description = " ".join(descriptive_sentences)
        context_lines.append(utils.format_text_with_paragraph_breaks(full_description, sentences_per_paragraph=3))

    # --- 3. Scan for Nearby Occupants ---
    occupants_found = []
    features_to_scan = {current_feature_tag} | nearby_tags
    other_entities = [e for e in game_state.entities if e.name != character_name]

    for other_entity in other_entities:
        occupant_loc_tag = get_narrative_location_for_position(other_entity.x, other_entity.y,
                                                              engine.generation_state)
        if occupant_loc_tag and occupant_loc_tag in features_to_scan:
            loc_data = engine.generation_state.placed_features.get(occupant_loc_tag, {})
            loc_name = loc_data.get('name', occupant_loc_tag)
            occupants_found.append(f"{other_entity.name} (in {loc_name})")

    if occupants_found:
        context_lines.append(f"\nYou can see: {', '.join(occupants_found)}.")

    return "\n".join(context_lines)


def _find_walkable_tile_in_area(game_state: GameState, x1, y1, x2, y2) -> tuple[int, int] | None:
    """Helper to find a random walkable tile within a bounding box."""
    potential_tiles = []
    for x in range(x1, x2):
        for y in range(y1, y2):
            if game_state.game_map.is_walkable(x, y):
                if not any(e.x == x and e.y == y for e in game_state.entities):
                    potential_tiles.append((x, y))
    if potential_tiles:
        return random.choice(potential_tiles)
    return None


def place_character_contextually(engine, game_state: GameState, character: dict, generation_state, placed_characters: list):
    """
    Uses an LLM to find a contextually appropriate location for a character
    from the generated features and places them there. It now also sets the
    initial character_state from the LLM's action description.
    """
    entity = game_state.get_entity(character['name'])
    if not entity or not generation_state or not generation_state.placed_features:
        return

    # --- REFACTORED LOGIC: Pre-filter for valid locations ---
    valid_location_tags = []
    for tag, details in generation_state.placed_features.items():
        if bounds := details.get('bounding_box'):
            x1, y1, w, h = [int(c) for c in bounds]
            if _find_walkable_tile_in_area(game_state, x1, y1, x1 + w, y1 + h):
                valid_location_tags.append(details.get('narrative_tag', tag))

    placer_agent = engine.config.agents.get('DIRECTOR')
    chosen_narrative_tag = None
    action_description = "observing the scene."  # Default state

    if valid_location_tags and placer_agent:
        placed_chars_str = "\n".join(
            [f"- {c['name']} is in '{get_narrative_location_for_position(game_state.get_entity(c['name']).x, game_state.get_entity(c['name']).y, generation_state)}' and is currently '{c.get('character_state', 'waiting')}'"
             for c in placed_characters]
        ) or "None."

        prompt_kwargs = {
            "scene_prompt": engine.scene_prompt,
            "character_name": character['name'],
            "character_description": character.get('description', 'An adventurer.'),
            "location_tags": ", ".join(f"'{tag}'" for tag in valid_location_tags),
            "placed_characters_list": placed_chars_str
        }

        raw_response = llm_api.execute_task(
            engine, placer_agent, 'GET_CHARACTER_PLACEMENT', [], task_prompt_kwargs=prompt_kwargs
        )
        command = command_parser.parse_structured_command(engine, raw_response, 'DIRECTOR')

        # --- Set Initial Character State from LLM Response ---
        # This happens regardless of whether the placement is successful.
        if command and command.get("action_description"):
            action_description = command.get("action_description").strip()

        character['character_state'] = action_description
        utils.log_message('full', f"[INITIAL STATE] {character['name']}: '{action_description}'")

        chosen_narrative_tag = command.get("location_tag")

    # --- Placement Logic ---
    placed_successfully = False
    if chosen_narrative_tag:
        actual_tag = next((tag for tag, details in generation_state.placed_features.items() if
                           details.get('narrative_tag', tag) == chosen_narrative_tag), None)

        if actual_tag:
            bounds = generation_state.placed_features[actual_tag].get('bounding_box')
            if bounds:
                x1, y1, w, h = [int(c) for c in bounds]
                if pos := _find_walkable_tile_in_area(game_state, x1, y1, x1 + w, y1 + h):
                    entity.x, entity.y = pos
                    utils.log_message('game',
                                      f"[POSITION] Placed '{entity.name}' in '{chosen_narrative_tag}' at {pos}.")
                    placed_successfully = True

    # --- Robust Fallback ---
    if not placed_successfully:
        utils.log_message('debug',
                          f"[PEG Placement WARNING] Could not place '{character['name']}' contextually. Placing randomly on map.")
        for _ in range(500):
            rand_x = random.randint(1, game_state.game_map.width - 2)
            rand_y = random.randint(1, game_state.game_map.height - 2)
            if game_state.game_map.is_walkable(rand_x, rand_y) and not any(
                    e.x == rand_x and e.y == rand_y for e in game_state.entities):
                entity.x, entity.y = rand_x, rand_y
                break


def process_movement_intent(engine, game_state: GameState, character_name: str, movement_action_text: str,
                            movement_mode: str):
    """
    Takes a character's narrated action, asks an LLM for a structured command,
    then executes movement based on whether the target is a feature or an entity.
    """
    pos_manager_agent = config.agents['POSITION_MANAGER']
    moving_character = roster_manager.find_character(engine, character_name)
    mover_entity = game_state.get_entity(character_name)

    if not moving_character or not mover_entity or not engine.generation_state:
        return

    # --- 1. Gather all possible targets (Entities and adjacent Features) ---
    other_entities = [e.name for e in game_state.entities if e.name.lower() != character_name.lower()]
    current_feature_tag = get_narrative_location_for_position(mover_entity.x, mover_entity.y,
                                                              engine.generation_state)
    adjacent_feature_names = []
    if current_feature_tag and engine.layout_graph:
        adjacent_tags = set()
        for p, c, _ in engine.layout_graph.edges:
            if p == current_feature_tag: adjacent_tags.add(c)
            if c == current_feature_tag: adjacent_tags.add(p)
        for tag in adjacent_tags:
            if name := engine.generation_state.placed_features.get(tag, {}).get('name'):
                adjacent_feature_names.append(name)

    all_targets = other_entities + adjacent_feature_names
    if not all_targets:
        return

    # --- 2. Call LLM with unified prompt and fallbacks ---
    target_list_str = "\n".join(f"- '{t}'" for t in all_targets)
    prompt_kwargs = {"target_list": target_list_str, "user_message_content": movement_action_text}

    raw_response = llm_api.execute_task(engine, pos_manager_agent, 'POSITION_PARSE_MOVEMENT', [],
                                        task_prompt_kwargs=prompt_kwargs)
    command = command_parser.parse_structured_command(engine, raw_response, 'POSITION_MANAGER',
                                                      fallback_task_key='POSITION_FIX_MOVEMENT_JSON',
                                                      fallback_prompt_kwargs={'target_list': target_list_str})
    if not command or not command.get("action"):
        keyword_response = llm_api.execute_task(engine, moving_character, 'POSITION_GET_MOVEMENT_KEYWORD', [],
                                                task_prompt_kwargs={"target_list": target_list_str,
                                                                    "raw_text": movement_action_text})
        match = re.search(r"MOVE:\s*(.+)", keyword_response.strip(), re.IGNORECASE)
        if match:
            target_name = match.group(1).strip()
            resolved_target = roster_manager.resolve_character_from_description(engine, target_name, all_targets)
            if resolved_target:
                command = {"action": "MOVE", "mover": character_name, "target": resolved_target, "relation": "NEXT_TO"}

    if not command or command.get("action", "").upper() != "MOVE":
        utils.log_message('debug', f"[POSITION] No valid MOVE command could be determined.")
        return

    # --- 3. Process the command based on target type ---
    target_name = command.get("target")
    if not target_name:
        return

    # Case A: Target is another entity
    if target_name in other_entities:
        game_functions.execute_movement_command(game_state, command, movement_mode)
        return

    # Case B: Target is a map feature
    if target_name in adjacent_feature_names:
        target_feature_tag = next((tag for tag, data in engine.generation_state.placed_features.items() if
                                   data.get('name') == target_name), None)
        if not target_feature_tag: return

        target_feature_data = engine.generation_state.placed_features.get(target_feature_tag)
        if not target_feature_data or 'bounding_box' not in target_feature_data: return

        bounds = target_feature_data['bounding_box']
        x1, y1, w, h = [int(c) for c in bounds]
        destination_pos = _find_walkable_tile_in_area(game_state, x1, y1, x1 + w, y1 + h)

        if not destination_pos:
            utils.log_message('debug',
                              f"[POSITION] No walkable destination found in target feature '{target_feature_tag}'.")
            return

        destination_x, destination_y = destination_pos
        cost_map = np.array(game_state.game_map.tiles["movement_cost"], dtype=np.int32)
        graph = tcod.path.SimpleGraph(cost=cost_map, cardinal=1, diagonal=0)
        pathfinder = tcod.path.Pathfinder(graph)
        pathfinder.add_root((mover_entity.x, mover_entity.y))
        path = pathfinder.path_to((destination_x, destination_y)).tolist()

        if not path:
            utils.log_message('debug', f"[POSITION] No physical path found for '{mover_entity.name}'.")
            return

        current_speed_feet = mover_entity.movement_remaining
        feet_per_tile = 5
        for i, (next_x, next_y) in enumerate(path):
            move_cost_feet = game_state.game_map.tiles["movement_cost"][next_x, next_y] * feet_per_tile
            if current_speed_feet >= move_cost_feet:
                current_speed_feet -= move_cost_feet
            else:
                path = path[:i]
                break

        if path:
            final_pos_x, final_pos_y = path[-1]
            mover_entity.x, mover_entity.y = final_pos_x, final_pos_y
            mover_entity.movement_remaining = current_speed_feet
            utils.log_message('game',
                              f"[POSITION] Moved '{mover_entity.name}' to ({mover_entity.x}, {mover_entity.y}) in '{target_name}'. {mover_entity.movement_remaining}ft remaining.")