# /core/components/position_manager.py

import random
import re
import tcod
import numpy as np
from ..llm import llm_api
from ..common.config_loader import config
from ..common.game_state import GameState
from ..common import utils
from ..common import command_parser
from . import roster_manager
from . import game_functions

FIRST_PERSON_PRONOUNS = {"i", "me", "myself", "my"}


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
    and any other characters present in those locations.
    """
    entity = game_state.get_entity(character_name)
    if not entity or not engine.generation_state or not engine.layout_graph:
        return ""

    current_feature_tag = get_narrative_location_for_position(entity.x, entity.y, engine.generation_state)
    if not current_feature_tag:
        return ""

    # Find adjacent features from the layout graph
    adjacent_tags = set()
    for parent, child, _ in engine.layout_graph.edges:
        if parent == current_feature_tag:
            adjacent_tags.add(child)
        elif child == current_feature_tag:
            adjacent_tags.add(parent)

    features_to_scan = {current_feature_tag} | adjacent_tags
    context_lines = []

    for tag in features_to_scan:
        feature_data = engine.generation_state.placed_features.get(tag)
        if not feature_data: continue

        feature_name = feature_data.get('name', tag)
        feature_desc = feature_data.get('description', 'an area')
        prefix = "You are in" if tag == current_feature_tag else "Nearby is"
        line = f"- {prefix} **{feature_name}**: {feature_desc}"

        # Find entities within this feature's bounding box
        occupants = []
        if bounds := feature_data.get('bounding_box'):
            x1, y1, w, h = [int(c) for c in bounds]
            x2, y2 = x1 + w, y1 + h
            for other_entity in game_state.entities:
                if other_entity.name != character_name and x1 <= other_entity.x < x2 and y1 <= other_entity.y < y2:
                    occupants.append(other_entity.name)

        if occupants:
            line += f" (Occupants: {', '.join(occupants)})"
        context_lines.append(line)

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


def place_character_contextually(engine, game_state: GameState, character: dict, generation_state):
    """
    Uses an LLM to find a contextually appropriate location for a character
    from the generated features and places them there. Includes a robust fallback.
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

    if valid_location_tags and placer_agent:
        prompt_kwargs = {
            "character_name": character['name'],
            "character_description": character.get('description', 'An adventurer.'),
            "location_tags": ", ".join(f"'{tag}'" for tag in valid_location_tags)
        }

        raw_response = llm_api.execute_task(
            engine, placer_agent, 'GET_CHARACTER_PLACEMENT', [], task_prompt_kwargs=prompt_kwargs
        )
        command = command_parser.parse_structured_command(engine, raw_response, 'DIRECTOR')
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
                    utils.log_message('debug',
                                      f"[PEG Placement] Placed '{entity.name}' in '{chosen_narrative_tag}' at {pos}.")
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
            utils.log_message('debug',
                              f"[POSITION] Moved '{mover_entity.name}' to ({mover_entity.x}, {mover_entity.y}) in '{target_name}'. {mover_entity.movement_remaining}ft remaining.")