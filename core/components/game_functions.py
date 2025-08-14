# /core/components/game_functions.py

import tcod
import numpy as np
from ..common.game_state import GameState, GameMap
from ..common import utils
from ..common import file_io
from ..common.config_loader import config

VALID_RELATIONS = {
    "NEXT_TO", "STRIKING_RANGE", "LONG_RANGE", "BEHIND",
    "ABOVE", "BELOW", "INSIDE"
}


def find_path(game_map: 'GameMap', start_pos: tuple[int, int], end_pos: tuple[int, int],
              allowed_pass_methods: set = None) -> list[tuple[int, int]] | None:
    """
    A generic A* pathfinding function that uses a data-driven cost map.

    Args:
        game_map: The GameMap object to pathfind on.
        start_pos: The (x, y) starting coordinates.
        end_pos: The (x, y) ending coordinates.
        allowed_pass_methods: A set of strings (e.g., {'GROUND', 'FLYING'}) defining
                                  which pass_methods are considered walkable for this path.

    Returns:
        A list of (x, y) tuples representing the path, or None if no path is found.
    """
    if allowed_pass_methods is None:
        allowed_pass_methods = {'GROUND'}

    # Create a cost map where 0 is impassable and 1 is passable.
    cost_map = np.zeros((game_map.width, game_map.height), dtype=np.int8)

    # Iterate through all defined tile types
    for type_name, type_def in config.tile_types.items():
        # Check if this tile type is traversable by the allowed methods
        if allowed_pass_methods.intersection(type_def.get("pass_methods", [])):
            type_index = config.tile_type_map.get(type_name)
            if type_index is not None:
                # Find all tiles of this type and mark them as walkable (cost 1)
                tile_indices = np.where(game_map.tiles["terrain_type"] == type_index)
                cost_map[tile_indices] = 1

    # Transpose cost map for tcod's (y, x) indexing
    graph = tcod.path.SimpleGraph(cost=cost_map.T, cardinal=1, diagonal=0)
    pathfinder = tcod.path.Pathfinder(graph)
    pathfinder.add_root((start_pos[1], start_pos[0]))  # tcod uses (y, x)

    path_yx = pathfinder.path_to((end_pos[1], end_pos[0])).tolist()

    if not path_yx:
        return None

    # Convert path back to (x, y) coordinates
    return [tuple(reversed(p)) for p in path_yx]


def execute_movement_command(game_state: GameState, command: dict, movement_mode: str):
    """
    Takes a parsed command and a movement mode, and updates character coordinates
    by performing pathfinding and applying game rules. This is CPU-driven.
    """
    mover_name = command.get("mover")
    target_name = command.get("target")
    relation = command.get("relation")

    mover_entity = game_state.get_entity(mover_name)
    target_entity = game_state.get_entity(target_name)

    if not mover_entity:
        utils.log_message('debug', f"[POSITION] Mover '{mover_name}' not found on board.")
        return
    if not target_entity:
        utils.log_message('debug', f"[POSITION] Target '{target_name}' not found on board.")
        return

    current_speed_feet = mover_entity.movement_remaining
    if movement_mode == "DASH":
        current_speed_feet += mover_entity.speed
    if movement_mode == "CRAWL" or "prone" in mover_entity.conditions:
        current_speed_feet = min(current_speed_feet, mover_entity.speed / 2)
    if movement_mode == "SNEAK":
        mover_entity.conditions.add("sneaking")

    utils.log_message('debug',
                      f"[POSITION] '{mover_name}' is attempting a '{movement_mode}' move towards '{target_name}' with {current_speed_feet}ft of movement.")

    # --- Data-driven Cost Map Generation ---
    cost_map = np.zeros_like(game_state.game_map.tiles["movement_cost"], dtype=np.int8)
    for type_index, type_name in config.tile_type_map_reverse.items():
        tile_def = config.tile_types.get(type_name, {})
        pass_methods = set(tile_def.get("pass_methods", []))

        # Find all tiles of this type
        tile_indices = np.where(game_state.game_map.tiles["terrain_type"] == type_index)

        # Check if the mover can traverse this tile type
        if mover_entity.movement_types.intersection(pass_methods):
            # Assign the movement cost from the tile definition
            cost_map[tile_indices] = int(tile_def.get("movement_cost", 1))
        else:
            cost_map[tile_indices] = 0  # 0 cost means impassable for tcod.path

    # Add dynamic obstacles (other entities)
    for entity in game_state.entities:
        if entity is not mover_entity and entity is not target_entity:
            if "ethereal" not in mover_entity.conditions:
                cost_map[entity.x, entity.y] = 0 # Impassable

    # --- Pathfinding ---
    graph = tcod.path.SimpleGraph(cost=cost_map.T, cardinal=1, diagonal=0) # Transposed for tcod
    pathfinder = tcod.path.Pathfinder(graph)
    pathfinder.add_root((mover_entity.y, mover_entity.x))

    destination_x, destination_y = target_entity.x, target_entity.y

    if relation not in VALID_RELATIONS:
        utils.log_message('debug',
                          f"[POSITION WARNING] Received unsupported relation '{relation}'. Defaulting to 'NEXT_TO'.")
        context_msg = f"Mover: '{mover_name}', Target: '{target_name}', Full Command: {str(command)}"
        file_io.log_to_ideation_file("POSITION RELATION", relation, context=context_msg)
        relation = "NEXT_TO"

    if relation == "INSIDE":
        pass
    else:
        found_spot = False
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            adj_x, adj_y = target_entity.x + dx, target_entity.y + dy
            if game_state.game_map.is_walkable(adj_x, adj_y):
                destination_x, destination_y = adj_x, adj_y
                found_spot = True
                break
        if not found_spot:
            utils.log_message('debug', f"[POSITION] No walkable adjacent tiles found for '{target_name}'.")
            return

    path_yx = pathfinder.path_to((destination_y, destination_x)).tolist()
    path = [tuple(reversed(p)) for p in path_yx]

    if not path:
        utils.log_message('debug', f"[POSITION] No path found for '{mover_name}' to '{target_name}'.")
        return

    feet_per_tile = 5
    for i, (next_x, next_y) in enumerate(path):
        tile_cost_multiplier = game_state.game_map.tiles["movement_cost"][next_x, next_y]
        move_cost_feet = tile_cost_multiplier * feet_per_tile

        terrain_idx = game_state.game_map.tiles["terrain_type"][next_x, next_y]
        tile_type_name = config.tile_type_map_reverse.get(terrain_idx, "")
        tile_def = config.tile_types.get(tile_type_name, {})
        pass_methods = set(tile_def.get("pass_methods", []))

        # Apply difficult terrain penalty if the mover's primary method isn't the best one for the tile
        if 'GROUND' in mover_entity.movement_types and 'SWIM' in pass_methods and 'GROUND' not in pass_methods:
             move_cost_feet *= 2 # Example: Walking through water

        if current_speed_feet >= move_cost_feet:
            current_speed_feet -= move_cost_feet
        else:
            path = path[:i]
            utils.log_message('debug', f"[POSITION] '{mover_name}' ran out of movement.")
            break

    if path:
        final_pos_x, final_pos_y = path[-1]
        mover_entity.x, mover_entity.y = final_pos_x, final_pos_y
        mover_entity.movement_remaining = current_speed_feet
        utils.log_message('debug',
                          f"[POSITION] Moved '{mover_name}' to ({mover_entity.x}, {mover_entity.y}). {mover_entity.movement_remaining}ft remaining.")
    else:
        utils.log_message('debug', f"[POSITION] '{mover_name}' could not move.")