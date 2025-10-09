import numpy as np
import random
from .config_loader import config
from typing import Optional, Dict, List, Set, Tuple


class LayoutGraph:
    """Represents the abstract, logical layout of a map as a graph."""

    def __init__(self):
        self.nodes = {}  # key: feature_tag, value: feature_data
        self.edges = []  # list of tuples (parent_tag, child_tag, attachment_hint)

    def add_feature(self, feature_tag: str, feature_data: dict, parent_tag: str = None, attachment_hint: str = 'any'):
        """Adds a feature node and its connecting edge to the graph."""
        self.nodes[feature_tag] = feature_data
        if parent_tag and (parent_tag, feature_tag) not in [(e[0], e[1]) for e in self.edges]:
            self.edges.append((parent_tag, feature_tag, attachment_hint))


class GenerationState:
    """
    Manages the state of an in-progress procedural generation. It holds the final
    abstract graph, the narrative log, and the dictionary of physically placed features.
    """

    def __init__(self, game_map):
        self.layout_graph = LayoutGraph()
        self.narrative_log = ""
        self.game_map = game_map
        self.placed_features = {}
        self.character_creation_queue = []
        self.interior_feature_queue = []
        self.door_locations = []
        self.blended_hallways = []
        self.exterior_tile_type = "DEFAULT_FLOOR"  # Default fallback
        self.feature_embeddings = {}
        self.clearance_mask = None


graphic_dt = np.dtype(
    [
        ("ch", np.int32),
        ("fg", "3B"),
        ("bg", "3B"),
    ]
)

tile_dt = np.dtype(
    [
        ("walkable", np.bool_),
        ("transparent", np.bool_),
        ("graphic", graphic_dt),
        ("movement_cost", np.float32),
        ("terrain_type", np.int8)
    ]
)


class Entity:
    """A generic object to represent players, NPCs, items, etc."""

    def __init__(self, name: str, x: int, y: int, char: str, color: tuple[int, int, int]):
        self.name = name
        self.x = x
        self.y = y
        self.char = char
        self.color = color
        self.speed = 30
        self.movement_remaining = 30
        self.conditions = set()
        self.movement_types = {"walk"}


class MapArtist:
    """
    Translates a GenerationState into a tile-based map and handles rendering of visual effects.
    """

    def __init__(self):
        self._color_cache = {}
        self._material_tile_cache = {}

    def _find_material_override_tile(self, target_material: str, move_capability: str) -> Optional[str]:
        """Finds a suitable tile type that matches a material and movement capability."""
        cache_key = (target_material, move_capability)
        if cache_key in self._material_tile_cache:
            return self._material_tile_cache[cache_key]

        for tile_name, tile_def in config.tile_types.items():
            if target_material in tile_def.get('materials', []) and move_capability in tile_def.get('pass_methods', []):
                self._material_tile_cache[cache_key] = tile_name
                return tile_name

        self._material_tile_cache[cache_key] = None
        return None

    def _get_nature_modified_tile_type(self, default_tile_name: str, nature_names: List[str],
                                       used_natures: Set[str]) -> Tuple[Optional[str], Optional[str]]:
        """
        Determines the final tile type for a feature part (floor/border) based on its natures,
        applying conflict resolution policies.
        """
        if not nature_names or not default_tile_name:
            return default_tile_name, None

        candidates = []
        default_tile_def = config.tile_types.get(default_tile_name, {})
        default_materials = default_tile_def.get('materials', [])
        default_pass_methods = default_tile_def.get('pass_methods', [])

        for nature_name in nature_names:
            nature_def = config.natures.get(nature_name, {})
            modifiers = nature_def.get('tile_modifiers', {})

            # 1. Check for direct overrides
            if direct_override := modifiers.get('direct_overrides', {}).get(default_tile_name):
                candidates.append({'nature': nature_name, 'tile': direct_override})
                continue

            # 2. Check for material overrides
            for rule in modifiers.get('material_overrides', []):
                source_material = rule.get('source_material')
                if source_material in default_materials:
                    move_match = rule.get('movement_capability_match')
                    if move_match in default_pass_methods:
                        target_tile = self._find_material_override_tile(rule.get('target_material'), move_match)
                        if target_tile:
                            candidates.append({'nature': nature_name, 'tile': target_tile})

        if not candidates:
            return default_tile_name, None

        if len(candidates) == 1:
            return candidates[0]['tile'], candidates[0]['nature']

        # Conflict Resolution
        policy = config.natures.get(candidates[0]['nature'], {}).get('conflict_resolution_policy', 'GREEDY')

        if policy == 'RANDOM':
            chosen = random.choice(candidates)
            return chosen['tile'], chosen['nature']

        if policy == 'BALANCED':
            # Prioritize a nature that hasn't been used yet for this feature
            for candidate in candidates:
                if candidate['nature'] not in used_natures:
                    return candidate['tile'], candidate['nature']

        # Default to GREEDY (first one found)
        return candidates[0]['tile'], candidates[0]['nature']

    def _get_tile_data_from_type(self, tile_type_key: str) -> tuple | None:
        tile_def = config.tile_types.get(tile_type_key)
        if not tile_def:
            # Fallback to a visually distinct error tile to avoid crashing
            tile_def = config.tile_types.get("DEFAULT_WALL")

        color = tile_def.get("colors", [[255, 0, 255]])[0]
        char = tile_def.get("characters", ["?"])[0]
        pass_methods = tile_def.get("pass_methods", [])
        walkable = "GROUND" in pass_methods
        transparent = tile_def.get("is_transparent", True)
        movement_cost = tile_def.get("movement_cost", 1.0)
        terrain_type_index = config.tile_type_map.get(tile_type_key, config.tile_type_map.get("DEFAULT_WALL", -1))

        return (
            walkable, transparent,
            (ord(char), tuple(color), (0, 0, 0)),
            movement_cost,
            terrain_type_index
        )

    def _draw_single_feature(self, game_map, feature_data, features_definitions, generation_state):
        feature_type_key = feature_data.get('type')
        if not feature_type_key or feature_type_key not in features_definitions: return
        feature_def = features_definitions[feature_type_key]

        if 'modified_tile_type' in feature_data:
            final_floor_name = feature_data['modified_tile_type']
            final_border_name = feature_data.get('modified_border_tile_type', feature_def.get('border_tile_type'))
        else:
            nature_names = feature_data.get('natures', [])
            used_natures_for_feature = set()
            default_floor_name = feature_def.get('tile_type')
            default_border_name = feature_def.get('border_tile_type')
            final_floor_name, used_nature_floor = self._get_nature_modified_tile_type(default_floor_name, nature_names,
                                                                                      used_natures_for_feature)
            if used_nature_floor: used_natures_for_feature.add(used_nature_floor)
            final_border_name, _ = self._get_nature_modified_tile_type(default_border_name, nature_names,
                                                                       used_natures_for_feature)

        # Handle SPEC_REPLACEMENT fallback
        if final_floor_name and config.tile_types.get(final_floor_name, {}).get("is_special_replacement", False):
            final_floor_name = generation_state.exterior_tile_type
        if final_border_name and config.tile_types.get(final_border_name, {}).get("is_special_replacement", False):
            final_border_name = generation_state.exterior_tile_type

        footprint = set(map(tuple, feature_data.get('footprint', [])))
        interior_footprint = set(map(tuple, feature_data.get('interior_footprint', [])))
        border_footprint = footprint - interior_footprint

        if not footprint: return

        if border_footprint and final_border_name:
            if border_tile_data := self._get_tile_data_from_type(final_border_name):
                border_coords = np.array(list(border_footprint))
                if border_coords.size > 0:
                    valid_mask = (border_coords[:, 0] >= 0) & (border_coords[:, 0] < game_map.width) & (
                                border_coords[:, 1] >= 0) & (border_coords[:, 1] < game_map.height)
                    valid_coords = border_coords[valid_mask]
                    if valid_coords.size > 0:
                        game_map.tiles[valid_coords[:, 0], valid_coords[:, 1]] = border_tile_data

        if interior_footprint and final_floor_name:
            if floor_tile_data := self._get_tile_data_from_type(final_floor_name):
                floor_coords = np.array(list(interior_footprint))
                if floor_coords.size > 0:
                    valid_mask = (floor_coords[:, 0] >= 0) & (floor_coords[:, 0] < game_map.width) & (
                                floor_coords[:, 1] >= 0) & (floor_coords[:, 1] < game_map.height)
                    valid_coords = floor_coords[valid_mask]
                    if valid_coords.size > 0:
                        game_map.tiles[valid_coords[:, 0], valid_coords[:, 1]] = floor_tile_data

        if tile_overrides_str := feature_data.get('tile_overrides'):
            tile_overrides = {tuple(map(int, k.split(','))): v for k, v in tile_overrides_str.items()}
            for (x, y), tile_name in tile_overrides.items():
                if override_tile_data := self._get_tile_data_from_type(tile_name):
                    if game_map.is_in_bounds(x, y):
                        game_map.tiles[x, y] = override_tile_data

    def draw_map(self, game_map: 'GameMap', generation_state: 'GenerationState', features_definitions: dict):
        if not generation_state or not features_definitions: return

        void_space_data = self._get_tile_data_from_type("VOID_SPACE")
        if not void_space_data:
            print("FATAL ERROR: 'VOID_SPACE' not found")
            exit()
        game_map.tiles[...] = void_space_data

        # Draw all features
        for tag, feature_data in generation_state.placed_features.items():
            self._draw_single_feature(game_map, feature_data, features_definitions, generation_state)

        # Draw hallways and doors
        if generation_state.blended_hallways:
            for hallway in generation_state.blended_hallways:
                type_a = hallway['type_a']
                type_b = hallway['type_b']
                tile_a_def = features_definitions.get(type_a, {})
                tile_b_def = features_definitions.get(type_b, {})

                floor_tile = self._get_tile_data_from_type(tile_a_def.get('tile_type', 'DEFAULT_FLOOR'))
                wall_tile = self._get_tile_data_from_type(tile_b_def.get('border_tile_type', 'DEFAULT_WALL'))

                # This check prevents a crash if a feature in a connection has no valid tile type
                if not floor_tile or not wall_tile:
                    continue

                for x, y in hallway['tiles']:
                    if game_map.is_in_bounds(x, y):
                        game_map.tiles[x, y] = floor_tile
                        for dx, dy in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
                            nx, ny = x + dx, y + dy
                            if game_map.is_in_bounds(nx, ny) and (nx, ny) not in hallway['tiles']:
                                game_map.tiles[nx, ny] = wall_tile

        if generation_state.door_locations:
            for door_info in generation_state.door_locations:
                pos, tile_type = door_info.get('pos'), door_info.get('type')
                if not pos or not tile_type: continue
                if game_map.is_in_bounds(pos[0], pos[1]):
                    if door_tile_data := self._get_tile_data_from_type(tile_type):
                        game_map.tiles[pos[0], pos[1]] = door_tile_data

        # Fill any remaining void space with the exterior tile
        exterior_tile_name = getattr(generation_state, 'exterior_tile_type', 'DEFAULT_FLOOR')
        if exterior_tile_data := self._get_tile_data_from_type(exterior_tile_name):
            void_mask = game_map.tiles["terrain_type"] == config.tile_type_map.get("VOID_SPACE", -1)
            game_map.tiles[void_mask] = exterior_tile_data


class GameMap:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.tiles = np.zeros((width, height), dtype=tile_dt, order="F")
        self.tiles["transparent"] = True

    def is_in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def is_walkable(self, x: int, y: int) -> bool:
        if not self.is_in_bounds(x, y): return False
        return self.tiles["walkable"][x, y]


class GameState:
    """Central class for holding all game world data."""

    def __init__(self):
        from .config_loader import config
        map_width = config.settings.get("MAP_WIDTH", 120)
        map_height = config.settings.get("MAP_HEIGHT", 80)
        self.game_map = GameMap(width=map_width, height=map_height)
        self.entities: list[Entity] = []

    def add_entity(self, entity: Entity):
        self.entities.append(entity)

    def get_entity(self, name: str) -> Entity | None:
        if name is None:
            return None
        return next((e for e in self.entities if e.name is not None and e.name.lower() == name.lower()), None)

    def reset_entity_turn_stats(self, entity_name: str):
        entity = self.get_entity(entity_name)
        if entity:
            entity.movement_remaining = entity.speed