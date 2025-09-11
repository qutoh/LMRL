# /core/common/game_state.py

import numpy as np
import random
from .config_loader import config


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

    def _get_color_for_feature(self, feature_name: str) -> tuple[int, int, int]:
        if feature_name in self._color_cache:
            return self._color_cache[feature_name]

        seed = hash(feature_name)
        r = (seed & 0xFF0000) >> 16
        g = (seed & 0x00FF00) >> 8
        b = seed & 0x0000FF
        r, g, b = max(128, r), max(128, g), max(128, b)
        color = (r, g, b)
        self._color_cache[feature_name] = color
        return color

    def _get_tile_data_from_type(self, tile_type_key: str) -> tuple | None:
        tile_def = config.tile_types.get(tile_type_key)
        if not tile_def:
            return None

        color = tile_def.get("colors", [[255, 0, 255]])[0]
        char = tile_def.get("characters", ["?"])[0]
        pass_methods = tile_def.get("pass_methods", [])
        walkable = "GROUND" in pass_methods
        transparent = tile_def.get("is_transparent", True)
        movement_cost = tile_def.get("movement_cost", 1.0)
        terrain_type_index = config.tile_type_map.get(tile_type_key, -1)

        return (
            walkable, transparent,
            (ord(char), tuple(color), (0, 0, 0)),
            movement_cost,
            terrain_type_index
        )

    def draw_map(self, game_map: 'GameMap', generation_state: 'GenerationState', features_definitions: dict):
        if not generation_state or not features_definitions: return
        void_space_data = self._get_tile_data_from_type("VOID_SPACE")
        if not void_space_data:
            print("FATAL ERROR: 'VOID_SPACE' not found")
            exit()
        game_map.tiles[...] = void_space_data

        for feature_tag, feature_data in generation_state.placed_features.items():
            feature_type_key = feature_data.get('type')
            if not feature_type_key or feature_type_key not in features_definitions: continue
            feature_def = features_definitions[feature_type_key]

            footprint = set(map(tuple, feature_data.get('footprint', [])))
            interior_footprint = set(map(tuple, feature_data.get('interior_footprint', [])))
            border_footprint = footprint - interior_footprint

            if not footprint: continue

            if border_footprint:
                if border_tile_data := self._get_tile_data_from_type(feature_def.get('border_tile_type')):
                    border_coords = np.array(list(border_footprint))
                    if border_coords.size > 0:
                        valid_mask = (border_coords[:, 0] >= 0) & (border_coords[:, 0] < game_map.width) & \
                                     (border_coords[:, 1] >= 0) & (border_coords[:, 1] < game_map.height)
                        valid_coords = border_coords[valid_mask]
                        if valid_coords.size > 0:
                            game_map.tiles[valid_coords[:, 0], valid_coords[:, 1]] = border_tile_data

            if interior_footprint:
                if floor_tile_data := self._get_tile_data_from_type(feature_def.get('tile_type', 'VOID_SPACE')):
                    floor_coords = np.array(list(interior_footprint))
                    if floor_coords.size > 0:
                        valid_mask = (floor_coords[:, 0] >= 0) & (floor_coords[:, 0] < game_map.width) & \
                                     (floor_coords[:, 1] >= 0) & (floor_coords[:, 1] < game_map.height)
                        valid_coords = floor_coords[valid_mask]
                        if valid_coords.size > 0:
                            game_map.tiles[valid_coords[:, 0], valid_coords[:, 1]] = floor_tile_data

            if tile_overrides_str := feature_data.get('tile_overrides'):
                tile_overrides = {tuple(map(int, k.split(','))): v for k, v in tile_overrides_str.items()}
                for (x, y), tile_name in tile_overrides.items():
                    if override_tile_data := self._get_tile_data_from_type(tile_name):
                        if game_map.is_in_bounds(x, y):
                            game_map.tiles[x, y] = override_tile_data

        if generation_state.blended_hallways:
            for hallway in generation_state.blended_hallways:
                type_a = hallway['type_a']
                type_b = hallway['type_b']
                tile_a_def = features_definitions.get(type_a, {})
                tile_b_def = features_definitions.get(type_b, {})

                floor_tile = self._get_tile_data_from_type(tile_a_def.get('tile_type', 'DEFAULT_FLOOR'))
                wall_tile = self._get_tile_data_from_type(tile_b_def.get('border_tile_type', 'DEFAULT_WALL'))

                for x, y in hallway['tiles']:
                    if game_map.is_in_bounds(x, y):
                        # Simple blend: floor tile for hallway, surrounded by border
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