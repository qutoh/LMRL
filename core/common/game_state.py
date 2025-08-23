# /core/common/game_state.py

import numpy as np

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
        self.exterior_tile_type = "DEFAULT_FLOOR" # Default fallback


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

    def draw_feature_trails(self, renderer, generation_state: 'GenerationState'):
        for feature_data in generation_state.placed_features.values():
            name = feature_data.get('name', 'unknown')
            rect = feature_data.get('bounding_box')
            if not rect: continue

            color = self._get_color_for_feature(name)
            x, y, w, h = rect

            renderer.draw_rect(x=x, y=y, width=w, height=h, bg=(*color, 255))

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

    def _draw_path_feature(self, game_map: 'GameMap', path_coords: set, feature_def: dict):
        """Draws a path feature with a border and floor using a two-pass method."""
        floor_tile_type = feature_def.get('tile_type', 'PATH_FLOOR')
        border_tile_type = feature_def.get('border_tile_type', 'DEFAULT_WALL')

        floor_tile_data = self._get_tile_data_from_type(floor_tile_type)
        border_tile_data = self._get_tile_data_from_type(border_tile_type)

        if not floor_tile_data or not border_tile_data:
            return

        # Pass 1: Draw the entire path with its floor tile.
        for x, y in path_coords:
            if game_map.is_in_bounds(x, y):
                game_map.tiles[x, y] = floor_tile_data

        # Pass 2: Draw the border on any adjacent tile that is NOT part of the path.
        for x, y in path_coords:
            for nx in range(x - 1, x + 2):
                for ny in range(y - 1, y + 2):
                    if (nx, ny) == (x, y):
                        continue
                    if game_map.is_in_bounds(nx, ny) and (nx, ny) not in path_coords:
                        # This neighbor is outside the path, so it's a border wall.
                        game_map.tiles[nx, ny] = border_tile_data

    def draw_map(self, game_map: 'GameMap', generation_state: 'GenerationState', features_definitions: dict):
        if not generation_state or not features_definitions:
            return

        # Step 1: Initialize the entire map with primordial, unwalkable space.
        void_space_data = self._get_tile_data_from_type("VOID_SPACE")
        if not void_space_data:
            print("FATAL ERROR: 'VOID_SPACE' not found in tile_types.json")
            exit()
        game_map.tiles[...] = void_space_data

        # Step 2: Draw all placed features onto the map.
        for feature_tag, feature_data in generation_state.placed_features.items():
            feature_type_key = feature_data.get('type')
            if not feature_type_key or feature_type_key not in features_definitions:
                continue
            feature_def = features_definitions[feature_type_key]

            # Check for path coordinates from generator ('path_coords') or loaded file ('path_tiles')
            path_coords = feature_data.get("path_tiles") or feature_data.get("path_coords")
            if path_coords:
                # When loaded from JSON, path_tiles will be a list of lists.
                # It must be converted to a set of tuples to be hashable and for fast lookups.
                path_coords_set = set(map(tuple, path_coords))
                self._draw_path_feature(game_map, path_coords_set, feature_def)
                continue

            bounds = feature_data.get('bounding_box')
            if not bounds: continue

            x1, y1, w, h = [int(c) for c in bounds]
            x2, y2 = x1 + w, y1 + h

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(game_map.width, x2), min(game_map.height, y2)
            if x1 >= x2 or y1 >= y2: continue

            floor_tile_type = feature_def.get('tile_type', 'VOID_SPACE')
            floor_tile_data = self._get_tile_data_from_type(floor_tile_type)
            if floor_tile_data:
                game_map.tiles[x1:x2, y1:y2] = floor_tile_data

            border_thickness = feature_def.get('border_thickness', 1)
            if border_thickness > 0:
                border_tile_type = feature_def.get('border_tile_type', 'VOID_SPACE')
                border_tile_data = self._get_tile_data_from_type(border_tile_type)
                if not border_tile_data: continue

                effective_border = min(border_thickness, (x2 - x1) // 2, (y2 - y1) // 2)
                if effective_border > 0:
                    game_map.tiles[x1:x2, y1:y1 + effective_border] = border_tile_data
                    game_map.tiles[x1:x2, y2 - effective_border:y2] = border_tile_data
                    game_map.tiles[x1:x1 + effective_border, y1:y2] = border_tile_data
                    game_map.tiles[x2 - effective_border:x2, y1:y2] = border_tile_data

        # Step 3: Draw doors.
        door_tile_data = self._get_tile_data_from_type("DEFAULT_DOOR")
        if door_tile_data and generation_state.door_locations:
            valid_coords = [(x, y) for x, y in generation_state.door_locations if game_map.is_in_bounds(x, y)]
            if valid_coords:
                door_xs, door_ys = zip(*valid_coords)
                game_map.tiles[door_xs, door_ys] = door_tile_data

        # Step 4: Post-processing to replace any remaining void space with the chosen exterior tile.
        exterior_tile_name = getattr(generation_state, 'exterior_tile_type', 'DEFAULT_FLOOR')
        exterior_tile_data = self._get_tile_data_from_type(exterior_tile_name)
        if exterior_tile_data:
            void_space_index = config.tile_type_map.get("VOID_SPACE", -1)
            void_mask = game_map.tiles["terrain_type"] == void_space_index
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
        return next((e for e in self.entities if e.name.lower() == name.lower()), None)

    def reset_entity_turn_stats(self, entity_name: str):
        entity = self.get_entity(entity_name)
        if entity:
            entity.movement_remaining = entity.speed