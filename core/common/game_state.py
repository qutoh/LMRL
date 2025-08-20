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
        ("terrain_type", np.int8)  # Maps to an index in config.tile_type_map
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
    Translates a completed GenerationState object into a tile-based map
    on a GameMap instance.
    """

    def _get_tile_data_from_type(self, tile_type_key: str) -> tuple | None:
        """Helper to convert a tile type from JSON into a numpy-compatible tuple."""
        tile_def = config.tile_types.get(tile_type_key)
        if not tile_def:
            return None

        color = tile_def.get("colors", [[255, 0, 255]])[0]
        char = tile_def.get("characters", ["?"])[0]
        pass_methods = tile_def.get("pass_methods", [])
        walkable = "GROUND" in pass_methods
        transparent = tile_def.get("is_transparent", False)
        movement_cost = tile_def.get("movement_cost", 1.0)
        terrain_type_index = config.tile_type_map.get(tile_type_key, -1)

        return (
            walkable, transparent,
            (ord(char), tuple(color), (0, 0, 0)),
            movement_cost,
            terrain_type_index
        )

    def draw_map(self, game_map: 'GameMap', generation_state: 'GenerationState', features_definitions: dict):
        """
        Renders the entire map from the generation state's placed_features using a two-pass system:
        1. Fill map with default background (e.g., walkable floor).
        2. Draw all features (floors and interior walls).
        3. Overlay pathfinding doors.
        """
        if not generation_state or not features_definitions:
            return

        # 1. Initialize the entire map with a default floor
        default_floor_data = self._get_tile_data_from_type("DEFAULT_FLOOR")
        if not default_floor_data:
            print("FATAL ERROR: 'DEFAULT_FLOOR' not found in tile_types.json")
            exit()
        game_map.tiles[...] = default_floor_data

        # 2. Draw each feature, including its floor and border
        for feature_tag, feature_data in generation_state.placed_features.items():
            bounds = feature_data.get('bounding_box')
            feature_type_key = feature_data.get('type')
            if not feature_type_key or feature_type_key not in features_definitions or not bounds:
                continue

            feature_def = features_definitions[feature_type_key]
            x1, y1, w, h = [int(c) for c in bounds]
            x2, y2 = x1 + w, y1 + h

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(game_map.width, x2), min(game_map.height, y2)
            if x1 >= x2 or y1 >= y2: continue

            # --- Draw Floor ---
            floor_tile_type = feature_def.get('tile_type', 'DEFAULT_FLOOR')
            floor_tile_data = self._get_tile_data_from_type(floor_tile_type)
            if floor_tile_data:
                game_map.tiles[x1:x2, y1:y2] = floor_tile_data

            # --- Draw Border ---
            border_thickness = feature_def.get('border_thickness', 1)
            if border_thickness > 0:
                border_tile_type = feature_def.get('border_tile_type', 'DEFAULT_WALL')
                border_tile_data = self._get_tile_data_from_type(border_tile_type)
                if not border_tile_data: continue

                effective_border = min(border_thickness, (x2 - x1) // 2, (y2 - y1) // 2)
                if effective_border > 0:
                    game_map.tiles[x1:x2, y1:y1 + effective_border] = border_tile_data
                    game_map.tiles[x1:x2, y2 - effective_border:y2] = border_tile_data
                    game_map.tiles[x1:x1 + effective_border, y1:y2] = border_tile_data
                    game_map.tiles[x2 - effective_border:x2, y1:y2] = border_tile_data

        # 3. Overlay pathfinding doors
        door_tile_data = self._get_tile_data_from_type("DEFAULT_DOOR")
        if door_tile_data and generation_state.door_locations:
            valid_coords = [(x, y) for x, y in generation_state.door_locations if game_map.is_in_bounds(x, y)]
            if valid_coords:
                door_xs, door_ys = zip(*valid_coords)
                game_map.tiles[door_xs, door_ys] = door_tile_data


class GameMap:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.tiles = np.zeros((width, height), dtype=tile_dt, order="F")
        self.tiles["transparent"] = True  # Start with all transparent

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