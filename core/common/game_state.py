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

    def _draw_blended_hallway(self, game_map: 'GameMap', hallway_data: dict):
        """Draws a hallway by blending the tiles of its two source features."""
        path_coords = set(map(tuple, hallway_data.get('path_coords', [])))
        if not path_coords: return

        type_a = hallway_data.get('source_a_type')
        type_b = hallway_data.get('source_b_type')
        def_a = config.features.get(type_a, {})
        def_b = config.features.get(type_b, {})

        tile_a_floor = self._get_tile_data_from_type(def_a.get('tile_type', 'DEFAULT_FLOOR'))
        tile_a_border = self._get_tile_data_from_type(def_a.get('border_tile_type', 'DEFAULT_WALL'))
        tile_b_floor = self._get_tile_data_from_type(def_b.get('tile_type', 'DEFAULT_FLOOR'))
        tile_b_border = self._get_tile_data_from_type(def_b.get('border_tile_type', 'DEFAULT_WALL'))
        if not all([tile_a_floor, tile_a_border, tile_b_floor, tile_b_border]): return

        for x, y in path_coords:
            if not game_map.is_in_bounds(x, y):
                continue
            game_map.tiles[x, y] = tile_a_floor if random.random() < 0.5 else tile_b_floor

        midpoint_index = len(hallway_data['path_coords']) // 2
        path_list = hallway_data['path_coords']

        for i, (x, y) in enumerate(path_list):
            if not game_map.is_in_bounds(x, y):
                continue
            border_tile = tile_a_border if i < midpoint_index else tile_b_border
            for nx in range(x - 1, x + 2):
                for ny in range(y - 1, y + 2):
                    if (nx, ny) != (x, y) and game_map.is_in_bounds(nx, ny) and (nx, ny) not in path_coords:
                        game_map.tiles[nx, ny] = border_tile

    def _draw_path_feature(self, game_map: 'GameMap', path_coords: set, feature_data: dict, all_features: dict):
        """Draws a path with context-aware tiles, ensuring it doesn't overwrite existing walls."""
        feature_def = config.features.get(feature_data.get('type'), {})
        default_floor_type = feature_def.get('tile_type', 'PATH_FLOOR')
        default_border_type = feature_def.get('border_tile_type', 'DEFAULT_WALL')
        intersection_rules = {rule['type']: rule for rule in feature_def.get('intersects_ok', [])}

        # Pass 1: Draw the floor, considering what it intersects.
        for x, y in path_coords:
            if not game_map.is_in_bounds(x, y): continue
            current_terrain_index = game_map.tiles["terrain_type"][x, y]
            intersected_type_key = config.tile_type_map_reverse.get(current_terrain_index)
            rule = intersection_rules.get(intersected_type_key)
            floor_type = rule.get('replaces_with_floor', default_floor_type) if rule else default_floor_type
            if floor_tile_data := self._get_tile_data_from_type(floor_type):
                game_map.tiles[x, y] = floor_tile_data

        # Pass 2: Draw the border, respecting intersection rules.
        void_space_index = config.tile_type_map.get("VOID_SPACE", -1)
        for x, y in path_coords:
            for nx in range(x - 1, x + 2):
                for ny in range(y - 1, y + 2):
                    if (nx, ny) == (x, y) or not game_map.is_in_bounds(nx, ny): continue
                    if (nx, ny) not in path_coords:
                        target_terrain_index = game_map.tiles["terrain_type"][nx, ny]

                        # Only draw border on void space or on a tile type that is explicitly allowed in intersects_ok
                        if target_terrain_index == void_space_index or config.tile_type_map_reverse.get(
                                target_terrain_index) in intersection_rules:
                            intersected_type_key = config.tile_type_map_reverse.get(target_terrain_index)
                            rule = intersection_rules.get(intersected_type_key)
                            border_type = rule.get('replaces_with_border',
                                                   default_border_type) if rule else default_border_type
                            if border_tile_data := self._get_tile_data_from_type(border_type):
                                game_map.tiles[nx, ny] = border_tile_data

    def _carve_portal_openings(self, game_map: 'GameMap', generation_state: 'GenerationState'):
        """Finds connections involving a PORTAL feature and removes the border between them."""
        if not generation_state.layout_graph: return

        for p_tag, c_tag, _ in generation_state.layout_graph.edges:
            parent_data = generation_state.placed_features.get(p_tag)
            child_data = generation_state.placed_features.get(c_tag)
            if not parent_data or not child_data: continue

            parent_def = config.features.get(parent_data.get('type'), {})
            child_def = config.features.get(child_data.get('type'), {})

            if parent_def.get('feature_type') != 'PORTAL' and child_def.get('feature_type') != 'PORTAL':
                continue

            portal_data = parent_data if parent_def.get('feature_type') == 'PORTAL' else child_data
            other_data = child_data if parent_def.get('feature_type') == 'PORTAL' else parent_data

            portal_rect = portal_data.get('bounding_box')
            other_rect = other_data.get('bounding_box')
            if not portal_rect or not other_rect: continue

            px1, py1, pw, ph = [int(c) for c in portal_rect]
            ox1, oy1, ow, oh = [int(c) for c in other_rect]
            px2, py2 = px1 + pw, py1 + ph
            ox2, oy2 = ox1 + ow, oy1 + oh

            # Find shared border coordinates
            x_overlap_start, x_overlap_end = max(px1, ox1), min(px2, ox2)
            y_overlap_start, y_overlap_end = max(py1, oy1), min(py2, oy2)

            border_to_carve = []
            if px2 == ox1 or ox2 == px1:  # Vertical border
                border_to_carve = [(px2 - 1, y) for y in range(y_overlap_start, y_overlap_end)]
            elif py2 == oy1 or oy2 == py1:  # Horizontal border
                border_to_carve = [(x, py2 - 1) for x in range(x_overlap_start, x_overlap_end)]

            if len(border_to_carve) > 2:
                border_to_carve = border_to_carve[1:-1]  # Trim corners

            portal_floor_type = config.features.get(portal_data.get('type'), {}).get('tile_type', 'DEFAULT_FLOOR')
            carving_tile_data = self._get_tile_data_from_type(portal_floor_type)
            if carving_tile_data:
                for x, y in border_to_carve:
                    if game_map.is_in_bounds(x, y): game_map.tiles[x, y] = carving_tile_data
                    if game_map.is_in_bounds(x + 1, y): game_map.tiles[x + 1, y] = carving_tile_data

    def draw_map(self, game_map: 'GameMap', generation_state: 'GenerationState', features_definitions: dict):
        if not generation_state or not features_definitions:
            return

        void_space_data = self._get_tile_data_from_type("VOID_SPACE")
        if not void_space_data:
            print("FATAL ERROR: 'VOID_SPACE' not found in tile_types.json")
            exit()
        game_map.tiles[...] = void_space_data

        standard_features = {}
        path_features = {}
        for tag, data in generation_state.placed_features.items():
            if data.get("path_tiles") or data.get("path_coords"):
                path_features[tag] = data
            else:
                standard_features[tag] = data

        for feature_tag, feature_data in standard_features.items():
            feature_type_key = feature_data.get('type')
            if not feature_type_key or feature_type_key not in features_definitions: continue
            feature_def = features_definitions[feature_type_key]
            bounds = feature_data.get('bounding_box')
            if not bounds: continue
            x1, y1, w, h = [int(c) for c in bounds]
            x2, y2 = x1 + w, y1 + h
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(game_map.width, x2), min(game_map.height, y2)
            if x1 >= x2 or y1 >= y2: continue
            floor_tile_type = feature_def.get('tile_type', 'VOID_SPACE')
            if floor_tile_data := self._get_tile_data_from_type(floor_tile_type):
                game_map.tiles[x1:x2, y1:y2] = floor_tile_data
            border_thickness = feature_def.get('border_thickness', 1)
            if border_thickness > 0:
                border_tile_type = feature_def.get('border_tile_type', 'VOID_SPACE')
                if border_tile_data := self._get_tile_data_from_type(border_tile_type):
                    effective_border = min(border_thickness, (x2 - x1) // 2, (y2 - y1) // 2)
                    if effective_border > 0:
                        game_map.tiles[x1:x2, y1:y1 + effective_border] = border_tile_data
                        game_map.tiles[x1:x2, y2 - effective_border:y2] = border_tile_data
                        game_map.tiles[x1:x1 + effective_border, y1:y2] = border_tile_data
                        game_map.tiles[x2 - effective_border:x2, y1:y2] = border_tile_data

        if generation_state.blended_hallways:
            for hallway_data in generation_state.blended_hallways:
                self._draw_blended_hallway(game_map, hallway_data)

        for feature_tag, feature_data in path_features.items():
            path_coords = feature_data.get("path_tiles") or feature_data.get("path_coords")
            path_coords_set = set(map(tuple, path_coords))
            self._draw_path_feature(game_map, path_coords_set, feature_data, generation_state.placed_features)

        # New Pass: Carve openings for portals
        self._carve_portal_openings(game_map, generation_state)

        if generation_state.door_locations:
            for door_info in generation_state.door_locations:
                pos, tile_type = door_info.get('pos'), door_info.get('type')
                if not pos or not tile_type: continue
                x, y = pos
                if game_map.is_in_bounds(x, y):
                    if door_tile_data := self._get_tile_data_from_type(tile_type):
                        game_map.tiles[x, y] = door_tile_data

        exterior_tile_name = getattr(generation_state, 'exterior_tile_type', 'DEFAULT_FLOOR')
        if exterior_tile_data := self._get_tile_data_from_type(exterior_tile_name):
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
        if name is None:
            return None
        return next((e for e in self.entities if e.name is not None and e.name.lower() == name.lower()), None)

    def reset_entity_turn_stats(self, entity_name: str):
        entity = self.get_entity(entity_name)
        if entity:
            entity.movement_remaining = entity.speed