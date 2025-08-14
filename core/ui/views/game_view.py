# /core/ui/views/game_view.py

import tcod
import numpy as np
import random
from ..ui_framework import View, DynamicTextBox, EventLog
from ...common.config_loader import config

class GameView(View):
    """Renders the main game state, including map, entities, and logs."""

    def __init__(self, event_log: EventLog, game_log_box: DynamicTextBox, is_debug_mode: bool = False):
        super().__init__()
        self.event_log = event_log
        self.game_log_box = game_log_box
        self.game_state = None
        self.generation_state = None
        self.widgets = [game_log_box] if game_log_box else []
        self.mouse_pos = (0, 0)
        self.is_debug_mode = is_debug_mode

    def update_state(self, game_state, generation_state=None, path_tracer=None):
        self.game_state = game_state
        self.generation_state = generation_state
        # The path_tracer is now handled by the base View's SDL primitives

    def handle_event(self, event: tcod.event.Event):
        if isinstance(event, tcod.event.MouseMotion):
            self.mouse_pos = event.tile

    def render(self, console: tcod.console.Console):
        if self.game_state:
            # 1. Blit the static base map
            console.rgb[...] = self.game_state.game_map.tiles["graphic"]

            # 2. Apply dynamic "shimmer" effects for specific tile types
            terrain_types = self.game_state.game_map.tiles["terrain_type"]
            for type_name, type_def in config.tile_types.items():
                # Check if this tile type has multiple colors or characters for animation
                if len(type_def.get("colors", [])) > 1 or len(type_def.get("characters", [])) > 1:
                    type_index = config.tile_type_map.get(type_name)
                    if type_index is not None:
                        # Find all tiles of this type on the map
                        tile_indices = np.where(terrain_types == type_index)

                        # If there are any, apply random visual changes
                        if tile_indices[0].size > 0:
                            # Choose a random character from the list if available
                            if len(type_def["characters"]) > 1:
                                random_char = ord(random.choice(type_def["characters"]))
                                console.rgb["ch"][tile_indices[1], tile_indices[0]] = random_char

                            # Choose a random color for each tile instance if available
                            if len(type_def["colors"]) > 1:
                                num_tiles = len(tile_indices[0])
                                color_indices = np.random.randint(len(type_def["colors"]), size=num_tiles)
                                random_colors = np.array(type_def["colors"])[color_indices]
                                console.rgb["fg"][tile_indices[1], tile_indices[0]] = random_colors

            # 3. Draw entities on top of everything
            for entity in self.game_state.entities:
                console.print(x=entity.x, y=entity.y, string=entity.char, fg=entity.color)

        if self.is_debug_mode:
            if self.generation_state and self.generation_state.placed_features:
                # Draw slices first, so they are in the background
                for tag, feature in self.generation_state.placed_features.items():
                    if sb := feature.get('slice_box'):
                        sx1, sy1, sx2, sy2 = [int(c) for c in sb]
                        console.rgb["bg"][sx1:sx2, sy1:sy2] = (40, 40, 40)

                # Then draw tags on top of features
                for tag, feature in self.generation_state.placed_features.items():
                    if bb := feature.get('bounding_box'):
                        # The first two elements of the bounding_box are always the top-left (x,y)
                        x, y = int(bb[0]), int(bb[1])
                        # The bounding box frame is no longer drawn.
                        if 0 <= x < console.width and 0 <= y < console.height:
                            console.print(x, y, tag, fg=(255, 255, 0))

            coord_text = f"({self.mouse_pos[0]}, {self.mouse_pos[1]})"
            console.print(x=console.width - len(coord_text) - 1, y=console.height - 1, string=coord_text,
                          fg=(255, 255, 255))

        self.event_log.render(console=console, x=1, y=0, width=console.width - 2, height=10)
        super().render(console)