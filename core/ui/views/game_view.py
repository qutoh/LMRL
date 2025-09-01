# /core/ui/views/game_view.py

import random
from typing import Optional

import numpy as np
import tcod

from ..ui_framework import View, DynamicTextBox, EventLog
from ...common.config_loader import config
from ..app_states import AppState


class GameView(View):
    """Renders the main game state, including map, entities, and a streaming log."""

    def __init__(self, event_log: EventLog, console_width: int, console_height: int, is_debug_mode: bool = False):
        super().__init__()
        self.event_log = event_log
        self.console_width = console_width
        self.console_height = console_height
        self.is_debug_mode = is_debug_mode
        self.help_context_key = AppState.GAME_RUNNING

        self.game_state = None
        self.generation_state = None
        self.mouse_pos = (0, 0)

        self.log_boxes: list[DynamicTextBox] = []
        self.active_streaming_box: Optional[DynamicTextBox] = None
        self.active_streaming_buffer = ""

        # Dedicate the bottom part of the screen to the scrollable log
        self.log_area_height = 15
        self.log_area_y = self.console_height - self.log_area_height

    def update_state(self, game_state, generation_state=None, path_tracer=None):
        self.game_state = game_state
        self.generation_state = generation_state
        # The path_tracer is now handled by the base View's SDL primitives

    def start_new_log_entry(self, speaker: str):
        """Creates a new, empty text box for an incoming stream."""
        self.finalize_active_log()  # Finalize any previous entry first.
        self.active_streaming_buffer = ""
        new_box = DynamicTextBox(
            text="",
            title=speaker,
            x=0,
            y=0,  # y-position is calculated dynamically during render
            max_width=self.console_width,
            max_height=self.log_area_height  # A single box can't be taller than the whole area
        )
        self.active_streaming_box = new_box
        self.log_boxes.append(new_box)

    def append_to_active_log(self, text_delta: str, is_retry_clear: bool = False):
        """Appends a token to the currently streaming text box and forces it to resize."""
        if not self.active_streaming_box:
            return

        if is_retry_clear:
            self.active_streaming_buffer = ""
        else:
            self.active_streaming_buffer += text_delta

        # We call set_text to force the widget to recalculate its dimensions
        self.active_streaming_box.set_text(self.active_streaming_buffer)

    def finalize_active_log(self):
        """Finalizes the current stream, making the text box a permanent log entry."""
        self.active_streaming_box = None
        self.active_streaming_buffer = ""

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
                if len(type_def.get("colors", [])) > 1 or len(type_def.get("characters", [])) > 1:
                    type_index = config.tile_type_map.get(type_name)
                    if type_index is not None:
                        tile_indices = np.where(terrain_types == type_index)
                        if tile_indices[0].size > 0:
                            if len(type_def["characters"]) > 1:
                                random_char = ord(random.choice(type_def["characters"]))
                                console.rgb["ch"][tile_indices[1], tile_indices[0]] = random_char
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
                for tag, feature in self.generation_state.placed_features.items():
                    if sb := feature.get('slice_box'):
                        sx1, sy1, sx2, sy2 = [int(c) for c in sb]
                        console.rgb["bg"][sx1:sx2, sy1:sy2] = (40, 40, 40)
                for tag, feature in self.generation_state.placed_features.items():
                    if bb := feature.get('bounding_box'):
                        x, y = int(bb[0]), int(bb[1])
                        if 0 <= x < console.width and 0 <= y < console.height:
                            console.print(x, y, tag, fg=(255, 255, 0))
            coord_text = f"({self.mouse_pos[0]}, {self.mouse_pos[1]})"
            console.print(x=console.width - len(coord_text) - 1, y=console.height - 1, string=coord_text,
                          fg=(255, 255, 255))

        # --- New Log Rendering Logic ---
        console.draw_rect(x=0, y=self.log_area_y, width=self.console_width, height=self.log_area_height, ch=0,
                          bg=(5, 5, 15))
        current_y = self.console_height - 1

        for box in reversed(self.log_boxes):
            box_height = box.height
            box.y = current_y - box_height + 1
            box.x = 0
            if box.y < self.console_height:
                box.render(console)
            current_y -= box_height
            if current_y < self.log_area_y:
                break

        # Prune boxes that are scrolled way off-screen to save memory/performance
        if len(self.log_boxes) > 20:
            self.log_boxes = self.log_boxes[-20:]

        self.event_log.render(console=console, x=1, y=0, width=console.width - 2, height=10)
        super().render(console)