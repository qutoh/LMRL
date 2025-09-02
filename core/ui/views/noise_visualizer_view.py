# /core/ui/views/noise_visualizer_view.py

import tcod
import numpy as np
from typing import Callable

from ..ui_framework import View
from ...common.game_state import GameState, GenerationState
from ..app_states import AppState


class NoiseVisualizerView(View):
    """An interactive view for visualizing and tuning tcod noise parameters."""

    def __init__(self, on_exit: Callable, console_width: int, console_height: int):
        super().__init__()
        self.on_exit = on_exit
        self.console_width = console_width
        self.console_height = console_height
        self.help_context_key = AppState.NOISE_VISUALIZER_TEST

        self.base_game_state: GameState | None = None
        self.base_gen_state: GenerationState | None = None

        self.noise_console = tcod.console.Console(console_width, console_height, order="F")

        # Define noise parameters, their ranges, and step values
        self.params = {
            'lacunarity': 2.0,
            'scale': 0.1
        }
        self.param_config = {
            'lacunarity': {'min': 0.1, 'max': 10.0, 'step': 0.1},
            'scale': {'min': 0.01, 'max': 1.0, 'step': 0.01}
        }
        self.param_keys = list(self.params.keys())
        self.selected_param_index = 0

    def set_base_map(self, game_state: GameState, generation_state: GenerationState):
        """Receives the pre-generated map to draw as a background."""
        self.base_game_state = game_state
        self.base_gen_state = generation_state
        self.regenerate_noise_texture()

    def regenerate_noise_texture(self):
        """Generates a new noise map based on current parameters and draws it to the noise console."""
        noise = tcod.noise.Noise(
            dimensions=2,
            algorithm=tcod.noise.Algorithm.PERLIN,
            implementation=tcod.noise.Implementation.FBM,
            hurst=0.0,
            lacunarity=self.params['lacunarity'],
            octaves=2,
            seed=None
        )

        grid_y, grid_x = np.ogrid[0:self.console_height, 0:self.console_width]
        samples = noise.sample_ogrid([grid_x * self.params['scale'], grid_y * self.params['scale']])

        normalized_samples = ((samples + 1.0) * 127.5).astype(np.uint8)

        self.noise_console.clear()
        grayscale_color = np.empty((*normalized_samples.shape, 3), dtype=np.uint8)
        grayscale_color[..., 0] = normalized_samples
        grayscale_color[..., 1] = normalized_samples
        grayscale_color[..., 2] = normalized_samples
        self.noise_console.bg[...] = grayscale_color

    def handle_event(self, event: tcod.event.Event):
        if not isinstance(event, tcod.event.KeyDown):
            return

        key = event.sym

        if key == tcod.event.KeySym.ESCAPE:
            self.on_exit()
            return

        key_handled = False
        if key == tcod.event.KeySym.RIGHT or key == tcod.event.KeySym.LEFT:
            self.selected_param_index = (self.selected_param_index + 1) % len(self.param_keys)
            key_handled = True
        else:
            param_key = self.param_keys[self.selected_param_index]
            config = self.param_config[param_key]
            current_value = self.params[param_key]

            if key == tcod.event.KeySym.UP:
                self.params[param_key] = min(config['max'], current_value + config['step'])
                key_handled = True
            elif key == tcod.event.KeySym.DOWN:
                self.params[param_key] = max(config['min'], current_value - config['step'])
                key_handled = True

        if key_handled:
            param_key = self.param_keys[self.selected_param_index]
            self.params[param_key] = round(self.params[param_key], 4)
            self.regenerate_noise_texture()

    def render(self, console: tcod.console.Console):
        if self.base_game_state:
            console.rgb[...] = self.base_game_state.game_map.tiles["graphic"]
            for entity in self.base_game_state.entities:
                console.print(x=entity.x, y=entity.y, string=entity.char, fg=entity.color)

        self.noise_console.blit(
            console,
            dest_x=0, dest_y=0,
            src_x=0, src_y=0,
            width=self.console_width, height=self.console_height,
            bg_alpha=0.5
        )

        y_offset = 2
        console.print_box(
            x=0, y=0, width=self.console_width, height=len(self.param_keys) + 3,
            string="", bg=(0, 0, 0)
        )
        console.print(x=2, y=1, string="Noise Parameters (Perlin FBM)", fg=(255, 255, 255))

        for i, key in enumerate(self.param_keys):
            fg = (255, 255, 0) if i == self.selected_param_index else (200, 200, 200)
            value_str = f"{self.params[key]:.2f}"
            console.print(x=2, y=y_offset + i, string=f"{key.capitalize()}: {value_str}", fg=fg)

        super().render(console)