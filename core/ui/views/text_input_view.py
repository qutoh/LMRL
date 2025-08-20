# /core/ui/views/text_input_view.py

from typing import Callable

import tcod

from ..input_handler import TextInputHandler
from ..ui_framework import View


class TextInputView(View):
    """A modal view that renders the state of a TextInputHandler."""

    def __init__(self, handler: TextInputHandler, on_submit: Callable[[str], None]):
        super().__init__()
        self.handler = handler
        self.on_submit = on_submit
        self._cursor_timer = 0
        self._show_cursor = True

    def handle_event(self, event: tcod.event.Event):
        result = self.handler.handle_event(event)
        if result is not None:
            self.on_submit(result)

    def render(self, console: tcod.console.Console):
        self._cursor_timer = (self._cursor_timer + 1) % 40
        self._show_cursor = self._cursor_timer < 20

        width = console.width
        wrapped_lines = self.handler.wrapped_lines
        has_counter = self.handler.max_tokens is not None
        height = len(wrapped_lines) + 2
        if has_counter: height += 1

        x, y = 0, console.height - height

        console.draw_frame(x=x, y=y, width=width, height=height, title=self.handler.prompt, fg=(255, 255, 255),
                           bg=(0, 0, 0))

        selection = self.handler._get_selection()

        for i, line in enumerate(wrapped_lines):
            line_start_display_pos = self.handler.line_start_indices[i]
            for j, char in enumerate(line):
                char_display_pos = line_start_display_pos + j
                text_pos = char_display_pos - len(self.handler.prefix)
                is_selected = selection and selection[0] <= text_pos < selection[1] and char_display_pos >= len(
                    self.handler.prefix)
                fg = (0, 0, 0) if is_selected else (255, 255, 255)
                bg = (200, 200, 200) if is_selected else (0, 0, 0)
                console.print(x=x + 1 + j, y=y + 1 + i, string=char, fg=fg, bg=bg)

        if has_counter:
            token_color = (255, 100, 100) if self.handler.limit_exceeded else (150, 150, 150)
            token_count_str = f"[{self.handler.token_count}/{self.handler.max_tokens} tokens]"
            console.print(x=x + width - len(token_count_str) - 1, y=y + height - 2, string=token_count_str,
                          fg=token_color)

        if self.handler.active and self._show_cursor:
            cursor_display_pos = self.handler.cursor_pos + len(self.handler.prefix)
            line_idx, col_idx = self.handler._map_pos_to_coord(cursor_display_pos)
            cursor_x, cursor_y = x + 1 + col_idx, y + 1 + line_idx

            if cursor_x < x + width - 1:
                console.rgb["bg"][cursor_x, cursor_y] = (255, 255, 255)
                console.rgb["fg"][cursor_x, cursor_y] = (0, 0, 0)