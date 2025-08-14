# /core/ui/views/text_input_view.py

from typing import Callable, Optional
import tcod

from ..ui_framework import View, Frame, ScrollableTextBox
from ..input_handler import TextInputHandler


class TextInputView(View):
    """A modal view that renders a scrollable context box and a scrollable text input handler."""

    def __init__(self, handler: TextInputHandler, on_submit: Callable[[str], None], context_text: Optional[str] = None):
        super().__init__()
        self.handler = handler
        self.on_submit = on_submit
        self._cursor_timer = 0
        self._show_cursor = True

        self.context_box = None
        if context_text:
            self.context_box = ScrollableTextBox(text=context_text)
            self.widgets.append(self.context_box)

        # The handler is not a widget in the traditional sense, but it is focusable.
        self.widgets.append(self.handler)
        self._update_focusable_widgets()

    def handle_event(self, event: tcod.event.Event):
        # Let the view handle focus passing
        super().handle_event(event)

        # Check for submission after the focused handler has processed the event
        if self.handler.is_focused:
            result = self.handler.handle_event(event)
            if result is not None:
                self.on_submit(result)

    def render(self, console: tcod.console.Console):
        self._cursor_timer = (self._cursor_timer + 1) % 40
        self._show_cursor = self._cursor_timer < 20

        console_width, console_height = console.width, console.height
        has_context = self.context_box is not None

        # --- Dynamic Layout ---
        input_height = (console_height - 4) // 3
        context_height = console_height - input_height - 5  # 2 for top, 2 for bottom frame, 1 for padding

        input_y = context_height + 3

        if has_context:
            self.context_box.x = 1
            self.context_box.y = 1
            self.context_box.width = console_width - 2
            self.context_box.height = context_height

            context_frame_fg = (255, 255, 0) if self.context_box.is_focused else (255, 255, 255)
            context_frame = Frame(self.context_box, title="Full Context", fg=context_frame_fg)
            context_frame.render(console)
        else:
            # If no context, let input take up more space
            input_height = console_height - 4
            input_y = 1

        # --- Render Input Handler ---
        self.handler.height = input_height
        self.handler.width = console_width - 4  # Space for frame and potential scrollbar

        input_frame_fg = (255, 255, 0) if self.handler.is_focused else (255, 255, 255)
        console.draw_frame(x=0, y=input_y, width=console_width, height=input_height + 2, title=self.handler.prompt,
                           fg=input_frame_fg)

        # Render visible lines
        for i in range(input_height):
            line_idx = self.handler.view_y_offset + i
            if 0 <= line_idx < len(self.handler.wrapped_lines):
                line = self.handler.wrapped_lines[line_idx]
                console.print(x=1, y=input_y + 1 + i, string=line)

        # Render selection
        selection = self.handler._get_selection()
        if selection:
            for i in range(input_height):
                line_idx = self.handler.view_y_offset + i
                if 0 <= line_idx < len(self.handler.wrapped_lines):
                    line = self.handler.wrapped_lines[line_idx]
                    line_start_display_pos = self.handler.line_start_indices[line_idx]
                    for j, char in enumerate(line):
                        char_display_pos = line_start_display_pos + j
                        text_pos = char_display_pos - len(self.handler.prefix)
                        if selection[0] <= text_pos < selection[1] and char_display_pos >= len(self.handler.prefix):
                            console.rgb["bg"][1 + j, input_y + 1 + i] = (200, 200, 200)
                            console.rgb["fg"][1 + j, input_y + 1 + i] = (0, 0, 0)

        # Render token counter
        if self.handler.max_tokens is not None:
            token_color = (255, 100, 100) if self.handler.limit_exceeded else (150, 150, 150)
            token_count_str = f"[{self.handler.token_count}/{self.handler.max_tokens} tokens]"
            console.print(x=console_width - len(token_count_str) - 1, y=input_y + input_height, string=token_count_str,
                          fg=token_color)

        # Render cursor
        if self.handler.active and self.handler.is_focused and self._show_cursor:
            cursor_display_pos = self.handler.cursor_pos + len(self.handler.prefix)
            line_idx, col_idx = self.handler._map_pos_to_coord(cursor_display_pos)

            # Check if cursor is within the visible area
            if self.handler.view_y_offset <= line_idx < self.handler.view_y_offset + self.handler.height:
                cursor_x, cursor_y = 1 + col_idx, input_y + 1 + (line_idx - self.handler.view_y_offset)
                if cursor_x < console_width - 1:
                    console.rgb["bg"][cursor_x, cursor_y] = (255, 255, 255)
                    console.rgb["fg"][cursor_x, cursor_y] = (0, 0, 0)