# /core/ui/views/prometheus_view.py

from typing import Callable, Optional

import tcod

from ..ui_framework import View, DynamicTextBox


class PrometheusView(View):
    """A view for toggling Prometheus tool decisions."""

    def __init__(self, tools: list[str], instruction_text: str, context_text: str,
                 on_submit: Callable[[Optional[dict]], None], console_width: int, console_height: int):
        super().__init__()
        self.tools = tools
        self.on_submit = on_submit
        self.selected_index = 0
        self.tool_states = {tool: False for tool in tools}
        self.help_context_key = "PROMETHEUS_TAKEOVER"

        # --- Layout Calculations ---
        self.menu_width = max(len(t) for t in tools) + 15 if tools else 30
        self.menu_height = len(tools) + 2

        available_text_height = console_height - self.menu_height - 5  # 2 for top, 2 for padding, 1 for help bar

        instructions_width = (console_width - 6) // 2
        context_width = instructions_width

        # --- Widget Creation ---
        self.instructions_box = DynamicTextBox(
            title="Instructions", text=instruction_text,
            x=2, y=2,
            max_width=instructions_width,
            max_height=available_text_height
        )
        self.context_box = DynamicTextBox(
            title="Narrative Context", text=context_text,
            x=4 + instructions_width, y=2,
            max_width=context_width,
            max_height=available_text_height
        )

        self.widgets = [self.instructions_box, self.context_box]

        # Position menu below the taller of the two text boxes
        taller_box_height = max(self.instructions_box.height, self.context_box.height)
        self.menu_y = 2 + taller_box_height + 2
        self.menu_x = 2


    def handle_event(self, event: tcod.event.Event):
        if not isinstance(event, tcod.event.KeyDown):
            return

        key_handled = False
        if event.sym == tcod.event.KeySym.UP:
            self.selected_index = (self.selected_index - 1) % len(self.tools)
            key_handled = True
        elif event.sym == tcod.event.KeySym.DOWN:
            self.selected_index = (self.selected_index + 1) % len(self.tools)
            key_handled = True
        elif event.sym in (tcod.event.KeySym.LEFT, tcod.event.KeySym.RIGHT):
            key = self.tools[self.selected_index]
            self.tool_states[key] = not self.tool_states[key]
            key_handled = True
        elif event.sym in (tcod.event.KeySym.RETURN, tcod.event.KeySym.KP_ENTER):
            self.on_submit(self.tool_states)
            key_handled = True
        elif event.sym == tcod.event.KeySym.ESCAPE:
            self.on_submit(None)  # Cancel
            key_handled = True

    def render(self, console: tcod.console.Console):
        # Render the widgets (context boxes) first
        super().render(console)

        # Now, render the menu frame and its contents manually
        console.draw_frame(x=self.menu_x, y=self.menu_y, width=self.menu_width, height=self.menu_height,
                           title="Prometheus Tool Control", fg=(255, 255, 255), bg=(20, 20, 20))

        for i, tool_name in enumerate(self.tools):
            is_selected = (i == self.selected_index)
            y = self.menu_y + 1 + i

            fg = (255, 255, 0) if is_selected else (255, 255, 255)
            bg = (50, 50, 70) if is_selected else (20, 20, 20)

            console.print(x=self.menu_x + 2, y=y, string=f"{'>>' if is_selected else '  '}{tool_name}", fg=fg, bg=bg)

            state_str = str(self.tool_states[tool_name]).upper()
            state_fg = (150, 255, 150) if self.tool_states[tool_name] else (255, 150, 150)

            console.print(x=self.menu_x + self.menu_width - len(state_str) - 3, y=y, string=f"< {state_str} >", fg=state_fg,
                          bg=bg)