# /core/ui_framework.py

from typing import Callable, Optional, Tuple
import tcod
import tcod.event
from tcod import libtcodpy
import textwrap
from math import sqrt


class Widget:
    """Base class for all UI elements."""

    def __init__(self, x=0, y=0, width=0, height=0):
        self.x, self.y, self.width, self.height = x, y, width, height
        self.parent: Optional['Widget'] = None
        self.is_focusable = False
        self.is_focused = False

    def handle_event(self, event: tcod.event.Event):
        pass

    def render(self, console: tcod.console.Console):
        raise NotImplementedError()

    def get_absolute_pos(self) -> Tuple[int, int]:
        """Calculates the widget's absolute position on the console."""
        if self.parent:
            px, py = self.parent.get_absolute_pos()
            return px + self.x, py + self.y
        return self.x, self.y


class Label(Widget):
    """A simple text label."""

    def __init__(self, text: str, fg=(255, 255, 255), **kwargs):
        super().__init__(**kwargs)
        self.text = text
        self.fg = fg
        self.width = len(text)
        self.height = 1

    def render(self, console: tcod.console.Console):
        abs_x, abs_y = self.get_absolute_pos()
        console.print(x=abs_x, y=abs_y, string=self.text, fg=self.fg)


class Button(Widget):
    """A clickable button that performs an action."""

    def __init__(self, text: str, on_click: Callable, fg=(255, 255, 255), **kwargs):
        super().__init__(width=len(text) + 4, height=1, **kwargs)
        self.text = text
        self.on_click = on_click
        self.fg = fg
        self.mouse_over = False
        self.selected = False
        self.is_focusable = True

    def handle_event(self, event: tcod.event.Event):
        abs_x, abs_y = self.get_absolute_pos()
        if isinstance(event, tcod.event.MouseMotion):
            self.mouse_over = abs_x <= event.tile.x < abs_x + self.width and abs_y == event.tile.y
        elif isinstance(event, tcod.event.MouseButtonDown):
            if self.mouse_over and event.button == tcod.BUTTON_LEFT:
                self.on_click()

    def render(self, console: tcod.console.Console):
        abs_x, abs_y = self.get_absolute_pos()

        final_fg = self.fg
        if self.selected or self.is_focused:
            final_fg = (255, 255, 0)  # Yellow for keyboard selection/focus

        bg = (50, 50, 50) if self.mouse_over else (0, 0, 0)

        display_text = f"{'> ' if self.selected or self.is_focused else ' '}{self.text}{' <' if self.selected or self.is_focused else ' '}"

        console.print_box(x=abs_x, y=abs_y, width=self.width, height=1, string=display_text, fg=final_fg, bg=bg,
                          alignment=libtcodpy.CENTER)


class VBox(Widget):
    """A container that arranges its children vertically."""

    def __init__(self, children: Optional[list[Widget]] = None, **kwargs):
        super().__init__(**kwargs)
        self.children = children or []
        for child in self.children:
            child.parent = self
        self.pack()

    def add_child(self, widget: Widget):
        widget.parent = self
        self.children.append(widget)
        self.pack()

    def pack(self):
        current_y = 0
        max_child_width = 0
        for child in self.children:
            child.y = current_y
            current_y += child.height
            if child.width > max_child_width:
                max_child_width = child.width

        self.width = max_child_width
        self.height = current_y

        for child in self.children:
            child.x = (self.width - child.width) // 2

    def handle_event(self, event: tcod.event.Event):
        for child in self.children:
            child.handle_event(event)

    def render(self, console: tcod.console.Console):
        for child in self.children:
            child.render(console)


class Frame(Widget):
    """A widget that draws a frame around another widget."""

    def __init__(self, child: Widget, title: str = "", fg=(255, 255, 255), **kwargs):
        super().__init__(**kwargs)
        self.child = child
        self.child.parent = self
        self.title = title
        self.fg = fg
        self.update_size()

    def update_size(self):
        self.width = self.child.width + 2
        self.height = self.child.height + 2
        self.child.x = 1
        self.child.y = 1

    def center_in_parent(self, parent_width: int, parent_height: int):
        self.x = (parent_width - self.width) // 2
        self.y = (parent_height - self.height) // 2

    def handle_event(self, event: tcod.event.Event):
        self.child.handle_event(event)

    def render(self, console: tcod.console.Console):
        abs_x, abs_y = self.get_absolute_pos()
        console.draw_frame(abs_x, abs_y, self.width, self.height, title=self.title, fg=self.fg)
        self.child.render(console)


class ScrollableTextBox(Widget):
    """A read-only text box that supports scrolling."""

    def __init__(self, text: str, **kwargs):
        super().__init__(**kwargs)
        self.view_y_offset = 0
        self.is_focusable = True
        self.lines = []
        self.set_text(text)

    def set_text(self, text: str):
        self.lines = []
        # Preserve explicit newlines by splitting and then wrapping each part.
        for paragraph in text.split('\n'):
            self.lines.extend(textwrap.wrap(paragraph, width=max(1, self.width - 2)))  # -2 for borders
        self.view_y_offset = 0

    def scroll(self, lines: int):
        """Scrolls the view by a number of lines."""
        max_offset = max(0, len(self.lines) - self.height)
        self.view_y_offset = max(0, min(self.view_y_offset + lines, max_offset))

    def handle_event(self, event: tcod.event.Event):
        if not self.is_focused:
            return

        if isinstance(event, tcod.event.MouseWheel):
            self.scroll(-event.y)
        elif isinstance(event, tcod.event.KeyDown):
            if event.sym == tcod.event.KeySym.UP:
                self.scroll(-1)
            elif event.sym == tcod.event.KeySym.DOWN:
                self.scroll(1)
            elif event.sym == tcod.event.KeySym.PAGEUP:
                self.scroll(-self.height)
            elif event.sym == tcod.event.KeySym.PAGEDOWN:
                self.scroll(self.height)

    def render(self, console: tcod.console.Console):
        abs_x, abs_y = self.get_absolute_pos()

        # Draw visible text
        for i in range(self.height):
            line_index = self.view_y_offset + i
            if 0 <= line_index < len(self.lines):
                console.print(x=abs_x, y=abs_y + i, string=self.lines[line_index], fg=(200, 200, 255))

        # --- Draw Scrollbar ---
        can_scroll_up = self.view_y_offset > 0
        can_scroll_down = self.view_y_offset < max(0, len(self.lines) - self.height)

        scrollbar_x = abs_x + self.width - 1

        # Up arrow
        console.print(scrollbar_x, abs_y, "▲", fg=(255, 255, 0) if can_scroll_up else (100, 100, 100))
        # Down arrow
        console.print(scrollbar_x, abs_y + self.height - 1, "▼",
                      fg=(255, 255, 0) if can_scroll_down else (100, 100, 100))

        # Scrollbar track
        for i in range(1, self.height - 1):
            console.print(scrollbar_x, abs_y + i, "│")

        # Scrollbar thumb
        if len(self.lines) > self.height:
            thumb_pos_float = (self.view_y_offset / (len(self.lines) - self.height)) * (self.height - 3)
            thumb_y = abs_y + 1 + int(thumb_pos_float)
            console.print(scrollbar_x, thumb_y, "█", fg=(255, 255, 0))


class DynamicTextBox(Widget):
    """A widget that displays text in a dynamically sized box."""

    def __init__(self, text: str, title: str = "", max_width: int = 80, max_height: int = 20,
                 aspect_preference: float = 4.0, **kwargs):
        super().__init__(**kwargs)
        self.title = title
        self.aspect_preference = aspect_preference
        self.max_width = max_width
        self.max_height = max_height
        self.set_text(text)

    def set_text(self, text: str):
        self.text = text
        self.width, self.height, self.wrapped_lines = self._calculate_optimal_box_size()

    def _calculate_optimal_box_size(self) -> tuple[int, int, list[str]]:
        if not self.text: return 0, 0, []

        # Try to find a reasonable width
        text_len = len(self.text)
        target_aspect = self.aspect_preference
        target_h = sqrt((text_len * 1.2) / target_aspect) if text_len > 0 else 0
        target_w = target_h * target_aspect
        box_width = min(self.max_width, int(target_w) + 2)

        # Wrap text based on the calculated width, respecting newlines
        wrapped_lines = []
        for paragraph in self.text.split('\n'):
            wrapped_lines.extend(textwrap.wrap(paragraph, width=max(1, box_width - 2), drop_whitespace=True))

        box_height = min(self.max_height, len(wrapped_lines) + 2)

        final_lines = wrapped_lines[:max(0, box_height - 2)]
        if final_lines and len(wrapped_lines) > len(final_lines):
            # Truncate the last visible line if content is cut off
            last_line = final_lines[-1]
            final_lines[-1] = last_line[:max(0, box_width - 5)] + "..."

        return box_width, box_height, final_lines

    def render(self, console: tcod.console.Console):
        if self.width <= 0 or self.height <= 0: return
        abs_x, abs_y = self.get_absolute_pos()
        console.draw_frame(x=abs_x, y=abs_y, width=self.width, height=self.height, title=self.title, fg=(220, 220, 220),
                           bg=(0, 0, 0))
        for i, line in enumerate(self.wrapped_lines):
            console.print(x=abs_x + 1, y=abs_y + 1 + i, string=line, fg=(200, 200, 255))


class HelpBar(Widget):
    """A context-aware help bar at the bottom of the screen."""

    def __init__(self, help_texts: dict, **kwargs):
        super().__init__(**kwargs)
        self.help_texts = help_texts
        self.current_context = "DEFAULT"

    def set_context(self, context_key):
        self.current_context = context_key

    def render(self, console: tcod.console.Console):
        help_text = self.help_texts.get(self.current_context, self.help_texts.get("DEFAULT", ""))

        self.x = 0
        self.y = console.height - 1
        self.width = console.width
        self.height = 1

        console.draw_rect(x=self.x, y=self.y, width=self.width, height=self.height, ch=0, bg=(20, 20, 20))
        console.print(x=self.x + 1, y=self.y, string=help_text, fg=(200, 200, 200))


class View:
    """Manages a screen full of widgets and focus."""

    def __init__(self):
        self.widgets: list[Widget] = []
        self.sdl_primitives = []
        self.focusable_widgets: list[Widget] = []
        self.focused_widget_index: int = 0

    def _update_focusable_widgets(self):
        """Should be called whenever the view's widget list changes."""
        self.focusable_widgets = [w for w in self.widgets if w.is_focusable]
        self.set_focus_by_index(0)

    def set_focus_by_index(self, index: int):
        if not self.focusable_widgets:
            return
        self.focused_widget_index = index % len(self.focusable_widgets)
        for i, widget in enumerate(self.focusable_widgets):
            widget.is_focused = (i == self.focused_widget_index)

    def set_focus_by_widget(self, widget: Widget):
        if widget in self.focusable_widgets:
            index = self.focusable_widgets.index(widget)
            self.set_focus_by_index(index)

    def cycle_focus(self, direction: int = 1):
        if self.focusable_widgets:
            self.set_focus_by_index(self.focused_widget_index + direction)

    def add_line(self, start_xy: Tuple[int, int], end_xy: Tuple[int, int], color: Tuple[int, int, int]):
        self.sdl_primitives.append({'type': 'line', 'start': start_xy, 'end': end_xy, 'color': color})

    def handle_event(self, event: tcod.event.Event):
        # Pass the event ONLY to the focused widget
        if self.focusable_widgets:
            focused_widget = self.focusable_widgets[self.focused_widget_index]
            focused_widget.handle_event(event)

    def render(self, console: tcod.console.Console):
        for widget in self.widgets:
            widget.render(console)


class EventLog:
    """A scrollable log for displaying messages."""

    def __init__(self, max_lines: int = 10):
        self.messages: list[Tuple[str, Tuple[int, int, int]]] = []
        self.max_lines = max_lines

    def add_message(self, text: str, fg: Tuple[int, int, int] = (255, 255, 255)):
        for line in text.split("\n"):
            self.messages.append((line, fg))
        if len(self.messages) > self.max_lines:
            self.messages = self.messages[len(self.messages) - self.max_lines:]

    def render(self, console: tcod.console.Console, x: int, y: int, width: int, height: int) -> None:
        y_offset = height - 1
        for text, fg in reversed(self.messages):
            for line in reversed(textwrap.wrap(text, width)):
                if y_offset < 0:
                    return
                console.print(x=x, y=y + y_offset, string=line, fg=fg)
                y_offset -= 1