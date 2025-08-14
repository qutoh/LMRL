# /core/ui/input_handler.py

import tcod
from typing import Optional, Tuple
import pyperclip
from ..common.utils import count_tokens
import textwrap


class MenuInputHandler:
    """A pure logic class to handle state and events for menu selection."""

    def __init__(self, options: list[str]):
        self.options = options
        self.selected_index = 0
        self.active = True

    def handle_event(self, event: tcod.event.Event) -> Optional[str]:
        if not self.active or not isinstance(event, tcod.event.KeyDown):
            return None

        if event.sym == tcod.event.KeySym.UP:
            self.selected_index = (self.selected_index - 1) % len(self.options)
        elif event.sym == tcod.event.KeySym.DOWN:
            self.selected_index = (self.selected_index + 1) % len(self.options)
        elif event.sym in (tcod.event.KeySym.RETURN, tcod.event.KeySym.KP_ENTER):
            self.active = False
            return self.options[self.selected_index]
        elif event.sym == tcod.event.KeySym.ESCAPE:
            self.active = False
            return None  # Return None to indicate cancellation
        return None


class TextInputHandler:
    """
    A pure logic class to handle state and events for text input.
    Now supports a scrollable viewport.
    """

    def __init__(self, prompt: str, width: int, height: int, prefix: str = "", max_tokens: Optional[int] = None,
                 initial_text: str = ""):
        self.prompt = prompt
        self.width = width
        self.height = height
        self.prefix = prefix
        self.max_tokens = max_tokens
        self.is_focusable = True
        self.is_focused = False

        self.text: str = initial_text
        self.active: bool = True
        self.cursor_pos: int = len(self.text)
        self.selection_anchor: Optional[int] = None

        self.token_count: int = 0
        self.limit_exceeded: bool = False

        self.wrapped_lines: list[str] = [""]
        self.line_start_indices: list[int] = [0]
        self.view_y_offset: int = 0

        self.key_down_time: float = 0.0

        self._update_text(self.text)

    def _get_selection(self) -> Optional[tuple[int, int]]:
        if self.selection_anchor is None: return None
        start, end = sorted((self.selection_anchor, self.cursor_pos))
        return (start, end) if start != end else None

    def _delete_selection(self):
        selection = self._get_selection()
        if not selection: return
        start, end = selection
        self.text = self.text[:start] + self.text[end:]
        self.cursor_pos = start
        self.selection_anchor = None

    def _update_text(self, new_text: str):
        self.text = new_text
        self.token_count = count_tokens(self.text)
        self.limit_exceeded = self.max_tokens is not None and self.token_count > self.max_tokens
        self._recalculate_lines()
        self._scroll_to_cursor()

    def _recalculate_lines(self):
        display_text = self.prefix + self.text
        self.wrapped_lines = []
        for paragraph in display_text.split('\n'):
            self.wrapped_lines.extend(textwrap.wrap(paragraph, self.width, drop_whitespace=False) or [""])

        self.line_start_indices = [0]
        text_pos = 0
        for line in self.wrapped_lines[:-1]:
            text_pos += len(line)
            # Handle newlines correctly in position mapping
            if text_pos < len(display_text) and display_text[text_pos] == '\n':
                text_pos += 1
            self.line_start_indices.append(text_pos)

    def _scroll_to_cursor(self):
        """Ensures the cursor is visible in the viewport."""
        cursor_display_pos = self.cursor_pos + len(self.prefix)
        cursor_line, _ = self._map_pos_to_coord(cursor_display_pos)

        if cursor_line < self.view_y_offset:
            self.view_y_offset = cursor_line
        elif cursor_line >= self.view_y_offset + self.height:
            self.view_y_offset = cursor_line - self.height + 1

    def scroll(self, lines: int):
        max_offset = max(0, len(self.wrapped_lines) - self.height)
        self.view_y_offset = max(0, min(self.view_y_offset + lines, max_offset))

    def _map_pos_to_coord(self, pos: int) -> Tuple[int, int]:
        line_idx = len(self.line_start_indices) - 1
        for i, start_idx in enumerate(self.line_start_indices):
            if pos < start_idx:
                line_idx = i - 1
                break

        start_of_line = self.line_start_indices[line_idx]
        col_idx = pos - start_of_line
        return line_idx, col_idx

    def _map_coord_to_pos(self, line_idx: int, col_idx: int) -> int:
        line_idx = max(0, min(line_idx, len(self.wrapped_lines) - 1))
        start_of_line = self.line_start_indices[line_idx]
        line_len = len(self.wrapped_lines[line_idx])
        col_idx = max(0, min(col_idx, line_len))
        return start_of_line + col_idx

    def handle_event(self, event: tcod.event.Event) -> Optional[str]:
        if not self.active: return None
        if not self.is_focused:
            if isinstance(event, tcod.event.MouseWheel):
                self.scroll(-event.y)
            return None

        if isinstance(event, tcod.event.MouseWheel):
            self.scroll(-event.y)
            return None

        if isinstance(event, tcod.event.TextInput):
            potential_text = self.text[:self.cursor_pos] + event.text + self.text[self.cursor_pos:]
            if self.max_tokens and count_tokens(potential_text) > self.max_tokens: return None
            self._delete_selection()
            new_text = self.text[:self.cursor_pos] + event.text + self.text[self.cursor_pos:]
            self.cursor_pos += len(event.text)
            self._update_text(new_text)
            return None

        if isinstance(event, tcod.event.KeyDown):
            shift, ctrl = bool(event.mod & tcod.event.Modifier.SHIFT), bool(event.mod & tcod.event.Modifier.CTRL)
            key = event.sym

            # --- Navigation and Scrolling ---
            if key in (tcod.event.KeySym.LEFT, tcod.event.KeySym.RIGHT, tcod.event.KeySym.HOME, tcod.event.KeySym.END,
                       tcod.event.KeySym.UP, tcod.event.KeySym.DOWN, tcod.event.KeySym.PAGEUP,
                       tcod.event.KeySym.PAGEDOWN):
                if shift and self.selection_anchor is None:
                    self.selection_anchor = self.cursor_pos
                elif not shift:
                    self.selection_anchor = None

                display_pos = self.cursor_pos + len(self.prefix)
                line_idx, col_idx = self._map_pos_to_coord(display_pos)

                if key == tcod.event.KeySym.LEFT:
                    self.cursor_pos = max(0, self.cursor_pos - 1)
                elif key == tcod.event.KeySym.RIGHT:
                    self.cursor_pos = min(len(self.text), self.cursor_pos + 1)
                elif key == tcod.event.KeySym.UP:
                    if line_idx > 0:
                        display_pos = self._map_coord_to_pos(line_idx - 1, col_idx)
                    else:
                        self.scroll(-1)  # Scroll if at top
                elif key == tcod.event.KeySym.DOWN:
                    if line_idx < len(self.wrapped_lines) - 1:
                        display_pos = self._map_coord_to_pos(line_idx + 1, col_idx)
                    else:
                        self.scroll(1)  # Scroll if at bottom
                elif key == tcod.event.KeySym.HOME:
                    display_pos = self.line_start_indices[line_idx]
                elif key == tcod.event.KeySym.END:
                    if line_idx < len(self.wrapped_lines):
                        display_pos = self.line_start_indices[line_idx] + len(self.wrapped_lines[line_idx])
                elif key == tcod.event.KeySym.PAGEUP:
                    self.scroll(-self.height)
                elif key == tcod.event.KeySym.PAGEDOWN:
                    self.scroll(self.height)

                if key not in (tcod.event.KeySym.PAGEUP, tcod.event.KeySym.PAGEDOWN):
                    self.cursor_pos = max(0, display_pos - len(self.prefix))

                self._scroll_to_cursor()
                return None

            # --- Editing ---
            if key == tcod.event.KeySym.BACKSPACE:
                if self._get_selection():
                    self._delete_selection()
                elif self.cursor_pos > 0:
                    self.text = self.text[:self.cursor_pos - 1] + self.text[self.cursor_pos:]
                    self.cursor_pos -= 1
                self._update_text(self.text)
                return None
            if key == tcod.event.KeySym.DELETE:
                if self._get_selection():
                    self._delete_selection()
                elif self.cursor_pos < len(self.text):
                    self.text = self.text[:self.cursor_pos] + self.text[self.cursor_pos + 1:]
                self._update_text(self.text)
                return None

            # --- Clipboard and Selection ---
            if ctrl:
                if key == tcod.event.KeySym.A:
                    self.selection_anchor, self.cursor_pos = 0, len(self.text)
                elif key == tcod.event.KeySym.C:
                    if selection := self._get_selection(): pyperclip.copy(self.text[selection[0]:selection[1]])
                elif key == tcod.event.KeySym.V:
                    try:
                        pasted = pyperclip.paste()  # Keep newlines for pasting
                        if not pasted: return None

                        potential_text = self.text[:self.cursor_pos] + pasted + self.text[self.cursor_pos:]
                        if self.max_tokens and count_tokens(potential_text) > self.max_tokens: return None

                        self._delete_selection()
                        new_text = self.text[:self.cursor_pos] + pasted + self.text[self.cursor_pos:]
                        self.cursor_pos += len(pasted)
                        self._update_text(new_text)
                    except Exception:
                        pass
                return None

            # --- Submission ---
            if key in (tcod.event.KeySym.RETURN,
                       tcod.event.KeySym.KP_ENTER) and not shift: self.active = False; return self.text
            if key == tcod.event.KeySym.ESCAPE: self.active = False; return ""
        return None