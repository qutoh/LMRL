# /core/ui/input_handler.py

import textwrap
from typing import Optional, Tuple

import pyperclip
import tcod

from ..common.utils import count_tokens


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
    A pure logic class to handle state and events for text input, mimicking
    standard text editor functionality. This class does NO rendering.
    """

    def __init__(self, prompt: str, width: int, prefix: str = "", max_tokens: Optional[int] = None, initial_text: str = ""):
        self.prompt = prompt
        self.width = width
        self.prefix = prefix
        self.max_tokens = max_tokens

        self.text: str = initial_text
        self.active: bool = True
        self.cursor_pos: int = len(self.text)
        self.selection_anchor: Optional[int] = None

        self.token_count: int = 0
        self.limit_exceeded: bool = False

        self.wrapped_lines: list[str] = [""]
        self.line_start_indices: list[int] = [0]

        self.key_down_time: float = 0.0  # Initialize the missing attribute

        self._update_text(self.text) # Use update_text to correctly initialize all derived state

    def _get_selection(self) -> Optional[tuple[int, int]]:
        """Returns the start and end of the selection, correctly ordered."""
        if self.selection_anchor is None: return None
        start, end = sorted((self.selection_anchor, self.cursor_pos))
        return (start, end) if start != end else None

    def _delete_selection(self):
        """Deletes the selected text and updates state."""
        selection = self._get_selection()
        if not selection: return
        start, end = selection
        self.text = self.text[:start] + self.text[end:]
        self.cursor_pos = start
        self.selection_anchor = None

    def _update_text(self, new_text: str):
        """Central method to update text and all derived state."""
        self.text = new_text
        self.token_count = count_tokens(self.text)
        if self.max_tokens is not None:
            self.limit_exceeded = self.token_count > self.max_tokens
        else:
            self.limit_exceeded = False
        self._recalculate_lines()

    def _recalculate_lines(self):
        """Wraps the display text and creates a map of line start indices."""
        display_text = self.prefix + self.text
        self.wrapped_lines = textwrap.wrap(display_text, self.width, drop_whitespace=False)
        if not self.wrapped_lines: self.wrapped_lines = [""]

        self.line_start_indices = [0]
        text_pos = 0
        for line in self.wrapped_lines[:-1]:
            text_pos += len(line)
            while text_pos < len(display_text) and display_text[text_pos].isspace():
                text_pos += 1
            if text_pos < len(display_text):
                self.line_start_indices.append(text_pos)

    def _map_pos_to_coord(self, pos: int) -> Tuple[int, int]:
        """Maps a 1D position in the display string to a 2D (line, col) coordinate."""
        line_idx = 0
        for i, start_idx in enumerate(self.line_start_indices):
            if pos < start_idx:
                break
            line_idx = i
        else:  # pos is on or after the last line's start
            if self.line_start_indices: line_idx = len(self.line_start_indices) - 1

        start_of_line = self.line_start_indices[line_idx]
        col_idx = pos - start_of_line
        return line_idx, col_idx

    def _map_coord_to_pos(self, line_idx: int, col_idx: int) -> int:
        """Maps a 2D (line, col) coordinate to a 1D position in the display string."""
        line_idx = max(0, min(line_idx, len(self.wrapped_lines) - 1))
        start_of_line = self.line_start_indices[line_idx]
        line_len = len(self.wrapped_lines[line_idx])
        col_idx = max(0, min(col_idx, line_len))
        return start_of_line + col_idx

    def handle_event(self, event: tcod.event.Event) -> Optional[str]:
        if not self.active: return None

        if isinstance(event, tcod.event.TextInput):
            potential_text = self.text[:self.cursor_pos] + event.text + self.text[self.cursor_pos:]
            if self.max_tokens and count_tokens(potential_text) > self.max_tokens: return None

            self._delete_selection()
            new_text = self.text[:self.cursor_pos] + event.text + self.text[self.cursor_pos:]
            self.cursor_pos += len(event.text)
            self._update_text(new_text)
            return None

        if isinstance(event, tcod.event.KeyDown):
            key, mod, shift, ctrl = event.sym, event.mod, bool(event.mod & tcod.event.Modifier.SHIFT), bool(
                event.mod & tcod.event.Modifier.CTRL)

            if key in (tcod.event.KeySym.LEFT, tcod.event.KeySym.RIGHT, tcod.event.KeySym.HOME, tcod.event.KeySym.END,
                       tcod.event.KeySym.UP, tcod.event.KeySym.DOWN):
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
                elif key == tcod.event.KeySym.HOME:
                    display_pos = self.line_start_indices[line_idx]
                elif key == tcod.event.KeySym.END:
                    display_pos = self.line_start_indices[line_idx] + len(self.wrapped_lines[line_idx])
                elif key == tcod.event.KeySym.UP:
                    display_pos = self._map_coord_to_pos(line_idx - 1, col_idx)
                elif key == tcod.event.KeySym.DOWN:
                    display_pos = self._map_coord_to_pos(line_idx + 1, col_idx)

                if key in (tcod.event.KeySym.HOME, tcod.event.KeySym.END, tcod.event.KeySym.UP, tcod.event.KeySym.DOWN):
                    self.cursor_pos = max(0, display_pos - len(self.prefix))
                return None

            if key == tcod.event.KeySym.BACKSPACE:
                if self._get_selection():
                    self._delete_selection()
                    self._update_text(self.text)
                elif self.cursor_pos > 0:
                    new_text = self.text[:self.cursor_pos - 1] + self.text[self.cursor_pos:]
                    self.cursor_pos -= 1
                    self._update_text(new_text)
                return None

            if key == tcod.event.KeySym.DELETE:
                if self._get_selection():
                    self._delete_selection()
                    self._update_text(self.text)
                elif self.cursor_pos < len(self.text):
                    new_text = self.text[:self.cursor_pos] + self.text[self.cursor_pos + 1:]
                    self._update_text(new_text)
                return None

            if ctrl:
                if key == tcod.event.KeySym.A:
                    self.selection_anchor, self.cursor_pos = 0, len(self.text)
                elif key == tcod.event.KeySym.C:
                    if selection := self._get_selection(): pyperclip.copy(self.text[selection[0]:selection[1]])
                elif key == tcod.event.KeySym.V:
                    try:
                        pasted = pyperclip.paste().replace('\n', ' ').replace('\r', '')
                        if not pasted: return None

                        selection = self._get_selection()
                        temp_text = self.text
                        temp_cursor = self.cursor_pos
                        if selection:
                            start, end = selection
                            temp_text = temp_text[:start] + temp_text[end:]
                            temp_cursor = start

                        potential_text = temp_text[:temp_cursor] + pasted + temp_text[temp_cursor:]
                        if self.max_tokens and count_tokens(potential_text) > self.max_tokens: return None

                        self._delete_selection()
                        new_text = self.text[:self.cursor_pos] + pasted + self.text[self.cursor_pos:]
                        self.cursor_pos += len(pasted)
                        self._update_text(new_text)
                    except Exception:
                        pass  # Fail silently if clipboard is unavailable
                return None

            if key in (tcod.event.KeySym.RETURN, tcod.event.KeySym.KP_ENTER): self.active = False; return self.text
            if key == tcod.event.KeySym.ESCAPE: self.active = False; return ""
        return None