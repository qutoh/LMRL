# /core/ui/views/settings_view.py

from typing import Callable, Optional
import tcod

from ..ui_framework import View
from ..input_handler import TextInputHandler
from ...common.config_loader import config


class SettingsView(View):
    """A view for editing application settings from settings.json."""

    def __init__(self, on_back: Callable[[], None], console_height: int):
        super().__init__()
        self.on_back = on_back
        self.setting_keys = sorted([k for k in config.settings.keys() if k != 'calibrated_sampling_params'])
        self.selected_index = 0
        self.is_editing = False
        self.edit_handler: Optional[TextInputHandler] = None
        self._cursor_timer = 0
        self._show_cursor = True

        self.keyword_options = {
            "LOG_LEVEL": ["game", "story", "debug", "full"],
            "LLM_PREFIX_CLEANUP_MODE": ["strip", "rewrite"],
            "PROCGEN_ALGORITHM": ["CONVERSATIONAL", "PARTITIONING"]
        }
        self.view_y_offset = 0
        self.console_height = console_height

    def _save_and_exit(self):
        config.save_settings()
        self.on_back()

    def _start_editing(self):
        self.is_editing = True
        key = self.setting_keys[self.selected_index]
        value = config.settings[key]

        if key not in self.keyword_options:
            self.edit_handler = TextInputHandler(prompt=f"Editing {key}", width=40, prefix="")
            self.edit_handler.text = str(value)
            self.edit_handler.cursor_pos = len(self.edit_handler.text)

    def _stop_editing(self, new_value_str: Optional[str] = None):
        if new_value_str is not None:
            key = self.setting_keys[self.selected_index]
            original_value = config.settings[key]
            try:
                if isinstance(original_value, bool):
                    new_value = new_value_str.lower() in ['true', '1', 'yes']
                else:
                    new_value = type(original_value)(new_value_str)
                config.settings[key] = new_value
            except (ValueError, TypeError):
                pass

        self.is_editing = False
        self.edit_handler = None

    def _restore_default(self):
        key = self.setting_keys[self.selected_index]
        if key in config.default_settings:
            config.settings[key] = config.default_settings[key]
        self._stop_editing()

    def _cycle_keyword(self, direction: int):
        key = self.setting_keys[self.selected_index]
        options = self.keyword_options.get(key)
        if not options: return

        current_value = config.settings[key]
        try:
            current_idx = options.index(current_value)
            new_idx = (current_idx + direction) % len(options)
            config.settings[key] = options[new_idx]
        except ValueError:
            config.settings[key] = options[0]

    def handle_event(self, event: tcod.event.Event):
        if self.is_editing:
            key = self.setting_keys[self.selected_index]
            if self.edit_handler:
                if isinstance(event, tcod.event.KeyDown) and event.sym == tcod.event.KeySym.ESCAPE:
                    self._restore_default()
                else:
                    result = self.edit_handler.handle_event(event)
                    if result is not None: self._stop_editing(result)
            else:
                if isinstance(event, tcod.event.KeyDown):
                    if event.sym == tcod.event.KeySym.LEFT:
                        self._cycle_keyword(-1)
                    elif event.sym == tcod.event.KeySym.RIGHT:
                        self._cycle_keyword(1)
                    elif event.sym in (tcod.event.KeySym.RETURN, tcod.event.KeySym.KP_ENTER):
                        self._stop_editing()
                    elif event.sym == tcod.event.KeySym.ESCAPE:
                        self._restore_default()
        else:
            if isinstance(event, tcod.event.KeyDown):
                if event.sym == tcod.event.KeySym.UP:
                    self.selected_index = max(0, self.selected_index - 1)
                elif event.sym == tcod.event.KeySym.DOWN:
                    self.selected_index = min(len(self.setting_keys) - 1, self.selected_index + 1)
                elif event.sym in (tcod.event.KeySym.RETURN, tcod.event.KeySym.KP_ENTER):
                    self._start_editing()
                elif event.sym == tcod.event.KeySym.ESCAPE:
                    self._save_and_exit()

    def render(self, console: tcod.console.Console):
        console.draw_frame(0, 0, console.width, console.height, "Settings", fg=(255, 255, 255))

        max_visible_items = self.console_height - 4
        if self.selected_index >= self.view_y_offset + max_visible_items:
            self.view_y_offset = self.selected_index - max_visible_items + 1
        if self.selected_index < self.view_y_offset:
            self.view_y_offset = self.selected_index

        y = 2
        for i in range(self.view_y_offset, len(self.setting_keys)):
            if y >= self.console_height - 2: break
            key = self.setting_keys[i]
            value = config.settings.get(key, 'N/A')
            is_selected = (i == self.selected_index)

            fg = (255, 255, 0) if is_selected else (255, 255, 255)
            bg = (50, 50, 70) if is_selected else (0, 0, 0)

            console.print(x=2, y=y, string=f"{'>>' if is_selected else '  '}{key}", fg=fg, bg=bg)

            value_str = str(value)

            if is_selected and self.is_editing:
                if self.edit_handler:
                    value_str = self.edit_handler.prefix + self.edit_handler.text
                    self._cursor_timer = (self._cursor_timer + 1) % 40
                    self._show_cursor = self._cursor_timer < 20
                elif key in self.keyword_options:
                    value_str = f"< {value_str} >"

            max_val_width = console.width - 45
            if len(value_str) > max_val_width: value_str = value_str[:max_val_width - 3] + "..."
            console.print(x=40, y=y, string=value_str, fg=fg, bg=bg)

            if is_selected and self.is_editing and self.edit_handler and self._show_cursor:
                cursor_x = 40 + self.edit_handler.cursor_pos + len(self.edit_handler.prefix)
                if 40 <= cursor_x < console.width:
                    console.rgb['bg'][cursor_x, y] = (255, 255, 255)
                    console.rgb['fg'][cursor_x, y] = (0, 0, 0)
            y += 1

        footer = "[UP/DOWN] Navigate | [ENTER] Edit | [ESC] Back & Save"
        if self.is_editing:
            footer = "[ENTER] Confirm | [ESC] Restore Default | [LEFT/RIGHT] Cycle Keywords"
        console.print(2, console.height - 2, footer, fg=(200, 200, 200))