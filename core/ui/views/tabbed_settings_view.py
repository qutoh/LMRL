# /core/ui/views/tabbed_settings_view.py

import tcod
import copy
from typing import Callable, Optional, List, Dict, Any

from ...common import file_io, utils
from ...common.config_loader import config
from ...common.localization import loc
from ..input_handler import TextInputHandler
from ..ui_framework import View, Widget, Frame, VBox, Button
from .menu_view import MenuView
from .text_input_view import TextInputView


# --- Base Class for all Settings Pages ---

class SettingsTabPage(Widget):
    """
    An abstract base class for a single page/tab within the settings view.
    Manages its own list of settings, selection, and editing logic.
    """

    def __init__(self, x: int, y: int, width: int, height: int, setting_keys: List[str],
                 parent_view: 'TabbedSettingsView'):
        super().__init__(x, y, width, height)
        self.parent_view = parent_view
        self.setting_keys = sorted([key for key in setting_keys if key in config.settings])
        self.selected_index = 0
        self.view_y_offset = 0
        self.is_editing = False
        self.edit_handler: Optional[TextInputHandler] = None
        self.error_message: Optional[str] = None
        self.keyword_options: Dict[str, List[str]] = {}

    def _start_editing(self):
        if not self.setting_keys: return
        key = self.setting_keys[self.selected_index]
        if key == 'calibrated_sampling_params':
            self.error_message = loc('settings_error_uneditable')
            return

        value = config.settings[key]
        if key in self.keyword_options or isinstance(value, bool):
            return

        self.is_editing = True
        self.edit_handler = TextInputHandler(prompt=f"Editing {key}", width=self.width - 45, prefix="")
        self.edit_handler.text = str(value) if not isinstance(value, list) else ", ".join(map(str, value))
        self.edit_handler.cursor_pos = len(self.edit_handler.text)
        self.error_message = None

    def _stop_editing(self, new_value_str: Optional[Any] = None):
        if new_value_str is not None and self.setting_keys:
            key = self.setting_keys[self.selected_index]
            original_value = config.settings[key]
            try:
                if isinstance(original_value, bool):
                    new_value = new_value_str.lower() in ['true', '1', 'yes', 't']
                elif isinstance(original_value, list):
                    new_value = [item.strip() for item in new_value_str.split(',') if item.strip()]
                else:
                    new_value = type(original_value)(new_value_str)
                config.settings[key] = new_value
            except (ValueError, TypeError):
                self.error_message = loc('settings_error_invalid_value', original_value=original_value)

        self.is_editing = False
        self.edit_handler = None

    def _restore_default(self):
        if not self.setting_keys: return
        key = self.setting_keys[self.selected_index]
        if key in config.default_settings:
            config.settings[key] = config.default_settings[key]
        self.is_editing = False
        self.edit_handler = None
        self.error_message = None

    def _revert_to_original(self):
        if not self.setting_keys: return
        key = self.setting_keys[self.selected_index]
        if key in self.parent_view.original_settings:
            config.settings[key] = self.parent_view.original_settings[key]
        self.is_editing = False
        self.edit_handler = None
        self.error_message = None

    def _cycle_value(self, direction: int):
        if not self.setting_keys: return
        key = self.setting_keys[self.selected_index]
        original_value = config.settings.get(key)

        if isinstance(original_value, bool):
            config.settings[key] = not original_value
            return

        options = self.keyword_options.get(key)
        if not options: return

        try:
            current_idx = options.index(original_value)
            new_idx = (current_idx + direction) % len(options)
            config.settings[key] = options[new_idx]
        except ValueError:
            config.settings[key] = options[0]

    def handle_event(self, event: tcod.event.Event):
        if not self.setting_keys: return

        if self.is_editing:
            if self.edit_handler:
                if isinstance(event, tcod.event.KeyDown) and event.sym == tcod.event.KeySym.ESCAPE:
                    self._stop_editing()
                else:
                    result = self.edit_handler.handle_event(event)
                    if result is not None: self._stop_editing(result)
        else:
            if isinstance(event, tcod.event.KeyDown):
                if event.sym == tcod.event.KeySym.W:
                    self.selected_index = max(0, self.selected_index - 1)
                    self.error_message = None
                elif event.sym == tcod.event.KeySym.S:
                    self.selected_index = min(len(self.setting_keys) - 1, self.selected_index + 1)
                    self.error_message = None
                elif event.sym == tcod.event.KeySym.A:
                    self._cycle_value(-1)
                elif event.sym == tcod.event.KeySym.D:
                    self._cycle_value(1)
                elif event.sym in (tcod.event.KeySym.RETURN, tcod.event.KeySym.KP_ENTER):
                    self._start_editing()
                elif event.sym == tcod.event.KeySym.DELETE:
                    self._restore_default()
                elif event.sym == tcod.event.KeySym.BACKSPACE:
                    self._revert_to_original()

    def render(self, console: tcod.console.Console):
        if not self.setting_keys:
            console.print(self.x + 2, self.y + 2, loc('settings_page_no_settings'), fg=(150, 150, 150))
            return

        max_visible_items = self.height - 2
        if self.selected_index >= self.view_y_offset + max_visible_items:
            self.view_y_offset = self.selected_index - max_visible_items + 1
        if self.selected_index < self.view_y_offset:
            self.view_y_offset = self.selected_index

        y = self.y + 1
        for i in range(self.view_y_offset, len(self.setting_keys)):
            if y >= self.y + self.height - 1: break
            key = self.setting_keys[i]
            value = config.settings.get(key, 'N/A')
            is_selected = (i == self.selected_index)

            fg = (255, 255, 0) if is_selected else (255, 255, 255)
            bg = (50, 50, 70) if is_selected else (0, 0, 0)

            console.print(x=self.x + 1, y=y, string=f"{'>>' if is_selected else '  '}{key}", fg=fg, bg=bg)

            value_str = str(value)
            if isinstance(value, list):
                value_str = ", ".join(map(str, value))

            if is_selected and self.is_editing and self.edit_handler:
                value_str = self.edit_handler.prefix + self.edit_handler.text
            elif is_selected and (key in self.keyword_options or isinstance(value, bool)):
                value_str = f"< {value_str} >"

            max_val_width = self.width - 45
            if len(value_str) > max_val_width: value_str = value_str[:max_val_width - 3] + "..."
            console.print(x=self.x + 40, y=y, string=value_str, fg=fg, bg=bg)
            y += 1


# --- Concrete Page Implementations ---
class GeneralSettingsPage(SettingsTabPage):
    def __init__(self, x, y, width, height, parent_view):
        keys = ["LOG_LEVEL", "DIRECTOR_CONFIRMATION_REQUIRED", "ADAPTIVE_RESOLUTION_MODE", "MAX_CYCLES",
                "MAX_CONSECUTIVE_TURNS_PER_CYCLE"]
        super().__init__(x, y, width, height, keys, parent_view)
        self.keyword_options = {"LOG_LEVEL": ["game", "story", "debug", "full"]}


class LLMSettingsPage(SettingsTabPage):
    def __init__(self, x, y, width, height, parent_view):
        keys = ["DEFAULT_INPUT_TOKEN_LIMIT", "LLM_PREFIX_CLEANUP_ENABLED", "LLM_PREFIX_CLEANUP_MODE",
                "ENABLE_REWRITE_STAGE", "GEMINI_API_KEY_ENV_VAR", "GEMINI_MODEL_STRING", "GEMINI_MODEL_NAME",
                "TOKEN_SUMMARY_OFFSET", "NEXT_TURN_KEYWORD", "STREAMING_BLACKLIST_RETRIES",
                "NONSTREAMING_BLACKLIST_RETRIES"]
        super().__init__(x, y, width, height, keys, parent_view)
        self.keyword_options = {"LLM_PREFIX_CLEANUP_MODE": ["strip", "rewrite"]}


class ModelSettingsPage(SettingsTabPage):
    def __init__(self, x, y, width, height, parent_view):
        keys = sorted([k for k in config.settings.keys() if
                       k.startswith("DEFAULT_") or (k.startswith("ENABLE_") and "REASONING" in k)])
        super().__init__(x, y, width, height, keys, parent_view)


class PEGSettingsPage(SettingsTabPage):
    def __init__(self, x, y, width, height, parent_view):
        keys = ["PEG_RECONCILIATION_METHOD", "MAX_PROCGEN_FEATURES", "PREGENERATE_LEVELS_ON_FIRST_RUN",
                "PEG_V3_JITTER_SCALE_BIAS", "PEG_V3_JITTER_DENSITY_FACTOR"]
        super().__init__(x, y, width, height, keys, parent_view)
        self.keyword_options = {"PEG_RECONCILIATION_METHOD": ["CONVERSATIONAL", "PARTITIONING", "ITERATIVE_PLACEMENT"]}


class AtlasSettingsPage(SettingsTabPage):
    def __init__(self, x, y, width, height, parent_view):
        keys = ["ATLAS_SCENE_PLACEMENT_STRATEGY", "ATLAS_AUTONOMOUS_EXPLORATION_STEPS",
                "ATLAS_PLAYER_FALLBACK_ON_NAME_FAIL"]
        super().__init__(x, y, width, height, keys, parent_view)
        self.keyword_options = {"ATLAS_SCENE_PLACEMENT_STRATEGY": ["SEARCH", "RANDOM", "ROOT"]}


class GameplaySettingsPage(SettingsTabPage):
    def __init__(self, x, y, width, height, parent_view):
        keys = ["ENABLE_MULTIPLE_DMS", "ENABLE_PAPER_DOLL_MODE", "FORCE_RANDOM_NAMES_FOR_ROLES",
                "BYPASS_PROMETHEUS_TOOL_USAGE", "MEMORY_RETRIEVAL_COUNT", "EMBEDDING_MODEL_NAME", "VECTOR_DB_PATH"]
        super().__init__(x, y, width, height, keys, parent_view)


class CalibrationSettingsPage(SettingsTabPage):
    def __init__(self, x, y, width, height, parent_view):
        keys = ["CALIBRATION_CONFIDENCE_LEVEL", "CALIBRATION_MIN_SAMPLES", "CALIBRATION_MAX_SAMPLES",
                "CALIBRATION_CI_WIDTH_THRESHOLD", "calibrated_sampling_params"]
        super().__init__(x, y, width, height, keys, parent_view)


class UISettingsPage(SettingsTabPage):
    def __init__(self, x, y, width, height, parent_view):
        keys = ["UI_THEME_BOX_ASPECT_PREFERENCE", "MAP_WIDTH", "MAP_HEIGHT"]
        super().__init__(x, y, width, height, keys, parent_view)


class SettingsProfilePage(Widget):
    def __init__(self, x, y, width, height, parent_view: 'TabbedSettingsView'):
        super().__init__(x, y, width, height)
        self.parent_view = parent_view
        self.selected_index = 0
        self.view_y_offset = 0
        self.error_message: Optional[str] = None
        self._update_options()

    def _update_options(self):
        self.profiles = file_io.list_settings_profiles()
        self.options = [
                           loc('settings_profiles_save'),
                           loc('settings_profiles_delete')
                       ] + self.profiles
        self.selected_index = min(self.selected_index, len(self.options) - 1)
        self.error_message = None

    def handle_event(self, event: tcod.event.Event):
        if isinstance(event, tcod.event.KeyDown):
            if event.sym == tcod.event.KeySym.W:
                self.selected_index = max(0, self.selected_index - 1)
            elif event.sym == tcod.event.KeySym.S:
                self.selected_index = min(len(self.options) - 1, self.selected_index + 1)
            elif event.sym in (tcod.event.KeySym.RETURN, tcod.event.KeySym.KP_ENTER):
                self._execute_selection()

    def _execute_selection(self):
        if not self.options: return

        selection = self.options[self.selected_index]
        if selection == loc('settings_profiles_save'):
            self.parent_view._prompt_to_save_profile()
        elif selection == loc('settings_profiles_delete'):
            self.parent_view._prompt_to_delete_profile()
        else:  # It's a profile name
            self.parent_view._load_profile(selection)

    def render(self, console: tcod.console.Console):
        max_visible_items = self.height - 2
        if self.selected_index >= self.view_y_offset + max_visible_items:
            self.view_y_offset = self.selected_index - max_visible_items + 1
        if self.selected_index < self.view_y_offset:
            self.view_y_offset = self.selected_index

        y = self.y + 1
        console.print_box(self.x + 1, y, self.width - 2, 1, loc('settings_profiles_menu_title'), alignment=tcod.CENTER)
        y += 2

        for i in range(self.view_y_offset, len(self.options)):
            if y >= self.y + self.height - 1: break

            option_text = self.options[i]
            is_selected = (i == self.selected_index)

            fg = (255, 255, 0) if is_selected else (255, 255, 255)
            bg = (50, 50, 70) if is_selected else (0, 0, 0)

            console.print(x=self.x + 2, y=y, string=f"{'>>' if is_selected else '  '}{option_text}", fg=fg, bg=bg)
            y += 1


# --- Main Tabbed View Container ---

class TabbedSettingsView(View):
    def __init__(self, on_back: Callable[[], None], console_width: int, console_height: int):
        super().__init__()
        self.on_back = on_back
        self.console_width = console_width
        self.console_height = console_height
        self.original_settings = copy.deepcopy(config.settings)
        self.show_exit_confirmation = False
        self.modal_view: Optional[View] = None
        self.info_message: Optional[str] = None
        self.info_message_timer = 0

        self._rebuild_pages()
        self._build_confirmation_dialog()

    def _rebuild_pages(self):
        content_x, content_y = 1, 4
        content_width, content_height = self.console_width - 2, self.console_height - 6
        self.pages: List[Widget] = [
            GeneralSettingsPage(content_x, content_y, content_width, content_height, self),
            LLMSettingsPage(content_x, content_y, content_width, content_height, self),
            ModelSettingsPage(content_x, content_y, content_width, content_height, self),
            PEGSettingsPage(content_x, content_y, content_width, content_height, self),
            AtlasSettingsPage(content_x, content_y, content_width, content_height, self),
            GameplaySettingsPage(content_x, content_y, content_width, content_height, self),
            CalibrationSettingsPage(content_x, content_y, content_width, content_height, self),
            UISettingsPage(content_x, content_y, content_width, content_height, self),
            SettingsProfilePage(content_x, content_y, content_width, content_height, self)
        ]
        self.tab_names = [
            loc('settings_tab_general'), loc('settings_tab_llm'), loc('settings_tab_models'),
            loc('settings_tab_peg'), loc('settings_tab_atlas'), loc('settings_tab_gameplay'),
            loc('settings_tab_calibration'), loc('settings_tab_ui'), loc('settings_tab_profiles')
        ]
        self.active_page_index = 0

    def _build_confirmation_dialog(self):
        vbox = VBox()
        vbox.add_child(Button(loc('settings_exit_confirm_save'), lambda: self._save_and_exit()))
        vbox.add_child(Button(loc('settings_exit_confirm_discard'), lambda: self._discard_and_exit()))
        vbox.add_child(Button(loc('settings_exit_confirm_cancel'), lambda: self._cancel_exit()))
        self.confirmation_dialog = Frame(vbox, title=loc('settings_exit_confirm_title'))
        self.confirmation_dialog.center_in_parent(self.console_width, self.console_height)

    def _has_unsaved_changes(self) -> bool:
        return self.original_settings != config.settings

    def _revert_all_changes(self):
        config.settings = copy.deepcopy(self.original_settings)
        self._rebuild_pages()

    def _save_and_exit(self):
        config.save_settings()
        self.on_back()

    def _discard_and_exit(self):
        self._revert_all_changes()
        self.on_back()

    def _cancel_exit(self):
        self.show_exit_confirmation = False
        if isinstance(self.confirmation_dialog.child, VBox):
            for child in self.confirmation_dialog.child.children:
                if isinstance(child, Button):
                    child.selected = False

    def _show_info_message(self, message: str):
        self.info_message = message
        self.info_message_timer = 120

    def _load_profile(self, name: str):
        data = file_io.read_settings_profile(name)
        if data:
            config.settings.update(data)
            self._rebuild_pages()
            self._show_info_message(loc('settings_profiles_load_success', name=name))

    def _prompt_to_save_profile(self):
        def on_submit(name: str):
            self.modal_view = None
            if utils.is_valid_filename(name):
                file_io.write_settings_profile(name, config.settings)
                profile_page = self.pages[-1]
                if isinstance(profile_page, SettingsProfilePage):
                    profile_page._update_options()
                self._show_info_message(loc('settings_profiles_save_success', name=name))
            else:
                profile_page = self.pages[-1]
                if isinstance(profile_page, SettingsProfilePage):
                    profile_page.error_message = loc('settings_profiles_error_invalid_name')

        handler = TextInputHandler(prompt=loc('settings_profiles_prompt_save_name'), width=50, prefix="> ")
        self.modal_view = TextInputView(handler, on_submit)

    def _prompt_to_delete_profile(self):
        profiles = file_io.list_settings_profiles()
        profile_page = self.pages[-1]
        if not isinstance(profile_page, SettingsProfilePage): return

        if not profiles:
            profile_page.error_message = loc('settings_profiles_no_profiles')
            return

        def on_submit(name: Optional[str]):
            self.modal_view = None
            if name:
                file_io.delete_settings_profile(name)
                profile_page._update_options()
                self._show_info_message(loc('settings_profiles_delete_success', name=name))

        self.modal_view = MenuView(loc('settings_profiles_prompt_delete'), profiles, on_submit, self.console_width,
                                   self.console_height)

    def handle_event(self, event: tcod.event.Event):
        if self.modal_view:
            self.modal_view.handle_event(event)
            return

        if self.show_exit_confirmation:
            if isinstance(event, tcod.event.KeyDown):
                if event.sym == tcod.event.KeySym.S:
                    self._save_and_exit()
                elif event.sym == tcod.event.KeySym.D:
                    self._discard_and_exit()
                elif event.sym == tcod.event.KeySym.ESCAPE:
                    self._cancel_exit()
            return

        active_page = self.pages[self.active_page_index]
        if isinstance(event, tcod.event.KeyDown) and not getattr(active_page, 'is_editing', False):
            if event.sym == tcod.event.KeySym.LEFT:
                self.active_page_index = (self.active_page_index - 1) % len(self.pages)
                return
            if event.sym == tcod.event.KeySym.RIGHT:
                self.active_page_index = (self.active_page_index + 1) % len(self.pages)
                return
            if event.sym == tcod.event.KeySym.ESCAPE:
                if self._has_unsaved_changes():
                    self.show_exit_confirmation = True
                else:
                    self.on_back()
                return
        active_page.handle_event(event)

    def render(self, console: tcod.console.Console):
        console.draw_frame(0, 0, self.console_width, self.console_height, loc('settings_view_title'),
                           fg=(255, 255, 255))
        current_x = 2
        for i, name in enumerate(self.tab_names):
            is_active = (i == self.active_page_index)
            fg = (255, 255, 0) if is_active else (150, 150, 150)
            bg = (40, 40, 60) if is_active else (10, 10, 10)
            tab_text = f" {name} "
            console.print(x=current_x, y=2, string=tab_text, fg=fg, bg=bg)
            if i < len(self.tab_names) - 1:
                console.print(x=current_x + len(tab_text), y=2, string="│", fg=(150, 150, 150), bg=(0, 0, 0))
            current_x += len(tab_text) + 1
        console.draw_rect(x=1, y=3, width=self.console_width - 2, height=1, ch=ord('─'), fg=(100, 100, 100))

        active_page = self.pages[self.active_page_index]
        active_page.render(console)

        error_msg = getattr(active_page, 'error_message', None)
        if error_msg:
            console.print(2, self.console_height - 3, error_msg, fg=(255, 100, 100))
        elif self.info_message and self.info_message_timer > 0:
            console.print(2, self.console_height - 3, self.info_message, fg=(100, 255, 100))
            self.info_message_timer -= 1

        footer = loc('settings_footer_nav')
        if isinstance(active_page, SettingsProfilePage):
            footer = loc('help_bar_settings_profiles')
        elif getattr(active_page, 'is_editing', False):
            footer = loc('settings_footer_edit')
        console.print(2, self.console_height - 2, footer, fg=(200, 200, 200))

        if self.show_exit_confirmation:
            console.rgb["bg"] = (console.rgb["bg"] * 0.5).astype("u1")
            self.confirmation_dialog.render(console)
            console.print_box(0, self.console_height - 2, self.console_width, 1, loc('help_bar_settings_confirm_exit'),
                              fg=(200, 200, 200), alignment=tcod.CENTER)

        if self.modal_view:
            console.rgb["bg"] = (console.rgb["bg"] * 0.5).astype("u1")
            self.modal_view.render(console)