# /core/ui/views/scene_selection_view.py

from typing import Callable
import tcod
import random

from ..ui_framework import VBox, Button, Frame, View, DynamicTextBox

class SceneSelectionView(View):
    """A view for selecting a starting scene for a chosen world."""

    def __init__(self, scenes: list[dict], anachronisms: list[dict], on_choice: Callable[[dict | str], None],
                 on_new: Callable[[], None], on_back: Callable[[], None], console_width: int, console_height: int):
        super().__init__()
        self.scenes = scenes
        self.anachronisms = anachronisms
        self.on_choice = on_choice
        self.on_new = on_new
        self.on_back = on_back
        self.selected_index = 0
        self.active_list = "main"

        self.console_width = console_width
        self.console_height = console_height

        self.prompt_box = DynamicTextBox(title="Scene Prompt", text="", max_width=console_width - 4)

        self._build_main_menu()
        self._update_selection()

    def _get_truncated_prompt(self, prompt: str) -> str:
        return f"{prompt[:70]}..." if len(prompt) > 70 else prompt

    def _layout_widgets(self):
        """Recalculates positions and updates the main widget list."""
        self.menu_frame.center_in_parent(self.console_width, self.console_height)
        self.menu_frame.y = 2

        prompt_box_y = self.menu_frame.y + self.menu_frame.height + 1
        self.prompt_box.y = prompt_box_y
        self.prompt_box.x = 2
        self.prompt_box.max_height = self.console_height - prompt_box_y - 2

        self.widgets = [self.menu_frame, self.prompt_box]

    def _build_main_menu(self):
        self.active_list = "main"
        menu_items = [self._get_truncated_prompt(s.get('scene_prompt', '')) for s in self.scenes]
        menu_items.extend([
            "Select a random scene for this world",
            "Write a new scene...",
            "Anachronisms - Scenes not meant for this world",
            "Back to World Selection"
        ])
        self.menu_options = menu_items
        self._rebuild_container()

    def _build_anachronisms_menu(self):
        self.active_list = "anachronisms"
        menu_items = [self._get_truncated_prompt(s.get('scene_prompt', '')) for s in self.anachronisms]
        menu_items.extend([
            "Select a random anachronism",
            "Back to main scene list"
        ])
        self.menu_options = menu_items
        self._rebuild_container()

    def _rebuild_container(self):
        """Recreates the button container, the frame, and relinks the main widget list."""
        self.container = VBox()
        for i, item_text in enumerate(self.menu_options):
            self.container.add_child(Button(item_text, on_click=lambda i=i: self._handle_click(i)))
        self.selected_index = 0
        self.menu_frame = Frame(self.container, title="Select a Starting Scene")
        self._layout_widgets()

    def _handle_click(self, index: int):
        self.selected_index = index
        self._execute_selection()

    def _select_and_confirm(self, choice: dict):
        self.on_choice(choice)

    def _update_selection(self):
        for i, child in enumerate(self.container.children):
            if isinstance(child, Button):
                child.selected = (i == self.selected_index)

        prompt = ""
        if self.active_list == "main":
            if self.selected_index < len(self.scenes):
                prompt = self.scenes[self.selected_index].get('scene_prompt', '')
        elif self.active_list == "anachronisms":
            if self.selected_index < len(self.anachronisms):
                prompt = self.anachronisms[self.selected_index].get('scene_prompt', '')

        self.prompt_box.set_text(prompt)

    def _execute_selection(self):
        choice_text = self.menu_options[self.selected_index]

        if self.active_list == "main":
            if self.selected_index < len(self.scenes):
                self.on_choice(self.scenes[self.selected_index])
            elif choice_text == "Select a random scene for this world":
                self.on_choice("RANDOM_NORMAL")
            elif choice_text == "Write a new scene...":
                self.on_new()
            elif choice_text == "Anachronisms - Scenes not meant for this world":
                self._build_anachronisms_menu()
            elif choice_text == "Back to World Selection":
                self.on_back()

        elif self.active_list == "anachronisms":
            if self.selected_index < len(self.anachronisms):
                self.on_choice(self.anachronisms[self.selected_index])
            elif choice_text == "Select a random anachronism":
                self.on_choice("RANDOM_ANACHRONISM")
            elif choice_text == "Back to main scene list":
                self._build_main_menu()

        self._update_selection()

    def handle_event(self, event: tcod.event.Event):
        super().handle_event(event)
        if isinstance(event, tcod.event.KeyDown):
            key_handled = False
            if event.sym == tcod.event.KeySym.UP:
                self.selected_index = (self.selected_index - 1) % len(self.menu_options)
                key_handled = True
            elif event.sym == tcod.event.KeySym.DOWN:
                self.selected_index = (self.selected_index + 1) % len(self.menu_options)
                key_handled = True
            elif event.sym in (tcod.event.KeySym.RETURN, tcod.event.KeySym.KP_ENTER):
                self._execute_selection()
                key_handled = True
            elif event.sym == tcod.event.KeySym.ESCAPE:
                if self.active_list == "anachronisms":
                    self._build_main_menu()
                else:
                    self.on_back()
                key_handled = True

            if key_handled:
                self._update_selection()