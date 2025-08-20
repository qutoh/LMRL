# /core/ui/views/save_selection_view.py

from typing import Callable

import tcod

from ..ui_framework import VBox, Button, Frame, View


class SaveSelectionView(View):
    """A view for selecting a specific save game to load."""

    def __init__(self, saves: list[str], on_choice: Callable[[str], None], on_back: Callable[[], None],
                 console_width: int, console_height: int):
        super().__init__()
        self.saves = saves
        self.on_choice = on_choice
        self.on_back = on_back
        self.selected_index = 0

        self.menu_options = saves + ["Back"]

        self.container = VBox()
        for item_text in self.menu_options:
            if item_text == "Back":
                button = Button(item_text, on_click=self.on_back)
                button.fg = (255, 150, 150)
            else:
                button = Button(item_text, on_click=lambda choice=item_text: self._select_and_confirm(choice))
            self.container.add_child(button)

        self.menu_frame = Frame(self.container, title="Select a Save File")
        self.menu_frame.center_in_parent(console_width, console_height)
        self.menu_frame.y = 2

        self.widgets = [self.menu_frame]
        self._update_selection()

    def _select_and_confirm(self, choice: str):
        self.on_choice(choice)

    def _update_selection(self):
        for i, child in enumerate(self.container.children):
            if isinstance(child, Button):
                child.selected = (i == self.selected_index)

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
                key_handled = True
                choice = self.menu_options[self.selected_index]
                if choice == "Back":
                    self.on_back()
                else:
                    self._select_and_confirm(choice)
            elif event.sym == tcod.event.KeySym.ESCAPE:
                self.on_back()
                key_handled = True

            if key_handled:
                self._update_selection()