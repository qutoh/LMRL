# /core/ui/views/load_or_new_view.py

from typing import Callable
import tcod

from ..ui_framework import VBox, Button, Frame, View

class LoadOrNewView(View):
    """A view that asks the player to load a game or start a new one."""

    def __init__(self, on_choice: Callable[[str], None], console_width: int, console_height: int):
        super().__init__()
        self.on_choice = on_choice
        self.selected_index = 0
        self.menu_options = ["Load Game", "New Game", "Back to World Selection"]

        self.container = VBox()
        for i, item_text in enumerate(self.menu_options):
            button = Button(item_text, on_click=lambda choice=item_text: self._select_and_confirm(choice))
            if "New" in item_text: button.fg = (150, 255, 150)
            if "Back" in item_text: button.fg = (255, 150, 150)
            self.container.add_child(button)

        self.menu_frame = Frame(self.container, title="Load or New?")
        self.menu_frame.center_in_parent(console_width, console_height)
        self.menu_frame.y = 2

        self.widgets = [self.menu_frame]
        self._update_selection()

    def _select_and_confirm(self, choice: str):
        self.on_choice(choice.split(' ')[0].upper())  # Returns LOAD, NEW, or BACK

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
                self._select_and_confirm(self.menu_options[self.selected_index])
                key_handled = True
            elif event.sym == tcod.event.KeySym.ESCAPE:
                self.on_choice("BACK")
                key_handled = True

            if key_handled:
                self._update_selection()