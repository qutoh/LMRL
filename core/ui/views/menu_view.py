# /core/ui/views/menu_view.py

from typing import Callable, Optional
import tcod

from ..ui_framework import VBox, Button, Frame, View


class MenuView(View):
    """A generic, modal menu view for in-game choices."""

    def __init__(self, title: str, options: list[str], on_choice: Callable[[Optional[str]], None], console_width: int,
                 console_height: int):
        super().__init__()
        self.on_choice = on_choice
        self.selected_index = 0
        self.menu_options = options

        self.container = VBox()
        for item_text in self.menu_options:
            button = Button(item_text, on_click=lambda choice=item_text: self._select_and_confirm(choice))
            self.container.add_child(button)

        self.menu_frame = Frame(self.container, title=title)
        self.menu_frame.center_in_parent(console_width, console_height)
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
                self._select_and_confirm(self.menu_options[self.selected_index])
                key_handled = True
            elif event.sym == tcod.event.KeySym.ESCAPE:
                self.on_choice(None)  # Cancel by sending None
                key_handled = True

            if key_handled:
                self._update_selection()