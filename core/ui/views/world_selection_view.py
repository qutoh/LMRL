# /core/ui/views/world_selection_view.py

from typing import Callable
import tcod

from ..ui_framework import VBox, Button, Frame, View, DynamicTextBox

class WorldSelectionView(View):
    """The main menu view for selecting a world or other options."""

    def __init__(self, worlds: list[dict], on_choice: Callable[[str], None], console_width: int, console_height: int):
        super().__init__()
        self.worlds = worlds
        self.on_choice = on_choice
        self.selected_index = 0

        menu_items = [world['name'] for world in worlds] + [
            "Build new...",
            "Settings",
            "Calibrate Task Temperatures",
            "PEG V3 Iterative Test",
            "Character Generation Test"
        ]
        self.menu_options = menu_items

        self.container = VBox()
        for i, item in enumerate(menu_items):
            button = Button(item, on_click=lambda choice=item: self._select_and_confirm(choice))
            if item == "Build new...": button.fg = (150, 255, 150)
            elif item == "Settings": button.fg = (200, 200, 200)
            elif item == "Calibrate Task Temperatures": button.fg = (150, 200, 255)
            elif item == "PEG V3 Iterative Test": button.fg = (200, 255, 150)
            elif item == "Character Generation Test": button.fg = (255, 150, 200)
            self.container.add_child(button)

        self.menu_frame = Frame(self.container, title="Select a World")
        self.menu_frame.center_in_parent(console_width, console_height)
        self.menu_frame.y = 2

        theme_box_y = self.menu_frame.y + self.menu_frame.height + 1
        self.theme_box = DynamicTextBox(
            title="World Theme", text="", x=2, y=theme_box_y,
            max_width=console_width - 4,
            max_height=console_height - theme_box_y - 2
        )

        self.widgets = [self.menu_frame, self.theme_box]
        self._update_selection()

    def _select_and_confirm(self, choice: str):
        self.selected_index = self.menu_options.index(choice)
        self.on_choice(choice)

    def _update_selection(self):
        for i, child in enumerate(self.container.children):
            if isinstance(child, Button):
                child.selected = (i == self.selected_index)

        if self.selected_index < len(self.worlds):
            theme_text = self.worlds[self.selected_index].get('theme', 'No theme description found.')
            self.theme_box.set_text(theme_text)
        else:
            self.theme_box.set_text("")

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
                self.on_choice("EXIT")
                key_handled = True

            if key_handled:
                self._update_selection()