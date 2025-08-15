# /core/ui/views/character_generation_test_view.py

from typing import List
import tcod

from ..ui_framework import View, DynamicTextBox


class CharacterGenerationTestView(View):
    """A view to display the results of the character generation test."""

    def __init__(self, context_text: str, console_width: int, console_height: int):
        super().__init__()
        self.console_width = console_width
        self.console_height = console_height
        self.context_text = context_text
        self.characters = []
        self._layout_widgets()

    def update_characters(self, characters: List[dict]):
        self.characters = characters
        self._layout_widgets()

    def _layout_widgets(self):
        self.widgets = []

        context_box = DynamicTextBox(
            title="Generation Context", text=self.context_text, x=2, y=1,
            max_width=self.console_width - 4, max_height=10
        )
        self.widgets.append(context_box)

        current_y = context_box.y + context_box.height + 1

        for char_data in self.characters:
            if current_y >= self.console_height - 2:
                break

            char_text = (
                f"Name: {char_data.get('name', 'N/A')}\n"
                f"Desc: {char_data.get('description', 'N/A')}\n"
                f"Phys Desc: {char_data.get('physical_description', 'N/A')}\n"
                f"Instructions: {char_data.get('instructions', 'N/A')}"
            )

            char_box = DynamicTextBox(
                title=f"Generated Character #{len(self.widgets)}", text=char_text, x=2, y=current_y,
                max_width=self.console_width - 4,
                max_height=self.console_height - current_y - 2
            )
            self.widgets.append(char_box)
            current_y += char_box.height + 1

    def render(self, console: tcod.console.Console):
        console.clear()
        super().render(console)