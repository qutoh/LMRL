# /core/ui/views/role_creator_view.py

import json
from typing import Callable, Optional

import tcod

from . import TextInputView
from ..input_handler import TextInputHandler
from ..ui_framework import View, VBox, Button, Frame, Label, DynamicTextBox


class RoleCreatorView(View):
    """A view for interactively creating character role groups."""

    def __init__(self, on_submit: Callable[[Optional[str]], None], context_text: str, console_width: int,
                 console_height: int, parent_ui_manager: 'UIManager'):
        super().__init__()
        self.on_submit = on_submit
        self.console_width = console_width
        self.console_height = console_height
        self.parent_ui_manager = parent_ui_manager

        self.groups = [{"group_description": "Initial group of adventurers.", "roles": []}]
        self.context_text = context_text
        self.on_text_submit_callback = lambda text: None

        self.focusable_buttons: list[Button] = []
        self.selected_index = 0

        self._rebuild_widgets()

    def _rebuild_widgets(self):
        self.widgets.clear()
        self.focusable_buttons.clear()

        main_vbox = VBox(x=1, y=1)

        context_width = self.console_width // 3
        context_box = DynamicTextBox(
            text=self.context_text, title="Scene Context", x=self.console_width - context_width - 1, y=1,
            max_width=context_width, max_height=self.console_height - 4
        )
        self.widgets.append(context_box)

        creation_panel_width = self.console_width - context_width - 4

        for i, group in enumerate(self.groups):
            group_label = Label(f"--- Group {i + 1}: {group['group_description']} ---", fg=(255, 255, 0))
            main_vbox.add_child(group_label)

            if not group['roles']:
                main_vbox.add_child(Label("  (No roles added yet)"))
            for role_text in group['roles']:
                main_vbox.add_child(Label(f"  - {role_text}"))

            add_role_btn = Button(f"Add Role", on_click=lambda idx=i: self._prompt_for_add_role(idx))
            self.focusable_buttons.append(add_role_btn)
            main_vbox.add_child(add_role_btn)

            edit_desc_btn = Button(f"Edit Description", on_click=lambda idx=i: self._prompt_for_edit_desc(idx))
            self.focusable_buttons.append(edit_desc_btn)
            main_vbox.add_child(edit_desc_btn)

            if len(self.groups) > 1:
                delete_group_btn = Button(f"Delete Group", on_click=lambda idx=i: self._delete_group(idx))
                self.focusable_buttons.append(delete_group_btn)
                main_vbox.add_child(delete_group_btn)

            main_vbox.add_child(Label(""))

        add_group_btn = Button("Add New Group", on_click=self._add_new_group)
        self.focusable_buttons.append(add_group_btn)
        main_vbox.add_child(add_group_btn)
        main_vbox.add_child(Label(""))

        submit_btn = Button("Done - Submit Roles", on_click=self._submit_form, fg=(100, 255, 100))
        self.focusable_buttons.append(submit_btn)
        main_vbox.add_child(submit_btn)

        self.frame = Frame(main_vbox, title="Player Control: Define Lead Roles", width=creation_panel_width,
                           height=self.console_height - 2)
        self.widgets.insert(0, self.frame)
        self._update_selection()

    def _update_selection(self):
        for i, btn in enumerate(self.focusable_buttons):
            btn.selected = (i == self.selected_index)

    def _submit_form(self):
        valid_groups = [g for g in self.groups if g['roles']]
        if not valid_groups:
            self.on_submit(None)
            return
        final_json_obj = {"groups": valid_groups}
        self.on_submit(json.dumps(final_json_obj, indent=2))

    def _add_new_group(self):
        self.groups.append({"group_description": "A new group.", "roles": []})
        self._rebuild_widgets()

    def _delete_group(self, index: int):
        if 0 <= index < len(self.groups):
            self.groups.pop(index)
            self._rebuild_widgets()

    def _prompt_for_add_role(self, group_index: int):
        self.on_text_submit_callback = lambda text: self._submit_new_role(text, group_index)
        handler = TextInputHandler(prompt="Enter a new role description (e.g., 'a grizzled veteran')",
                                   width=self.console_width - 2, prefix="> ")
        self.parent_ui_manager.active_view = TextInputView(handler, self.on_text_submit_callback)

    def _submit_new_role(self, text: str, group_index: int):
        if text and 0 <= group_index < len(self.groups):
            self.groups[group_index]['roles'].append(text)
        self._rebuild_widgets()
        self.parent_ui_manager.active_view = self

    def _prompt_for_edit_desc(self, group_index: int):
        self.on_text_submit_callback = lambda text: self._submit_new_description(text, group_index)
        initial_text = self.groups[group_index]['group_description']
        handler = TextInputHandler(prompt="Enter the group's shared context/description",
                                   width=self.console_width - 2, prefix="> ", initial_text=initial_text)
        self.parent_ui_manager.active_view = TextInputView(handler, self.on_text_submit_callback)

    def _submit_new_description(self, text: str, group_index: int):
        if text and 0 <= group_index < len(self.groups):
            self.groups[group_index]['group_description'] = text
        self._rebuild_widgets()
        self.parent_ui_manager.active_view = self

    def handle_event(self, event: tcod.event.Event):
        for widget in self.widgets:
            widget.handle_event(event)

        if isinstance(event, tcod.event.KeyDown):
            key = event.sym
            if not self.focusable_buttons:
                if key == tcod.event.KeySym.ESCAPE:
                    self.on_submit(None)
                return

            if key == tcod.event.KeySym.UP:
                self.selected_index = (self.selected_index - 1) % len(self.focusable_buttons)
                self._update_selection()
            elif key == tcod.event.KeySym.DOWN:
                self.selected_index = (self.selected_index + 1) % len(self.focusable_buttons)
                self._update_selection()
            elif key in (tcod.event.KeySym.RETURN, tcod.event.KeySym.KP_ENTER):
                self.focusable_buttons[self.selected_index].on_click()
            elif key == tcod.event.KeySym.ESCAPE:
                self.on_submit(None)