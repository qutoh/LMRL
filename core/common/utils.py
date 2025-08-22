# /core/common/utils.py

import re
from datetime import datetime, timedelta

import tiktoken

from .config_loader import config
from .localization import loc
import numpy as np

LOG_HIERARCHY = {'game': 0, 'story': 1, 'debug': 2, 'full': 3}


def format_text_with_paragraph_breaks(text: str, sentences_per_paragraph: int = 3) -> str:
    """
    Formats a block of text by inserting paragraph breaks every N sentences.
    Also ensures that literal '\\n' from JSON are converted to newlines.
    """
    if not text or not isinstance(text, str):
        return ""

    # First, replace literal '\n' with actual newlines
    processed_text = text.replace('\\n', '\n')

    # Split into sentences using a regex that looks for punctuation followed by space
    sentences = re.split(r'(?<=[.?!])\s+', processed_text)
    if not sentences:
        return processed_text

    formatted_parts = []
    for i in range(0, len(sentences), sentences_per_paragraph):
        paragraph = ' '.join(sentences[i:i + sentences_per_paragraph]).strip()
        if paragraph:
            formatted_parts.append(paragraph)

    return "\n\n".join(formatted_parts)


class _Logger:
    """A simple logger service that can be configured to push UI updates."""

    def __init__(self):
        self.render_queue = None

    def configure(self, render_queue=None):
        """Injects the render queue for UI communication."""
        self.render_queue = render_queue

    def log(self, level, message, **kwargs):
        """Prints to console and pushes 'game' level messages to the UI queue."""
        log_level_setting = config.settings.get('LOG_LEVEL', 'debug')
        setting_val = LOG_HIERARCHY.get(log_level_setting, 2)

        # Always send 'game' logs to the UI widget
        if level == 'game' and self.render_queue:
            try:
                self.render_queue.put_nowait(('GAME_LOG_UPDATE', message))
            except Exception:
                pass

        # Determine if the message should be printed to the console
        should_print = False
        if level == 'story':
            # Story messages print on all active log levels, as they are the core narrative.
            should_print = True
        elif LOG_HIERARCHY.get(level, 99) <= setting_val:
            # Other messages (game, debug, full) follow the hierarchy.
            should_print = True

        if should_print:
            # Use special formatting for story messages for readability.
            if level == 'story':
                print(message, end=" ", **kwargs)
            else:
                print(message, **kwargs)


_logger_instance = _Logger()


def configure_logger(render_queue=None):
    """Public function to configure the logger service."""
    _logger_instance.configure(render_queue)


def log_message(level, message, **kwargs):
    """A wrapper that directs all logging calls to the singleton logger instance."""
    _logger_instance.log(level, message, **kwargs)


def clean_json_from_llm(raw_text: str) -> str | None:
    """
    Extracts a JSON object from a string, even if it's embedded in markdown.
    """
    if not raw_text:
        return None
    markdown_match = re.search(r"```(?:json)?\s*\n(.*?)\n```", raw_text, re.DOTALL)
    if markdown_match:
        return markdown_match.group(1).strip()
    brace_match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if brace_match:
        return brace_match.group(0)
    return None


def serialize_character_for_prompt(character: dict) -> str:
    """
    Intelligently serializes a character dictionary into a readable string
    suitable for an LLM system prompt.
    """
    if instructions := character.get('instructions'):
        if isinstance(instructions, str):
            return instructions

    if persona := character.get('persona'):
        if isinstance(persona, str):
            return persona

    lines = []
    if name := character.get('name'):
        lines.append(f"You are {name}.")
    if desc := character.get('description'):
        lines.append(f"Your description is: \"{desc}\"")

    for key, value in sorted(character.items()):
        if key in ['name', 'description', 'model', 'temperature', 'scaling_factor', 'role_type', 'instructions',
                   'persona'] or key.startswith('is_'):
            continue
        if isinstance(value, list):
            lines.append(f"\nYour {key} traits are:")
            lines.extend([f"- {item}" for item in value])
        elif isinstance(value, str):
            lines.append(f"\nYour {key} is: {value}")
    return "\n".join(lines)


def format_timedelta_natural_language(delta: timedelta) -> str:
    """
    Converts a timedelta object into a human-readable string like '5 minutes, 10 seconds ago'
    with up to two levels of precision.
    """
    seconds = int(delta.total_seconds())

    def _plural(n, unit):
        return f"{n} {unit}{'s' if n > 1 else ''}"

    if seconds < 1: return "just now"
    if seconds < 60: return f"{_plural(seconds, 'second')} ago"

    minutes = seconds // 60
    if minutes < 60:
        rem_seconds = seconds % 60
        return f"{_plural(minutes, 'minute')}, {_plural(rem_seconds, 'second')} ago" if rem_seconds else f"{_plural(minutes, 'minute')} ago"

    hours = minutes // 60
    if hours < 24:
        rem_minutes = minutes % 60
        return f"{_plural(hours, 'hour')}, {_plural(rem_minutes, 'minute')} ago" if rem_minutes else f"{_plural(hours, 'hour')} ago"

    days = hours // 24
    if days < 7:
        rem_hours = hours % 24
        return f"{_plural(days, 'day')}, {_plural(rem_hours, 'hour')} ago" if rem_hours else f"{_plural(days, 'day')} ago"

    weeks = days // 7
    if weeks < 4:
        rem_days = days % 7
        return f"{_plural(weeks, 'week')}, {_plural(rem_days, 'day')} ago" if rem_days else f"{_plural(weeks, 'week')} ago"

    months = days // 30
    if months < 12:
        rem_weeks = (days % 30) // 7
        return f"{_plural(months, 'month')}, {_plural(rem_weeks, 'week')} ago" if rem_weeks else f"{_plural(months, 'month')} ago"

    years = days // 365
    rem_months = (days % 365) // 30
    return f"{_plural(years, 'year')}, {_plural(rem_months, 'month')} ago" if rem_months else f"{_plural(years, 'year')} ago"


class PromptBuilder:
    """
    A class to cleanly assemble the message history for an LLM prompt.
    """

    def __init__(self, engine, character):
        self.engine = engine
        self.character = character
        self.message_channels = []

    def add_world_theme(self):
        if self.engine.world_theme:
            formatted_theme = format_text_with_paragraph_breaks(self.engine.world_theme)
            theme_text = f"--- WORLD THEME ---\n{formatted_theme}"
            self.message_channels.append([{"role": "user", "content": theme_text}])
        return self

    def add_summary(self, summaries):
        if summaries:
            formatted_summary = format_text_with_paragraph_breaks(summaries[-1])
            summary_text = loc('prompt_summary_context') + formatted_summary
            self.message_channels.append([{"role": "user", "content": summary_text}])
        return self

    def add_local_context(self, local_context_str: str):
        if local_context_str:
            context_text = f"{loc('prompt_local_context_header')}\n{local_context_str}"
            self.message_channels.append([{"role": "user", "content": context_text}])
        return self

    def add_physical_context(self):
        """Adds the character's physical description and equipment to the prompt."""
        if phys_desc := self.character.get('physical_description'):
            context_text = f"--- YOUR APPEARANCE & GEAR ---\n{phys_desc}\n"
            if equipment := self.character.get('equipment', {}).get('equipped'):
                item_lines = [f"- {item['name']}: {item['description']}" for item in equipment]
                context_text += "You are carrying/wearing:\n" + "\n".join(item_lines)

            self.message_channels.append([{"role": "user", "content": context_text.strip()}])
        return self

    def add_long_term_memory(self, memories: list[dict]):
        if memories:
            formatted_lines = [
                f"({format_timedelta_natural_language(self.engine.current_game_time - datetime.fromisoformat(mem['timestamp']))}) {mem['text']}" if mem.get(
                    "timestamp") else mem['text'] for mem in memories]
            memory_text = f"{loc('prompt_memory_header')}\n" + "\n".join(
                formatted_lines) + f"\n{loc('prompt_memory_footer')}"
            self.message_channels.append([{"role": "user", "content": memory_text}])
        return self

    def add_dialogue_log(self, dialogue_log):
        if dialogue_log:
            dialogue_messages = [{"role": "user", "content": loc('prompt_dialogue_header')}]
            for entry in dialogue_log:
                role = 'assistant' if entry["speaker"] == self.character['name'] else 'user'
                content = f"({entry['speaker']}): {entry['content']}" if role == 'user' else entry["content"]
                dialogue_messages.append({"role": role, "content": content})
            dialogue_messages.append({"role": "user", "content": loc('prompt_dialogue_footer')})
            self.message_channels.append(dialogue_messages)
        return self

    def clear_dialogue(self):
        if self.message_channels and loc('prompt_dialogue_header') in self.message_channels[-1][0]['content']:
            self.message_channels.pop()
        return self

    def build(self) -> list:
        return [message for channel in self.message_channels for message in channel]

    def build_for_token_count(self) -> tuple[str, list]:
        persona = serialize_character_for_prompt(self.character)
        messages = self.build()
        return persona, messages


def count_tokens(text):
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


def get_text_from_messages(messages):
    return "\n".join(f"[{msg.get('role', '')}] {msg.get('content', '')}" for msg in messages)


def print_ascii_local_view(title: str, grid: np.ndarray, center_x: int, center_y: int, radius: int,
                           other_markers: dict = None) -> str:
    """
    Generates a string containing a simple ASCII representation of a local grid area for debugging.

    Args:
        title: The title to print above the grid.
        grid: A 2D numpy array where different integer values represent different object types.
        center_x: The x-coordinate of the center of the view.
        center_y: The y-coordinate of the center of the view.
        radius: The radius of the view (e.g., radius 2 creates a 5x5 view).
        other_markers: A dict of {(x, y): char} for additional markers.

    Returns:
        A formatted, multi-line string representing the local view.
    """
    if other_markers is None:
        other_markers = {}

    char_map = {
        0: '.',  # Empty space
        1: '#',  # Generic obstacle
        2: 'P',  # Parent feature
        3: 'T'  # Target feature
    }

    header = f"--- {title} at ({center_x}, {center_y}) ---"
    lines = [header]

    start_y, end_y = center_y - radius, center_y + radius
    start_x, end_x = center_x - radius, center_x + radius

    for y in range(start_y, end_y + 1):
        line = ""
        for x in range(start_x, end_x + 1):
            if x == center_x and y == center_y:
                line += "@"
                continue
            if (x, y) in other_markers:
                line += other_markers[(x, y)]
                continue

            if 0 <= y < grid.shape[0] and 0 <= x < grid.shape[1]:
                value = grid[y, x]
                line += char_map.get(value, '?')
            else:
                line += ' '  # Out of bounds
        lines.append(line)

    lines.append("-" * len(header))
    return "\n".join(lines)

def print_ascii_cost_grid(title: str, cost_grid_yx: np.ndarray, start_xy: tuple, end_xy: tuple) -> str:
    """
    Generates a string containing a simple ASCII representation of a float cost grid for debugging.
    Used to visualize what a pathfinder sees.
    Args:
        title: The title to print above the grid.
        cost_grid_yx: A 2D numpy array of floats with (height, width) shape, where np.inf is an obstacle.
        start_xy: The (x, y) start coordinate.
        end_xy: The (x, y) end coordinate.

    Returns:
        A formatted, multi-line string representing the cost grid.
    """


    header = f"--- {title} ---"
    lines = [header]
    height, width = cost_grid_yx.shape
    start_x, start_y = start_xy
    end_x, end_y = end_xy

    for y in range(height):
        line = ""
        for x in range(width):
            if x == start_x and y == start_y:
                line += "S"
            elif x == end_x and y == end_y:
                line += "E"
            else:
                # Corrected Indexing: Use [y, x] for the (height, width) array
                cost = cost_grid_yx[y, x]
                line += "#" if np.isinf(cost) else "."
        lines.append(line)

    lines.append("-" * len(header))
    return "\n".join(lines)


def get_chosen_name_from_response(response_text):
    if not response_text: return None
    match = re.search(rf"^\s*{config.settings['NEXT_TURN_KEYWORD']}\s*(.*)$", response_text,
                      re.MULTILINE | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


def is_valid_filename(filename):
    """Checks if a filename is valid for profile saving."""
    if not filename:
        return False
    # Disallow file system characters
    if re.search(r'[<>:"/\\|?*]', filename):
        return False
    # Enforce a reasonable length and word count
    if not (1 <= len(filename) <= 50):
        return False
    if not (1 <= len(filename.split()) <= 5):
        return False
    return True