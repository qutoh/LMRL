# /core/ui/views/calibration_view.py

from collections import Counter
import tcod
from ..ui_framework import View, DynamicTextBox, EventLog


class CalibrationView(View):
    """A view to display real-time feedback during the LLM calibration process."""

    def __init__(self, console_width: int, console_height: int):
        super().__init__()
        self.console_width = console_width
        self.console_height = console_height

        self.status_box = DynamicTextBox(
            title="Calibration Status", text="Initializing...", x=2, y=1,
            max_width=console_width - 4, max_height=5
        )
        self.output_log = EventLog(max_lines=50)
        self.widgets = [self.status_box]

        self.current_phase = None
        self.top_p_responses = Counter()
        self.top_p_total_samples = 0
        self.top_p_valid_samples = 0

    def set_phase(self, phase_name: str, status_text: str):
        """Switches the view between 'top_p' and 'temperature' modes."""
        self.current_phase = phase_name
        self.status_box.set_text(status_text)
        if phase_name == 'top_p':
            self.top_p_responses.clear()
            self.top_p_total_samples = 0
            self.top_p_valid_samples = 0
        elif phase_name == 'temperature':
            self.output_log.messages.clear()

    def update_data(self, data: dict):
        """Processes incoming data from the calibration thread."""
        if data.get('type') == 'top_p_sample':
            self.top_p_total_samples += 1
            if data.get('is_valid'):
                self.top_p_valid_samples += 1
                response_key = data.get('response', 'INVALID').strip()
                # Truncate for display
                if len(response_key) > self.console_width - 25:
                    response_key = response_key[:self.console_width - 28] + "..."
                self.top_p_responses[response_key] += 1

        elif data.get('type') == 'temp_sample':
            self.output_log.add_message(data.get('response', '...'), fg=(200, 200, 220))

    def _render_top_p_histogram(self, console: tcod.console.Console):
        """Draws the diversity histogram for the Top-P calibration phase."""
        y_start = self.status_box.y + self.status_box.height + 1
        x_start = 2
        max_bar_width = self.console_width - 25

        validity_rate = (self.top_p_valid_samples / self.top_p_total_samples) if self.top_p_total_samples > 0 else 0
        diversity_score = len(self.top_p_responses) / self.top_p_valid_samples if self.top_p_valid_samples > 0 else 0

        header = f"Samples: {self.top_p_total_samples} | Valid: {self.top_p_valid_samples} ({validity_rate:.0%}) | Unique Responses: {len(self.top_p_responses)} ({diversity_score:.0%})"
        console.print(x=x_start, y=y_start, string=header, fg=(255, 255, 0))

        y = y_start + 2
        max_count = max(self.top_p_responses.values()) if self.top_p_responses else 1

        sorted_responses = self.top_p_responses.most_common()

        for response_text, count in sorted_responses:
            if y >= self.console_height - 2:
                console.print(x=x_start, y=y, string="...", fg=(150, 150, 150))
                break

            bar_width = int((count / max_count) * max_bar_width) if max_count > 0 else 0

            # Draw the bar background
            for i in range(bar_width):
                console.print(x=x_start + i, y=y, string=" ", bg=(50, 80, 50))

            # Draw the bar text on top
            display_text = f"[{count}] {response_text}"
            console.print(x=x_start, y=y, string=display_text, fg=(200, 255, 200))
            y += 1

    def _render_temperature_log(self, console: tcod.console.Console):
        """Renders the event log for the Temperature calibration phase."""
        log_y = self.status_box.y + self.status_box.height + 1
        log_height = self.console_height - log_y - 2
        self.output_log.render(console, x=2, y=log_y, width=self.console_width - 4, height=log_height)

    def render(self, console: tcod.console.Console):
        super().render(console)  # Renders the status box
        if self.current_phase == 'top_p':
            self._render_top_p_histogram(console)
        elif self.current_phase == 'temperature':
            self._render_temperature_log(console)