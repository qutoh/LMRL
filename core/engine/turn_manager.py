# /core/engine/turn_manager.py

import re

from ..common import command_parser
from ..common import utils
from ..components import position_manager
from ..components import roster_manager
from ..components import utility_tasks
from ..llm import llm_api


class TurnManager:
    """
    A stateful manager that handles the entire process of executing a single
    character's turn, whether AI or human-controlled.
    """

    def __init__(self, engine, game_state, player_interface):
        self.engine = engine
        self.game_state = game_state
        self.player_interface = player_interface
        self.movement_classifiers = {
            "DASH": r"\b(dash|dashes|dashed|rush|rushes|rushed|charge|charges|charged|sprint|sprints|sprinted|bolt|bolts|bolted)\b",
            "SNEAK": r"\b(sneak|sneaks|snuck|creep|creeps|crept|stalk|stalks|stalked|sidle|sidles|sidled|tiptoe|tiptoes|tiptoed)\b",
            "CRAWL": r"\b(crawl|crawls|crawled)\b",
            "CLIMB": r"\b(climb|climbs|climbed|ascend|ascends|ascended|scrambles up)\b",
            "SWIM": r"\b(swim|swims|swam|wade|wades|waded)\b",
            "JUMP": r"\b(jump|jumps|jumped|leap|leaps|leapt)\b",
            "FALLBACK": r"\b(retreat|retreats|retreated|withdraw|withdraws|withdrew|backs away|backed away|fall back|falls back|fell back)\b",
            "RUN": r"\b(run|runs|ran)\b",
            "WALK": r"\b(walk|walks|walked|go|goes|went|move|moves|moved|proceed|proceeds|proceeded|advance|advances|advanced|step|steps|stepped|sets off|set off|position|positions|positioned)\b",
        }

    def _classify_movement_mode(self, text: str) -> str | None:
        """Iterates through classifiers to find the dominant movement mode."""
        for mode, pattern in self.movement_classifiers.items():
            if re.search(pattern, text, re.IGNORECASE):
                return mode
        return None

    def _execute_ai_turn(self, current_actor, unacted_roles):
        """
        Executes a full turn for an AI-controlled character. This is now the
        primary location where full prompt context is built.
        """
        self.game_state.reset_entity_turn_stats(current_actor['name'])

        # Build the full narrative context for the actor.
        memories = self.engine.memory_manager.retrieve_memories(self.engine.dialogue_log, self.engine.summaries)
        local_context_str = position_manager.get_local_context_for_character(self.engine, self.game_state,
                                                                             current_actor['name'])
        builder = utils.PromptBuilder(self.engine, current_actor) \
            .add_world_theme() \
            .add_summary(self.engine.summaries) \
            .add_local_context(local_context_str) \
            .add_physical_context()

        if memories: builder.add_long_term_memory(memories)
        builder.add_dialogue_log(self.engine.dialogue_log)
        messages = builder.build()

        self.engine.render_queue.put(('ADD_EVENT_LOG', f"{current_actor['name']} is thinking...", (150, 150, 150)))

        next_actor = None
        events = {}

        # A character's main turn is a generic task; the prompt is already built.
        raw_llm_response = llm_api.execute_task(
            self.engine,
            current_actor,
            'GENERIC_TURN',
            messages
        )
        prose = command_parser.post_process_llm_response(self.engine, current_actor, raw_llm_response,
                                                         is_dm_role=current_actor.get('role_type') == 'dm')

        if prose:
            dialogue_entry = {
                "speaker": current_actor['name'],
                "content": prose,
                "timestamp": self.engine.current_game_time.isoformat()
            }

            next_actor_name, events = self.engine.prometheus_manager.analyze_and_dispatch(
                prose, dialogue_entry, unacted_roles, messages
            )

            if events.get('refusal'):
                # The turn was handled by the refusal system. Overwrite prose and next_actor.
                prometheus_result = self.engine.prometheus_manager._call_handle_refusal(dialogue_entry, messages,
                                                                                        unacted_roles=unacted_roles)
                prose = prometheus_result.get('new_prose', '')
                next_actor_name = prometheus_result.get('next_actor_name')
                events = prometheus_result.get('events', {})

            if next_actor_name and isinstance(next_actor_name, str):
                next_actor = roster_manager.find_character(self.engine, next_actor_name)

            if prose:
                dialogue_entry["content"] = prose  # Update with new prose if reprompted
                utils.log_message('story', prose)
                self.engine.dialogue_log.append(dialogue_entry)

                duration_seconds = events.get('time_passed_seconds',
                                              utility_tasks.get_duration_for_action(self.engine, prose))
                self.engine.advance_time(duration_seconds)

                if current_actor.get('is_positional'):
                    if movement_mode := self._classify_movement_mode(prose):
                        position_manager.process_movement_intent(self.engine, self.game_state, current_actor['name'],
                                                                 prose, movement_mode)

                if not events.get('refusal'):
                    self.engine.character_manager.update_character_state(self.engine, current_actor, dialogue_entry, events)

        return next_actor

    def execute_turn_for(self, current_actor, unacted_roles):
        """Public method to execute a turn for any actor."""
        next_actor = None
        events = {}

        if current_actor.get("controlled_by") == "human":
            prose = self.player_interface.execute_turn(current_actor)
        else:
            return self._execute_ai_turn(current_actor, unacted_roles)

        if prose:
            dialogue_entry = {"speaker": current_actor['name'], "content": prose,
                              "timestamp": self.engine.current_game_time.isoformat()}
            self.engine.dialogue_log.append(dialogue_entry)

            next_actor_name, events = self.engine.prometheus_manager.analyze_and_dispatch(prose, dialogue_entry,
                                                                                          unacted_roles, [])
            duration_seconds = events.get('time_passed_seconds',
                                          utility_tasks.get_duration_for_action(self.engine, prose))
            self.engine.advance_time(duration_seconds)

            if not events.get('refusal'):
                self.engine.character_manager.update_character_state(self.engine, current_actor, dialogue_entry, events)

            if next_actor_name and isinstance(next_actor_name, str):
                next_actor = roster_manager.find_character(self.engine, next_actor_name)

        return next_actor