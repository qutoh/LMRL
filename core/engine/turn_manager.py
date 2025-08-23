# /core/engine/turn_manager.py

import re
import random

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

    def prepare_turn_queue(self, all_characters: list) -> list:
        """
        Hydrates nearby NPCs, culls distant ones, and shuffles the remaining
        characters to create the turn queue for the current cycle.
        """
        # Note to dev: This function should be called by the main StoryEngine loop
        # at the start of each cycle, before iterating through turns.
        # e.g., turn_queue = self.turn_manager.prepare_turn_queue(turn_order)

        self._hydrate_proximate_npcs()

        # Get an updated list of all characters after potential hydration
        current_roster = roster_manager.get_all_characters(self.engine)
        npcs_to_cull = position_manager.get_npcs_to_cull(self.engine, self.game_state)

        if npcs_to_cull:
            utils.log_message('debug', f"[TURN MANAGER] Culling distant NPCs from turn order: {', '.join(npcs_to_cull)}")
            turn_queue = [char for char in current_roster if char['name'] not in npcs_to_cull]
        else:
            turn_queue = current_roster[:]

        random.shuffle(turn_queue)
        return turn_queue

    def _hydrate_proximate_npcs(self):
        """
        Checks for dehydrated NPCs and generates them if they are determined
        to be in a location visible to any lead character.
        """
        if not self.engine.dehydrated_npcs:
            return

        visible_tags = position_manager._get_nearby_feature_tags_for_leads(self.engine, self.game_state)
        if not visible_tags:
            return

        visible_locations = {
            data.get('name', tag) for tag, data in self.engine.generation_state.placed_features.items() if tag in visible_tags
        }
        visible_locations_list_str = "\n".join(f"- {name}" for name in sorted(list(visible_locations)))

        hydrated_any = False
        # Iterate over a copy, as we may modify the original list
        for concept in self.engine.dehydrated_npcs[:]:
            concept_dict = {}
            if isinstance(concept, str):
                parts = concept.split(' - ', 1)
                concept_dict['name'] = parts[0].strip()
                concept_dict['description'] = parts[1].strip() if len(parts) > 1 else "An individual."
            else:
                concept_dict = concept

            char_name = concept_dict.get('name', 'Unknown')
            char_desc = concept_dict.get('description', concept_dict.get('source_sentence', 'An individual.'))

            kwargs = {
                "character_name": char_name,
                "character_description": char_desc,
                "visible_locations_list": visible_locations_list_str
            }
            raw_response = llm_api.execute_task(self.engine, self.engine.config.agents['DIRECTOR'],
                                                'DIRECTOR_SHOULD_SPAWN_INHABITANT', [], task_prompt_kwargs=kwargs)
            command = command_parser.parse_structured_command(self.engine, raw_response, 'DIRECTOR',
                                                              fallback_task_key='CH_FIX_SPAWN_INHABITANT_JSON')

            if command and command.get('spawn'):
                location_name = command.get('location_name')
                utils.log_message('debug', f"[TURN MANAGER] Hydrating NPC '{char_name}' in/near '{location_name}'.")

                location_context = { "world_theme": self.engine.world_theme, "scene_prompt": self.engine.scene_prompt }
                full_profile, role = roster_manager.get_or_create_character_from_concept(self.engine, concept, location_context)

                if full_profile and role:
                    roster_manager.decorate_and_add_character(self.engine, full_profile, role)
                    roster_manager.spawn_single_entity(self.engine, self.game_state, full_profile['name'], location_name)
                    self.engine.dehydrated_npcs.remove(concept)
                    hydrated_any = True

        if hydrated_any:
            # Re-initialize states for all characters to account for new arrivals
            self.engine.character_manager.initialize_all_character_states(self.engine)


    def _execute_ai_turn(self, current_actor, unacted_roles):
        """
        Executes a full turn for an AI-controlled character. This is now the
        primary location where full prompt context is built.
        """
        self.game_state.reset_entity_turn_stats(current_actor['name'])

        if current_actor.get('role_type') == 'lead':
            position_manager.log_character_perspective(self.engine, self.game_state, current_actor['name'])

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
                    position_manager.handle_turn_movement(self.engine, self.game_state, current_actor['name'], prose)

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