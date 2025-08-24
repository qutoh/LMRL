# /core/engine/turn_manager.py

from ..common import command_parser
from ..common import utils
from ..components import position_manager
from ..components import roster_manager
from ..llm import llm_api


class TurnManager:
    """
    A library of stateful helper functions for executing character turns and
    processing their outcomes. It is orchestrated by a Game Mode.
    """

    def __init__(self, engine, game_state, player_interface):
        self.engine = engine
        self.game_state = game_state
        self.player_interface = player_interface

    def hydrate_proximate_npcs(self):
        if not self.engine.dehydrated_npcs:
            return
        visible_tags = position_manager._get_nearby_feature_tags_for_leads(self.engine, self.game_state)
        if not visible_tags:
            return
        visible_locations = {
            data.get('name', tag) for tag, data in self.engine.generation_state.placed_features.items() if
            tag in visible_tags
        }
        visible_locations_list_str = "\n".join(f"- {name}" for name in sorted(list(visible_locations)))
        hydrated_any = False
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
                location_context = {"world_theme": self.engine.world_theme, "scene_prompt": self.engine.scene_prompt}
                full_profile, role = roster_manager.get_or_create_character_from_concept(self.engine, concept,
                                                                                         location_context)
                if full_profile and role:
                    roster_manager.decorate_and_add_character(self.engine, full_profile, role)
                    roster_manager.spawn_single_entity(self.engine, self.game_state, full_profile['name'],
                                                       location_name)
                    self.engine.dehydrated_npcs.remove(concept)
                    hydrated_any = True
        if hydrated_any:
            self.engine.character_manager.initialize_all_character_states(self.engine)

    def generate_ai_prose(self, current_actor, unacted_roles) -> tuple[str, list]:
        """
        Generates the prose for an AI-controlled character's turn.
        Returns the prose string and the message context used to generate it.
        """
        self.game_state.reset_entity_turn_stats(current_actor['name'])

        if current_actor.get('role_type') == 'lead':
            position_manager.log_character_perspective(self.engine, self.game_state, current_actor['name'])

        memories = self.engine.memory_manager.retrieve_memories(self.engine.dialogue_log, self.engine.summaries)
        local_context_str = position_manager.get_local_context_for_character(self.engine, self.game_state,
                                                                             current_actor['name'])

        builder = utils.PromptBuilder(self.engine, current_actor) \
            .add_world_theme() \
            .add_scene_prompt() \
            .add_summary(self.engine.summaries) \
            .add_local_context(local_context_str) \
            .add_physical_context()

        if memories: builder.add_long_term_memory(memories)
        builder.add_dialogue_log(self.engine.dialogue_log)
        messages = builder.build()

        self.engine.render_queue.put(('ADD_EVENT_LOG', f"{current_actor['name']} is thinking...", (150, 150, 150)))

        task_key = 'DM_TURN' if current_actor.get('role_type') == 'dm' else 'CHARACTER_TURN'

        raw_llm_response = llm_api.execute_task(
            self.engine,
            current_actor,
            task_key,
            messages
        )

        prose = command_parser.post_process_llm_response(self.engine, current_actor, raw_llm_response,
                                                         is_dm_role=current_actor.get('role_type') == 'dm')
        return prose, messages

    def process_turn_results(self, current_actor, prose: str, messages: list, unacted_roles: list) -> tuple[
        dict | None, str]:
        dialogue_entry = {"speaker": current_actor['name'], "content": prose,
                          "timestamp": self.engine.current_game_time.isoformat()}
        next_actor_name, events = self.engine.prometheus_manager.analyze_and_dispatch(
            prose, dialogue_entry, unacted_roles, messages)
        if events.get('refusal'):
            reprompt_result = self.engine.prometheus_manager._call_handle_refusal(dialogue_entry, messages,
                                                                                  unacted_roles=unacted_roles)
            prose = reprompt_result.get('new_prose', '')
            next_actor_name = reprompt_result.get('next_actor_name')
            events = reprompt_result.get('events', {})
        elif events.get('role_break'):
            reprompt_result = self.engine.prometheus_manager._call_handle_role_break(dialogue_entry,
                                                                                     unacted_roles=unacted_roles)
            prose = reprompt_result.get('new_prose', '')
            next_actor_name = reprompt_result.get('next_actor_name')
            events = reprompt_result.get('events', {})
        if prose:
            utils.log_message('story', f"({current_actor['name']}) {prose}")
            dialogue_entry["content"] = prose
            self.engine.dialogue_log.append(dialogue_entry)
            self.engine.advance_time(events.get('time_passed_seconds', 0))
            if current_actor.get('is_positional'):
                position_manager.handle_turn_movement(self.engine, self.game_state, current_actor['name'], prose)
            self.engine.character_manager.update_character_state(self.engine, current_actor, dialogue_entry, events)
            if next_actor_name and isinstance(next_actor_name, str):
                next_actor = roster_manager.find_character(self.engine, next_actor_name)
                return next_actor, prose
        return None, prose