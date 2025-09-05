# /core/engine/game_modes/narrative_simulation_mode.py

import random
from core.common import utils
from core.components import roster_manager
from core.components import position_manager  # IMPORT ADDED


class NarrativeSimulationMode:
    def __init__(self, engine):
        self.engine = engine
        self.turn_manager = engine.turn_manager
        self.player_interface = engine.player_interface

    def _prepare_turn_queue(self) -> list:
        """
        Prepares the turn queue for the narrative simulation mode by hydrating,
        culling, and shuffling characters.
        """
        self.turn_manager.hydrate_proximate_npcs()

        # --- FULL CULLING LOGIC PRESERVED ---
        current_roster = roster_manager.get_all_characters(self.engine)
        npcs_to_cull = position_manager.get_npcs_to_cull(self.engine, self.engine.game_state)

        if npcs_to_cull:
            utils.log_message('debug',
                              f"[NARRATIVE MODE] Culling distant NPCs from turn order: {', '.join(npcs_to_cull)}")
            turn_queue = [char for char in current_roster if char['name'] not in npcs_to_cull]
        else:
            turn_queue = current_roster[:]
        # --- END OF PRESERVED LOGIC ---

        random.shuffle(turn_queue)
        return turn_queue

    def run_cycle(self):
        """
        Executes a full cycle of turns using the semi-random, narrative
        simulation method.
        """
        turn_queue = self._prepare_turn_queue()

        if not turn_queue:
            utils.log_message('debug', "No characters in the turn queue. Ending cycle.")
            # This return will be caught by the StoryEngine to break the main loop if needed.
            return

        while turn_queue:
            self.engine._check_for_interrupts()
            if self.engine.interrupted or self.engine.player_interrupted or self.engine.cast_manager_interrupted:
                break

            current_actor = turn_queue.pop(0)
            unacted_roles_in_queue = turn_queue[:]
            prose = ""
            messages = []

            # Step 1: Get Prose from AI or Player
            if current_actor.get("controlled_by") == "human":
                prose = self.player_interface.execute_turn(current_actor)
            else:
                prose, messages = self.turn_manager.generate_ai_prose(current_actor, unacted_roles_in_queue)

            # Step 2: Process the Prose to get results
            if prose:
                next_actor, _ = self.turn_manager.process_turn_results(
                    current_actor, prose, messages, unacted_roles_in_queue
                )
                self.engine.render_queue.put(self.engine.game_state)

                if next_actor:
                    utils.log_message('debug', f"Turn passed from {current_actor['name']} to {next_actor['name']}.")
                    # Modify the current queue directly for the hand-off
                    turn_queue = [r for r in turn_queue if r.get('name') != next_actor.get('name')]
                    turn_queue.insert(0, next_actor)