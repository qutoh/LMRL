# /core/procgen_manager.py

from ..common import file_io, utils
from core.common.config_loader import config
from core.llm.llm_api import execute_task
from core.common.game_state import GenerationState
from .map_architect import MapArchitect
from .map_architect_v2 import MapArchitectV2
from .map_architect_v3 import MapArchitectV3


class ProcGenManager:
    """
    A high-level manager that orchestrates the procedural generation process
    by delegating to a MapArchitect.
    """

    def __init__(self, engine):
        self.engine = engine
        self.level_generator = config.agents['LEVEL_GENERATOR']

    def generate(self, scene_prompt: str, game_map, ui_callback: callable = None) -> GenerationState:
        utils.log_message('debug', "\n--- [SYSTEM] Starting Procedural Environment Generation (PEG) ---")

        algorithm = config.settings.get("PEG_RECONCILIATION_METHOD", "CONVERSATIONAL").strip().upper()
        utils.log_message('debug', f"[PEG] Using algorithm: {algorithm}")

        def default_callback(state):
            pass
        callback = ui_callback if ui_callback else default_callback

        state = None  # Initialize state

        if algorithm == "PARTITIONING":
            architect_v2 = MapArchitectV2(self.engine, game_map, getattr(self.engine, 'world_theme', 'fantasy'),
                                          scene_prompt)
            state = architect_v2.generate_layout(callback)
        elif algorithm == "ITERATIVE_PLACEMENT":
            architect_v3 = MapArchitectV3(self.engine, game_map, getattr(self.engine, 'world_theme', 'fantasy'),
                                          scene_prompt)
            # The non-test path exhausts the generator immediately to get the final state.
            gen = architect_v3.generate_layout_in_steps(callback)
            for _, final_state_obj in gen:
                state = final_state_obj
        else:  # Default to Conversational (V1)
            architect = MapArchitect(self.engine, game_map, getattr(self.engine, 'world_theme', 'fantasy'),
                                     scene_prompt)
            state = architect.generate_layout(callback)

        utils.log_message('debug', f"--- [SYSTEM] PEG process ({algorithm}) complete. ---")
        return state