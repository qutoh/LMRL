# /core/worldgen/procgen_manager.py

import re
from core.common.config_loader import config
from core.common.game_state import GenerationState
from .map_architect import MapArchitect
from .map_architect_v2 import MapArchitectV2
from .map_architect_v3 import MapArchitectV3
from ..common import utils, file_io
from ..ui.ui_messages import AddEventLogMessage
from .v3_components.v3_llm import V3_LLM


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

        # --- Exterior Tile Selection ---
        llm = V3_LLM(self.engine)
        tiles = {name: data for name, data in self.engine.config.tile_types.items() if name != "VOID_SPACE"}
        tile_options_str = "\n".join(
            f"- `{name}`: {data.get('description', 'No description.')}" for name, data in tiles.items())
        raw_llm_choice = llm.choose_exterior_tile(scene_prompt, tile_options_str)

        chosen_tile = None
        valid_tile_keys = tiles.keys()

        for key in valid_tile_keys:
            if key.lower() == raw_llm_choice.strip().lower():
                chosen_tile = key
                break

        if not chosen_tile:
            for key in valid_tile_keys:
                if re.search(rf"\b{re.escape(key)}\b", raw_llm_choice, re.IGNORECASE):
                    chosen_tile = key
                    utils.log_message('debug',
                                      f"[PEG Setup] Extracted tile keyword '{chosen_tile}' from chatty response.")
                    break

        if not chosen_tile or chosen_tile == "VOID_SPACE":
            utils.log_message('debug', "[PEG Setup] Invalid or no tile chosen. Triggering new tile generation.")
            self.engine.render_queue.put(
                AddEventLogMessage("That's an interesting idea... creating a new ground tile for this scene..."))

            new_tile_data = llm.create_new_tile_type(scene_prompt)
            if new_tile_data:
                world_tiles_path = file_io.join_path(self.engine.config.data_dir, 'worlds', self.engine.world_name,
                                                     'generated_tiles.json')
                existing_world_tiles = file_io.read_json(world_tiles_path, default={})
                existing_world_tiles.update(new_tile_data)
                file_io.write_json(world_tiles_path, existing_world_tiles)

                self.engine.config.tile_types.update(new_tile_data)
                self.engine.config.tile_type_map = {name: i for i, name in
                                                    enumerate(self.engine.config.tile_types.keys())}
                self.engine.config.tile_type_map_reverse = {i: name for name, i in
                                                            self.engine.config.tile_type_map.items()}

                chosen_tile = list(new_tile_data.keys())[0]
                utils.log_message('debug', f"[PEG Setup] Successfully created and selected new tile: {chosen_tile}")
                self.engine.render_queue.put(
                    AddEventLogMessage(f"New tile '{chosen_tile}' created for {self.engine.world_name}!",
                                       (150, 255, 150)))
            else:
                chosen_tile = "DEFAULT_FLOOR"
                utils.log_message('debug', "[PEG Setup] New tile generation failed. Falling back to default.")

        utils.log_message('debug', f"[PEG] LLM chose '{chosen_tile}' as the exterior tile.")

        # --- Algorithm Selection & Execution ---
        algorithm = config.settings.get("PEG_RECONCILIATION_METHOD", "CONVERSATIONAL").strip().upper()
        utils.log_message('debug', f"[PEG] Using algorithm: {algorithm}")

        def default_callback(state):
            pass

        callback = ui_callback if ui_callback else default_callback

        state = None
        if algorithm == "PARTITIONING":
            architect_v2 = MapArchitectV2(self.engine, game_map, getattr(self.engine, 'world_theme', 'fantasy'),
                                          scene_prompt)
            state = architect_v2.generate_layout(callback)
        elif algorithm == "ITERATIVE_PLACEMENT":
            architect_v3 = MapArchitectV3(self.engine, game_map, getattr(self.engine, 'world_theme', 'fantasy'),
                                          scene_prompt)
            gen = architect_v3.generate_layout_in_steps(callback)
            for _, final_state_obj in gen:
                state = final_state_obj
        else:
            architect = MapArchitect(self.engine, game_map, getattr(self.engine, 'world_theme', 'fantasy'),
                                     scene_prompt)
            state = architect.generate_layout(callback)

        if state:
            state.exterior_tile_type = chosen_tile

        utils.log_message('debug', f"--- [SYSTEM] PEG process ({algorithm}) complete. ---")
        return state