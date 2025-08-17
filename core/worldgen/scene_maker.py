# /core/worldgen/scene_maker.py

import random
from core.common import file_io, utils
from core.common.config_loader import config
from core.llm.llm_api import execute_task


class SceneMakerManager:
    def __init__(self, engine):
        self.engine = engine
        self.scene_maker_agent = config.agents.get('SCENE_MAKER')
        self.world_data = self.engine.config.world

    def _select_random_location(self) -> dict | None:
        if not self.world_data: return None
        all_locations = []

        def recurse_locations(node):
            all_locations.append(node)
            if "Children" in node and node["Children"]:
                for child_node in node["Children"].values():
                    recurse_locations(child_node)

        for root_node in self.world_data.values():
            recurse_locations(root_node)
        return random.choice(all_locations) if all_locations else None

    def _save_generated_scene(self, scene_prompt: str, location_data: dict, world_name: str):
        scenes_path = file_io.join_path(self.engine.config.data_dir, 'worlds', world_name, 'generated_scenes.json')
        all_scenes = file_io.read_json(scenes_path, default=[])
        new_scene = {
            "scene_prompt": scene_prompt,
            "source_location_name": location_data.get("Name"),
            "source_location": location_data
        }
        all_scenes.append(new_scene)
        file_io.write_json(scenes_path, all_scenes)

    def generate_scene(self, target_location: dict | None = None, world_name: str | None = None) -> dict:
        location = target_location or self._select_random_location()
        if not location: return {"scene_prompt": "A story begins."}
        utils.log_message('debug', f"[SCENE MAKER] Generating a new scene for location: '{location.get('Name')}'.")

        world_theme = getattr(self.engine, 'world_theme', 'A generic fantasy world.')

        prompt_kwargs = {
            "world_theme": world_theme,
            "location_name": location.get("Name", "Unnamed"),
            "location_type": location.get("Type", "Place"),
            "location_description": location.get("Description", "No description.")
        }
        scene_prompt = execute_task(self.engine, self.scene_maker_agent, 'GENERATE_NEW_SCENE', [],
                                    task_prompt_kwargs=prompt_kwargs)

        if scene_prompt and world_name:
            self._save_generated_scene(scene_prompt, location, world_name)
        elif scene_prompt and not world_name:
            utils.log_message('debug',
                              "[SCENE MAKER WARNING] Generated a scene but no world_name was provided. Scene will not be saved.")

        return {"scene_prompt": scene_prompt or "A story begins.", "source_location": location}