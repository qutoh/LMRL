# /core/ui/ui_data_provider.py

from ..common import file_io
from ..common.config_loader import config

def get_available_worlds() -> list[dict]:
    """Scans the data directory and returns a list of world data dictionaries."""
    worlds_data = []
    worlds_path = file_io.join_path(file_io.PROJECT_ROOT, 'data', 'worlds')
    if not file_io.path_exists(worlds_path): return []
    for world_name in file_io.list_directory_contents(worlds_path):
        world_json_path = file_io.join_path(worlds_path, world_name, 'world.json')
        if file_io.path_exists(world_json_path):
            world_data = file_io.read_json(world_json_path)
            if world_data:
                root_key = next(iter(world_data), None)
                theme = world_data[root_key].get('theme', 'No theme description found.') if root_key else 'No theme description found.'
                worlds_data.append({'name': world_name, 'theme': theme})
    return worlds_data

def get_available_scenes() -> list[dict]:
    """Returns a de-duplicated list of scenes for the currently loaded world."""
    world_scenes = config.generated_scenes
    seen_prompts = set()
    combined = []
    for scene in world_scenes:
        prompt = scene.get('scene_prompt')
        if prompt and prompt not in seen_prompts:
            combined.append(scene)
            seen_prompts.add(prompt)
    return combined

def get_anachronisms(current_world_name: str) -> list[dict]:
    """
    Returns a list of scenes that are NOT from the current world, to be used
    as 'anachronistic' or out-of-context starting points.
    """
    anachronisms = []
    current_world_prompts = {s.get('scene_prompt') for s in get_available_scenes()}
    seen_prompts = set(current_world_prompts)

    # 1. Add global default scenes
    for scene in config.scene:
        prompt = scene.get('scene_prompt')
        if prompt and prompt not in seen_prompts:
            anachronisms.append({"scene_prompt": prompt})  # No source location
            seen_prompts.add(prompt)

    # 2. Add scenes from all other worlds
    worlds_path = file_io.join_path(file_io.PROJECT_ROOT, 'data', 'worlds')
    for world_name in file_io.list_directory_contents(worlds_path):
        if world_name == current_world_name:
            continue
        scenes_path = file_io.join_path(worlds_path, world_name, 'generated_scenes.json')
        if file_io.path_exists(scenes_path):
            other_world_scenes = file_io.read_json(scenes_path, default=[])
            for scene in other_world_scenes:
                prompt = scene.get('scene_prompt')
                if prompt and prompt not in seen_prompts:
                    anachronisms.append({"scene_prompt": prompt})  # No source location
                    seen_prompts.add(prompt)
    return anachronisms