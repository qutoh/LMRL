# /core/worldgen/world_components/world_generator.py

from core.common import file_io, utils
from core.common.config_loader import config
from .atlas_logic import AtlasLogic
from .content_generator import ContentGenerator


class WorldGenerator:
    """
    A service class responsible for the end-to-end process of creating a new world,
    including initial setup and autonomous exploration.
    """

    def __init__(self, engine, content_generator: ContentGenerator, atlas_logic: AtlasLogic):
        self.engine = engine
        self.content_generator = content_generator
        self.atlas_logic = atlas_logic
        self._skip_exploration = False

    def create_new_world_generator(self, theme: str):
        """A generator that yields after each step of world creation and exploration."""
        utils.log_message('game', f"[ATLAS] Atlas is forging a new world with theme: {theme}")
        world_name = self.content_generator.get_world_name(theme)
        utils.log_message('game', f"[ATLAS] Atlas has named this world: {world_name}")

        origin_location = self.content_generator.create_location(theme, [], {"Name": "The Cosmos"})
        if not origin_location:
            utils.log_message('game', "[ATLAS] ERROR: Failed to create origin location.")
            yield {'status': 'complete', 'world_name': None}
            return

        origin_location.pop("Relationship", None)
        origin_location['theme'] = theme
        utils.log_message('game', f"[ATLAS] The center of the world: {origin_location.get('Name')}")

        world_data = {origin_location['Name']: origin_location}
        world_dir = file_io.join_path(file_io.PROJECT_ROOT, 'data', 'worlds', world_name)
        file_io.create_directory(world_dir)
        world_filepath = file_io.join_path(world_dir, 'world.json')
        file_io.write_json(world_filepath, world_data)
        file_io.write_json(file_io.join_path(world_dir, 'levels.json'), {})
        file_io.write_json(file_io.join_path(world_dir, 'generated_scenes.json'), [])
        file_io.write_json(file_io.join_path(world_dir, 'casting_npcs.json'), [])
        file_io.write_json(file_io.join_path(world_dir, 'inhabitants.json'), [])

        yield {'status': 'update', 'world_data': world_data}

        exploration_steps = config.settings.get("ATLAS_AUTONOMOUS_EXPLORATION_STEPS", 0)
        if exploration_steps > 0:
            utils.log_message('game',
                              f"\n--- [ATLAS] Starting {exploration_steps}-step world exploration... ---")
            config.load_world_data(world_name)
            current_breadcrumb = [origin_location['Name']]
            visited_breadcrumbs = [current_breadcrumb]

            for i in range(exploration_steps):
                if self._skip_exploration:
                    utils.log_message('game', "[ATLAS] Exploration skipped by user.")
                    break

                current_node = file_io._find_node_by_breadcrumb(config.world, current_breadcrumb)
                if not current_node:
                    utils.log_message('debug',
                                      f"[ATLAS ERROR] Breadcrumb {current_breadcrumb} broke mid-exploration.")
                    break

                utils.log_message('game', f"\n[ATLAS EXPLORATION] Step {i + 1}/{exploration_steps}")
                breadcrumb, node, status = self.atlas_logic.run_exploration_step(
                    world_name, theme, current_breadcrumb, current_node, visited_breadcrumbs=visited_breadcrumbs
                )

                current_breadcrumb = breadcrumb
                if current_breadcrumb not in visited_breadcrumbs:
                    visited_breadcrumbs.append(current_breadcrumb)

                yield {'status': 'update', 'world_data': config.world}

                utils.log_message('game', f'Location: {current_node["Name"]}')
                utils.log_message('game', f'Description: {current_node["Description"]}')
                if status == "ACCEPT":
                    utils.log_message('game', "[ATLAS] Atlas just thinks this place is neat :3")
                    breadcrumb, node, status = self.atlas_logic.run_exploration_step(
                        world_name, theme, current_breadcrumb,
                        current_node, visited_breadcrumbs=visited_breadcrumbs
                    )
                    current_breadcrumb = breadcrumb
                    if current_breadcrumb not in visited_breadcrumbs:
                        visited_breadcrumbs.append(current_breadcrumb)

                    yield {'status': 'update', 'world_data': config.world}

            utils.log_message('game', "--- [ATLAS] Exploration complete. ---\n")

        yield {'status': 'complete', 'world_name': world_name}