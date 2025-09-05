# /core/worldgen/atlas_manager.py

from core.common.config_loader import config
from core.worldgen.world_components.atlas_logic import AtlasLogic
from core.worldgen.world_components.content_generator import ContentGenerator
from core.worldgen.world_components.location_finder import LocationFinder
from core.worldgen.world_components.world_actions import WorldActions
from core.worldgen.world_components.world_generator import WorldGenerator
from core.worldgen.world_components.world_graph_navigator import WorldGraphNavigator


class AtlasManager:
    """
    A façade that orchestrates the procedural generation of a world by delegating
    tasks to specialized service classes for navigation, content creation, and logic.
    """

    def __init__(self, engine):
        self.engine = engine
        self.atlas_agent = config.agents.get('ATLAS')
        if not self.atlas_agent:
            raise ValueError("ATLAS agent not found in agents.json")

        # Initialize and compose the service classes
        self.navigator = WorldGraphNavigator()
        self.content_generator = ContentGenerator(engine, self.navigator)
        self.world_actions = WorldActions(engine, self.navigator, self.content_generator)
        self.atlas_logic = AtlasLogic(engine, self.navigator, self.content_generator, self.world_actions)
        self.finder = LocationFinder(engine, self.navigator, self.content_generator, self.atlas_logic,
                                     self.world_actions)
        self.world_generator = WorldGenerator(engine, self.content_generator, self.atlas_logic)

    def find_or_create_location_for_scene(self, world_name: str, world_theme: str, scene_prompt: str) -> tuple[
                                                                                                                dict, list] | tuple[
                                                                                                                None, None]:
        """Delegates the entire scene placement process to the LocationFinder service."""
        return self.finder.find_or_create_location_for_scene(world_name, world_theme, scene_prompt)

    def create_new_world_generator(self, theme: str):
        """Delegates the world creation process to the WorldGenerator service, returning a generator."""
        # Ensure the skip flag is reset before starting a new generation process
        setattr(self.world_generator, '_skip_exploration', False)
        return self.world_generator.create_new_world_generator(theme)