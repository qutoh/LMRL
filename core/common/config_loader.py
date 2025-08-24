# /core/common/config_loader.py

from . import file_io
from .localization import loc


class Config:
    def __init__(self, data_dir_name='data'):
        self.data_dir = file_io.join_path(file_io.PROJECT_ROOT, data_dir_name)

        self.settings = self._load_json('settings.json')
        self.default_settings = self._load_json('default_settings.json', default={})

        self.leads = self._load_json('leads.json')
        self.dm_roles = self._load_json('dm_roles.json')
        self.agents = self._load_json('agents.json', default={})
        self.scene = self._load_json('scene.json', default=[{"scene_prompt": "A story begins."}])
        self.model_tuning = self._load_json('model_tuning.json', default={})

        self.levels = {}
        self.generated_scenes = []
        self.world = {}

        self.features = self._load_json('features.json', default={})
        self.tile_types = self._load_json('tile_types.json', default={})

        # --- Programmatically ensure core definitions exist ---
        self._ensure_core_definitions()

        self.tile_type_map = {name: i for i, name in enumerate(self.tile_types.keys())}
        self.tile_type_map_reverse = {i: name for name, i in self.tile_type_map.items()}

        self.casting_leads = self._load_json('casting_leads.json')
        self.casting_npcs = self._load_json('casting_npcs.json')
        self.casting_dms = self._load_json('casting_dms.json')

        self.task_parameters = self._load_task_parameters()

        self.calibration_groups = self._load_json('calibration_groups.json', default={})
        self.calibrated_temperatures = self._load_json('calibrated_temperatures.json', default={})
        self.calibration_tasks = self._load_json('calibration_tasks.json', default=[])
        self.ideation_blacklist = self._load_json('ideation_blacklist.json', default={})

        self._apply_defaults_to_agents()
        self._apply_api_keys()

    def _ensure_core_definitions(self):
        """Ensures that critical, hard-coded features and tiles exist."""
        if "VOID_SPACE" not in self.tile_types:
            self.tile_types["VOID_SPACE"] = {
                "description": "The primordial, unformed space of the map before generation.",
                "movement_cost": 1.0,
                "is_transparent": True,
                "materials": ["NONE"],
                "pass_methods": [],  # Not walkable
                "colors": [[10, 0, 10]],
                "characters": [" "]
            }

        if "CHARACTER" not in self.features:
            self.features["CHARACTER"] = {
                "display_name": "A character or creature.",
                "feature_type": "CHARACTER",
                "placement_strategy": "SPECIAL_FUNC",
                "border_thickness": 0
            }

    def _load_task_parameters(self) -> dict:
        """Loads and merges all task parameter JSON files from the dedicated directory."""
        params_dir = file_io.join_path(self.data_dir, 'task_parameters')
        merged_params = {}
        if not file_io.path_exists(params_dir):
            print(f"WARNING: Task parameters directory not found at {params_dir}")
            return {}

        for filename in file_io.list_directory_contents(params_dir):
            if filename.endswith('.json'):
                file_path = file_io.join_path(params_dir, filename)
                params_data = file_io.read_json(file_path, default={})
                for key in params_data:
                    if key in merged_params:
                        print(
                            f"WARNING: Duplicate task parameter key '{key}' found in '{filename}'. It will be overwritten.")
                merged_params.update(params_data)

        return merged_params

    def save_settings(self):
        """Saves the current settings dictionary to settings.json."""
        path = file_io.join_path(self.data_dir, 'settings.json')
        self.settings.pop('GEMINI_API_KEY', '') # Yeah lol
        return file_io.write_json(path, self.settings)

    def load_world_data(self, world_name: str):
        """Loads all data from the specified world's directory."""
        world_data_path = file_io.join_path(self.data_dir, 'worlds', world_name)

        world_file = file_io.join_path(world_data_path, 'world.json')
        self.world = file_io.read_json(world_file, default={})
        if not self.world:
            print(f"WARNING: Could not load world.json for '{world_name}' from path: {world_file}")

        levels_file = file_io.join_path(world_data_path, 'levels.json')
        self.levels = file_io.read_json(levels_file, default={})

        scenes_file = file_io.join_path(world_data_path, 'generated_scenes.json')
        self.generated_scenes = file_io.read_json(scenes_file, default=[])

        world_npcs_file = file_io.join_path(world_data_path, 'casting_npcs.json')
        world_npcs = file_io.read_json(world_npcs_file, default=[])

        self.casting_npcs = self._load_json('casting_npcs.json')

        existing_npc_names = {npc['name'].lower() for npc in self.casting_npcs}
        for npc in world_npcs:
            if npc['name'].lower() not in existing_npc_names:
                self.casting_npcs.append(npc)

        # --- MERGE DM CASTING LISTS ---
        world_dms_file = file_io.join_path(world_data_path, 'casting_dms.json')
        world_dms = file_io.read_json(world_dms_file, default=[])
        existing_dm_names = {dm['name'].lower() for dm in self.casting_dms}
        for dm in world_dms:
            if dm['name'].lower() not in existing_dm_names:
                self.casting_dms.append(dm)

    def _load_json(self, filename, default=None):
        """Helper function to load a JSON file with error handling."""
        path = file_io.join_path(self.data_dir, filename)

        if 'casting' in filename and not file_io.path_exists(path):
            print(f"INFO: Casting file '{filename}' not found, proceeding without it.")
            return []

        data = file_io.read_json(path, default=default)
        if data is None:
            if default is None and 'casting' not in filename:
                print(loc('error_json_not_found', path=path))
                exit()
            return default
        return data

    def _apply_defaults_to_agents(self):
        """
        Applies role-based default model, temperature, and scaling factors
        to all loaded agent configurations.
        """
        default_temp = self.settings.get("DEFAULT_AGENT_TEMPERATURE", 0.5)
        default_scale = self.settings.get("DEFAULT_AGENT_SCALING_FACTOR", 1.0)
        default_model = self.settings.get("DEFAULT_AGENT_MODEL")

        for agent_data in self.agents.values():
            agent_data['role_type'] = 'agent'
            if not agent_data.get('model') and default_model:
                agent_data['model'] = default_model
            agent_data['temperature'] = agent_data.get('temperature', default_temp)
            agent_data['scaling_factor'] = agent_data.get('scaling_factor', default_scale)

    def _apply_api_keys(self):
        """Get API keys from environment variables via the file_io module."""
        gemini_env_var = self.settings.get("GEMINI_API_KEY_ENV_VAR")
        if gemini_env_var: # This will spill your API key to settings, recommended to leave uncommented but
            self.settings['GEMINI_API_KEY'] = file_io.get_env_variable(gemini_env_var)


config = Config()