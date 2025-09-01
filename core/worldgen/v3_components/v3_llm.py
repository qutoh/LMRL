# /core/worldgen/v3_components/v3_llm.py

import json
import random
import re
from typing import List

from core.common import command_parser, utils, file_io
from core.common.config_loader import config
from core.common.localization import loc
from core.llm.llm_api import execute_task
from core.worldgen.procgen_utils import parse_dimensions_from_text


class V3_LLM:
    """
    Handles all direct LLM interactions for the MapArchitectV3 generation process.
    This includes conceptualizing initial features and generating sub-features conversationally.
    """

    def __init__(self, engine):
        self.engine = engine
        self.level_generator = config.agents['LEVEL_GENERATOR']
        self.world_theme = engine.world_theme
        self.scene_prompt = engine.scene_prompt

    def _get_world_scene_context_str(self) -> str:
        """Helper to build the standardized world/scene context string."""
        return loc('prompt_substring_world_scene_context', world_theme=self.world_theme, scene_prompt=self.scene_prompt)

    def _get_feature_menu_by_strategy(self, allowed_strategies: List[str]) -> str:

        menu_items = []
        for key, data in self.engine.config.features.items():
            if data.get('placement_strategy') in allowed_strategies:
                display_name = data.get('display_name', key)
                menu_items.append(f"- **{key}:** {display_name}")
        return "\n".join(menu_items)

    def _generate_dynamic_example(self, allowed_strategies: List[str], num_examples: int, output_format: str) -> str:

        valid_features = {
            key: data for key, data in self.engine.config.features.items()
            if data.get('placement_strategy') in allowed_strategies and key != 'CHARACTER'
        }
        if not valid_features:
            if output_format == 'list':
                fallback_obj = {"features": [{"name": "Example Chamber", "type": "GENERIC_ROOM", "size_tier": "large",
                                              "description_sentence": "A large, empty chamber stands here."}]}
                return json.dumps(fallback_obj, indent=2)
            else:
                fallback_obj = {"name": "Example Object", "description": "An object of some importance.",
                                "type": "GENERIC_ROOM", "dimensions": "medium"}
                return json.dumps(fallback_obj, indent=2)
        num_to_select = min(num_examples, len(valid_features))
        selected_keys = random.sample(list(valid_features.keys()), num_to_select)
        examples = []
        for key in selected_keys:
            name_prefix = key.replace("_", " ").title().replace("Generic ", "")
            example = {"name": f"The {name_prefix}", "type": key}
            if output_format == 'list':
                example["size_tier"] = random.choice(['small', 'medium', 'large'])
                example["description_sentence"] = f"A {example['size_tier']} {name_prefix.lower()} is here."
            else:
                example["description"] = f"A simple {name_prefix.lower()}."
                example["dimensions"] = "medium"
            examples.append(example)
        if output_format == 'list':
            return json.dumps({"features": examples}, indent=2)
        else:
            return json.dumps(examples[0], indent=2)

    def define_feature_from_sentence(self, sentence: str) -> dict | None:
        allowed_strategies = ['EXTERIOR', 'CONNECTOR', 'INTERIOR', 'BLOCKING', 'PATHING']
        type_list_str = self._get_feature_menu_by_strategy(allowed_strategies)
        example_response = self._generate_dynamic_example(allowed_strategies, 1, 'single')
        kwargs = {
            "world_scene_context": self._get_world_scene_context_str(),
            "sentence": sentence,
            "type_list_str": type_list_str,
            "example_response": example_response
        }
        raw_response = execute_task(self.engine, self.level_generator, 'PEG_CREATE_FEATURE_JSON', [],
                                    task_prompt_kwargs=kwargs)
        command = command_parser.parse_structured_command(
            self.engine, raw_response, 'LEVEL_GENERATOR',
            fallback_task_key='CH_FIX_PEG_JSON',
            fallback_prompt_kwargs={'type_list_str': type_list_str}
        )
        if not command or not command.get('type'): return None
        if command['type'] == 'CHARACTER': return {'type': 'CHARACTER',
                                                   'name': command.get('name', 'Unnamed Character'),
                                                   'source_sentence': sentence}
        required_keys = ['name', 'description', 'type', 'dimensions']
        if not all(k in command for k in required_keys): return None
        data = command.copy()
        feature_def = self.engine.config.features.get(data['type'], {})
        data['placement_strategy'] = feature_def.get('placement_strategy')
        parsed_dims = parse_dimensions_from_text(data['dimensions'])
        data['size_tier'] = parsed_dims[2] if parsed_dims else 'medium'
        data['dimensions_tiles'] = (parsed_dims[0], parsed_dims[1]) if parsed_dims else ('medium', 'medium')
        if forced_tier := feature_def.get('force_size_tier'):
            data['size_tier'] = forced_tier
        return data

    def _get_initial_features_stepwise(self) -> list[dict]:
        utils.log_message('debug', "[PEGv3] Single-shot failed. Falling back to stepwise initial feature generation.")
        context_str = self._get_world_scene_context_str()
        areas_str = execute_task(self.engine, self.level_generator, 'PEG_V2_GET_AREA_NAMES', [],
                                 task_prompt_kwargs={"world_scene_context": context_str})
        area_names = [name.strip() for name in areas_str.split(';') if name.strip()]
        ranked_str = execute_task(self.engine, self.level_generator, 'PEG_V2_RANK_AREAS', [],
                                  task_prompt_kwargs={"area_list_str": ", ".join(area_names)})
        ranked_names = [name.strip() for name in ranked_str.split(';') if name.strip()]
        defined_features = []
        for name in ranked_names:
            type_list_str = self._get_feature_menu_by_strategy(['EXTERIOR', 'BLOCKING'])
            raw_response = execute_task(self.engine, self.level_generator, 'PEG_V2_DEFINE_AREA_DATA', [],
                                        task_prompt_kwargs={"world_scene_context": context_str,
                                                            "area_name": name,
                                                            "type_list_str": type_list_str})
            feature_data = command_parser.parse_structured_command(
                self.engine, raw_response, 'LEVEL_GENERATOR', 'CH_FIX_PEG_V2_JSON',
                {'type_list_str': type_list_str})
            if feature_data and all(k in feature_data for k in ['description', 'type', 'size_tier']):
                feature_data['name'] = name
                feature_data['description_sentence'] = self.get_narrative_seed(name) or f"This is the {name}."
                defined_features.append(feature_data)
        return defined_features

    def get_initial_features(self) -> list[dict]:
        """Primary Method: Uses a single JSON prompt to get all initial features at once."""
        utils.log_message('debug', "[PEGv3] Getting initial features via single-shot JSON...")
        allowed_strategies = ['EXTERIOR', 'BLOCKING']
        type_list_str = self._get_feature_menu_by_strategy(allowed_strategies)
        example_response = self._generate_dynamic_example(allowed_strategies, 2, 'list')
        kwargs = {
            "world_scene_context": self._get_world_scene_context_str(),
            "type_list_str": type_list_str,
            "example_response": example_response
        }
        raw_response = execute_task(self.engine, self.level_generator, 'PEG_V3_GET_INITIAL_FEATURES_JSON', [],
                                    task_prompt_kwargs=kwargs)
        command = command_parser.parse_structured_command(
            self.engine, raw_response, 'LEVEL_GENERATOR',
            fallback_task_key='CH_FIX_PEG_V3_JSON',
            fallback_prompt_kwargs={'type_list_str': type_list_str}
        )
        if command and isinstance(command.get('features'), list) and command['features']:
            valid_features = []
            for feat in command['features']:
                feat_type = feat.get('type')
                if feat_type:
                    feat_def = self.engine.config.features.get(feat_type, {})
                    if feat_def.get('placement_strategy') in allowed_strategies:
                        valid_features.append(feat)
            if valid_features:
                return valid_features
        return self._get_initial_features_stepwise()

    def _validate_tile_data(self, tile_data: dict) -> bool:
        """Checks if a generated tile data object meets all schema requirements."""
        if not isinstance(tile_data, dict) or len(tile_data) != 1: return False
        name, data = next(iter(tile_data.items()))
        if not isinstance(data, dict): return False

        required_keys = ["description", "movement_cost", "is_transparent", "materials", "pass_methods", "colors",
                         "characters"]
        if not all(key in data for key in required_keys): return False

        if not (isinstance(data["characters"], list) and data["characters"] and all(
            isinstance(c, str) for c in data["characters"])): return False
        if not (isinstance(data["colors"], list) and data["colors"] and all(
            isinstance(c, list) and len(c) == 3 for c in data["colors"])): return False
        if not (isinstance(data["materials"], list) and data["materials"]): return False

        return True

    def create_new_tile_type(self, scene_prompt: str) -> dict | None:
        """Orchestrates the creation of a new tile type using a single-shot -> fix -> stepwise fallback chain."""
        utils.log_message('debug', "[PEG Setup] Attempting to generate new tile type via single-shot JSON...")

        pass_methods_menu = "\n".join(
            f"- `{key}`: {desc}" for key, desc in config.travel.get('movement_capabilities', {}).items())
        materials_list_str = "\n".join(f"- `{mat}`" for mat in config.materials.get("materials", []))
        generation_reason = loc('prompt_substring_tile_reason_exterior')
        world_scene_context = self._get_world_scene_context_str()

        kwargs = {
            "world_scene_context": world_scene_context,
            "generation_reason": generation_reason,
            "pass_methods_menu": pass_methods_menu,
            "materials_list": materials_list_str
        }
        raw_response = execute_task(self.engine, self.level_generator, 'PEG_V3_CREATE_TILE_JSON', [],
                                    task_prompt_kwargs=kwargs)

        fix_kwargs = {
            "materials_list": ", ".join(config.materials.get("materials", [])),
            "pass_methods_list": ", ".join(config.travel.get("movement_capabilities", {}).keys())
        }
        command = command_parser.parse_structured_command(
            self.engine, raw_response, 'LEVEL_GENERATOR',
            fallback_task_key='CH_FIX_TILE_JSON',
            fallback_prompt_kwargs=fix_kwargs,
            validator=self._validate_tile_data
        )

        if command and command.get("action") != "NONE":
            return command

        utils.log_message('debug',
                          "[PEG Setup] JSON generation and repair failed. Falling back to stepwise tile generation.")
        new_tile = {"is_transparent": True, "movement_cost": 1.0}

        name_kwargs = {"world_scene_context": world_scene_context}
        tile_name = execute_task(self.engine, self.level_generator, 'PEG_V3_CREATE_TILE_NAME_STEPWISE', [],
                                 task_prompt_kwargs=name_kwargs).strip()
        if not tile_name or not tile_name.isupper() or " " in tile_name: return None

        desc_kwargs = {"world_scene_context": world_scene_context, "tile_name": tile_name}
        new_tile["description"] = execute_task(self.engine, self.level_generator, 'PEG_V3_CREATE_TILE_DESC_STEPWISE',
                                               [], task_prompt_kwargs=desc_kwargs)

        valid_pass_methods = list(config.travel.get('movement_capabilities', {}).keys())
        pass_methods_list_str = "\n".join(
            f"- `{key}`: {desc}" for key, desc in config.travel.get('movement_capabilities', {}).items())
        pass_methods_kwargs = {"tile_name": tile_name, "tile_description": new_tile["description"],
                               "pass_methods_list": pass_methods_list_str}
        pass_methods_response = execute_task(self.engine, self.level_generator,
                                             'PEG_V3_CREATE_TILE_PASS_METHODS_STEPWISE', [],
                                             task_prompt_kwargs=pass_methods_kwargs).strip().upper()

        chosen_methods = [m.strip() for m in pass_methods_response.split(';')]
        validated_methods = [m for m in chosen_methods if m in valid_pass_methods]

        if not validated_methods or "NONE" in chosen_methods:
            new_tile["pass_methods"] = []
            if pass_methods_response.upper() != "NONE":
                file_io.log_to_ideation_file("TILE_PASS_METHOD", pass_methods_response,
                                             context=f"Invalid methods for {tile_name}")
        else:
            new_tile["pass_methods"] = validated_methods

        materials_list_str = ", ".join(config.materials.get("materials", []))
        materials_kwargs = {"tile_name": tile_name, "tile_description": new_tile["description"],
                            "materials_list": materials_list_str}
        materials_response = execute_task(self.engine, self.level_generator, 'PEG_V3_CREATE_TILE_MATERIALS_STEPWISE',
                                          [], task_prompt_kwargs=materials_kwargs).strip().upper()

        chosen_materials = [mat.strip() for mat in materials_response.split(';') if mat.strip()]
        if not chosen_materials:
            new_tile["materials"] = ["NONE"]  # Safe default
        else:
            new_tile["materials"] = chosen_materials

        app_kwargs = {"tile_name": tile_name, "tile_description": new_tile["description"]}
        char_response = execute_task(self.engine, self.level_generator, 'PEG_V3_CREATE_TILE_CHAR_STEPWISE', [],
                                     task_prompt_kwargs=app_kwargs)
        char_match = re.search(r"['\"`´](.)['\"`´]", char_response)
        new_tile["characters"] = [char_match.group(1)] if char_match else ["?"]

        colors_path = file_io.join_path(file_io.PROJECT_ROOT, 'data', 'lang', 'en_us', 'colors.json')
        color_map = file_io.read_json(colors_path, default={"WHITE": [255, 255, 255]})
        color_kwargs = {
            "tile_name": tile_name, "tile_description": new_tile["description"],
            "color_keywords_list": ", ".join(color_map.keys())
        }
        color_keyword = execute_task(self.engine, self.level_generator, 'PEG_V3_CREATE_TILE_COLOR_STEPWISE', [],
                                     task_prompt_kwargs=color_kwargs).strip().upper()

        if color_keyword not in color_map:
            file_io.log_to_ideation_file("TILE_COLOR", color_keyword, context=f"Invalid color for {tile_name}")
            new_tile["colors"] = [[255, 255, 255]]
        else:
            new_tile["colors"] = [color_map.get(color_keyword)]

        return {tile_name: new_tile}

    def get_narrative_seed(self, area_name: str) -> str:
        kwargs = {"world_scene_context": self._get_world_scene_context_str(), "area_name": area_name}
        return execute_task(self.engine, self.level_generator, 'PEG_V3_GET_NARRATIVE_SEED', [],
                            task_prompt_kwargs=kwargs)

    def get_next_narrative_beat(self, context: str, other_features_context: str) -> str:
        kwargs = {"world_scene_context": self._get_world_scene_context_str(), "context": context,
                  "other_features_context": other_features_context}
        return execute_task(self.engine, self.level_generator, 'PROCGEN_GENERATE_NARRATIVE_BEAT', [],
                            task_prompt_kwargs=kwargs)

    def choose_parent_feature(self, narrative_log: str, new_feature_sentence: str, parent_options_list: str) -> str:
        choice_kwargs = {
            "narrative_log": narrative_log,
            "new_feature_sentence": new_feature_sentence,
            "parent_options_list": parent_options_list
        }
        return execute_task(self.engine, self.level_generator, 'PEG_V3_CHOOSE_PARENT', [],
                            task_prompt_kwargs=choice_kwargs)

    def choose_path_target_feature(self, narrative_log: str, new_feature_sentence: str,
                                   target_options_list: str) -> str:
        """Asks the LLM to choose a destination for a new path from a list of options."""
        choice_kwargs = {
            "narrative_log": narrative_log,
            "new_feature_sentence": new_feature_sentence,
            "target_options_list": target_options_list
        }
        return execute_task(self.engine, self.level_generator, 'PEG_V3_CHOOSE_PATH_TARGET', [],
                            task_prompt_kwargs=choice_kwargs)

    def get_nearby_feature_sentence(self, all_areas_narrative: str) -> str:
        """Asks the LLM to generate a sentence for a new, nearby feature."""
        kwargs = {
            "world_scene_context": self._get_world_scene_context_str(),
            "all_areas_narrative": all_areas_narrative
        }
        return execute_task(self.engine, self.level_generator, 'PEG_V3_CREATE_NEARBY_FEATURE_SENTENCE', [],
                            task_prompt_kwargs=kwargs)

    def choose_exterior_tile(self, scene_prompt: str, tile_options_str: str) -> str:
        """Asks the LLM to choose a base tile for the map's exterior."""
        kwargs = {
            "scene_prompt": scene_prompt,
            "tile_options_str": tile_options_str
        }
        return execute_task(self.engine, self.level_generator, 'PEG_V3_CHOOSE_EXTERIOR_TILE', [],
                            task_prompt_kwargs=kwargs)