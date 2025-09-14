# /core/worldgen/v3_components/v3_llm.py

import json
import random
import re
from typing import List, Dict, Optional

from core.common import command_parser, utils, file_io
from core.common.config_loader import config
from core.common.localization import loc
from core.llm.llm_api import execute_task
from core.worldgen.procgen_utils import parse_dimensions_from_text
from .feature_node import FeatureNode


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
        """Generates a markdown list of feature types filtered by placement strategy."""
        menu_items = []
        for key, data in self.engine.config.features.items():
            if data.get('placement_strategy') in allowed_strategies:
                display_name = data.get('display_name', key)
                menu_items.append(f"- **{key}:** {display_name}")
        return "\n".join(menu_items)

    def _get_nature_menu_str(self) -> str:
        """Generates a markdown list of available natures."""
        menu_items = []
        for key, data in config.natures.items():
            description = data.get('description', 'No description.')
            menu_items.append(f"- **{key}:** {description}")
        return "\n".join(menu_items)

    def _generate_dynamic_example(self, allowed_strategies: List[str], num_examples: int, output_format: str,
                                  available_size_tiers: Optional[List[str]] = None) -> str:
        """Creates a dynamic, valid JSON example for few-shot prompting."""
        valid_features = {
            key: data for key, data in self.engine.config.features.items()
            if data.get('placement_strategy') in allowed_strategies and key != 'CHARACTER'
        }
        size_tiers = available_size_tiers or ['small', 'medium', 'large']
        if not size_tiers: size_tiers = ['small']

        valid_natures = list(config.natures.keys())

        if not valid_features:
            if output_format == 'list':
                fallback_obj = {"features": [{"name": "Example Chamber", "type": "BUILT_ROOM", "size_tier": "large",
                                              "description_sentence": "A large, empty chamber stands here."}]}
                return json.dumps(fallback_obj, indent=2)
            else:
                fallback_obj = {"name": "Example Object", "description": "An object of some importance.",
                                "type": "BUILT_ROOM", "dimensions": "medium", "natures": ["BUILT"]}
                return json.dumps(fallback_obj, indent=2)

        if output_format == 'list':
            non_path_features = {k: v for k, v in valid_features.items() if v.get('placement_strategy') != 'PATHING'}
            path_features = {k: v for k, v in valid_features.items() if v.get('placement_strategy') == 'PATHING'}
            examples = []
            num_anchors = min(max(1, num_examples - 1), len(non_path_features))
            if num_anchors > 0:
                anchor_keys = random.sample(list(non_path_features.keys()), num_anchors)
                for key in anchor_keys:
                    name_prefix = key.replace("_", " ").title()
                    example = {"name": f"The {name_prefix}", "type": key,
                               "size_tier": random.choice(size_tiers),
                               "natures": random.sample(valid_natures, k=min(2, len(valid_natures))),
                               "description_sentence": f"A {random.choice(size_tiers)} {name_prefix.lower()} is here."}
                    examples.append(example)
            if path_features and len(examples) < num_examples:
                path_key = random.choice(list(path_features.keys()))
                path_prefix = path_key.replace("_", " ").title()
                path_example = {"name": f"The {path_prefix}", "type": path_key,
                                "size_tier": random.choice(size_tiers),
                                "natures": random.sample(valid_natures, k=min(2, len(valid_natures))),
                                "description_sentence": f"A winding {path_prefix.lower()} crosses the area."}
                possible_endpoints = [e['name'] for e in examples] + ["NORTH_BORDER", "SOUTH_BORDER", "EAST_BORDER",
                                                                      "WEST_BORDER"]
                if len(possible_endpoints) >= 2:
                    source, dest = random.sample(possible_endpoints, 2)
                    path_example["source"] = source
                    path_example["destination"] = dest
                    examples.append(path_example)
            while len(examples) < num_examples and non_path_features:
                key = random.choice(list(non_path_features.keys()))
                if any(e['type'] == key for e in examples):
                    del non_path_features[key]
                    if not non_path_features: break
                    continue
                name_prefix = key.replace("_", " ").title()
                example = {"name": f"The {name_prefix}", "type": key,
                           "size_tier": random.choice(size_tiers),
                           "natures": random.sample(valid_natures, k=min(2, len(valid_natures))),
                           "description_sentence": f"A {random.choice(size_tiers)} {name_prefix.lower()} is here."}
                examples.append(example)
                del non_path_features[key]
            return json.dumps({"features": examples}, indent=2)
        else:
            num_to_select = min(num_examples, len(valid_features))
            selected_keys = random.sample(list(valid_features.keys()), num_to_select)
            example_data = []
            for key in selected_keys:
                name_prefix = key.replace("_", " ").title()
                example = {"name": f"The {name_prefix}", "type": key}
                example["description"] = f"A simple {name_prefix.lower()}."
                example["dimensions"] = random.choice(size_tiers)
                example["natures"] = random.sample(valid_natures, k=min(2, len(valid_natures)))
                example_data.append(example)
            return json.dumps(example_data[0], indent=2)

    def define_feature_from_sentence(self, sentence: str,
                                     available_size_tiers: Optional[List[str]] = None) -> Dict | None:
        """Takes a narrative sentence and converts it into a structured feature dictionary."""
        allowed_strategies = ['BRANCHING', 'CONNECTOR', 'INTERIOR', 'PATHING']
        type_list_str = self._get_feature_menu_by_strategy(allowed_strategies)
        nature_menu_str = self._get_nature_menu_str()

        tiers_to_use = available_size_tiers or ['large', 'medium', 'small']
        example_response = self._generate_dynamic_example(allowed_strategies, 1, 'single', tiers_to_use)

        size_tier_examples = {
            'large': "large (e.g., 'a vast cavern', '80ft wide', '25m across')",
            'medium': "medium (e.g., 'a medium-sized room', '40ft by 40ft', '12 meters long')",
            'small': "small (e.g., 'a small alcove', '15 feet', '5m')"
        }
        size_tier_list_str = "\n- ".join([desc for tier, desc in size_tier_examples.items() if tier in tiers_to_use])
        if size_tier_list_str:
            size_tier_list_str = "- " + size_tier_list_str

        kwargs = {
            "world_scene_context": self._get_world_scene_context_str(),
            "sentence": sentence,
            "type_list_str": type_list_str,
            "nature_menu_str": nature_menu_str,
            "example_response": example_response,
            "size_tier_list_str": size_tier_list_str
        }
        raw_response = execute_task(self.engine, self.level_generator, 'PEG_CREATE_FEATURE_JSON', [],
                                    task_prompt_kwargs=kwargs)
        command = command_parser.parse_structured_command(
            self.engine, raw_response, 'LEVEL_GENERATOR',
            fallback_task_key='CH_FIX_PEG_JSON',
            fallback_prompt_kwargs={'type_list_str': type_list_str}
        )
        if command and command.get('type'):
            if command['type'] == 'CHARACTER':
                return {'type': 'CHARACTER', 'name': command.get('name', 'Unnamed Character'),
                        'source_sentence': sentence}

            required_keys = ['name', 'description', 'type', 'dimensions']
            if all(k in command for k in required_keys):
                data = command.copy()
                feature_def = self.engine.config.features.get(data['type'], {})
                data['placement_strategy'] = feature_def.get('placement_strategy')
                parsed_dims = parse_dimensions_from_text(data['dimensions'])
                data['size_tier'] = parsed_dims[2] if parsed_dims else 'medium'
                if forced_tier := feature_def.get('force_size_tier'):
                    data['size_tier'] = forced_tier
                return data

        utils.log_message('debug', "[PEGv3] JSON feature definition failed. Falling back to stepwise generation.")
        name_match = re.search(r'is a (.+?) here\.', sentence, re.IGNORECASE)
        area_name = name_match.group(1).strip() if name_match else "Unnamed Area"

        raw_response = execute_task(self.engine, self.level_generator, 'PEG_V2_DEFINE_AREA_DATA', [],
                                    task_prompt_kwargs={"world_scene_context": self._get_world_scene_context_str(),
                                                        "area_name": area_name,
                                                        "type_list_str": type_list_str})
        feature_data = command_parser.parse_structured_command(
            self.engine, raw_response, 'LEVEL_GENERATOR', 'CH_FIX_PEG_V2_JSON',
            {'type_list_str': type_list_str})

        if feature_data and all(k in feature_data for k in ['description', 'type', 'size_tier']):
            feature_data['name'] = area_name
            feature_data['description_sentence'] = sentence

            # Stepwise nature generation
            nature_menu_str = self._get_nature_menu_str()
            natures_raw = execute_task(self.engine, self.level_generator, 'PEG_V2_GET_NATURES', [],
                                       task_prompt_kwargs={"area_description": feature_data['description'],
                                                           "nature_menu_str": nature_menu_str})
            feature_data['natures'] = [n.strip() for n in natures_raw.split(';') if n.strip()]

            return feature_data

        return None

    def _get_initial_features_stepwise(self) -> List[Dict]:
        """Fallback method to generate initial features one step at a time."""
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
            type_list_str = self._get_feature_menu_by_strategy(['BRANCHING'])
            raw_response = execute_task(self.engine, self.level_generator, 'PEG_V2_DEFINE_AREA_DATA', [],
                                        task_prompt_kwargs={"world_scene_context": context_str, "area_name": name,
                                                            "type_list_str": type_list_str})
            feature_data = command_parser.parse_structured_command(
                self.engine, raw_response, 'LEVEL_GENERATOR', 'CH_FIX_PEG_V2_JSON',
                {'type_list_str': type_list_str})
            if feature_data and all(k in feature_data for k in ['description', 'type', 'size_tier']):
                feature_data['name'] = name
                feature_data['description_sentence'] = self.get_narrative_seed(name) or f"This is the {name}."

                nature_menu_str = self._get_nature_menu_str()
                natures_raw = execute_task(self.engine, self.level_generator, 'PEG_V2_GET_NATURES', [],
                                           task_prompt_kwargs={"area_description": feature_data['description'],
                                                               "nature_menu_str": nature_menu_str})
                feature_data['natures'] = [n.strip() for n in natures_raw.split(';') if n.strip()]
                defined_features.append(feature_data)
        return defined_features

    def get_initial_features(self) -> List[Dict]:
        """Uses a single-shot JSON prompt to get all initial features, with a stepwise fallback."""
        allowed_strategies = ['BRANCHING', 'PATHING']
        type_list_str = self._get_feature_menu_by_strategy(allowed_strategies)
        nature_menu_str = self._get_nature_menu_str()
        example_response = self._generate_dynamic_example(allowed_strategies, 3, 'list')
        kwargs = {
            "world_scene_context": self._get_world_scene_context_str(),
            "type_list_str": type_list_str,
            "nature_menu_str": nature_menu_str,
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
            validated_features = []
            for feat in command['features']:
                if not feat.get('type'): continue
                strategy = config.features.get(feat.get('type'), {}).get('placement_strategy')
                if strategy not in allowed_strategies: continue
                if strategy == 'PATHING':
                    if 'source' in feat and 'destination' in feat:
                        validated_features.append(feat)
                else:
                    validated_features.append(feat)
            return validated_features
        return self._get_initial_features_stepwise()

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
            new_tile["pass_methods"] = ["GROUND"]  # Default
            if pass_methods_response.upper() != "NONE" and validated_methods:
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
        new_tile["materials"] = chosen_materials if chosen_materials else ["DIRT"]

        app_kwargs = {"tile_name": tile_name, "tile_description": new_tile["description"]}
        char_response = execute_task(self.engine, self.level_generator, 'PEG_V3_CREATE_TILE_CHAR_STEPWISE', [],
                                     task_prompt_kwargs=app_kwargs)
        char_match = re.search(r"['\"`´](.)['\"`´]", char_response)
        new_tile["characters"] = [char_match.group(1)] if char_match and char_match.group(1) else ["."]

        colors_path = file_io.join_path(file_io.PROJECT_ROOT, 'data', 'lang', 'en_us', 'colors.json')
        color_map = file_io.read_json(colors_path, default={"WHITE": [255, 255, 255]})
        color_kwargs = {
            "tile_name": tile_name, "tile_description": new_tile["description"],
            "color_keywords_list": ", ".join(color_map.keys())
        }
        color_keyword = execute_task(self.engine, self.level_generator, 'PEG_V3_CREATE_TILE_COLOR_STEPWISE', [],
                                     task_prompt_kwargs=color_kwargs).strip().upper()

        new_tile["colors"] = [color_map.get(color_keyword, [128, 128, 128])]
        if color_keyword not in color_map:
            file_io.log_to_ideation_file("TILE_COLOR", color_keyword, context=f"Invalid color for {tile_name}")

        return {tile_name: new_tile}

    def _validate_tile_data(self, tile_data: dict) -> bool:
        """Checks if a generated tile data object meets all schema requirements."""
        if not isinstance(tile_data, dict) or len(tile_data) != 1: return False
        name, data = next(iter(tile_data.items()))
        if not isinstance(data, dict): return False
        required_keys = ["description", "movement_cost", "is_transparent", "materials", "pass_methods", "colors",
                         "characters"]
        if not all(key in data for key in required_keys): return False
        if not (isinstance(data["characters"], list) and data["characters"] and all(
                isinstance(c, str) and len(c) > 0 for c in data["characters"])): return False  # CHECK FOR EMPTY STRING
        if not (isinstance(data["colors"], list) and data["colors"] and all(
                isinstance(c, list) and len(c) == 3 for c in data["colors"])): return False
        if not (isinstance(data["materials"], list) and data["materials"]): return False
        valid_pass = list(config.travel.get("movement_capabilities", {}).keys())
        if not (isinstance(data["pass_methods"], list) and all(
                p in valid_pass for p in data["pass_methods"])): return False
        return True

    def get_narrative_seed(self, area_name: str) -> str:
        """Generates the first descriptive sentence for a new area."""
        kwargs = {"world_scene_context": self._get_world_scene_context_str(), "area_name": area_name}
        return execute_task(self.engine, self.level_generator, 'PEG_V3_GET_NARRATIVE_SEED', [],
                            task_prompt_kwargs=kwargs)

    def get_next_narrative_beat(self, context: str, other_features_context: str) -> str:
        """Generates the next descriptive sentence to add a new feature."""
        kwargs = {"world_scene_context": self._get_world_scene_context_str(), "context": context,
                  "other_features_context": other_features_context}
        return execute_task(self.engine, self.level_generator, 'PROCGEN_GENERATE_NARRATIVE_BEAT', [],
                            task_prompt_kwargs=kwargs)

    def choose_parent_feature(self, narrative_log: str, new_feature_sentence: str, parent_options_list: str) -> str:
        """Asks the LLM to choose a parent for a new feature."""
        choice_kwargs = {"narrative_log": narrative_log, "new_feature_sentence": new_feature_sentence,
                         "parent_options_list": parent_options_list}
        return execute_task(self.engine, self.level_generator, 'PEG_V3_CHOOSE_PARENT', [],
                            task_prompt_kwargs=choice_kwargs)

    def choose_path_target_feature(self, narrative_log: str, new_feature_sentence: str,
                                   target_options_list: str) -> str:
        """Asks the LLM to choose a destination for a new path."""
        choice_kwargs = {"narrative_log": narrative_log, "new_feature_sentence": new_feature_sentence,
                         "target_options_list": target_options_list}
        return execute_task(self.engine, self.level_generator, 'PEG_V3_CHOOSE_PATH_TARGET', [],
                            task_prompt_kwargs=choice_kwargs)

    def choose_path_source_feature(self, source_options_list: str) -> str:
        """Asks the LLM to choose a source for a new path."""
        kwargs = {"world_scene_context": self._get_world_scene_context_str(),
                  "source_options_list": source_options_list}
        return execute_task(self.engine, self.level_generator, 'PEG_V3_CHOOSE_PATH_SOURCE', [],
                            task_prompt_kwargs=kwargs)

    def get_nearby_feature_sentence(self, all_areas_narrative: str) -> str:
        """Asks the LLM to generate a sentence for a new, nearby feature."""
        kwargs = {"world_scene_context": self._get_world_scene_context_str(),
                  "all_areas_narrative": all_areas_narrative}
        return execute_task(self.engine, self.level_generator, 'PEG_V3_CREATE_NEARBY_FEATURE_SENTENCE', [],
                            task_prompt_kwargs=kwargs)

    def choose_exterior_tile(self, scene_prompt: str, tile_options_str: str) -> str:
        """Asks the LLM to choose a base tile for the map's exterior."""
        kwargs = {"scene_prompt": scene_prompt, "tile_options_str": tile_options_str}
        return execute_task(self.engine, self.level_generator, 'PEG_V3_CHOOSE_EXTERIOR_TILE', [],
                            task_prompt_kwargs=kwargs)

    def decide_connector_strategy(self, connector_name: str, connector_desc: str, nearby_targets: List[str]) -> str:
        """Decides whether to bridge a connector to an existing feature or create a new one."""
        kwargs = {
            "world_scene_context": self._get_world_scene_context_str(),
            "connector_name": connector_name,
            "connector_description": connector_desc,
            "nearby_targets_list": "\n".join(
                f"- {name}" for name in nearby_targets) if nearby_targets else "None available."
        }
        return execute_task(self.engine, self.level_generator, 'PEG_V3_DECIDE_CONNECTOR_STRATEGY', [],
                            task_prompt_kwargs=kwargs)

    def create_connector_child(self, grandparent_node: 'FeatureNode', connector_node: 'FeatureNode') -> dict | None:
        """Generates the definition for a new feature at the end of a seeded connector."""
        context_kwargs = {
            "world_scene_context": self._get_world_scene_context_str(),
            "starting_area_description": grandparent_node.narrative_log,
            "passage_description": connector_node.narrative_log
        }
        result = execute_task(self.engine, self.level_generator, 'PEG_V3_CREATE_CONNECTOR_CHILD', [],
                              task_prompt_kwargs=context_kwargs)
        if not result or "none" in result.lower(): return None
        try:
            data = json.loads(result)
            if isinstance(data, dict) and 'name' in data and 'type' in data:
                data['description_sentence'] = data.get('description_sentence', 'Generated from mock data.')
                return data
        except (json.JSONDecodeError, TypeError):
            pass
        return self.define_feature_from_sentence(result)