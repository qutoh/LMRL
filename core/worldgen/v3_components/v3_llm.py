# /core/worldgen/v3_components/v3_llm.py

import json
import random
from typing import List

from core.common import command_parser, utils
from core.common.config_loader import config
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

    def _get_feature_menu_by_strategy(self, allowed_strategies: List[str]) -> str:

        menu_items = []
        for key, data in self.engine.config.features.items():
            if data.get('placement_strategy') in allowed_strategies:
                display_name = data.get('display_name', key)
                display_sentence = data.get('sentence', key)
                menu_items.append(f"- **{key}:** {display_name}")
        return "\n".join(menu_items)

    def _generate_dynamic_example(self, allowed_strategies: List[str], num_examples: int, output_format: str) -> str:

        valid_features = {
            key: data for key, data in self.engine.config.features.items()
            if data.get('placement_strategy') in allowed_strategies and key != 'CHARACTER'
        }
        if not valid_features:
            if output_format == 'list':
                fallback_obj = {"features": [{"name": "Example Chamber", "type": "GENERIC_ROOM", "size_tier": "large", "description_sentence": "A large, empty chamber stands here."}]}
                return json.dumps(fallback_obj, indent=2)
            else:
                fallback_obj = {"name": "Example Object", "description": "An object of some importance.", "type": "GENERIC_ROOM", "dimensions": "medium"}
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
            "world_theme": self.world_theme,
            "scene_prompt": self.scene_prompt,
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
        areas_str = execute_task(self.engine, self.level_generator, 'PEG_V2_GET_AREA_NAMES', [],
                                 task_prompt_kwargs={"scene_prompt": self.scene_prompt})
        area_names = [name.strip() for name in areas_str.split(';') if name.strip()]
        ranked_str = execute_task(self.engine, self.level_generator, 'PEG_V2_RANK_AREAS', [],
                                  task_prompt_kwargs={"area_list_str": ", ".join(area_names)})
        ranked_names = [name.strip() for name in ranked_str.split(';') if name.strip()]
        defined_features = []
        for name in ranked_names:
            type_list_str = self._get_feature_menu_by_strategy(['EXTERIOR', 'BLOCKING'])
            raw_response = execute_task(self.engine, self.level_generator, 'PEG_V2_DEFINE_AREA_DATA', [],
                                        task_prompt_kwargs={"scene_prompt": self.scene_prompt, "area_name": name,
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
            "world_theme": self.world_theme,
            "scene_prompt": self.scene_prompt,
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
            # Post-process to ensure only valid strategies are included as a safety net.
            valid_features = []
            for feat in command['features']:
                feat_type = feat.get('type')
                if feat_type:
                    feat_def = self.engine.config.features.get(feat_type, {})
                    if feat_def.get('placement_strategy') in allowed_strategies:
                        valid_features.append(feat)
            if valid_features:
                return valid_features

        # If the primary method fails or returns no valid features, fall back.
        return self._get_initial_features_stepwise()

    def get_narrative_seed(self, area_name: str) -> str:

        seed_kwargs = {"world_theme": self.world_theme, "scene_prompt": self.scene_prompt, "area_name": area_name}
        return execute_task(self.engine, self.level_generator, 'PEG_V3_GET_NARRATIVE_SEED', [],
                            task_prompt_kwargs=seed_kwargs)

    def get_next_narrative_beat(self, context: str, other_features_context: str) -> str:

        return execute_task(self.engine, self.level_generator, 'PROCGEN_GENERATE_NARRATIVE_BEAT', [],
                              task_prompt_kwargs={"world_theme": self.world_theme, "context": context, "other_features_context": other_features_context})

    def choose_parent_feature(self, narrative_log: str, new_feature_sentence: str, parent_options_list: str) -> str:

        choice_kwargs = {
            "narrative_log": narrative_log,
            "new_feature_sentence": new_feature_sentence,
            "parent_options_list": parent_options_list
        }
        return execute_task(self.engine, self.level_generator, 'PEG_V3_CHOOSE_PARENT', [],
                            task_prompt_kwargs=choice_kwargs)

    def choose_path_target_feature(self, narrative_log: str, new_feature_sentence: str, target_options_list: str) -> str:
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
            "world_theme": self.world_theme,
            "scene_prompt": self.scene_prompt,
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