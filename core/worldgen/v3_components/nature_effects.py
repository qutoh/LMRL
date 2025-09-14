# /core/worldgen/v3_components/nature_effects.py

from typing import Optional, Dict, List, Set, Tuple
import random

from ...common.config_loader import config
from ...common.game_state import GenerationState

_material_tile_cache: Dict[Tuple[str, str], Optional[str]] = {}


def _find_material_override_tile(target_material: str, move_capability: str) -> Optional[str]:
    """Finds a suitable tile type that matches a material and movement capability."""
    cache_key = (target_material, move_capability)
    if cache_key in _material_tile_cache:
        return _material_tile_cache[cache_key]

    for tile_name, tile_def in config.tile_types.items():
        if target_material in tile_def.get('materials', []) and move_capability in tile_def.get('pass_methods', []):
            _material_tile_cache[cache_key] = tile_name
            return tile_name

    _material_tile_cache[cache_key] = None
    return None


def _get_nature_modified_tile_type(default_tile_name: str, nature_names: List[str],
                                   used_natures: Set[str]) -> Tuple[Optional[str], Optional[str]]:
    """
    Determines the final tile type for a feature part (floor/border) based on its natures,
    applying conflict resolution policies.
    """
    if not nature_names or not default_tile_name:
        return default_tile_name, None

    candidates = []
    default_tile_def = config.tile_types.get(default_tile_name, {})
    default_materials = default_tile_def.get('materials', [])
    default_pass_methods = default_tile_def.get('pass_methods', [])

    for nature_name in nature_names:
        nature_def = config.natures.get(nature_name, {})
        modifiers = nature_def.get('tile_modifiers', {})

        # 1. Check for direct overrides
        if direct_override := modifiers.get('direct_overrides', {}).get(default_tile_name):
            candidates.append({'nature': nature_name, 'tile': direct_override})
            continue

        # 2. Check for material overrides
        for rule in modifiers.get('material_overrides', []):
            source_material = rule.get('source_material')
            if source_material in default_materials:
                move_match = rule.get('movement_capability_match')
                if move_match in default_pass_methods:
                    target_tile = _find_material_override_tile(rule.get('target_material'), move_match)
                    if target_tile:
                        candidates.append({'nature': nature_name, 'tile': target_tile})

    if not candidates:
        return default_tile_name, None

    if len(candidates) == 1:
        return candidates[0]['tile'], candidates[0]['nature']

    # Conflict Resolution
    policy = config.natures.get(candidates[0]['nature'], {}).get('conflict_resolution_policy', 'GREEDY')

    if policy == 'RANDOM':
        chosen = random.choice(candidates)
        return chosen['tile'], chosen['nature']

    if policy == 'BALANCED':
        # Prioritize a nature that hasn't been used yet for this feature
        for candidate in candidates:
            if candidate['nature'] not in used_natures:
                return candidate['tile'], candidate['nature']

    # Default to GREEDY (first one found)
    return candidates[0]['tile'], candidates[0]['nature']


def bake_nature_modifications_into_state(generation_state: GenerationState):
    """
    Iterates through all placed features in a GenerationState and applies nature-driven
    tile modifications, saving the result directly into the feature's data dictionary.
    This "bakes" the dynamic effects into the saved state.
    """
    if not generation_state or not generation_state.placed_features:
        return

    for feature_tag, feature_data in generation_state.placed_features.items():
        feature_type_key = feature_data.get('type')
        if not feature_type_key:
            continue

        feature_def = config.features.get(feature_type_key, {})
        nature_names = feature_def.get('natures', [])
        used_natures_for_feature = set()

        default_floor_name = feature_def.get('tile_type')
        default_border_name = feature_def.get('border_tile_type')

        final_floor_name, used_nature_floor = _get_nature_modified_tile_type(
            default_floor_name, nature_names, used_natures_for_feature
        )
        if used_nature_floor:
            used_natures_for_feature.add(used_nature_floor)

        final_border_name, _ = _get_nature_modified_tile_type(
            default_border_name, nature_names, used_natures_for_feature
        )

        if final_floor_name != default_floor_name:
            feature_data['modified_tile_type'] = final_floor_name
        if final_border_name != default_border_name:
            feature_data['modified_border_tile_type'] = final_border_name