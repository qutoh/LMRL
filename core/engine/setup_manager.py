# /core/engine/setup_manager.py

from datetime import datetime

from core.common import file_io, utils
from core.common.game_state import GenerationState, MapArtist, LayoutGraph
from core.common.localization import loc
from core.components import position_manager
from core.components import roster_manager
from core.components.memory_manager import MemoryManager
from core.worldgen.procgen_manager import ProcGenManager
from core.worldgen.v3_components.v3_llm import V3_LLM


class SetupManager:
    """
    Manages the entire setup and initialization process for the story engine.
    """

    def __init__(self, engine):
        self.engine = engine
        self.game_state = engine.game_state

    def _reconstruct_generation_state_from_cache(self, cached_level_data: dict) -> GenerationState | None:
        """Helper function to build a GenerationState object from cached level data."""
        if not cached_level_data or "placed_features" not in cached_level_data:
            return None

        generation_state = GenerationState(self.game_state.game_map)
        generation_state.placed_features = cached_level_data.get("placed_features", {})
        generation_state.narrative_log = cached_level_data.get("narrative_log", "")
        generation_state.exterior_tile_type = cached_level_data.get("exterior_tile_type", "DEFAULT_FLOOR")
        generation_state.physics_layout = cached_level_data.get("physics_layout")
        generation_state.feature_embeddings = cached_level_data.get("feature_embeddings", {})

        if 'layout_graph' in cached_level_data:
            graph_data = cached_level_data['layout_graph']
            generation_state.layout_graph = LayoutGraph()
            generation_state.layout_graph.nodes = graph_data.get('nodes', {})
            generation_state.layout_graph.edges = graph_data.get('edges', [])

        return generation_state

    def initialize_run(self) -> bool:
        try:
            import google.generativeai as genai
            if key := self.engine.config.settings.get('GEMINI_API_KEY'): genai.configure(api_key=key)
        except Exception as e:
            utils.log_message('debug', loc('error_gemini_config', e=e))

        success = self._load_existing_run() if self.engine.load_path else self._initialize_new_run()

        if success:
            db_path = file_io.join_path(self.engine.run_path, 'vectordb')
            self.engine.memory_manager = MemoryManager(db_path=db_path, embedding_model=self.engine.embedding_model)
            if self.engine.load_path:
                state_data = file_io.read_json(file_io.join_path(self.engine.run_path, 'story_state.json'), default={})
                if time_str := state_data.get('current_game_time'):
                    try:
                        self.engine.current_game_time = datetime.fromisoformat(time_str)
                    except (ValueError, TypeError):
                        utils.log_message('debug', loc('warning_time_parse_fail'))

            self.engine.render_queue.put(self.game_state)
            if self.engine.dialogue_log:
                log_key = 'log_story_resumes' if self.engine.load_path else 'log_story_start'
                # The detailed game log is now the primary source of startup info, so we simplify this.
                if self.engine.load_path:
                    utils.log_message('game', loc(log_key))

        return success

    def _determine_context_limit(self):
        """
        Calculates the global context threshold based on the minimum context
        length of all models used for full, narrative context generation.
        """
        self.engine.render_queue.put(('ADD_EVENT_LOG', 'Determining context limit for narrative...'))
        full_context_models = set()
        for agent_data in self.engine.config.agents.values():
            if agent_data.get("build_full_context", True):
                if model := agent_data.get('model'):
                    full_context_models.add(model)
        for character in self.engine.characters:
            if model := character.get('model'):
                full_context_models.add(model)
        for key in ["DEFAULT_LEAD_MODEL", "DEFAULT_NPC_MODEL", "DEFAULT_DM_MODEL"]:
            if model := self.engine.config.settings.get(key):
                full_context_models.add(model)
        utils.log_message('debug', f"[SYSTEM] Models considered for context threshold: {list(full_context_models)}")
        valid_lengths = []
        for model_id in full_context_models:
            if length := self.engine.model_context_limits.get(model_id):
                if length > 0:
                    valid_lengths.append(length)
        self.engine.token_context_limit = min(valid_lengths) if valid_lengths else 8192
        offset = self.engine.config.settings.get('TOKEN_SUMMARY_OFFSET', 2000)
        self.engine.token_summary_threshold = max(500, self.engine.token_context_limit - offset)
        utils.log_message('debug', loc('log_initial_threshold', limit=self.engine.token_context_limit,
                                       threshold=self.engine.token_summary_threshold))

    def _cache_newly_generated_level(self, scene_prompt, generation_state):
        if not self.engine.config.settings.get("PREGENERATE_LEVELS_ON_FIRST_RUN"): return
        world_name = self.engine.world_name
        levels_path = file_io.join_path(self.engine.config.data_dir, 'worlds', world_name, 'levels.json')
        levels_cache = file_io.read_json(levels_path, default={})

        serializable_features = {}
        if hasattr(generation_state, 'placed_features'):
            for tag, feature in generation_state.placed_features.items():
                if 'bounding_box' in feature and feature.get('bounding_box'):
                    feature['bounding_box'] = [int(v) for v in feature['bounding_box']]
                serializable_features[tag] = feature

        level_data_to_cache = {
            "placed_features": serializable_features,
            "narrative_log": generation_state.narrative_log,
            "exterior_tile_type": getattr(generation_state, 'exterior_tile_type', 'DEFAULT_FLOOR'),
            "feature_embeddings": getattr(generation_state, 'feature_embeddings', {})
        }

        if hasattr(generation_state, 'layout_graph') and generation_state.layout_graph:
            level_data_to_cache['layout_graph'] = {
                'nodes': generation_state.layout_graph.nodes,
                'edges': generation_state.layout_graph.edges
            }

        if hasattr(generation_state, 'physics_layout') and generation_state.physics_layout:
            level_data_to_cache['physics_layout'] = generation_state.physics_layout

        levels_cache[scene_prompt] = level_data_to_cache

        if file_io.write_json(levels_path, levels_cache):
            self.engine.config.levels = levels_cache

    def _initialize_new_run(self) -> bool:
        self.engine.run_path = file_io.setup_new_run_directory(self.engine.config.data_dir, self.engine.world_name)
        roster_manager.load_initial_roster(self.engine)
        self.engine.lead_character_summary = ""

        scene_prompt = self.engine.scene_prompt or "A story begins."
        self.engine.render_queue.put(('ADD_EVENT_LOG', f"Scene: {scene_prompt[:50]}..."))
        self.engine.current_scene_key = scene_prompt

        chosen_scene = {
            "scene_prompt": scene_prompt,
            "source_location": self.engine.starting_location
        }
        run_scene_path = file_io.join_path(self.engine.run_path, 'scene.json')
        file_io.write_json(run_scene_path, [chosen_scene])

        # --- Generation Phase ---
        cached_level = self.engine.config.levels.get(scene_prompt)
        generation_state = self._reconstruct_generation_state_from_cache(cached_level)

        if not generation_state:
            # --- New: LLM chooses exterior tile ---
            llm = V3_LLM(self.engine)
            tiles = {
                name: data for name, data in self.engine.config.tile_types.items()
                if name != "VOID_SPACE"
            }
            tile_options_str = "\n".join(
                f"- `{name}`: {data.get('description', 'No description.')}" for name, data in tiles.items()
            )
            chosen_tile = llm.choose_exterior_tile(scene_prompt, tile_options_str)
            if chosen_tile not in tiles:
                chosen_tile = "VOID_SPACE"  # Fallback

            utils.log_message('debug', f"[PEG Setup] LLM chose '{chosen_tile}' as the exterior tile.")

            procgen = ProcGenManager(self.engine)

            def redraw_and_update_ui(gen_state):
                map_artist = MapArtist()
                map_artist.draw_map(self.game_state.game_map, gen_state, self.engine.config.features)
                self.engine.render_queue.put(self.game_state)
                import time
                time.sleep(0.05)

            generation_state = procgen.generate(scene_prompt, self.game_state.game_map,
                                                ui_callback=redraw_and_update_ui)
            generation_state.exterior_tile_type = chosen_tile
            self._cache_newly_generated_level(scene_prompt, generation_state)

        self.engine.generation_state = generation_state
        self.engine.layout_graph = generation_state.layout_graph

        map_artist = MapArtist()
        map_artist.draw_map(self.game_state.game_map, generation_state, self.engine.config.features)

        # --- Character and Casting Phase ---
        # Load leads and DMs, but defer inhabitants and proc-gen characters
        roster_manager.load_characters_from_scene(self.engine, chosen_scene, hydrate_inhabitants=False)

        # Store character concepts from world file and proc-gen for later hydration
        self.engine.dehydrated_npcs.extend(
            chosen_scene.get('source_location', {}).get('inhabitants', [])
        )
        self.engine.dehydrated_npcs.extend(generation_state.character_creation_queue)

        location_summary = f"{chosen_scene['source_location'].get('Name', '')}: {chosen_scene['source_location'].get('Description', '')}"
        character_groups = self.engine.director_manager.establish_initial_cast(scene_prompt, location_summary)

        self._determine_context_limit()

        if self.game_state:
            roster_manager.spawn_entities_from_roster(self.engine, self.game_state)

            # --- New Group-based Placement ---
            placed_character_names = set()
            for group in character_groups:
                position_manager.place_character_group_contextually(self.engine, self.game_state, group,
                                                                    generation_state)
                for char in group['characters']:
                    placed_character_names.add(char['name'])

            # --- Place remaining individual characters ---
            for character in self.engine.characters:
                if character.get('is_positional') and character['name'] not in placed_character_names:
                    position_manager.place_character_contextually(self.engine, self.game_state, character,
                                                                  generation_state,
                                                                  [])  # No context needed for individuals now

        # Build a descriptive summary of the entire generated level for the player.
        level_description_sentences = []
        if generation_state and generation_state.placed_features:
            if generation_state.narrative_log:
                level_description_sentences.append(generation_state.narrative_log)
            for tag, feature in sorted(generation_state.placed_features.items()):
                desc = feature.get('description')
                if desc:
                    level_description_sentences.append(desc)

        full_level_context = " ".join(level_description_sentences)
        formatted_level_context = utils.format_text_with_paragraph_breaks(full_level_context, 3)
        formatted_world_theme = utils.format_text_with_paragraph_breaks(self.engine.world_theme, 3)
        formatted_scene_prompt = utils.format_text_with_paragraph_breaks(scene_prompt, 3)

        log_header = f"---\n**WORLD:** {self.engine.world_name}\n\n**THEME:** {formatted_world_theme}\n\n**SCENE:** {formatted_scene_prompt}\n---"
        utils.log_message('game', f"{log_header}\n\n{formatted_level_context}\n---")

        narrative_intro = generation_state.narrative_log if generation_state and generation_state.narrative_log else ""
        enhanced_prompt = f"{narrative_intro}\n\n{scene_prompt}".strip()
        self.engine.dialogue_log.append({"speaker": "Scene Setter", "content": enhanced_prompt})
        file_io.save_active_character_files(self.engine)
        self.engine.render_queue.put(('ADD_EVENT_LOG', 'Setup complete. The story begins...'))
        return True

    def _load_existing_run(self) -> bool:
        self.engine.run_path = self.engine.load_path
        roster_manager.load_initial_roster(self.engine)
        self._determine_context_limit()

        state_data = file_io.read_json(file_io.join_path(self.engine.run_path, 'story_state.json'), default={})
        self.engine.dialogue_log = state_data.get('dialogue_log', [])
        self.engine.summaries = state_data.get('summaries', [])
        self.engine.lead_character_summary = state_data.get('lead_character_summary', "")
        self.engine.current_scene_key = state_data.get('current_scene_key')

        if self.engine.current_scene_key:
            cached_level = self.engine.config.levels.get(self.engine.current_scene_key)
            if generation_state := self._reconstruct_generation_state_from_cache(cached_level):
                self.engine.generation_state = generation_state
                self.engine.layout_graph = generation_state.layout_graph
                map_artist = MapArtist()
                map_artist.draw_map(self.game_state.game_map, generation_state, self.engine.config.features)
                utils.log_message('debug',
                                  f"Reconstructed generation state and drew map from cache for scene: '{self.engine.current_scene_key}'")

        if not self.engine.characters: return False
        if self.game_state:
            roster_manager.spawn_entities_from_roster(self.engine, self.game_state)
        return True