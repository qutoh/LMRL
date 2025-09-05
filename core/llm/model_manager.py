# /core/llm/model_manager.py

import threading

from sentence_transformers import SentenceTransformer

from core.common.config_loader import config
from .llm_api import get_model_context_length


class ModelManager:
    """
    Handles loading, unloading, and managing all required LLM and embedding models
    based on the application's current state.
    """
    def __init__(self):
        self.loading_thread = None
        self.models_loaded = threading.Event()
        self.active_llm_models = {}
        self.embedding_model = None
        self.is_busy = False

    def _get_required_models_for_state(self, state: str) -> set:
        """
        Determines the set of model IDs required for a given application state.
        """
        required_models = set()
        
        # 1. Check all base agents
        for agent_data in config.agents.values():
            if model := agent_data.get('model'):
                required_models.add(model)

        # 2. Check all character default models
        for key in ["DEFAULT_LEAD_MODEL", "DEFAULT_NPC_MODEL", "DEFAULT_DM_MODEL", "DEFAULT_AGENT_MODEL"]:
            if model := config.settings.get(key):
                required_models.add(model)
        
        # 3. Check all task-specific model overrides
        for task_key, task_params in config.task_parameters.items():
            if model_override := task_params.get('model_override'):
                required_models.add(model_override)

        return required_models

    def _update_loaded_models(self, required_models: set, state_name: str = "custom"):
        """The main logic for loading and unloading models. Runs in a background thread."""
        try:
            print(f"[MODEL_LOADER] Updating loaded models for state: {state_name}")
            
            if self.embedding_model is None:
                model_name = config.settings.get('EMBEDDING_MODEL_NAME', 'all-MiniLM-L6-v2')
                self.embedding_model = SentenceTransformer(model_name)
                print(f"[MODEL_LOADER] Embedding model '{model_name}' loaded.")

            print(f"[MODEL_LOADER] State '{state_name}' requires models: {required_models}")

            for model_id in required_models:
                if model_id not in self.active_llm_models:
                    print(f"[MODEL_LOADER] Loading metadata for '{model_id}'...")
                    context_len = get_model_context_length(model_id)
                    self.active_llm_models[model_id] = context_len or 8192
                    print(f"[MODEL_LOADER] ...'{model_id}' context length is {self.active_llm_models[model_id]}.")
            
            models_to_unload = set(self.active_llm_models.keys()) - required_models
            for model_id in models_to_unload:
                del self.active_llm_models[model_id]
                print(f"[MODEL_LOADER] Unloaded model '{model_id}' as it is no longer required.")

        except Exception as e:
            print(f"[MODEL_LOADER_ERROR] Failed during model update: {e}")
        finally:
            self.models_loaded.set()
            self.is_busy = False
            print(f"[MODEL_LOADER] Update for '{state_name}' complete. Active models: {list(self.active_llm_models.keys())}")

    def transition_to_state(self, state: str):
        """Starts a background thread to handle the model loading/unloading for a new high-level state."""
        if self.is_busy: return
        self.is_busy = True
        self.models_loaded.clear()
        
        print(f"[MODEL_LOADER] Starting background transition to state: {state}")
        required_models = self._get_required_models_for_state(state)
        self.loading_thread = threading.Thread(target=self._update_loaded_models, args=(required_models, state), daemon=True)
        self.loading_thread.start()

    def set_active_models(self, models_to_load: list[str]):
        """Starts a background thread to load a specific list of models, unloading any others."""
        if self.is_busy: return
        self.is_busy = True
        self.models_loaded.clear()

        print(f"[MODEL_LOADER] Setting active models to: {models_to_load}")
        self.loading_thread = threading.Thread(target=self._update_loaded_models, args=(set(models_to_load), "calibration"), daemon=True)
        self.loading_thread.start()