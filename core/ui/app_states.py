# /core/ui/app_states.py

from enum import Enum, auto

class AppState(Enum):
    WORLD_SELECTION = auto()
    SETTINGS = auto()
    LOAD_OR_NEW = auto()
    LOAD_GAME_SELECTION = auto()
    SCENE_SELECTION = auto()
    AWAITING_THEME_INPUT = auto()
    AWAITING_SCENE_INPUT = auto()
    WORLD_CREATING = auto()
    LOADING_GAME = auto()
    GAME_RUNNING = auto()
    SHUTTING_DOWN = auto()
    EXITING = auto()
    CALIBRATING = auto()
    PEG_V3_TEST = auto()
    CHARACTER_GENERATION_TEST = auto()
    NOISE_VISUALIZER_TEST = auto()
    ROLE_CREATOR_VIEW = auto()