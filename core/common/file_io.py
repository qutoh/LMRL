# /core/file_io.py

import os
import json
import shutil
import re
import random
from datetime import datetime

# --- SINGLE SOURCE OF TRUTH FOR PATHS ---
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_CORE_DIR = os.path.dirname(_CURRENT_DIR)
PROJECT_ROOT = os.path.dirname(_CORE_DIR)
SAVE_DATA_ROOT = os.path.join(PROJECT_ROOT, "save_data")
IDEATION_LOG_PATH = os.path.join(PROJECT_ROOT, "ideation_log.txt")
PEG_RECONCILIATION_LOG_PATH = os.path.join(PROJECT_ROOT, "peg_reconciliation_log.txt")
LOCALIZATION_ERROR_LOG_PATH = os.path.join(PROJECT_ROOT, "localization_errors.log")
NAMES_TXT_PATH = os.path.join(PROJECT_ROOT, "res", "names.txt")
WORLD_PREFIX_PATH = os.path.join(PROJECT_ROOT, "res", "world_name_prefix.txt")
WORLD_SUFFIX_PATH = os.path.join(PROJECT_ROOT, "res", "world_name_suffix.txt")


# ---

def get_random_world_name() -> str:
    """Reads from prefix and suffix files to generate a random world name."""
    try:
        with open(WORLD_PREFIX_PATH, 'r', encoding='utf-8') as f_prefix:
            prefixes = [line.strip() for line in f_prefix if line.strip()]
        with open(WORLD_SUFFIX_PATH, 'r', encoding='utf-8') as f_suffix:
            suffixes = [line.strip() for line in f_suffix if line.strip()]

        if prefixes and suffixes:
            return f"{random.choice(prefixes)} {random.choice(suffixes)}"
    except Exception as e:
        print(f"[ERROR] Could not generate random world name: {e}")
    # Fallback
    return f"World_{random.randint(100, 999)}"


def log_prompt_duplication_error(key: str, filename: str):
    """Logs a duplicate prompt key to the localization error file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] DUPLICATE PROMPT KEY: '{key}' found in '{filename}' and was ignored.\n"
    append_to_text_file(LOCALIZATION_ERROR_LOG_PATH, log_entry)


def get_random_name() -> str:
    """Reads all names from names.txt and returns a random one."""
    try:
        if path_exists(NAMES_TXT_PATH):
            with open(NAMES_TXT_PATH, 'r', encoding='utf-8') as f:
                names = [line.strip() for line in f if line.strip()]
                if names:
                    return random.choice(names)
    except Exception as e:
        print(f"[ERROR] Could not read from names.txt: {e}")
    # Fallback in case of file error or empty file
    return f"Character_{random.randint(100, 999)}"


def get_env_variable(variable_name):
    """Safely gets an environment variable."""
    return os.getenv(variable_name)


def join_path(*args):
    """Joins path components into a single path."""
    return os.path.join(*args)


def get_dirname(path):
    """Gets the directory name of a path."""
    return os.path.dirname(path)


def path_exists(path):
    """Checks if a path exists."""
    return os.path.exists(path)


def remove_file(path):
    """Removes a file if it exists."""
    if path_exists(path):
        os.remove(path)


def normalize_path(path):
    """Converts a user-provided path into an absolute path."""
    return os.path.abspath(path)


def create_directory(path, exist_ok=True):
    """Creates a directory if it does not exist."""
    os.makedirs(path, exist_ok=exist_ok)


def copy_file(source, destination):
    """Copies a single file from source to destination."""
    shutil.copy(source, destination)


def rename_directory(source, destination):
    """Renames a directory."""
    try:
        os.rename(source, destination)
        return True, None
    except OSError as e:
        return False, str(e)


def remove_directory(path):
    """Removes a directory and all its contents if it exists."""
    if path_exists(path):
        shutil.rmtree(path)


def list_directory_contents(path):
    """Lists the contents of a directory."""
    if os.path.isdir(path):
        return os.listdir(path)
    return []


def read_json(path, default=None):
    """Reads a JSON file, returning a default value if not found or invalid."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default if default is not None else {}


def write_json(path, data):
    """Writes data to a JSON file."""
    try:
        create_directory(get_dirname(path))
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
        return True
    except Exception:
        return False


def append_to_text_file(path, content):
    """Appends a string to a text file, creating it if necessary."""
    try:
        create_directory(get_dirname(path))
        with open(path, 'a', encoding='utf-8') as f:
            f.write(content)
        return True
    except Exception:
        return False


def log_to_ideation_file(category: str, item: str, context: str = None):
    """
    Logs a new, unique idea from the LLM to the ideation file, now with context.
    """
    try:
        if item.strip().startswith(f"[{category.upper()}]"):
            content = item.strip()
        else:
            content = f"[{category.upper()}] {item.strip()}"

        if context:
            content += f" - Context: {context}"
        content += "\n"

        if path_exists(IDEATION_LOG_PATH):
            with open(IDEATION_LOG_PATH, 'r', encoding='utf-8') as f:
                if any(content.strip() == line.strip() for line in f):
                    return

        append_to_text_file(IDEATION_LOG_PATH, content)
    except Exception as e:
        print(f"[ERROR] Could not write to ideation log: {e}")


def log_localization_error(key: str):
    """Logs a missing localization key to a dedicated file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] Missing localization key: '{key}'\n"
    append_to_text_file(LOCALIZATION_ERROR_LOG_PATH, log_entry)


def get_saved_runs_for_world(world_name: str) -> list[str]:
    """Lists all valid, non-temporary save directories for a given world."""
    sanitized_world_name = sanitize_filename(world_name)
    world_save_path = join_path(SAVE_DATA_ROOT, sanitized_world_name)

    if not path_exists(world_save_path):
        return []

    try:
        # A valid run directory is a directory.
        runs = [
            item for item in os.listdir(world_save_path)
            if os.path.isdir(join_path(world_save_path, item))
        ]
        return sorted(runs)
    except OSError:
        return []


def sanitize_filename(filename):
    """Returns a sanitized version of a string safe for use as a filename."""
    if not filename: return ""
    sanitized = re.sub(r'[<>:"/\\|?*]', '', filename)
    return re.sub(r'\s+', ' ', sanitized).strip()


def setup_new_run_directory(base_data_dir: str, world_name: str):
    """Creates a timestamped directory for a new run within a world-specific folder."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    world_save_path = join_path(SAVE_DATA_ROOT, sanitize_filename(world_name))
    run_path = join_path(world_save_path, f"run_{timestamp}")
    create_directory(run_path)

    for filename in ['leads.json', 'dm_roles.json', 'casting_dms.json']:
        source_path = join_path(base_data_dir, filename)
        if path_exists(source_path):
            copy_file(source_path, run_path)

    write_json(join_path(run_path, 'temporary_npcs.json'), [])
    write_json(join_path(run_path, 'story_state.json'), {})
    write_json(join_path(run_path, 'casting_leads.json'), [])

    return run_path


def finalize_run_directory(current_run_path, final_name):
    """Renames the run directory to a final, sanitized name within its parent folder."""
    clean_name = sanitize_filename(final_name)
    if not clean_name:
        clean_name = f"run_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

    parent_dir = get_dirname(current_run_path)
    new_run_path = join_path(parent_dir, clean_name)

    if path_exists(new_run_path) and new_run_path != current_run_path:
        # If the target exists and is not the source, delete the target first to ensure a clean overwrite.
        # This is safer than appending a timestamp in a debug/overwrite context.
        remove_directory(new_run_path)

    success, error = rename_directory(current_run_path, new_run_path)
    if success:
        return new_run_path, None
    else:
        return current_run_path, error


def save_character_to_world_casting(world_name: str, character: dict, role_type: str):
    """Saves a character's full profile to the persistent casting file for the world."""
    if not world_name or not role_type: return False

    filename_map = {
        'lead': 'casting_leads.json',
        'npc': 'casting_npcs.json',
        'dm': 'casting_dms.json'
    }
    filename = filename_map.get(role_type)
    if not filename: return False

    file_path = join_path(PROJECT_ROOT, 'data', 'worlds', world_name, filename)
    casting_list = read_json(file_path, default=[])

    char_name_lower = character.get('name', '').lower()
    if any(c.get('name', '').lower() == char_name_lower for c in casting_list):
        return True  # Character already exists, no need to save again.

    char_copy = {k: v for k, v in character.items() if not k.startswith('is_')}
    # Clean keys that are added dynamically during a run
    for key in ['role_type', 'controlled_by']:
        if key in char_copy:
            del char_copy[key]

    casting_list.append(char_copy)
    return write_json(file_path, casting_list)


def save_active_character_files(engine):
    """
    Saves the current state of characters from the unified roster back
    to their original, separate files.
    """
    leads, dms, npcs = [], [], []

    clean_characters = []
    for char in engine.characters:
        char_copy = {k: v for k, v in char.items() if not k.startswith('is_')}

        # Convert tuple keys in fused_personas to strings for JSON serialization.
        if 'fused_personas' in char_copy and isinstance(char_copy['fused_personas'], dict):
            # Using str() is safe and sufficient for this purpose.
            string_keyed_personas = {str(key): value for key, value in char_copy['fused_personas'].items()}
            char_copy['fused_personas'] = string_keyed_personas

        if 'role_type' in char_copy:
            clean_characters.append(char_copy)

    for char in clean_characters:
        role_type = char.pop('role_type', None)
        if role_type == 'lead':
            leads.append(char)
        elif role_type == 'dm':
            dms.append(char)
        elif role_type == 'npc':
            npcs.append(char)

    write_json(join_path(engine.run_path, 'leads.json'), leads)
    write_json(join_path(engine.run_path, 'dm_roles.json'), dms)
    write_json(join_path(engine.run_path, 'temporary_npcs.json'), npcs)


def _find_node_by_breadcrumb(world_data: dict, breadcrumb: list[str]) -> dict | None:
    """Helper to traverse the nested world structure and return a specific node."""
    if not breadcrumb or not world_data:
        return None

    current_level = world_data
    for i, step in enumerate(breadcrumb):
        if i == 0:  # The first step is a key in the root dictionary
            if step not in current_level: return None
            current_level = current_level[step]
        else:  # Subsequent steps are keys in the 'Children' dictionary
            if "Children" not in current_level or step not in current_level["Children"]:
                return None
            current_level = current_level["Children"][step]
    return current_level


def add_inhabitant_to_location(world_name: str, breadcrumb_trail: list[str], character_concept: dict):
    """Reads the world file, traverses a path, adds a new inhabitant concept, and saves."""
    if not world_name or not breadcrumb_trail or not character_concept:
        return False

    world_filepath = join_path(PROJECT_ROOT, 'data', 'worlds', world_name, 'world.json')
    world_data = read_json(world_filepath)
    if not world_data: return False

    node_to_modify = _find_node_by_breadcrumb(world_data, breadcrumb_trail)
    if not node_to_modify:
        return False

    if "inhabitants" not in node_to_modify:
        node_to_modify["inhabitants"] = []

    char_name_lower = character_concept.get('name', '').lower()
    existing_names = set()
    for inhabitant in node_to_modify['inhabitants']:
        if isinstance(inhabitant, dict):
            existing_names.add(inhabitant.get('name', '').lower())
        elif isinstance(inhabitant, str):
            existing_names.add(inhabitant.split(' - ')[0].strip().lower())

    if char_name_lower not in existing_names:
        node_to_modify['inhabitants'].append(character_concept)

    return write_json(world_filepath, world_data)


def add_relationship_to_node(world_name: str, breadcrumb: list[str], relationship: str, target_name: str) -> bool:
    """Finds a node by its breadcrumb and adds a new key to its 'relationships' dictionary."""
    if not world_name or not breadcrumb or not relationship or not target_name:
        return False

    world_filepath = join_path(PROJECT_ROOT, 'data', 'worlds', world_name, 'world.json')
    world_data = read_json(world_filepath)
    if not world_data: return False

    node_to_modify = _find_node_by_breadcrumb(world_data, breadcrumb)
    if not node_to_modify: return False

    if "relationships" not in node_to_modify:
        node_to_modify["relationships"] = {}

    node_to_modify["relationships"][relationship] = target_name
    return write_json(world_filepath, world_data)


def add_child_to_world(world_name: str, parent_breadcrumb: list[str], new_child_node: dict) -> bool:
    """Adds a new location as a child of a specified parent."""
    world_filepath = join_path(PROJECT_ROOT, 'data', 'worlds', world_name, 'world.json')
    world_data = read_json(world_filepath)
    if not world_data: return False

    parent_node = _find_node_by_breadcrumb(world_data, parent_breadcrumb)
    if not parent_node: return False

    if "Children" not in parent_node:
        parent_node["Children"] = {}

    parent_node["Children"][new_child_node["Name"]] = new_child_node
    return write_json(world_filepath, world_data)


def update_world_data(world_name: str, new_world_data: dict) -> bool:
    """Saves a new world data structure to the world's world.json file."""
    world_filepath = join_path(PROJECT_ROOT, 'data', 'worlds', world_name, 'world.json')
    return write_json(world_filepath, new_world_data)