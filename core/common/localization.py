# /core/common/localization.py

from core.common import file_io


class Localization:
    def __init__(self, lang_code='en_us'):
        self.strings = {}
        self.logged_missing_keys = set()
        lang_dir = file_io.join_path(file_io.PROJECT_ROOT, 'data', 'lang', lang_code)

        try:
            if not file_io.path_exists(lang_dir):
                raise FileNotFoundError(f"Language directory not found at {lang_dir}")

            # Load single-file configs from the root language directory
            for filename in file_io.list_directory_contents(lang_dir):
                if filename.endswith('.json'):
                    file_path = file_io.join_path(lang_dir, filename)
                    lang_data = file_io.read_json(file_path)
                    if lang_data:
                        # Check for duplicate keys before updating
                        for key, value in lang_data.items():
                            if key in self.strings:
                                print(
                                    f"DUPLICATE_PROMPT_KEY: '{key}' in '{filename}' conflicts with an existing key. Ignoring.")
                                file_io.log_prompt_duplication_error(key, filename)
                            else:
                                self.strings[key] = value
                    else:
                        print(f"WARNING: Could not load or parse language file: {file_path}")

            # Load split prompt files from the 'prompts' subdirectory
            prompts_dir = file_io.join_path(lang_dir, 'prompts')
            if file_io.path_exists(prompts_dir):
                for filename in file_io.list_directory_contents(prompts_dir):
                    if filename.endswith('.json'):
                        file_path = file_io.join_path(prompts_dir, filename)
                        prompt_data = file_io.read_json(file_path)
                        if not prompt_data:
                            print(f"WARNING: Could not load or parse prompt file: {file_path}")
                            continue

                        for key, value in prompt_data.items():
                            if key in self.strings:
                                print(
                                    f"DUPLICATE_PROMPT_KEY: '{key}' in '{filename}' conflicts with an existing key. Ignoring.")
                                file_io.log_prompt_duplication_error(key, filename)
                            else:
                                self.strings[key] = value

        except FileNotFoundError:
            print(f"FATAL ERROR: Language directory not found at {lang_dir}")
            exit()

    def get_raw(self, key, default=None):
        """Retrieves a raw value (string, list, dict) by key without formatting."""
        value = self.strings.get(key)
        if value is None:
            if key not in self.logged_missing_keys:
                print(f"LOC_KEY_NOT_FOUND: '{key}' (logging to file)")
                file_io.log_localization_error(key)
                self.logged_missing_keys.add(key)
            return default if default is not None else {}
        return value

    def get(self, key, **kwargs):
        """Retrieves a string by key and formats it."""
        template = self.strings.get(key)
        if template is None:
            if key not in self.logged_missing_keys:
                print(f"LOC_KEY_NOT_FOUND: '{key}' (logging to file)")
                file_io.log_localization_error(key)
                self.logged_missing_keys.add(key)
            return f"LOC_KEY_NOT_FOUND: {key}"

        try:
            return template.format(**kwargs)
        except (KeyError, TypeError) as e:
            # TypeError can happen if the value is not a string (e.g., a dict from the new file)
            error_key = f"{key} (Format Error)"
            if error_key not in self.logged_missing_keys:
                print(f"LOC_FORMAT_ERROR: Missing key {e} for '{key}' or value is not a string. (logging to file)")
                file_io.log_localization_error(f"{key} - Formatting failed, missing placeholder: {e} or not a string.")
                self.logged_missing_keys.add(error_key)
            return f"LOC_FORMAT_ERROR: for '{key}'"


_localization_instance = Localization()


def loc_raw(key, default=None):
    return _localization_instance.get_raw(key, default=default)


def loc(key, **kwargs):
    return _localization_instance.get(key, **kwargs)