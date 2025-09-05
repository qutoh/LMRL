# /core/common/annotation_manager.py

from core.common import file_io, utils


class AnnotationManager:
    """
    Handles finding and appending metadata annotations to existing LLM interaction logs.
    """
    def __init__(self, engine):
        self.engine = engine

    def annotate_last_log_as_failure(self, reason: str, details: str = ""):
        """
        Retrieves the context of the last LLM interaction and annotates its
        log entry as a failure.
        """
        last_log_info = self.engine.last_interaction_log
        if not last_log_info:
            utils.log_message('debug', "[ANNOTATION WARNING] annotate_last_log_as_failure called but no last_interaction_log was found.")
            return

        log_id = last_log_info.get('log_id')
        log_path = last_log_info.get('log_path')

        if not log_id or not log_path or not file_io.path_exists(log_path):
            utils.log_message('debug', f"[ANNOTATION WARNING] Log file not found at {log_path}")
            return

        logs = file_io.read_json(log_path, default=None)
        if logs is None: return

        log_found = False
        
        if isinstance(logs, list): # Agent log format
            for entry in logs:
                if isinstance(entry, dict) and entry.get("log_id") == log_id:
                    entry["annotation"] = {"status": "FAIL", "reason": reason, "details": details}
                    log_found = True
                    break
        elif isinstance(logs, dict): # Narrative log format
            for session_id, session_data in logs.items():
                if isinstance(session_data, dict) and 'interactions' in session_data:
                    for interaction in session_data['interactions']:
                        if isinstance(interaction, dict) and interaction.get("log_id") == log_id:
                            interaction["annotation"] = {"status": "FAIL", "reason": reason, "details": details}
                            log_found = True
                            break
                if log_found: break
        
        if log_found:
            file_io.write_json(log_path, logs)
        else:
            utils.log_message('debug', f"[ANNOTATION WARNING] Log ID '{log_id}' not found in {log_path}")