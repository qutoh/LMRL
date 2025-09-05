# /core/common/command_parser.py

import json
import re
from typing import Callable, Optional

from .config_loader import config
from .localization import loc
from .utils import log_message, get_chosen_name_from_response, clean_json_from_llm
from ..llm.llm_api import execute_task


def _clean_json_from_text(raw_text: str) -> str:
    """
    Strips markdown and other garbage from a potential JSON string using a multi-step process.
    """
    if not raw_text: return ""

    # 1. Prioritize finding a Markdown JSON block.
    markdown_match = re.search(r"```(?:json)?\s*\n(.*?)\n```", raw_text, re.DOTALL)
    if markdown_match:
        return markdown_match.group(1).strip()

    # 2. If no markdown, fall back to finding the first and last curly brace.
    brace_match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if brace_match:
        return brace_match.group(0)

    # 3. If no JSON object is found, return the stripped text.
    return raw_text.strip()


def _context_aware_fallback(engine, raw_text: str, fallback_task_key: str, fallback_prompt_kwargs: dict):
    """
    Calls the Command Handler with a specific 'fix-it' prompt based on the failure context.
    """
    log_message('debug', f"[PARSER] Primary parse failed. Using context-aware fallback: '{fallback_task_key}'")
    handler_agent = config.agents['COMMAND_HANDLER']

    if fallback_prompt_kwargs is None: fallback_prompt_kwargs = {}
    fallback_prompt_kwargs['raw_text'] = raw_text

    handler_response = execute_task(
        engine,
        handler_agent,
        fallback_task_key,
        [],
        task_prompt_kwargs=fallback_prompt_kwargs
    )

    if not handler_response:
        log_message('debug', "[PARSER] Context-aware fallback returned no response.")
        return None

    clean_handler_response = _clean_json_from_text(handler_response)
    try:
        command = json.loads(clean_handler_response)
        if isinstance(command, dict) and command:
            log_message('debug', "[PARSER] Context-aware fallback SUCCEEDED.")
            return command
    except (json.JSONDecodeError, TypeError):
        log_message('debug', "[PARSER] Context-aware fallback FAILED to produce valid JSON.")
        return None


def parse_dm_command_with_legacy_regex(raw_text):
    """
    Parses DM output for prose and a 'NEXT: <Actor>' command.
    """
    if not raw_text: return {"prose": "", "next_action": None}

    chosen_name = get_chosen_name_from_response(raw_text)

    if chosen_name:
        clean_prose = re.sub(rf"^\s*{config.settings['NEXT_TURN_KEYWORD']}.*$", "", raw_text,
                             flags=re.MULTILINE | re.IGNORECASE).strip()
        action = "CREATE_NPC" if "new" in chosen_name.lower() or "create" in chosen_name.lower() else "NEXT"
        target = re.sub(r'\((new|create)\)', '', chosen_name, flags=re.IGNORECASE).strip()
        return {"prose": clean_prose, "next_action": {"action": action, "target_name": target}}

    return {"prose": raw_text.strip(), "next_action": None}


def parse_structured_command(
        engine,
        raw_response: str,
        agent_type: str,
        fallback_task_key: str | None = None,
        fallback_prompt_kwargs: dict | None = None,
        validator: Optional[Callable[[dict], bool]] = None
):
    """
    Parses a structured command from raw text, using a new context-aware fallback system.
    """
    if not raw_response:
        return {"action": "NONE"}

    clean_response = _clean_json_from_text(raw_response)
    try:
        command = json.loads(clean_response)
        if isinstance(command, dict) and command:
            if validator is None or validator(command):
                return command
            else:
                log_message('debug', "[PARSER] Initial JSON parse passed but failed validation.")
                raise json.JSONDecodeError("Validation failed", clean_response, 0)
    except (json.JSONDecodeError, TypeError):
        pass

    if fallback_task_key:
        command_from_handler = _context_aware_fallback(
            engine, raw_response, fallback_task_key, fallback_prompt_kwargs
        )
        if command_from_handler:
             if validator is None or validator(command_from_handler):
                return command_from_handler
             else:
                log_message('debug', "[PARSER] Fallback JSON parse passed but failed validation.")

    if agent_type == 'DIRECTOR':
        log_message('debug', "[PARSER] All other methods failed. Falling back to legacy regex for Director.")
        if re.search(r'REMOVE', raw_response, re.IGNORECASE):
            return {"action": "REMOVE", "replacement_type": "CREATE"}

    log_message('debug', f"[PARSER] All parsing methods failed for agent type '{agent_type}'.")
    return {"action": "NONE"}


def rewrite_response_for_role_compliance(engine, character, original_response):
    log_message('debug', loc('system_rewrite_active'))

    corrected_response = execute_task(
        engine, character, 'REWRITE_FOR_ROLE_COMPLIANCE',
        [{"role": "user", "content": original_response}],
        task_prompt_kwargs={"original_response": original_response}
    )
    if corrected_response and corrected_response.strip() != original_response.strip():
        log_message('full',
                    f"--- REWRITE STAGE ---\n[ORIGINAL]\n{original_response}\n\n[CORRECTED]\n{corrected_response}\n--- END REWRITE ---")
        log_message('debug', loc('system_rewrite_corrected'))
        return corrected_response
    elif not corrected_response:
        log_message('debug', loc('system_rewrite_empty'))
        return original_response
    else:
        log_message('debug', loc('system_rewrite_no_change'))
        return original_response


def post_process_llm_response(engine, character, raw_response, is_dm_role):
    if not raw_response: return ""
    processed_response = raw_response.strip()
    markdown_match = re.search(r"```(?:json)?\s*\n(.*?)\n```", processed_response, re.DOTALL)
    if markdown_match:
        processed_response = markdown_match.group(1).strip()
    try:
        json.loads(processed_response)
        return processed_response
    except json.JSONDecodeError:
        pass
    prefix_pattern = r"^(?:\s*(?:\(|- )?\s*([A-Za-z0-9, ]+?)\s*\)?\s*[:：])\s*"
    if config.settings.get('LLM_PREFIX_CLEANUP_ENABLED', False):
        match = re.match(prefix_pattern, processed_response)
        if match:
            detected_name = match.group(1).strip().lower()
            all_active_characters = engine.characters if engine else []
            is_known_character_prefix = any(
                detected_name in char_obj['name'].lower() for char_obj in all_active_characters)
            if is_known_character_prefix:
                is_own_prefix = detected_name in character['name'].lower()
                log_key = 'system_prefix_detected' if is_own_prefix else 'system_role_breaking_prefix_detected'
                log_message('debug', loc(log_key, prefix=match.group(0), character_name=character['name']))
                cleanup_mode = config.settings.get('LLM_PREFIX_CLEANUP_MODE', 'strip').lower()
                if cleanup_mode == 'rewrite':
                    processed_response = rewrite_response_for_role_compliance(engine, character, processed_response)
                elif cleanup_mode == 'strip':
                    processed_response = re.sub(prefix_pattern, '', raw_response, 1).strip()
    if config.settings.get('ENABLE_REWRITE_STAGE', False) and config.settings.get(
            'LLM_PREFIX_CLEANUP_MODE') != 'rewrite':
        processed_response = rewrite_response_for_role_compliance(engine, character, processed_response)
    return processed_response