# /core/llm/llm_api.py

import re
import uuid
from datetime import datetime
from typing import Callable, Optional

import google.generativeai as genai
import lmstudio as lms
from lmstudio import json_api as lms_json_api

from core.common.config_loader import config
from . import sampling_manager
from ..common import file_io
from ..common.localization import loc
from ..common.utils import log_message, get_text_from_messages


def _parse_and_strip_thinking_block(raw_text: str, model_id: str) -> str:
    """
    Checks for model-specific parsing rules and processes the raw LLM output using a
    multi-step, case-insensitive approach to robustly isolate the final response.
    """
    parser_config = config.model_tuning.get(model_id)
    if not parser_config:
        return raw_text.strip()

    text_to_process = raw_text
    think_start = parser_config.get("think_start_str")
    think_end = parser_config.get("think_end_str")
    response_start = parser_config.get("response_start_str")
    response_end = parser_config.get("response_end_str")

    # Priority 1: Find and extract a dedicated response block if it exists.
    # This is considered the definitive answer if present.
    if response_start and response_end:
        pattern = re.compile(re.escape(response_start) + r'(.*?)' + re.escape(response_end), re.DOTALL | re.IGNORECASE)
        match = re.search(pattern, text_to_process)
        if match:
            return match.group(1).strip()

    # Priority 2: If no dedicated response block, aggressively remove any and all complete thinking blocks.
    # This handles the common case of: <think>...</think>Answer
    if think_start and think_end:
        pattern = re.compile(re.escape(think_start) + r'.*?' + re.escape(think_end), re.DOTALL | re.IGNORECASE)
        text_to_process = re.sub(pattern, '', text_to_process)

    # Priority 3: As a final cleanup, if an end-of-thought tag still exists,
    # take everything that comes after its last occurrence. This handles cases
    # where the model might have omitted a start tag.
    if think_end:
        # We split on the end tag, and if successful, the desired text is the last part.
        parts = re.split(re.escape(think_end), text_to_process, flags=re.IGNORECASE)
        if len(parts) > 1:
            # Take the last part of the split and strip it.
            potential_response = parts[-1].strip()
            if potential_response:
                return potential_response

    # Fallback: If all else fails, return the text (which may have had think blocks removed)
    return text_to_process.strip()


def _prepare_messages_for_llm(
        persona: str,
        task_prompt_key: str | None,
        messages: list,
        task_prompt_kwargs: dict | None,
        model_id: str,
        enable_reasoning: bool
) -> list[dict]:
    """
    Assembles the final list of messages to be sent to the LLM, respecting model-specific
    tuning preferences for persona placement and system prompt construction.
    """
    tuning_config = config.model_tuning.get(model_id, {})
    system_parts = []

    # 1. Add optional prefix
    if prefix := tuning_config.get("system_prompt_prefix"):
        system_parts.append(prefix)

    # 2. Handle persona placement
    final_messages = list(messages)
    if tuning_config.get("prefer_user_prompt_for_persona", False):
        user_message_found = False
        for msg in final_messages:
            if msg.get('role') == 'user':
                msg['content'] = f"{persona}\n\n---\n\n{msg['content']}"
                user_message_found = True
                break
        if not user_message_found:
            final_messages.insert(0, {"role": "user", "content": persona})
    else:
        system_parts.append(persona)

    # 3. Check for model-specific reasoning disable command
    if not enable_reasoning:
        if disable_cmd := tuning_config.get("disable_reasoning_command"):
            system_parts.append(disable_cmd)
            log_message('full', f"[LLM_API] Injected disable reasoning command for model {model_id}.")

    # 4. Add optional suffix
    if suffix := tuning_config.get("system_prompt_suffix"):
        system_parts.append(suffix)

    # 5. Assemble final system content
    system_content = "\n\n".join(part for part in system_parts if part)

    # --- REFINED LOGIC: Reliably create user message from task prompt key ---
    if task_prompt_key:
        task_content = loc(task_prompt_key, **(task_prompt_kwargs or {}))
        final_messages.append({"role": "user", "content": task_content})
    # --- END REFINEMENT ---

    # Final sanity checks
    if not any(msg.get('role') == 'user' for msg in final_messages):
        log_message('debug', "[LLM_API WARNING] No user messages in final prompt. Adding generic fallback.")
        final_messages.append({"role": "user", "content": "What do you do next?"})

    final_prompt = []
    if system_content:
        final_prompt.append({"role": "system", "content": system_content})

    final_prompt.extend(final_messages)
    return final_prompt


def _prepare_task_execution(
        engine, agent_or_character: dict, task_key: str, messages: list,
        task_prompt_kwargs: dict | None = None, temperature_override: float | None = None,
        top_k_override: int | None = None, top_p_override: float | None = None) -> dict:
    """Consolidates the logic for preparing all parameters for an LLM task execution."""
    agent_name = agent_or_character.get('name', 'unknown_agent')
    role_type = agent_or_character.get('role_type')
    task_params = config.task_parameters.get(task_key, {})

    if task_params.get("is_failure_handler"):
        engine.annotation_manager.annotate_last_log_as_failure('AUTOMATIC_HANDLER_TRIGGERED',
                                                               f"Triggered by task: {task_key}")

    # --- THIS IS THE FIX ---
    # Establish the base model from the agent/character.
    model_identifier = agent_or_character.get('model')
    is_turn_task = task_key in ['CHARACTER_TURN', 'DM_TURN']

    # Check for a task-specific override, but only if it's a non-empty string.
    if not is_turn_task and task_params:
        if model_override := task_params.get('model_override'):
            model_identifier = model_override
    # --- END FIX ---

    sampling_params = sampling_manager.get_sampling_params(
        task_key, model_identifier, agent_or_character,
        temperature_override, top_k_override, top_p_override
    )

    if persona_desc := agent_or_character.get('persona_description'):
        instructions = agent_or_character.get('instructions', '')
        persona = f"{persona_desc.strip()} {instructions}"
    else:
        persona = agent_or_character.get('persona', '')

    enable_reasoning_for_task = task_params.get(
        'enable_reasoning',
        agent_or_character.get(
            'enable_reasoning',
            config.settings.get(f"ENABLE_{role_type.upper()}_REASONING", True) if role_type else True
        )
    )

    task_prompt_key = task_params.get('task_prompt_key') if not is_turn_task else None
    full_messages = _prepare_messages_for_llm(
        persona, task_prompt_key, messages, task_prompt_kwargs, model_identifier, enable_reasoning_for_task
    )

    return {
        "model_identifier": model_identifier,
        "sampling_params": sampling_params,
        "full_messages": full_messages,
        "agent_name": agent_name,
        "role_type": role_type
    }


def _log_interaction(engine, agent_or_character: dict, task_key: str, execution_data: dict, raw_response: str,
                     log_id: str) -> Optional[str]:
    """Consolidates the logic for creating and writing an interaction log entry."""
    agent_name = execution_data['agent_name']
    role_type = execution_data['role_type']
    model_identifier = execution_data['model_identifier']
    sampling_params = execution_data['sampling_params']
    full_messages = execution_data['full_messages']

    log_entry = {
        "log_id": log_id, "timestamp": datetime.now().isoformat(),
        "model_identifier": model_identifier, "temperature": sampling_params['temperature'],
        "top_p": sampling_params['top_p'], "top_k": sampling_params['top_k'],
        "prompt": full_messages, "raw_response": raw_response, "annotation": {"status": "PASS"}
    }

    log_path = ""
    try:
        if role_type in ['lead', 'npc', 'dm'] and hasattr(engine, 'character_log_sessions'):
            sanitized_char_name = file_io.sanitize_filename(agent_name)
            log_path = file_io.join_path(file_io.PROJECT_ROOT, 'logs', 'Narrative', f"{sanitized_char_name}.json")
            session_info = engine.character_log_sessions.get(agent_name, {})
            current_instructions = agent_or_character.get('instructions', '')
            session_id = session_info.get('session_id')
            if not session_id or session_info.get('instructions') != current_instructions:
                session_id = str(uuid.uuid4())
                engine.character_log_sessions[agent_name] = {'session_id': session_id,
                                                             'instructions': current_instructions}
            char_logs = file_io.read_json(log_path, default={})
            if session_id not in char_logs:
                char_logs[session_id] = {'instructions': current_instructions, 'interactions': []}
            char_logs[session_id]['interactions'].append(log_entry)
            file_io.write_json(log_path, char_logs)
        else:
            sanitized_agent_name = file_io.sanitize_filename(agent_name)
            sanitized_task_name = file_io.sanitize_filename(task_key)
            log_path = file_io.join_path(file_io.PROJECT_ROOT, 'logs', sanitized_agent_name,
                                         f"{sanitized_task_name}.json")
            existing_logs = file_io.read_json(log_path, default=[])
            if not isinstance(existing_logs, list): existing_logs = [existing_logs]
            existing_logs.append(log_entry)
            file_io.write_json(log_path, existing_logs)
    except Exception as e:
        log_message('debug', f"[LLM LOGGING ERROR] Could not write log. Error: {e}")
        return None

    return log_path


def execute_task(engine, agent_or_character: dict, task_key: str, messages: list,
                 task_prompt_kwargs: dict | None = None, temperature_override: float | None = None,
                 top_k_override: int | None = None, top_p_override: float | None = None) -> str:
    """
    A wrapper for LLM calls that uses the sampling manager to determine
    the optimal parameters. It also logs the interaction.
    """
    task_params = config.task_parameters.get(task_key, {})
    if task_params.get("player_takeover_enabled") is True:
        persona = agent_or_character.get('persona') or agent_or_character.get('instructions', '')
        handler_kwargs = {
            "agent_name": agent_or_character.get('name', 'unknown_agent'), "task_key": task_key,
            "task_prompt_kwargs": task_prompt_kwargs, "persona": persona, "messages": messages
        }

        player_response = None
        if engine.render_queue and engine.player_interface:
            player_response = engine.player_interface.handle_task_takeover(**handler_kwargs)
        elif engine.ui_manager:
            player_response = engine.ui_manager.get_player_task_input(**handler_kwargs)

        if player_response is not None:
            log_message('debug', f"[PLAYER TAKEOVER] Task '{task_key}' intercepted. Using player response.")
            return player_response
        else:
            log_message('debug', f"[PLAYER TAKEOVER] Player cancelled or passed. Passing task '{task_key}' to LLM.")

    execution_data = _prepare_task_execution(
        engine, agent_or_character, task_key, messages,
        task_prompt_kwargs, temperature_override, top_k_override, top_p_override
    )

    raw_response = ""
    max_retries = config.settings.get("NONSTREAMING_BLACKLIST_RETRIES", 0)

    if max_retries > 0:
        banned_words = config.global_blacklist.get("words", [])
        banned_phrases = config.global_blacklist.get("phrases", [])
        blacklist_patterns = []
        if banned_words:
            blacklist_patterns.append(r'\b(' + '|'.join(re.escape(w) for w in banned_words) + r')\b')
        if banned_phrases:
            blacklist_patterns.append('(' + '|'.join(re.escape(p) for p in banned_phrases) + ')')
        blacklist_regex = re.compile('|'.join(blacklist_patterns), re.IGNORECASE) if blacklist_patterns else None

        current_messages = list(execution_data['full_messages'])
        for attempt in range(max_retries):
            response_attempt = get_llm_response(
                engine, agent_or_character, current_messages,
                temperature=execution_data['sampling_params']['temperature'],
                top_k=execution_data['sampling_params']['top_k'],
                top_p=execution_data['sampling_params']['top_p'],
                model_override=execution_data['model_identifier']
            )
            raw_response = response_attempt

            if blacklist_regex and blacklist_regex.search(raw_response):
                log_message('debug',
                            f"[BLACKLIST] Non-streaming blacklist triggered. Retrying ({attempt + 1}/{max_retries}).")
                current_messages.append({"role": "assistant", "content": raw_response})
                current_messages.append({"role": "user", "content": loc('system_prompt_blacklist_retry')})
                if attempt == max_retries - 1:
                    log_message('debug', "Max retries reached for non-streaming blacklist.")
                    break
            else:
                break  # Success
    else:
        raw_response = get_llm_response(
            engine, agent_or_character, execution_data['full_messages'],
            temperature=execution_data['sampling_params']['temperature'],
            top_k=execution_data['sampling_params']['top_k'],
            top_p=execution_data['sampling_params']['top_p'],
            model_override=execution_data['model_identifier']
        )

    log_id = str(uuid.uuid4())
    log_path = _log_interaction(engine, agent_or_character, task_key, execution_data, raw_response, log_id)
    if log_path and hasattr(engine, 'last_interaction_log'):
        engine.last_interaction_log = {"log_id": log_id, "log_path": log_path}

    return _parse_and_strip_thinking_block(raw_response, execution_data['model_identifier'])


def execute_task_streaming(
        engine, agent_or_character: dict, task_key: str, messages: list,
        on_token_stream: Callable[..., None],
        should_stop: Callable[[str], bool],
        task_prompt_kwargs: dict | None = None,
        temperature_override: float | None = None,
        top_k_override: int | None = None,
        top_p_override: float | None = None
) -> str:
    """
    A wrapper for streaming LLM calls that supports callbacks for token processing,
    early termination, a global word/phrase blacklist with retries, and falls back
    to a non-streaming call on failure. The `on_token_stream` callback may be called
    with `(delta="", is_retry_clear=True)` to signal a buffer wipe.
    """
    execution_data = _prepare_task_execution(
        engine, agent_or_character, task_key, messages,
        task_prompt_kwargs, temperature_override, top_k_override, top_p_override
    )

    max_retries = config.settings.get("STREAMING_BLACKLIST_RETRIES", 3)
    banned_words = config.global_blacklist.get("words", [])
    banned_phrases = config.global_blacklist.get("phrases", [])
    blacklist_patterns = []
    if banned_words:
        blacklist_patterns.append(r'\b(' + '|'.join(re.escape(w) for w in banned_words) + r')\b')
    if banned_phrases:
        blacklist_patterns.append('(' + '|'.join(re.escape(p) for p in banned_phrases) + ')')
    blacklist_regex = re.compile('|'.join(blacklist_patterns), re.IGNORECASE) if blacklist_patterns else None

    current_messages = list(execution_data['full_messages'])
    final_raw_response = ""

    for attempt in range(max_retries):
        raw_response_buffer = ""
        blacklist_triggered = False

        def internal_should_stop(current_buffer: str):
            nonlocal blacklist_triggered
            if should_stop(current_buffer): return True
            if blacklist_regex:
                match = blacklist_regex.search(current_buffer)
                if match:
                    log_message('debug', f"[BLACKLIST] Detected banned phrase: '{match.group(0)}'. Triggering retry.")
                    blacklist_triggered = True
                    return True
            return False

        try:
            raw_response_buffer = get_llm_response_streaming(
                engine, agent_or_character, current_messages,
                on_token_stream=on_token_stream,
                should_stop=internal_should_stop,
                temperature=execution_data['sampling_params']['temperature'],
                top_k=execution_data['sampling_params']['top_k'],
                top_p=execution_data['sampling_params']['top_p'],
                model_override=execution_data['model_identifier']
            )
            final_raw_response = raw_response_buffer
            if not blacklist_triggered:
                break
        except Exception as e:
            log_message('debug', f"[LLM_API_STREAMING] Streaming call failed: {e}. Falling back to non-streaming.")
            return execute_task(
                engine, agent_or_character, task_key, messages,
                task_prompt_kwargs, temperature_override, top_k_override, top_p_override
            )

        if attempt < max_retries - 1:
            log_message('debug', f"Retrying LLM call (Attempt {attempt + 2}/{max_retries}).")
            on_token_stream("", is_retry_clear=True)
            current_messages.append({"role": "assistant", "content": raw_response_buffer})
            current_messages.append({"role": "user", "content": loc('system_prompt_blacklist_retry')})
        else:
            log_message('debug', "Max retries reached for blacklist. Returning last bad response.")

    log_id = str(uuid.uuid4())
    log_path = _log_interaction(engine, agent_or_character, task_key, execution_data, final_raw_response, log_id)
    if log_path and hasattr(engine, 'last_interaction_log'):
        engine.last_interaction_log = {"log_id": log_id, "log_path": log_path}

    return _parse_and_strip_thinking_block(final_raw_response, execution_data['model_identifier'])


def get_model_context_length(model_identifier):
    if not model_identifier: return None
    if model_identifier == config.settings.get('GEMINI_MODEL_STRING'):
        try:
            model_info = genai.get_model(f"models/{config.settings.get('GEMINI_MODEL_NAME')}")
            return model_info.input_token_limit
        except Exception as e:
            log_message('debug', loc('warning_model_scan_fail', model_id=model_identifier, e=e))
            return None
    else:
        try:
            model = lms.llm(model_identifier)
            if hasattr(model, 'get_context_length'): return model.get_context_length()
            return 8192
        except Exception as e:
            log_message('debug', loc('warning_model_scan_fail', model_id=model_identifier, e=e))
            return None


def _get_gemini_gen_config(temperature, top_p, top_k):
    """Helper to create the Gemini GenerationConfig object."""
    gen_config = genai.GenerationConfig(temperature=temperature)
    if top_p is not None and top_p > 0:
        gen_config.top_p = top_p
    if top_k is not None and top_k > 0:
        gen_config.top_k = top_k
    return gen_config


def get_gemini_response(messages, temperature, top_p, top_k):
    system_instructions = None
    if messages and messages[0]['role'] == 'system':
        system_instructions = messages.pop(0)['content']
    gemini_messages = []
    for msg in messages:
        if not msg.get('content'): continue
        role = 'model' if msg['role'] == 'assistant' else 'user'
        gemini_messages.append({'role': role, 'parts': [msg['content']]})
    try:
        gen_config = _get_gemini_gen_config(temperature, top_p, top_k)
        model = genai.GenerativeModel(model_name=config.settings.get('GEMINI_MODEL_NAME'),
                                      system_instruction=system_instructions)
        response = model.generate_content(gemini_messages, generation_config=gen_config)
        return response.text.strip() if hasattr(response, 'text') and response.text else ""
    except Exception as e:
        log_message('debug', loc('error_gemini_config', e=e))
        return ""


def get_gemini_response_streaming(
        messages,
        on_token_stream: Callable[[str], None],
        should_stop: Callable[[str], bool],
        temperature, top_p, top_k
) -> str:
    """
    Gets a response from Gemini using streaming.
    Calls callbacks for each token and checks for early termination.
    """
    system_instructions = None
    if messages and messages[0]['role'] == 'system':
        system_instructions = messages.pop(0)['content']
    gemini_messages = []
    for msg in messages:
        if not msg.get('content'): continue
        role = 'model' if msg['role'] == 'assistant' else 'user'
        gemini_messages.append({'role': role, 'parts': [msg['content']]})

    full_response_content = ""
    try:
        gen_config = _get_gemini_gen_config(temperature, top_p, top_k)
        model = genai.GenerativeModel(
            model_name=config.settings.get('GEMINI_MODEL_NAME'),
            system_instruction=system_instructions
        )
        response_stream = model.generate_content(
            gemini_messages,
            generation_config=gen_config,
            stream=True
        )

        for chunk in response_stream:
            if hasattr(chunk, 'text') and chunk.text:
                delta = chunk.text
                full_response_content += delta
                on_token_stream(delta)
                if should_stop(full_response_content):
                    log_message('debug', "[LLM_API] Streaming stopped early by callback.")
                    break
    except Exception as e:
        log_message('debug', loc('error_gemini_config', e=e))
    return full_response_content.strip()


def _get_lmstudio_config_dict(temperature, top_p, top_k):
    """Helper to create the LM Studio configuration dictionary for the native API."""
    config_dict = {"temperature": temperature}
    if top_p is not None and top_p > 0.0:
        config_dict['top_p'] = top_p
        config_dict['top_k'] = 0  # Mutually exclusive
    elif top_k is not None:
        config_dict['top_k'] = top_k
    return config_dict


def _prepare_messages_for_lmstudio(messages: list) -> list:
    """
    Ensures the message list adheres to the strict user/assistant alternation
    required by the LM Studio API by converting any subsequent assistant messages
    into user messages.
    """
    if not messages:
        return []

    processed_messages = []
    first_assistant_found = False
    for msg in messages:
        msg_copy = msg.copy()
        current_role = msg_copy.get('role')

        if current_role == 'assistant':
            if not first_assistant_found:
                first_assistant_found = True
            else:
                msg_copy['role'] = 'user'

        processed_messages.append(msg_copy)

    return processed_messages


def get_lmstudio_response(model_identifier, messages, temperature=0.75, top_k=None, top_p=None):
    try:
        prepared_messages = _prepare_messages_for_lmstudio(messages)
        with lms.Client() as client:
            model = client.llm.model(model_identifier)
            chat_history = {"messages": prepared_messages}
            config_dict = _get_lmstudio_config_dict(temperature, top_p, top_k)
            prediction = model.respond(chat_history, config=config_dict)
            return prediction.content or ""
    except Exception as e:
        log_message('debug', loc('error_lmstudio_api', e=e))
        return ""


def get_lmstudio_response_streaming(
        model_identifier,
        messages,
        on_token_stream: Callable[[str], None],
        should_stop: Callable[[str], bool],
        temperature=0.75, top_k=None, top_p=None
) -> str:
    full_response_content = ""
    prediction_stream = None
    try:
        prepared_messages = _prepare_messages_for_lmstudio(messages)
        client = lms.Client()
        model = client.llm.model(model_identifier)
        chat_history = {"messages": prepared_messages}
        config_dict = _get_lmstudio_config_dict(temperature, top_p, top_k)

        def _handle_fragment(fragment: lms_json_api.LlmPredictionFragment):
            nonlocal full_response_content
            nonlocal prediction_stream
            delta = fragment.content
            if delta:
                full_response_content += delta
                on_token_stream(delta)
                if should_stop(full_response_content):
                    log_message('debug', "[LLM_API] Streaming stopped early by callback.")
                    if prediction_stream:
                        prediction_stream.cancel()

        prediction_stream = model.respond_stream(
            history=chat_history,
            config=config_dict,
            on_prediction_fragment=_handle_fragment
        )
        prediction_stream.wait_for_result()
    except Exception as e:
        log_message('debug', loc('error_lmstudio_api', e=e))
    return full_response_content.strip()


def get_llm_response(engine, agent_or_character: dict, full_messages: list, temperature: float = 0.7,
                     top_k: int | None = None, top_p: float | None = None, model_override: str | None = None):
    """
    The main entry point for making an LLM call.
    """
    model_identifier = model_override if model_override else agent_or_character.get('model')
    log_message('full',
                f"\n\n{'=' * 20} PROMPT FOR {model_identifier} {'=' * 20}\n{get_text_from_messages(full_messages)}\n{'=' * 20} END PROMPT {'=' * 20}\n")
    if model_identifier == config.settings.get('GEMINI_MODEL_STRING'):
        return get_gemini_response(full_messages, temperature, top_p, top_k)
    return get_lmstudio_response(model_identifier, full_messages, temperature, top_k, top_p)


def get_llm_response_streaming(
        engine, agent_or_character: dict, full_messages: list,
        on_token_stream: Callable[[str], None],
        should_stop: Callable[[str], bool],
        temperature: float = 0.7, top_k: int | None = None, top_p: float | None = None,
        model_override: str | None = None
) -> str:
    """
    The main entry point for making a streaming LLM call.
    """
    model_identifier = model_override if model_override else agent_or_character.get('model')
    log_message('full',
                f"\n\n{'=' * 20} STREAMING PROMPT FOR {model_identifier} {'=' * 20}\n{get_text_from_messages(full_messages)}\n{'=' * 20} END PROMPT {'=' * 20}\n")

    if model_identifier == config.settings.get('GEMINI_MODEL_STRING'):
        return get_gemini_response_streaming(
            full_messages, on_token_stream, should_stop, temperature, top_p, top_k
        )
    return get_lmstudio_response_streaming(
        model_identifier, full_messages, on_token_stream, should_stop, temperature, top_k, top_p
    )