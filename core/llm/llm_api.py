# /core/llm/llm_api.py

import re
import uuid
from datetime import datetime

import google.generativeai as genai
import lmstudio as lms

from core.common.config_loader import config
from . import sampling_manager
from ..common import file_io
from ..common.localization import loc
from ..common.utils import log_message, get_text_from_messages


def _parse_and_strip_thinking_block(raw_text: str, model_id: str) -> str:
    """
    Checks for model-specific parsing rules and processes the raw LLM output.
    - If response tags are found, it extracts the content between them.
    - If an end-of-thought tag is found, it treats everything after it as the response.
    - If no rules apply, it returns the original text.
    """
    parser_config = config.model_tuning.get(model_id)
    if not parser_config:
        return raw_text

    think_end = parser_config.get("think_end_str")
    response_start = parser_config.get("response_start_str")
    response_end = parser_config.get("response_end_str")

    # Priority 1: Extract content from a dedicated response block if tags are defined.
    if response_start and response_end:
        # Use a more robust regex that allows for whitespace around tags and content.
        pattern = re.compile(re.escape(response_start) + r'\s*(.*?)\s*' + re.escape(response_end),
                             re.DOTALL | re.IGNORECASE)
        match = re.search(pattern, raw_text)
        if match:
            return match.group(1).strip()

    # Priority 2: Split the response at the first end-of-thought marker.
    # This robustly handles cases where the start tag is missing.
    if think_end:
        parts = raw_text.split(think_end, 1)
        if len(parts) > 1:
            # We have successfully split the string. The response is the second part.
            return parts[1].strip()

    # Fallback: If no tags matched or the configuration was partial.
    return raw_text


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


def execute_task(engine, agent_or_character: dict, task_key: str, messages: list,
                 task_prompt_kwargs: dict | None = None, temperature_override: float | None = None,
                 top_k_override: int | None = None, top_p_override: float | None = None) -> str:
    """
    A wrapper for LLM calls that uses the sampling manager to determine
    the optimal parameters. It also logs the interaction.
    """
    log_id = str(uuid.uuid4())
    agent_name = agent_or_character.get('name', 'unknown_agent')
    role_type = agent_or_character.get('role_type')

    task_params = config.task_parameters.get(task_key, {})

    if task_params.get("player_takeover_enabled") is True:
        player_response = None
        persona = agent_or_character.get('persona') or agent_or_character.get('instructions', '')
        handler_kwargs = {
            "agent_name": agent_name, "task_key": task_key,
            "task_prompt_kwargs": task_prompt_kwargs, "persona": persona, "messages": messages
        }

        if engine.render_queue and engine.player_interface:
            player_response = engine.player_interface.handle_task_takeover(**handler_kwargs)
        elif engine.ui_manager:
            player_response = engine.ui_manager.get_player_task_input(**handler_kwargs)

        if player_response is not None:
            log_message('debug', f"[PLAYER TAKEOVER] Task '{task_key}' intercepted. Using player response.")
            return player_response
        else:
            log_message('debug', f"[PLAYER TAKEOVER] Player cancelled or passed. Passing task '{task_key}' to LLM.")

    if task_params.get("is_failure_handler"):
        engine.annotation_manager.annotate_last_log_as_failure('AUTOMATIC_HANDLER_TRIGGERED',
                                                               f"Triggered by task: {task_key}")

    model_identifier = agent_or_character.get('model')
    if task_key not in ['CHARACTER_TURN', 'DM_TURN'] and task_params:
        model_identifier = task_params.get('model_override', model_identifier)

    params = sampling_manager.get_sampling_params(
        task_key, model_identifier, agent_or_character,
        temperature_override, top_k_override, top_p_override
    )

    final_temperature = params['temperature']
    final_top_k = params['top_k']
    final_top_p = params['top_p']

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

    # Turn tasks (and any task that builds its messages manually) do not have a prompt key.
    task_prompt_key = task_params.get('task_prompt_key') if task_key not in ['CHARACTER_TURN', 'DM_TURN'] else None

    full_messages = _prepare_messages_for_llm(persona, task_prompt_key, messages, task_prompt_kwargs, model_identifier,
                                              enable_reasoning_for_task)

    raw_response = get_llm_response(
        engine, agent_or_character, full_messages,
        temperature=final_temperature, top_k=final_top_k, top_p=final_top_p,
        model_override=model_identifier
    )

    log_entry = {
        "log_id": log_id, "timestamp": datetime.now().isoformat(),
        "model_identifier": model_identifier, "temperature": final_temperature,
        "top_p": final_top_p, "top_k": final_top_k, "prompt": full_messages,
        "raw_response": raw_response, "annotation": {"status": "PASS"}
    }
    processed_response = _parse_and_strip_thinking_block(raw_response, model_identifier)
    log_path = ""
    if role_type in ['lead', 'npc', 'dm'] and hasattr(engine, 'character_log_sessions'):
        sanitized_char_name = file_io.sanitize_filename(agent_name)
        log_path = file_io.join_path(file_io.PROJECT_ROOT, 'logs', 'Narrative', f"{sanitized_char_name}.json")
        session_info = engine.character_log_sessions.get(agent_name, {})
        current_instructions = agent_or_character.get('instructions', '')
        session_id = session_info.get('session_id')
        if not session_id or session_info.get('instructions') != current_instructions:
            session_id = str(uuid.uuid4())
            engine.character_log_sessions[agent_name] = {'session_id': session_id, 'instructions': current_instructions}
        try:
            char_logs = file_io.read_json(log_path, default={})
            if session_id not in char_logs:
                char_logs[session_id] = {'instructions': current_instructions, 'interactions': []}
            char_logs[session_id]['interactions'].append(log_entry)
            file_io.write_json(log_path, char_logs)
        except Exception as e:
            log_message('debug', f"[LLM NARRATIVE LOGGING ERROR] Could not write to {log_path}. Error: {e}")
    else:
        sanitized_agent_name = file_io.sanitize_filename(agent_name)
        sanitized_task_name = file_io.sanitize_filename(task_key)
        log_path = file_io.join_path(file_io.PROJECT_ROOT, 'logs', sanitized_agent_name, f"{sanitized_task_name}.json")
        try:
            existing_logs = file_io.read_json(log_path, default=[])
            if not isinstance(existing_logs, list): existing_logs = [existing_logs]
            existing_logs.append(log_entry)
            file_io.write_json(log_path, existing_logs)
        except Exception as e:
            log_message('debug', f"[LLM AGENT LOGGING ERROR] Could not write to {log_path}. Error: {e}")

    if log_path and hasattr(engine, 'last_interaction_log'):
        engine.last_interaction_log = {"log_id": log_id, "log_path": log_path}
    return processed_response

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
        gen_config = genai.GenerationConfig(temperature=temperature)
        if top_p is not None and top_p > 0:
            gen_config.top_p = top_p
        if top_k is not None and top_k > 0:
            gen_config.top_k = top_k
        model = genai.GenerativeModel(model_name=config.settings.get('GEMINI_MODEL_NAME'),
                                      system_instruction=system_instructions)
        response = model.generate_content(gemini_messages, generation_config=gen_config)
        return response.text.strip() if hasattr(response, 'text') and response.text else ""
    except Exception as e:
        log_message('debug', loc('error_gemini_config', e=e))
        return ""


def get_lmstudio_response(model_identifier, messages, temperature=0.75, top_k=None, top_p=None):
    try:
        with lms.Client() as client:
            model = client.llm.model(model_identifier)
            chat_history = {"messages": messages}

            config_dict = {"temperature": temperature}
            if top_p is not None and top_p > 0.0:
                config_dict['top_p'] = top_p
                config_dict['top_k'] = 0
            elif top_k is not None:
                config_dict['top_k'] = top_k

            response_object = model.respond(chat_history, config=config_dict)

            if not response_object: return ""

            if isinstance(response_object, dict):
                try:
                    return response_object['choices'][0]['message']['content'] or ""
                except (KeyError, IndexError, TypeError):
                    return ""
            return str(response_object).strip()
    except Exception as e:
        log_message('debug', loc('error_lmstudio_api', e=e))
        return ""


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