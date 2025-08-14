# /core/llm/calibration_manager.py

import json
import numpy as np
import re
from scipy import stats
from . import llm_api
from core.common import file_io, utils
from core.common.config_loader import config
from core.engine.prometheus_manager import parse_and_log_tool_decisions

from sentence_transformers.util import cos_sim


class CalibrationManager:
    """
    Orchestrates the process of finding optimal temperature and top_k settings for LLM tasks.
    """

    def __init__(self, engine, embedding_model, event_log=None):
        self.engine = engine
        self.embedding_model = embedding_model
        self.event_log = event_log
        self.calibration_data = file_io.read_json(
            file_io.join_path(config.data_dir, 'calibration_tasks.json'), default={}
        )
        self.confidence_level = config.settings.get("CALIBRATION_CONFIDENCE_LEVEL", 0.95)
        self.min_samples = config.settings.get("CALIBRATION_MIN_SAMPLES", 3)
        self.max_samples = config.settings.get("CALIBRATION_MAX_SAMPLES", 15)
        self.ci_width_threshold = config.settings.get("CALIBRATION_CI_WIDTH_THRESHOLD", 0.1)
        self.structural_weight = 0.7
        self.semantic_weight = 0.3

    def _log_to_ui(self, message: str, color: tuple = (255, 255, 255)):
        if self.event_log:
            self.event_log.add_message(message, color)

    def _get_model_for_task(self, task_key: str) -> str:
        task_params = config.task_parameters.get(task_key, {})
        if model_override := task_params.get('model_override'):
            return model_override
        if source_agent_key := task_params.get('source_agent'):
            if agent_data := config.agents.get(source_agent_key):
                return agent_data.get('model')
        return config.settings.get("DEFAULT_AGENT_MODEL")

    # --- SCORING LOGIC ---

    def _get_structural_score(self, raw_output: str, task: dict) -> tuple[float, str]:
        validation_type = task.get("validation", {}).get("type", "json")
        if validation_type == "key_value_checklist":
            return self._score_key_value_checklist(raw_output, task)
        if validation_type == "keyword":
            return self._score_keyword_output(raw_output, task)
        if validation_type == "exact_keyword":
            return self._score_exact_keyword_output(raw_output, task)

        llm_data, cleaned_response = self._get_json_from_output(raw_output)
        if not llm_data: return 0.0, cleaned_response

        rules = task.get("validation", {}).get("rules", [])
        if not rules: return 1.0, cleaned_response
        passed_checks = sum(1 for rule in rules if self._check_rule(llm_data, rule))
        return (passed_checks / len(rules)), cleaned_response

    def _get_semantic_score(self, raw_output: str, task: dict) -> float:
        if not self.embedding_model or "semantic_exemplar" not in task:
            return 1.0

        exemplar_str = task["semantic_exemplar"]
        text_to_compare = raw_output

        try:
            if task.get("validation", {}).get("type") == "json":
                llm_data, _ = self._get_json_from_output(raw_output)
                if not llm_data: return 0.0
                exemplar_data = json.loads(exemplar_str)
                valid_alternatives = task.get("validation", {}).get("mutually_exclusive_actions", [])
                if valid_alternatives and llm_data.get("action") in valid_alternatives and llm_data.get(
                        "action") != exemplar_data.get("action"):
                    return 1.0
                creative_fields = task.get("creative_fields", [])
                text_to_compare = json.dumps(self._prepare_for_semantic_comparison(llm_data, creative_fields))
                exemplar_str = json.dumps(self._prepare_for_semantic_comparison(exemplar_data, creative_fields))

            vec1 = self.embedding_model.encode(exemplar_str)
            vec2 = self.embedding_model.encode(text_to_compare)
            return max(0, cos_sim(vec1, vec2).item())

        except (json.JSONDecodeError, Exception):
            return 0.0

    def _calculate_score(self, raw_output: str, task: dict, temp: float) -> tuple[float, str]:
        structural_score, cleaned_response = self._get_structural_score(raw_output, task)
        base_score = structural_score

        if "semantic_exemplar" in task:
            semantic_score = self._get_semantic_score(raw_output, task)
            base_score = (structural_score * self.structural_weight) + (semantic_score * self.semantic_weight)

        reward_factor = task.get("validation", {}).get("temperature_reward_factor", 0.0)
        return base_score * (1 + (temp * reward_factor)), cleaned_response

    def _score_key_value_checklist(self, raw_output: str, task: dict) -> tuple[float, str]:
        rules = task.get("validation", {})
        expected_pairs = {k.lower(): v for k, v in rules.get("expected_pairs", {}).items()}
        if not expected_pairs: return 1.0, raw_output

        context = task.get("prompt_args", {}).get("recent_events", "No context available.")
        extracted_pairs, _, _ = parse_and_log_tool_decisions(raw_output, list(expected_pairs.keys()), context)

        correct_count = 0
        for key, expected_value in expected_pairs.items():
            if key not in extracted_pairs: continue
            extracted_value = extracted_pairs[key]
            if isinstance(expected_value, list):
                if extracted_value in expected_value: correct_count += 1
            elif extracted_value == expected_value:
                correct_count += 1

        return (correct_count / len(expected_pairs)) if extracted_pairs else 0.0, raw_output

    def _score_keyword_output(self, raw_output: str, task: dict) -> tuple[float, str]:
        rules, text = task.get("validation", {}), raw_output.strip()
        prefix_ok = text.startswith(rules.get("expected_prefix", ""))
        substring_ok = any(sub in text for sub in rules.get("expected_substrings", []))
        return (1.0, text) if prefix_ok and substring_ok else (0.0, text)

    def _score_exact_keyword_output(self, raw_output: str, task: dict) -> tuple[float, str]:
        rules, text = task.get("validation", {}), raw_output.strip().upper()
        return (1.0, text) if text in rules.get("valid_keywords", []) else (0.0, text)

    def _get_json_from_output(self, raw_output: str) -> tuple[dict | None, str]:
        try:
            match = re.search(r"\{.*\}", raw_output, re.DOTALL)
            if match:
                cleaned_str = match.group(0)
                return json.loads(cleaned_str), cleaned_str
        except (json.JSONDecodeError, TypeError):
            pass
        return None, raw_output

    def _check_rule(self, llm_data: dict, rule: dict) -> bool:
        check = rule.get("check")
        if check == "has_key": return rule.get("key") in llm_data
        if check == "key_value_in": return llm_data.get(rule.get("key")) in rule.get("values", [])
        if check == "key_requires_key":
            return rule.get("then_key") in llm_data if llm_data.get(rule.get("if_key")) == rule.get(
                "if_value") else True
        if check == "has_nested_key":
            keys = rule.get("parent_key", "").split('.')
            data = llm_data
            for key in keys:
                if isinstance(data, dict) and key in data:
                    data = data[key]
                else:
                    return False
            return isinstance(data, dict) and rule.get("child_key") in data
        return False

    def _prepare_for_semantic_comparison(self, data: dict, creative_fields: list) -> dict:
        prepared_data = data.copy()
        for key in creative_fields:
            if key in prepared_data: prepared_data[key] = "[CREATIVE_CONTENT]"
        return prepared_data

    def _calculate_sequence_score(self, choices: list, task_definition: dict, temp: float) -> float:
        if len(set(choices)) == 1: return 0.0
        correct_choices = sum(1 for i, choice in enumerate(choices) if
                              i < len(task_definition['steps']) and choice in task_definition['steps'][i][
                                  'valid_options'])
        validity_score = correct_choices / len(task_definition['steps']) if task_definition['steps'] else 0.0
        valid_choices = [c for i, c in enumerate(choices) if
                         i < len(task_definition['steps']) and c in task_definition['steps'][i]['valid_options']]
        variety_score = len(set(valid_choices)) / len(valid_choices) if valid_choices else 0.0
        base_score = (validity_score * 0.8) + (variety_score * 0.2)
        reward_factor = task_definition.get("validation", {}).get("temperature_reward_factor", 0.0)
        return base_score * (1 + (temp * reward_factor))

    def _calculate_diversity_score(self, responses: list[str], task: dict) -> float:
        """Calculates a score based on structural validity and response diversity."""
        if not responses: return 0.0

        valid_responses = [self._get_structural_score(resp, task)[1] for resp in responses if
                           self._get_structural_score(resp, task)[0] > 0.99]

        validity_rate = len(valid_responses) / len(responses)

        if not valid_responses: return 0.0

        diversity_score = len(set(valid_responses)) / len(valid_responses)

        return validity_rate * diversity_score

    def _get_confidence_interval(self, scores: list) -> tuple[float, float, float]:
        n = len(scores)
        if n < 2: return (np.mean(scores),) * 3 if n == 1 else (0.0,) * 3
        mean, std_err = np.mean(scores), stats.sem(scores)
        if std_err == 0: return (mean,) * 3
        margin_of_error = std_err * stats.t.ppf((1 + self.confidence_level) / 2., n - 1)
        return mean, mean - margin_of_error, mean + margin_of_error

    def _get_stable_score(self, dummy_agent, task, temp, top_k, top_p, is_sequence):
        scores = []
        target_key = task["target_task_key"]
        for i in range(self.max_samples):
            if is_sequence:
                choices = []
                for step_def in task['steps']:
                    kwargs = step_def.get('prompt_args', {}).copy()
                    kwargs['user_message_content'] = step_def.get('user_message_content', '')
                    raw = llm_api.execute_task(self.engine, dummy_agent, target_key, [], kwargs, temp, top_k, top_p)
                    choices.append(next((opt for opt in step_def['valid_options'] if opt.lower() in raw.lower()), ""))
                score = self._calculate_sequence_score(choices, task, temp)
                log_msg = f"Task: {target_key[:10]} | T: {temp:.2f}, P: {top_p}, K: {top_k} (Run {i + 1}) | Score: {score:.3f}"
            else:
                kwargs = task.get("prompt_args", {}).copy()
                raw = llm_api.execute_task(self.engine, dummy_agent, target_key, [], kwargs, temp, top_k, top_p)
                score, cleaned = self._calculate_score(raw, task, temp)
                log_msg = f"Task: {target_key[:10]} | T: {temp:.2f}, P: {top_p}, K: {top_k} (Sample {i + 1}) | Score: {score:.3f}"
                utils.log_message('full', f"      | Response: {cleaned.strip()}")

            scores.append(score)
            self._log_to_ui(log_msg, (200, 200, 220))
            utils.log_message('full', f"    - {log_msg}")

            if i + 1 >= self.min_samples and (
                    self._get_confidence_interval(scores)[2] - self._get_confidence_interval(scores)[
                1]) < self.ci_width_threshold:
                break

        mean, lower, upper = self._get_confidence_interval(scores)
        return mean, lower, upper

    def _find_optimal_temperature(self, model_id: str, task: dict, is_sequence: bool, top_p: float) -> float:
        utils.log_message('debug',
                          f"  -> Phase 2: Calibrating Temperature for {task['target_task_key']} on '{model_id}' (Top_P={top_p})")
        dummy_agent = {"model": model_id, "persona": "You are a utility model."}

        temp_file_path = file_io.join_path(config.data_dir, 'temp_calibration.json')
        progress = file_io.read_json(temp_file_path, default={})

        if progress.get('phase') != 'temperature':
            progress = {'phase': 'temperature', 'results': []}

        results = progress.get('results', [])

        temps_to_test = [t for t in np.arange(0.2, 1.9, 0.2) if
                         round(t, 2) not in {r.get('temp') for r in results if 'temp' in r}]

        for temp in temps_to_test:
            py_temp = float(temp)
            mean, lower, upper = self._get_stable_score(dummy_agent, task, py_temp, 0, top_p, is_sequence)
            result_entry = {'temp': round(py_temp, 2), 'mean': mean, 'ci': (lower, upper)}
            results.append(result_entry)
            progress['results'] = results
            file_io.write_json(temp_file_path, progress)

            log_msg = f"    - Temp Search Result for T={py_temp:.2f}: Score = {mean:.3f}, CI = [{lower:.3f}, {upper:.3f}]"
            self._log_to_ui(log_msg, (220, 210, 180))
            utils.log_message('debug', log_msg)

        if not results: return 0.7

        best_result = max(results, key=lambda x: x['mean'])
        best_temp = best_result['temp']

        utils.log_message('debug',
                          f"  -> Optimal temperature for '{task['target_task_key']}' is {best_temp:.2f} (Best Score: {best_result['mean']:.3f})")
        file_io.remove_file(temp_file_path)
        return round(best_temp, 2)

    def _find_optimal_top_p(self, model_id: str, task: dict) -> float:
        utils.log_message('debug', f"  -> Phase 1: Calibrating Top_P for {task['target_task_key']} on '{model_id}'")
        dummy_agent = {"model": model_id, "persona": "You are a utility model."}

        temp_file_path = file_io.join_path(config.data_dir, 'temp_calibration.json')
        progress = file_io.read_json(temp_file_path, default={'phase': 'top_p', 'results': []})

        if progress.get('phase') != 'top_p':
            progress = {'phase': 'top_p', 'results': []}

        results = progress.get('results', [])

        fixed_temp = 0.7
        is_sequence = task.get("validation", {}).get("type") == "sequence"
        top_p_values_to_test = [p for p in [0.85, 0.90, 0.95, 0.98, 1.0] if
                                p not in {r.get('p') for r in results if 'p' in r}]

        for p in top_p_values_to_test:
            responses = []
            for _ in range(self.max_samples):
                kwargs_for_prompt = {}
                if is_sequence:
                    if task.get('steps'):
                        step_def = task['steps'][0]
                        kwargs_for_prompt = step_def.get('prompt_args', {}).copy()
                        kwargs_for_prompt['user_message_content'] = step_def.get('user_message_content', '')
                else:
                    kwargs_for_prompt = task.get("prompt_args", {}).copy()

                raw = llm_api.execute_task(self.engine, dummy_agent, task["target_task_key"], [], kwargs_for_prompt,
                                           fixed_temp, 0, p)
                responses.append(raw)

            score = self._calculate_diversity_score(responses, task)
            results.append({'p': p, 'score': score})
            progress['results'] = results
            file_io.write_json(temp_file_path, progress)

            log_msg = f"    - Top_P Search Result for P={p}: Diversity Score = {score:.3f}"
            self._log_to_ui(log_msg, (210, 220, 180))
            utils.log_message('debug', log_msg)

        if not results:
            utils.log_message('debug', "  -> All top_p values failed to produce results. Falling back to default 0.95.")
            return 0.95

        best_result = max(results, key=lambda x: x['score'])
        file_io.remove_file(temp_file_path)
        return best_result['p']

    def get_calibration_plan(self) -> list[dict]:
        plan = []
        all_cal_tasks = self.calibration_data.get("single_tasks", []) + self.calibration_data.get("sequence_tasks", [])

        for group_data in config.calibration_groups.values():
            rep_id = group_data.get("representative_task")
            if not rep_id: continue

            rep_task = next((t for t in all_cal_tasks if t.get("calibration_id") == rep_id), None)
            if not rep_task: continue

            models_for_group = set(model for member_key in group_data.get("members", []) if
                                   (model := self._get_model_for_task(member_key)))

            for model_id in models_for_group:
                job = {'model_id': model_id, 'task': rep_task, 'group_id': rep_id,
                       'calibrate_sampling': group_data.get('calibrate_sampling_params', False)}
                plan.append(job)
        return plan

    def run_calibration_test(self, job: dict):
        model_id = job['model_id']
        task = job['task']
        group_id = job['group_id']

        save_path_temp = file_io.join_path(config.data_dir, 'calibrated_temperatures.json')
        calibrated_temps = file_io.read_json(save_path_temp, default={})

        save_path_sampling = file_io.join_path(config.data_dir, 'calibrated_sampling_params.json')
        calibrated_sampling = file_io.read_json(save_path_sampling, default={})

        calibrated_sampling.setdefault(model_id, {}).setdefault(group_id, {})

        is_sequence = task.get("validation", {}).get("type") == "sequence"
        final_top_p = 0.95  # Default

        if job.get('calibrate_sampling'):
            if 'top_p' not in calibrated_sampling.get(model_id, {}).get(group_id, {}):
                optimal_top_p = self._find_optimal_top_p(model_id, task)
                calibrated_sampling[model_id][group_id]['top_p'] = optimal_top_p
                file_io.write_json(save_path_sampling, calibrated_sampling)

                if 'calibrated_sampling_params' not in config.settings: config.settings[
                    'calibrated_sampling_params'] = {}
                config.settings['calibrated_sampling_params'].setdefault(model_id, {})[group_id] = {
                    'top_p': optimal_top_p}

            final_top_p = calibrated_sampling[model_id][group_id]['top_p']

        if task['target_task_key'] not in calibrated_temps.get(model_id, {}):
            optimal_temp = self._find_optimal_temperature(model_id, task, is_sequence, final_top_p)
            calibrated_temps.setdefault(model_id, {})[task['target_task_key']] = optimal_temp
            file_io.write_json(save_path_temp, calibrated_temps)
            config.calibrated_temperatures = calibrated_temps