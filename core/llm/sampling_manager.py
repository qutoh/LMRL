# /core/llm/sampling_manager.py

from ..common.config_loader import config
from ..common.utils import log_message


def _get_calibration_group_for_task(task_key: str) -> dict | None:
    """Finds the calibration group a specific task belongs to."""
    for group_data in config.calibration_groups.values():
        if task_key in group_data.get("members", []):
            return group_data
    return None


def _determine_temperature(
        task_key: str,
        model_identifier: str,
        agent_or_character: dict,
        temperature_override: float | None
) -> float:
    """
    Determines the final temperature for an LLM call based on a clear hierarchy.
    """
    if temperature_override is not None:
        log_message('full',
                    f"[SAMPLING] Key: {task_key}, Model: {model_identifier}, Using CALIBRATION OVERRIDE Temp: {temperature_override:.2f}")
        return temperature_override

    calibrated_temp_data = None
    group = _get_calibration_group_for_task(task_key)

    if group and hasattr(config, 'calibrated_temperatures'):
        representative_id = group.get("representative_task")
        representative_task_key = None
        if representative_id:
            all_cal_tasks = config.calibration_tasks.get('single_tasks', []) + config.calibration_tasks.get(
                'sequence_tasks', [])
            for cal_task in all_cal_tasks:
                if cal_task.get("calibration_id") == representative_id:
                    representative_task_key = cal_task.get("target_task_key")
                    break

        if representative_task_key and (model_temps := config.calibrated_temperatures.get(model_identifier)):
            calibrated_temp_data = model_temps.get(representative_task_key)

    if calibrated_temp_data and isinstance(calibrated_temp_data, dict) and 'temperature' in calibrated_temp_data:
        calibrated_temp = calibrated_temp_data['temperature']
        log_message('full',
                    f"[SAMPLING] Key: {task_key} (Group Rep: {representative_task_key}), Model: {model_identifier}, Using CALIBRATED Temp: {calibrated_temp:.2f}")
        return calibrated_temp

    task_params = config.task_parameters.get(task_key, {})
    default_adjustment = task_params.get('default_adjustment', 0.0)
    base_temperature = agent_or_character.get('temperature', 0.7)
    scaling_factor = agent_or_character.get('scaling_factor', 1.0)
    adjustment = task_params.get('temperature_overrides', {}).get(model_identifier, default_adjustment)
    final_temperature = max(0.01, (scaling_factor * adjustment) + base_temperature)
    log_message('full',
                f"[SAMPLING] Key: {task_key}, Model: {model_identifier}, Using MANUAL Temp: {final_temperature:.2f} (Base: {base_temperature}, Scale: {scaling_factor}, Adj: {adjustment})")

    return final_temperature


def get_sampling_params(
        task_key: str,
        model_identifier: str,
        agent_or_character: dict,
        temperature_override: float | None,
        top_k_override: int | None,
        top_p_override: float | None
) -> dict:
    """
    Determines all sampling parameters (temp, top_k, top_p) for a given task.
    """
    final_temperature = _determine_temperature(
        task_key, model_identifier, agent_or_character, temperature_override
    )
    final_top_k = top_k_override
    final_top_p = top_p_override

    # If top_p is not manually overridden, check for a calibrated value.
    if final_top_p is None:
        group = _get_calibration_group_for_task(task_key)
        if group and group.get('calibrate_sampling_params', False):
            group_id = group.get("representative_task")
            if model_params := config.settings.get("calibrated_sampling_params", {}).get(model_identifier):
                if calibrated_p := model_params.get(group_id, {}).get("top_p"):
                    final_top_p = calibrated_p
                    final_top_k = 0  # top_p and top_k are mutually exclusive in many APIs
                    log_message('full',
                                f"[SAMPLING] Key: {task_key} (Group Rep: {group_id}), Model: {model_identifier}, Using CALIBRATED Top_P: {final_top_p}")

    return {
        "temperature": final_temperature,
        "top_k": final_top_k,
        "top_p": final_top_p
    }