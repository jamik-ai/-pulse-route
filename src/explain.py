"""Причины прогноза для диспетчера: вклад признаков (SHAP) → человекочитаемые формулировки.

Каждый признак отнесён к «фактору». Вклады SHAP (сек, для модели-поправки A) суммируются по факторам;
факторы с наибольшим положительным вкладом (в сторону опоздания) описываются текстом по значениям признаков.
"""
import numpy as np

# фактор → признаки
FACTORS = {
    "accumulated": ["cur_dev_s", "gps_dev_last", "gps_dev_med3", "gps_dev_med5", "gps_dev_med_10min",
                    "gps_dev_med5_auto", "delta_curdev_vs_gps", "arrival_at_last_after_T_s", "inst_dev_s",
                    "inst_dev_minus_cur", "inst_dev_age_adj", "dev_lower_bound_next", "dev_lower_bound_tgt"],
    "trend": ["gps_dev_trend5", "gps_dev_std5", "pace_ratio"],
    "standstill": ["stopped_share_60", "stopped_share_180", "stopped_share_300", "stopped_share_600",
                   "speed_last", "speed_mean_60", "speed_mean_180"],
    "slow_segment": ["speed_mean_300", "speed_mean_600", "moving_speed_30min", "speed_need_ratio",
                     "route_speed_kmh", "eta_gap_s", "plan_speed_to_tgt_kmh"],
    "history": ["hist_tgt_dev_last", "hist_tgt_dev_med", "hist_change_last", "hist_change_med",
                "hist_change_mean", "hist_plan_dur_ratio", "hist_tgt_dev_n", "hist_change_n"],
    "layover": ["max_plan_gap_to_tgt_s", "layover_between", "cur_dev_minus_layover", "tgt_idx_in_trip",
                "trip_started_before_T", "since_trip_start_s"],
    "place": ["tgt_stop_key", "tr_cat"],
    "telemetry": ["age_last_gps_s", "n_gps_30min"],
}
FEATURE_TO_FACTOR = {f: k for k, fs in FACTORS.items() for f in fs}


def _num(x, default=np.nan):
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def describe(factor: str, row: dict) -> str:
    g = row.get
    if factor == "accumulated":
        v = _num(g("gps_dev_med3"), _num(g("cur_dev_s"), 0))
        return f"Накопленное отклонение: {v:+.0f} с на последних остановках"
    if factor == "trend":
        v = _num(g("gps_dev_trend5"), 0)
        return f"Отставание растёт: {v:+.0f} с за последние 5 остановок" if v > 0 else "Неровный темп движения по графику"
    if factor == "standstill":
        share = _num(g("stopped_share_600"), 0)
        return f"Длительный простой: стоит {share * 10:.0f} из последних 10 минут"
    if factor == "slow_segment":
        need, have = _num(g("plan_speed_to_tgt_kmh")), _num(g("moving_speed_30min"))
        if np.isfinite(need) and np.isfinite(have) and need >= 5:
            return f"Низкая скорость: нужно ~{need:.0f} км/ч по графику, фактически ~{have:.0f} км/ч"
        return "Низкая скорость на подходе к остановке"
    if factor == "history":
        v = _num(g("hist_change_med"), _num(g("hist_tgt_dev_med"), 0))
        if abs(v) < 10:
            return "Проблемный участок по истории прошлых кругов"
        return f"Проблемный участок: на прошлых кругах здесь {v:+.0f} с"
    if factor == "layover":
        lay = _num(g("max_plan_gap_to_tgt_s"), 0)
        return f"Разворот на конечной (плановый отстой {lay / 60:.0f} мин)" if lay > 300 else "Начало рейса"
    if factor == "place":
        return "Особенность маршрута/остановки (по истории)"
    if factor == "telemetry":
        return f"Нет свежей телеметрии ({_num(g('age_last_gps_s'), 0):.0f} с)"
    return factor


def top_causes(shap_row: np.ndarray, feature_names: list[str], row: dict, k: int = 3, min_s: float = 5.0,
               direction: float = 1.0) -> list[dict]:
    """Факторы с наибольшим вкладом в сторону прогноза: к опозданию (direction>0) или опережению (<0)."""
    contrib: dict[str, float] = {}
    for name, v in zip(feature_names, shap_row):
        f = FEATURE_TO_FACTOR.get(name, "other")
        contrib[f] = contrib.get(f, 0.0) + float(v)
    contrib.pop("other", None)
    items = sorted(contrib.items(), key=lambda kv: -kv[1] * direction)
    out = []
    for f, v in items:
        if v * direction < min_s:
            break
        out.append({"factor": f, "contribution_s": round(v, 1), "text": describe(f, row)})
        if len(out) == k:
            break
    return out
