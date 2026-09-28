"""Признаки на момент T не должны зависеть от телеметрии после T и от фактов расписания."""
import numpy as np
import pandas as pd

from src.dataio import load_points, load_schedule, load_traffic
from src.features import build_contexts, features_at


def _same(a: dict, b: dict):
    assert a.keys() == b.keys()
    for k in a:
        if isinstance(a[k], str):
            assert a[k] == b[k], k
        else:
            assert (np.isnan(a[k]) and np.isnan(b[k])) or np.isclose(a[k], b[k]), k


def test_future_telemetry_does_not_change_features():
    pts = load_points("test").sample(40, random_state=0)
    plan = load_schedule("test").drop(columns=["time_fact_begin"])
    traffic = load_traffic("test")
    full = build_contexts(plan, traffic)
    for r in pts.itertuples():
        past = traffic[(traffic.tr_id == r.tr_id) & (traffic.event_time <= r.T)]
        cut = build_contexts(plan[plan.tr_id == r.tr_id], past)
        args = (r.T, r.target_stop_id, r.target_time_begin, r.cur_dev_s)
        _same(features_at(full[r.tr_id], *args), features_at(cut[r.tr_id], *args))


def test_schedule_facts_are_not_used():
    pts = load_points("test").head(30)
    s = load_schedule("test")
    traffic = load_traffic("test")
    a = build_contexts(s, traffic)
    s2 = s.assign(time_fact_begin=s.time_fact_begin + pd.Timedelta("7min"))
    b = build_contexts(s2, traffic)
    for r in pts.itertuples():
        args = (r.T, r.target_stop_id, r.target_time_begin, r.cur_dev_s)
        _same(features_at(a[r.tr_id], *args), features_at(b[r.tr_id], *args))
