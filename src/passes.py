"""Детекция проездов остановок по GPS.

Для каждой плановой остановки ищем первый «визит» — непрерывную серию GPS-точек
ближе VISIT_RADIUS_M к остановке в окне ±WINDOW вокруг плана. Время прибытия —
первая точка серии ближе ARRIVE_RADIUS_M (иначе точка минимума расстояния).

Причинность: визит считается известным на момент T только когда серия закончилась
(есть точка после неё) и `confirm_time <= T`. Время прибытия выбирается только
по точкам серии, т.е. тоже не позже confirm_time.
"""
import numpy as np
import pandas as pd

from src.dataio import haversine_km

VISIT_RADIUS_M = 150.0
ARRIVE_RADIUS_M = 40.0
WINDOW = pd.Timedelta("20min")


def detect_passes(schedule: pd.DataFrame, traffic: pd.DataFrame, arrive_radius_m: float = ARRIVE_RADIUS_M) -> pd.DataFrame:
    """Возвращает schedule + колонки gps_arrival, confirm_time, min_dist_m, gps_dev."""
    res = []
    tr_groups = dict(tuple(traffic.groupby("tr_id")))
    for tr, ss in schedule.groupby("tr_id", sort=False):
        tt = tr_groups.get(tr)
        n = len(ss)
        arr = np.full(n, np.datetime64("NaT"), dtype="datetime64[ns]")
        conf = arr.copy()
        mind = np.full(n, np.nan)
        if tt is not None and len(tt) > 2:
            et = tt["event_time"].values
            lo = tt["lon"].values
            la = tt["lat"].values
            for k, r in enumerate(ss.itertuples(index=False)):
                i0, i1 = np.searchsorted(et, [np.datetime64(r.time_begin - WINDOW), np.datetime64(r.time_begin + WINDOW)])
                if i1 - i0 < 2:
                    continue
                d = haversine_km(lo[i0:i1], la[i0:i1], r.stop_lon, r.stop_lat) * 1000
                near = np.flatnonzero(d < VISIT_RADIUS_M)
                if near.size == 0:
                    continue
                start = near[0]
                end = start
                while end + 1 < d.size and d[end + 1] < VISIT_RADIUS_M:
                    end += 1
                if end + 1 >= d.size:  # серия не закончилась внутри окна — ТС ещё у остановки
                    continue
                run = d[start:end + 1]
                close = np.flatnonzero(run < arrive_radius_m)
                j = start + (close[0] if close.size else run.argmin())
                arr[k] = et[i0 + j]
                conf[k] = et[i0 + end + 1]
                mind[k] = run.min()
        out = ss.copy()
        out["gps_arrival"] = arr
        out["confirm_time"] = conf
        out["min_dist_m"] = mind
        res.append(out)
    out = pd.concat(res, ignore_index=True)
    out["gps_dev"] = (out["gps_arrival"] - out["time_begin"]).dt.total_seconds()
    return out
