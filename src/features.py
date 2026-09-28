"""Признаки для прогнозной точки (tr_id, T).

Правило честности: используется только

* плановое расписание (time_begin, координаты, manual_fill) — известно заранее;
* телеметрия с event_time <= T;
* подсказка cur_dev_s.

Факты из schedule.csv (time_fact_begin) сюда НЕ передаются вообще.

Одна и та же функция `features_at` используется офлайн (train/test/validate) и в real-time.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.dataio import haversine_km

VISIT_RADIUS_M = 150.0     # серия точек ближе этого — «визит» остановки
ARRIVE_RADIUS_M = 25.0     # первая точка ближе этого — момент прибытия
EARLY_S = 7 * 60           # визит ищем в [план − 7 мин, план + 12 мин]:
LATE_S = 12 * 60           # реальные отклонения лежат в −6…+11 мин
PASS_LOOKBACK_S = 45 * 60  # какие остановки проверяем на проезд
GOOD_PASS_M = 60.0         # визит с min расстоянием больше — ненадёжен (встречка и т.п.)

PLAN_COLS = ["stop_id", "time_begin", "stop_lon", "stop_lat", "manual_fill"]


def _sec(ts) -> np.ndarray:
    return pd.to_datetime(ts).values.astype("datetime64[s]").astype(np.int64)


@dataclass
class TrContext:
    """Всё, что известно про одно ТС: план и (растущий) буфер телеметрии."""
    plan_t: np.ndarray        # плановое время, unix-сек, отсортировано
    plan_id: np.ndarray
    plan_lon: np.ndarray
    plan_lat: np.ndarray
    plan_manual: np.ndarray
    plan_cumdist: np.ndarray  # накопленное расстояние по цепочке остановок, км
    plan_key: np.ndarray      # id места остановки (округлённые координаты) — одинаков на разных кругах
    et: np.ndarray            # время телеметрии, unix-сек, отсортировано
    lon: np.ndarray
    lat: np.ndarray
    speed: np.ndarray
    tr_key: str = ""           # id ТС как категория
    _day_passes: tuple = None  # (n_hist, dev по всем остановкам) — кэш для исторических признаков

    def day_pass_dev(self) -> np.ndarray:
        """GPS-задержка проезда для каждой плановой остановки (NaN — не найден/ненадёжен).
        Вызывающий обязан брать только остановки с plan_t + LATE_S <= T: тогда окно поиска
        целиком в прошлом и результат не зависит от телеметрии после T."""
        n = len(self.et)
        if self._day_passes is None or self._day_passes[0] != n:
            dev = np.full(len(self.plan_t), np.nan)
            if n:
                T_end = int(self.et[-1])
                ks, arr, dmin = _passes_at(self, T_end, n, k_range=(0, len(self.plan_t)))
                ok = dmin < GOOD_PASS_M
                dev[ks[ok]] = arr[ok] - self.plan_t[ks[ok]]
            self._day_passes = (n, dev)
        return self._day_passes[1]

    @classmethod
    def build(cls, plan: pd.DataFrame, traffic: pd.DataFrame) -> "TrContext":
        plan = plan.sort_values(["time_begin", "stop_id"])
        lon, lat = plan["stop_lon"].values, plan["stop_lat"].values
        seg = np.r_[0.0, haversine_km(lon[:-1], lat[:-1], lon[1:], lat[1:])]
        traffic = traffic.sort_values("event_time")
        return cls(
            plan_t=_sec(plan["time_begin"]), plan_id=plan["stop_id"].values,
            plan_lon=lon, plan_lat=lat, plan_manual=plan["manual_fill"].values.astype(bool),
            plan_cumdist=np.cumsum(seg),
            plan_key=np.round(lon, 4) * 1e5 + np.round(lat, 4),
            et=_sec(traffic["event_time"]), lon=traffic["lon"].values,
            lat=traffic["lat"].values, speed=traffic["speed"].values.astype(float),
        )


def build_contexts(plan: pd.DataFrame, traffic: pd.DataFrame) -> dict:
    tg = dict(tuple(traffic.groupby("tr_id")))
    empty = traffic.iloc[:0]
    ctxs = {}
    for tr, p in plan.groupby("tr_id"):
        ctxs[tr] = TrContext.build(p[PLAN_COLS], tg.get(tr, empty))
        ctxs[tr].tr_key = str(tr)
    return ctxs


def _passes_at(ctx: TrContext, T: int, n_hist: int, k_range=None):
    """Проезды остановок, достоверно известные на момент T (по телеметрии <= n_hist)."""
    et, lo, la = ctx.et[:n_hist], ctx.lon[:n_hist], ctx.lat[:n_hist]
    k0, k1 = k_range or np.searchsorted(ctx.plan_t, [T - PASS_LOOKBACK_S, T + EARLY_S])
    ks, arr, dmin = [], [], []
    for k in range(k0, k1):
        pt = ctx.plan_t[k]
        i0, i1 = np.searchsorted(et, [pt - EARLY_S, pt + LATE_S], side="right")
        if i1 - i0 < 2:
            continue
        d = haversine_km(lo[i0:i1], la[i0:i1], ctx.plan_lon[k], ctx.plan_lat[k]) * 1000
        inside = d < VISIT_RADIUS_M
        if not inside.any():
            continue
        # серии подряд идущих точек внутри радиуса; последняя незавершённая — не считаем
        edges = np.flatnonzero(np.diff(np.r_[0, inside.astype(np.int8), 0]))
        best = None
        for s, e in zip(edges[::2], edges[1::2]):  # [s, e)
            if e >= d.size:  # ТС всё ещё у остановки на момент T
                continue
            m = d[s:e].min()
            if best is None or m < best[0]:
                close = np.flatnonzero(d[s:e] < ARRIVE_RADIUS_M)
                j = s + (close[0] if close.size else d[s:e].argmin())
                best = (m, et[i0 + j])
        if best is not None:
            ks.append(k); arr.append(best[1]); dmin.append(best[0])
    return np.array(ks, dtype=int), np.array(arr, dtype=np.int64), np.array(dmin)


def _position_dev(ctx: TrContext, T: int, lon: float, lat: float, dev_est: float,
                  max_dist_m: float = 80.0):
    """Проекция текущей точки на отрезки маршрута рядом по плану → «плановое время этого места».
    Возвращает (T − интерполированный план, расстояние до отрезка, индекс начала отрезка)."""
    k0, k1 = np.searchsorted(ctx.plan_t, [T - 15 * 60, T + 12 * 60])
    k0, k1 = max(int(k0), 0), min(int(k1), len(ctx.plan_t) - 1)
    if k1 <= k0:
        return None
    ax, ay = ctx.plan_lon[k0:k1], ctx.plan_lat[k0:k1]
    bx, by = ctx.plan_lon[k0 + 1:k1 + 1], ctx.plan_lat[k0 + 1:k1 + 1]
    cx = np.cos(np.radians(lat)) * 111_320.0
    cy = 110_540.0
    ux, uy = (bx - ax) * cx, (by - ay) * cy
    px, py = (lon - ax) * cx, (lat - ay) * cy
    L2 = ux * ux + uy * uy
    a = np.where(L2 > 0, np.clip((px * ux + py * uy) / np.where(L2 > 0, L2, 1), 0, 1), 0)
    dist = np.hypot(px - a * ux, py - a * uy)
    t0, t1 = ctx.plan_t[k0:k1], ctx.plan_t[k0 + 1:k1 + 1]
    dev = T - (t0 + a * (t1 - t0))
    ok = dist < max_dist_m
    if not ok.any():
        return None
    # из близких отрезков — тот, что согласуется с текущей оценкой задержки
    j = np.flatnonzero(ok)[np.abs(dev[ok] - dev_est).argmin()]
    return float(dev[j]), float(dist[j]), k0 + int(j)


def _history_features(ctx: TrContext, T: int, kt: int, kl: int) -> dict:
    f = {}
    dev = ctx.day_pass_dev()
    past = ctx.plan_t + LATE_S <= T
    prev = np.flatnonzero(past[:kt] & (ctx.plan_key[:kt] == ctx.plan_key[kt]))
    lvl = dev[prev][~np.isnan(dev[prev])]
    f["hist_tgt_dev_n"] = lvl.size
    if lvl.size:
        f["hist_tgt_dev_last"] = float(lvl[-1])
        f["hist_tgt_dev_med"] = float(np.median(lvl[-4:]))
    if kl < 0:
        return f
    off = kt - kl
    plan_dur = ctx.plan_t[kt] - ctx.plan_t[kl]
    ch, dur_ratio = [], []
    for k in prev:
        j = k - off
        if j < 0 or ctx.plan_key[j] != ctx.plan_key[kl]:
            continue
        if not (np.isnan(dev[k]) or np.isnan(dev[j])):
            ch.append(dev[k] - dev[j])
            dur_ratio.append((ctx.plan_t[k] - ctx.plan_t[j]) / plan_dur if plan_dur else np.nan)
    f["hist_change_n"] = len(ch)
    if ch:
        ch = np.array(ch)
        f["hist_change_last"] = float(ch[-1])
        f["hist_change_med"] = float(np.median(ch[-4:]))
        f["hist_change_mean"] = float(ch[-4:].mean())
        f["hist_plan_dur_ratio"] = float(dur_ratio[-1])
    return f


def features_at(ctx: TrContext, T, target_stop_id, target_time_begin, cur_dev_s) -> dict:
    T = int(_sec([T])[0]) if not isinstance(T, (int, np.integer)) else int(T)
    tgt_t = int(_sec([target_time_begin])[0]) if not isinstance(target_time_begin, (int, np.integer)) else int(target_time_begin)
    f = {"cur_dev_s": float(cur_dev_s), "plan_to_target_s": tgt_t - T}

    # --- расписание -------------------------------------------------------------
    hits = np.flatnonzero(ctx.plan_id == target_stop_id)
    kt = int(hits[0]) if hits.size else int(np.searchsorted(ctx.plan_t, tgt_t))
    kt = min(kt, len(ctx.plan_t) - 1)
    kl = int(np.searchsorted(ctx.plan_t, T, side="right")) - 1  # остановка, к которой относится cur_dev_s
    f["n_stops_plan_to_target"] = kt - kl
    f["tgt_manual"] = float(ctx.plan_manual[kt])
    f["prev_manual"] = float(ctx.plan_manual[kt - 1]) if kt > 0 else np.nan
    f["manual_share_around"] = float(ctx.plan_manual[max(kt - 5, 0):kt + 6].mean())
    f["last_plan_manual"] = float(ctx.plan_manual[kl]) if kl >= 0 else np.nan
    gap_prev = ctx.plan_t[kt] - ctx.plan_t[kt - 1] if kt > 0 else np.nan
    gap_next = ctx.plan_t[kt + 1] - ctx.plan_t[kt] if kt + 1 < len(ctx.plan_t) else np.nan
    f["tgt_gap_prev_s"], f["tgt_gap_next_s"] = gap_prev, gap_next
    # начало рейса: плановый разрыв > 5 мин
    brk = np.flatnonzero(np.diff(ctx.plan_t[: kt + 1]) > 300)
    trip_start = brk[-1] + 1 if brk.size else 0
    f["tgt_idx_in_trip"] = kt - trip_start
    f["trip_started_before_T"] = float(ctx.plan_t[trip_start] <= T)
    f["cur_dev_is_zero"] = float(cur_dev_s == 0)
    f["no_plan_stop_before_T"] = float(kl < 0)
    f["since_first_plan_s"] = T - ctx.plan_t[0]
    f["since_trip_start_s"] = T - ctx.plan_t[trip_start]
    f["hour"] = (T // 3600) % 24  # время в данных «наивное», как в CSV
    f["route_dist_last_plan_to_tgt_km"] = ctx.plan_cumdist[kt] - ctx.plan_cumdist[max(kl, 0)]
    if kl >= 0:
        # cur_dev_s относится к остановке kl; если cur_dev_s > возраста плана kl — ТС туда ещё не доехало
        f["last_plan_age_s"] = T - ctx.plan_t[kl]
        f["arrival_at_last_after_T_s"] = cur_dev_s - f["last_plan_age_s"]
        seg_gaps = np.diff(ctx.plan_t[kl:kt + 1])
        lay = float(seg_gaps.max()) if seg_gaps.size else 0.0
        f["max_plan_gap_to_tgt_s"] = lay
        f["layover_between"] = float(lay > 300)
        f["cur_dev_minus_layover"] = cur_dev_s - (lay if lay > 300 else 0.0)
        plan_dur = ctx.plan_t[kt] - ctx.plan_t[kl]
        f["plan_speed_to_tgt_kmh"] = f["route_dist_last_plan_to_tgt_km"] / (plan_dur / 3600) if plan_dur > 0 else np.nan
    f["tgt_stop_key"] = str(ctx.plan_key[kt])
    f["tr_cat"] = ctx.tr_key

    # --- прошлые круги того же ТС (только GPS, окно поиска целиком до T) ---------
    f.update(_history_features(ctx, T, kt, kl))

    # --- телеметрия <= T -------------------------------------------------------
    n = int(np.searchsorted(ctx.et, T, side="right"))
    f["n_gps_30min"] = n - int(np.searchsorted(ctx.et, T - 1800))
    if n == 0:
        return f
    f["age_last_gps_s"] = T - ctx.et[n - 1]
    f["speed_last"] = ctx.speed[n - 1]
    for w in (60, 180, 300, 600):
        i0 = int(np.searchsorted(ctx.et, T - w))
        sp = ctx.speed[i0:n]
        f[f"speed_mean_{w}"] = sp.mean() if sp.size else np.nan
        f[f"stopped_share_{w}"] = (sp < 3).mean() if sp.size else np.nan
    i0 = int(np.searchsorted(ctx.et, T - 1800))
    sp = ctx.speed[i0:n]
    moving = sp[sp >= 3]
    f["moving_speed_30min"] = moving.mean() if moving.size else np.nan
    if f.get("plan_speed_to_tgt_kmh") and f.get("moving_speed_30min"):
        f["speed_need_ratio"] = f["plan_speed_to_tgt_kmh"] / f["moving_speed_30min"]
    lon, lat = ctx.lon[n - 1], ctx.lat[n - 1]
    f["dist_to_tgt_km"] = float(haversine_km(lon, lat, ctx.plan_lon[kt], ctx.plan_lat[kt]))
    inst = _position_dev(ctx, T, lon, lat, cur_dev_s)
    if inst is not None:
        f["inst_dev_s"], f["inst_seg_dist_m"], k_seg = inst
        f["inst_dev_minus_cur"] = f["inst_dev_s"] - f["cur_dev_s"]
        f["inst_dev_age_adj"] = f["inst_dev_s"] - f["age_last_gps_s"]  # точка старше T
        f["n_stops_pos_to_target"] = kt - k_seg
        f["route_pos_to_tgt_km"] = ctx.plan_cumdist[kt] - ctx.plan_cumdist[k_seg]

    # --- проезды остановок по GPS ----------------------------------------------
    ks, arr, dmin = _passes_at(ctx, T, n)
    good = (dmin < GOOD_PASS_M) & (ks <= kt)
    ks, arr = ks[good], arr[good]
    if ks.size:
        # последний проезд, согласованный по порядку (убираем «прыжки» назад по маршруту)
        order = np.argsort(arr)
        ks, arr = ks[order], arr[order]
        mono = ks >= np.maximum.accumulate(ks)
        ks, arr = ks[mono], arr[mono]
        dev = arr - ctx.plan_t[ks]
        kp = ks[-1]
        f["gps_dev_last"] = float(dev[-1])
        f["gps_dev_med3"] = float(np.median(dev[-3:]))
        f["gps_dev_med5"] = float(np.median(dev[-5:]))
        f["gps_dev_trend5"] = float(dev[-1] - dev[-5:][0]) if dev.size > 1 else 0.0
        f["gps_dev_std5"] = float(dev[-5:].std())
        recent = dev[(T - arr) <= 600]
        f["gps_dev_med_10min"] = float(np.median(recent)) if recent.size else np.nan
        auto = ~ctx.plan_manual[ks]
        f["gps_dev_med5_auto"] = float(np.median(dev[auto][-5:])) if auto.any() else np.nan
        f["age_last_pass_s"] = T - arr[-1]
        f["n_stops_pass_to_target"] = kt - kp
        f["plan_pass_to_target_s"] = ctx.plan_t[kt] - ctx.plan_t[kp]
        f["n_passes"] = ks.size
        # нижняя граница задержки на следующей непройденной остановке
        kn = min(kp + 1, kt)
        f["dev_lower_bound_next"] = float(max(dev[-1], T - ctx.plan_t[kn]))
        f["dev_lower_bound_tgt"] = float(max(dev[-1], T - ctx.plan_t[kt]))
        # оставшийся путь: до следующей остановки + по цепочке до цели
        d_next = float(haversine_km(lon, lat, ctx.plan_lon[kn], ctx.plan_lat[kn]))
        route = d_next + ctx.plan_cumdist[kt] - ctx.plan_cumdist[kn]
        f["route_dist_to_tgt_km"] = route
        # темп: реальное время между проездами vs плановое (последние ~5 остановок)
        if ks.size > 1:
            j = max(0, ks.size - 6)
            real = arr[-1] - arr[j]
            plan = ctx.plan_t[ks[-1]] - ctx.plan_t[ks[j]]
            dist = ctx.plan_cumdist[ks[-1]] - ctx.plan_cumdist[ks[j]]
            f["pace_ratio"] = real / plan if plan > 0 else np.nan
            v = dist / (real / 3600) if real > 0 else np.nan
            f["route_speed_kmh"] = v
            if v and v > 1:
                eta = route / v * 3600
                f["eta_gap_s"] = float(T + eta - ctx.plan_t[kt])
    f["delta_curdev_vs_gps"] = f.get("gps_dev_last", np.nan) - f["cur_dev_s"]
    return f


def build_features(points: pd.DataFrame, plan: pd.DataFrame, traffic: pd.DataFrame) -> pd.DataFrame:
    ctxs = build_contexts(plan, traffic)
    rows = []
    for r in points.itertuples(index=False):
        rows.append(features_at(ctxs[r.tr_id], r.T, r.target_stop_id, r.target_time_begin, r.cur_dev_s))
    X = pd.DataFrame(rows, index=points.index)
    return X
