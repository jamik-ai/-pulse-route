"""Синтетический датасет «несколько автобусов на маршруте» для демонстрации дашборда.

На каждом маршруте реального ТС из validate идут три машины: само ТС и два дополнительных борта с тем же
расписанием, сдвинутым на 12 и 24 минуты. Телеметрия строится по реальной геометрии дорог (кэш путей рейсов):
стоянки на остановках, разгон и торможение, отстой на конечных, шум GPS, пропадания связи. Задержки —
случайный процесс, похожий на реальный: медленный дрейф, инциденты (пробка, долгая посадка), частичное
восстановление на конечных, иногда ранний проход. Формат файлов — как в датасете хакатона, поэтому backend
проигрывает синтетику тем же NDTP-потоком.

Запуск::

    python -m src.synth            # → output/synthetic/{schedule.csv, traffic.csv, meta.json}
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.dataio import DATA, load_schedule

OUT_DIR = Path(__file__).resolve().parent.parent / "output" / "synthetic"
PATTERNS = Path(__file__).resolve().parent.parent / "output" / "cache" / "trip_patterns_v2.json"
COPIES = 3            # машин на маршруте (включая исходное ТС)
HEADWAY_MIN = 12      # сдвиг расписания между машинами, мин
EXTRA_ID = 700_000    # tr_id дополнительных бортов: 700000 + 10·номер маршрута + k
UNIT_ID = 9_900_000   # unit_id терминалов синтетики


def _xy(lat, lon, lat0):
    return np.c_[np.asarray(lon) * 111_320 * np.cos(np.radians(lat0)), np.asarray(lat) * 110_540]


def _line_for(stops: np.ndarray, line: list | None):
    """Линия рейса (lat, lon), накопленная длина, м, и положение каждой остановки на ней, м."""
    L = np.asarray(line if line and len(line) > 1 else stops, dtype=float)
    lat0 = L[:, 0].mean()
    xy = _xy(L[:, 0], L[:, 1], lat0)
    cum = np.r_[0.0, np.cumsum(np.hypot(*np.diff(xy, axis=0).T))]
    sxy = _xy(stops[:, 0], stops[:, 1], lat0)
    pos, j0 = [], 0
    for p in sxy:  # ближайшая точка линии не раньше предыдущей остановки (кольцевые рейсы)
        d = np.hypot(*(xy[j0:] - p).T) + 0.02 * (cum[j0:] - cum[j0])
        j0 += int(np.argmin(d))
        pos.append(cum[j0])
    return L, cum, np.asarray(pos)


def _at(L, cum, s):
    s = np.clip(s, 0, cum[-1])
    return np.interp(s, cum, L[:, 0]), np.interp(s, cum, L[:, 1])


# параметры подобраны так, чтобы распределение отклонений было похоже на реальное (validate/points.csv):
# медиана ≈ 10–20 с, 90-й перцентиль ≈ 250–300 с, опоздания > 2 мин ≈ 25%, раньше плана > 1 мин ≈ 10–15%
P = dict(drift_mu=0.0, drift_sd=3.0, noise=8.0, p_inc=0.022, inc=(70, 300), p_early=0.035, early=(40, 110), keep=0.975)


def _delays(n: int, rng, d_start: float) -> np.ndarray:
    """Отклонение от графика на каждой остановке рейса, с."""
    d = np.empty(n)
    d[0] = d_start
    drift = rng.normal(P["drift_mu"], P["drift_sd"])  # у рейса своя «погода»: кто-то теряет время, кто-то нагоняет
    for k in range(1, n):
        step = drift + rng.normal(0, P["noise"])
        if rng.random() < P["p_inc"]:
            step += rng.uniform(*P["inc"])        # инцидент: пробка, долгая посадка, светофор
        if rng.random() < P["p_early"]:
            step -= rng.uniform(*P["early"])      # «пустая» дорога — ранний проход
        d[k] = np.clip(d[k - 1] * P["keep"] + step, -170, 900)
    return d


def build(seed: int = 7) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    from src.backend.geo import trip_patterns

    rng = np.random.default_rng(seed)
    raw = pd.read_csv(DATA / "validate" / "schedule_plan.csv")
    plan = load_schedule("validate")
    trips = trip_patterns(plan.drop(columns=["time_fact_begin"]))
    lines = json.loads(PATTERNS.read_text()) if PATTERNS.exists() else {}
    sched_rows, traf = [], []
    meta = {"routes": [], "seed": seed, "copies": COPIES, "headway_min": HEADWAY_MIN}
    devs = []
    t_day0 = pd.Timestamp("2026-01-06").value // 10**9
    for ri, (tr, g) in enumerate(plan.groupby("tr_id")):
        g = g.sort_values("time_begin").reset_index(drop=True)
        rg = raw[raw.tr_id == tr].copy()
        tsec = g.time_begin.values.astype("datetime64[s]").astype(np.int64)
        ids = [int(tr)] + [EXTRA_ID + 10 * ri + k for k in range(1, COPIES)]
        meta["routes"].append({"base_tr_id": int(tr), "tr_ids": ids})
        for k, vid in enumerate(ids):
            shift = k * HEADWAY_MIN * 60
            # расписание борта: те же остановки, сдвиг по времени
            rs = rg.copy()
            rs["tr_id"] = vid
            rs["time_begin"] = (pd.to_datetime(rs.time_begin) + pd.Timedelta(seconds=shift)).dt.strftime("%Y-%m-%d %H:%M:%S")
            rs["time_fact_begin"] = ""
            sched_rows.append(rs)
            unit = UNIT_ID + 10 * ri + k
            lost_from = t_day0 + rng.integers(11, 15) * 3600 if rng.random() < 0.08 else None  # связь пропала до конца дня
            gaps = [(s, s + rng.integers(8, 20) * 60) for s in t_day0 + rng.integers(7 * 3600, 18 * 3600, size=rng.poisson(0.6))]
            carry = rng.normal(20, 40)
            pts = []  # (t, lat, lon, speed_kmh)
            prev_end = None
            for a, b, pid in trips[int(tr)]["spans"]:
                ks = np.flatnonzero((tsec >= a) & (tsec <= b))
                if len(ks) < 2:
                    continue
                stops = g.loc[ks, ["stop_lat", "stop_lon"]].to_numpy(float)
                L, cum, spos = _line_for(stops, lines.get(pid))
                plan_t = tsec[ks] + shift
                # на конечной часть опоздания «съедает» отстой
                d = _delays(len(ks), rng, d_start=max(-60.0, carry * 0.35 + rng.normal(5, 35)))
                arr = plan_t + d
                dwell = rng.uniform(12, 40, size=len(ks))
                dep = arr + dwell
                for j in range(1, len(ks)):  # физика: не быстрее ~55 км/ч между остановками
                    arr[j] = max(arr[j], dep[j - 1] + max(25.0, (spos[j] - spos[j - 1]) / 15.3))
                    dep[j] = arr[j] + dwell[j]
                carry = arr[-1] - plan_t[-1]
                devs.extend((arr - plan_t).tolist())
                # отстой перед рейсом: стоит на первой остановке
                t0 = (prev_end + 60) if prev_end is not None else arr[0] - 900
                la, lo = _at(L, cum, spos[0])
                for t in np.arange(t0, arr[0], 20.0):
                    pts.append((t, la, lo, 0.0))
                for j in range(len(ks)):
                    la, lo = _at(L, cum, spos[j])
                    for t in np.arange(arr[j], dep[j], rng.uniform(4, 8)):
                        pts.append((t, la, lo, 0.0))
                    if j + 1 < len(ks):
                        T0, T1, s0, s1 = dep[j], arr[j + 1], spos[j], spos[j + 1]
                        ts = np.arange(T0, T1, rng.uniform(3, 7))
                        u = (ts - T0) / max(1.0, T1 - T0)
                        s = s0 + (s1 - s0) * (1 - np.cos(np.pi * u)) / 2  # разгон и торможение
                        v = (s1 - s0) / max(1.0, T1 - T0) * np.pi / 2 * np.sin(np.pi * u) * 3.6
                        la, lo = _at(L, cum, s)
                        pts.extend(zip(ts, la, lo, v))
                prev_end = dep[-1]
            if not pts:
                continue
            p = np.array(pts, dtype=float)
            p = p[np.argsort(p[:, 0])]
            keep = np.ones(len(p), bool)
            for s, e in gaps:
                keep &= ~((p[:, 0] >= s) & (p[:, 0] < e))
            if lost_from is not None:
                keep &= p[:, 0] < lost_from
            p = p[keep]
            if len(p) < 2:
                continue
            noise = rng.normal(0, 4.0, size=(len(p), 2))  # шум GPS ~4 м
            lat = p[:, 1] + noise[:, 0] / 110_540
            lon = p[:, 2] + noise[:, 1] / (111_320 * np.cos(np.radians(p[:, 1])))
            spd = np.clip(p[:, 3] + rng.normal(0, 1.5, len(p)) * (p[:, 3] > 0), 0, 90)
            dx, dy = np.diff(lon, prepend=lon[0]) * np.cos(np.radians(lat)), np.diff(lat, prepend=lat[0])
            hdg = (np.degrees(np.arctan2(dx, dy)) + 360) % 360
            et = pd.to_datetime(p[:, 0], unit="s")
            traf.append(pd.DataFrame({"packet_id": 0, "tr_id": vid, "unit_id": unit,
                                      "event_time": et.strftime("%Y-%m-%d %H:%M:%S.%f"), "device_event_id": 0,
                                      "location_valid": "True", "gps_time": et.strftime("%Y-%m-%d %H:%M:%S"),
                                      "lon": lon.round(7), "lat": lat.round(7), "alt": 150,
                                      "speed": spd.round(1), "heading": hdg.round(0),
                                      "receive_time": et.strftime("%Y-%m-%d %H:%M:%S.%f"), "is_hist_data": "False"}))
    sched = pd.concat(sched_rows, ignore_index=True)
    traffic = pd.concat(traf, ignore_index=True).sort_values("event_time", kind="stable")
    dv = np.asarray(devs)
    meta.update(vehicles=int(sched.tr_id.nunique()), points=int(len(traffic)),
                deviation_s={"p10": round(float(np.percentile(dv, 10))), "p50": round(float(np.median(dv))),
                             "p90": round(float(np.percentile(dv, 90))), "share_late_120": round(float((dv > 120).mean()), 2),
                             "share_early_60": round(float((dv < -60).mean()), 2)})
    return sched, traffic, meta


def main(seed: int = 7) -> Path:
    from src.dataio import CACHE
    sched, traffic, meta = build(seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cols = ["tt_action_item_id", "time_begin", "time_fact_begin", "order_date", "manual_fill", "tr_id", "geom", "building_address"]
    sched[cols].to_csv(OUT_DIR / "schedule.csv", index=False)
    traffic.to_csv(OUT_DIR / "traffic.csv", index=False)
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    (CACHE / "traffic_synthetic.parquet").unlink(missing_ok=True)  # кэш нормализованной телеметрии — пересоберётся
    print(f"синтетика: {meta['vehicles']} ТС на {len(meta['routes'])} маршрутах, {meta['points']} точек → {OUT_DIR}")
    print("отклонения от графика:", meta["deviation_s"])
    return OUT_DIR


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 7)
