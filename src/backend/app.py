"""Backend диспетчерской: приём NDTP, оркестрация прогнозов, REST API и дашборд.

uvicorn src.backend.app:app --port 8080       → дашборд: /, Swagger: /docs

Процесс:
  * TCP :9201 — NDTP-сервер (эмулятор, реальные терминалы, replay архива);
  * OnlineEngine — состояние ТС, признаки строго по данным ≤ T, прогноз каждые 5 мин времени потока;
  * ML-модуль — отдельный сервис (ML_URL); недоступен → деградация на правило, приём не падает;
  * онлайн-точность — когда ТС проезжает целевую остановку (по GPS), считаем фактическую ошибку.
"""
import asyncio
import base64
import hashlib
import hmac
import json
import secrets
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.backend.geo import PATTERNS_CACHE, PATTERNS_CACHE_V1, build_pattern_lines, build_routes, matched_routes, plan_table, stop_names, stop_routes, trip_patterns
from src.dataio import SYNTH_DIR
from src.backend.ml_client import MLClient, _clean
from src.dataio import CACHE, _ts
from src.features import LATE_S as PASS_WINDOW_LATE_S
from src.realtime import replay as replay_mod
from src.realtime.online import HintProvider, OnlineEngine
from src.realtime.server import NdtpServer, unit_registry

log = logging.getLogger("backend")
OUT = CACHE.parent
DATASET_FILE = OUT / os.getenv("DATASET_STATE", "dataset.txt")  # выбранный в настройках датасет (переживает перезапуск)
STATIC = Path(__file__).parent / "static"

ML_URL = os.getenv("ML_URL", "http://localhost:8001")
PLAN_SPLIT = os.getenv("PLAN_SPLIT", "validate")
HINTS = os.getenv("HINTS", "data/validate/points.csv")
NDTP_PORT = int(os.getenv("NDTP_PORT", "9201"))
AUTO_REPLAY = os.getenv("AUTO_REPLAY", "1") == "1"        # дашборд всегда работает на архиве validate
AUTO_REPLAY_SPEED = float(os.getenv("AUTO_REPLAY_SPEED", "1"))
# дневное окно архива: 06:00–19:00 на линии 10–12 ТС из 13 (вечером и ночью по расписанию — 3–7)
AUTO_REPLAY_START = os.getenv("AUTO_REPLAY_START", "2026-01-06 06:00")
AUTO_REPLAY_HOURS = float(os.getenv("AUTO_REPLAY_HOURS", "13"))
STALE_S = int(os.getenv("STALE_S", "180"))          # нет GPS дольше — ТС «без связи»
OFFLINE_WALL_S = float(os.getenv("OFFLINE_WALL_S", "20"))  # поток молчит дольше — баннер «нет связи»
RED_P, YELLOW_P = 0.7, 0.4


def ops_pred(rec: dict) -> float:
    """Прогноз для диспетчера. В сабмите у целей с manual_fill стоит 0 (так размечен факт), а диспетчеру нужна
    реальная оценка модели — иначе опаздывающая машина выглядит «по графику»."""
    v = rec.get("ops_prediction_s")
    return float(v) if v is not None else float(rec["prediction"])


def ops_p_late(rec: dict):
    if rec.get("ops_basis") == "gps":
        return None  # вероятность для ручных участков не оцениваем — риск по величине отставания
    return rec["ops_p_late"] if rec.get("ops_p_late") is not None else rec.get("p_late")
EARLY_S = -60  # прогноз раньше плана на минуту и больше — отдельное событие «раньше плана»


def _finite(x, scale: float = 1.0):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return round(x * scale, 1) if np.isfinite(x) else None


class Hub:
    """Всё состояние backend: движок, NDTP-сервер, справочники, журнал действий."""

    def __init__(self):
        self.ml = MLClient(ML_URL)
        chosen = DATASET_FILE.read_text().strip() if DATASET_FILE.exists() else PLAN_SPLIT  # выбор из настроек
        if chosen == "synthetic" and not (SYNTH_DIR / "traffic.csv").exists() and PLAN_SPLIT == "synthetic":
            from src import synth
            synth.main()  # отдельный стенд с синтетикой: сгенерировать при первом запуске
        self.load_dataset(chosen if chosen != "synthetic" or (SYNTH_DIR / "traffic.csv").exists() else PLAN_SPLIT)
        self.actions: list[dict] = []
        self.acks: dict[int, dict] = {}  # ТС → кто и когда принял событие в работу (видно всем диспетчерам)
        self.evaluated: dict[str, dict] = {}   # sample_id → {pred, fact, err}
        self.replay_task: asyncio.Task | None = None
        self.replay_info: dict = {}
        self.rate = {"t": time.time(), "points": 0, "pps": 0.0}
        self.perf: list[dict] = []
        self.reset()

    def load_dataset(self, split: str):
        """План, маршруты, справочники и подсказки выбранного датасета: архив хакатона или синтетика."""
        self.split = split
        synthetic = split == "synthetic"
        self.routes = stop_routes(split) if synthetic else matched_routes(build_routes(split))
        # варианты рейсов: путь по дорогам через остановки (кэш) — показываем тот, что ТС выполняет сейчас
        self.plan = plan_table(split)
        self.trips = trip_patterns(self.plan)
        self.load_patterns()  # заодно пересчитывает routes_json / routes_version
        self.names = stop_names(split)
        self.registry = unit_registry()
        self.tr_to_unit = {}
        for u, tr in self.registry.items():
            self.tr_to_unit.setdefault(tr, u)
        hints = pd.read_csv(HINTS) if not synthetic and HINTS and Path(HINTS).exists() else None
        if hints is not None:
            hints["T"] = _ts(hints["T"])
        self.hints = HintProvider(hints)  # синтетика: подсказок АСУ нет — отклонение и цель по GPS и плану
        self.acks.clear() if hasattr(self, "acks") else None

    def reset(self):
        self.engine = OnlineEngine(self.plan, self.registry, self.ml, self.hints,
                                   on_prediction=self._on_pred, watermark_lag_s=60)
        self.server = NdtpServer(self.engine) if not hasattr(self, "server") else self.server
        self.server.engine = self.engine
        self.evaluated.clear()
        self.last_packet_wall = None
        self.pending: list[dict] = []

    def tick_rate(self):
        now, pts = time.time(), self.engine.stats["points"]
        dt = now - self.rate["t"]
        if dt >= 2:
            self.rate = {"t": now, "points": pts, "pps": max(0.0, (pts - self.rate["points"]) / dt)}
            ml = self.ml.stats()
            self.perf.append({"t": round(now), "stream_time": self.engine.max_ts, "pps": round(self.rate["pps"], 1),
                              "ml_p50": ml["latency_ms_p50"], "ml_p95": ml["latency_ms_p95"],
                              "ml_ok": ml["status"] == "ok"})
            del self.perf[:-600]

    def _on_pred(self, rec: dict):
        rec["risk"] = self.risk(rec)
        self.pending.append(rec)

    # --- онлайн-оценка точности: факт = GPS-проезд целевой остановки ---------------
    def evaluate_pending(self):
        now = self.engine.max_ts
        if now is None:
            return
        keep = []
        for rec in self.pending:
            if rec["target_time_begin"] + PASS_WINDOW_LATE_S + 60 > now:
                keep.append(rec)
                continue
            v = self.engine.vehicles.get(rec["tr_id"])
            ctx = v.context()
            k = np.flatnonzero(ctx.plan_id == rec["target_stop_id"])
            if k.size:
                dev = ctx.day_pass_dev()[k[0]]
                if np.isfinite(dev):
                    self.evaluated[rec["sample_id"]] = {"tr_id": rec["tr_id"], "pred": ops_pred(rec),
                                                        "fact_gps": float(dev), "err": abs(ops_pred(rec) - dev)}
        self.pending = keep

    # --- представление для дашборда ------------------------------------------------
    def risk(self, rec: dict | None) -> str:
        if not rec:
            return "none"
        p = ops_p_late(rec)
        d = ops_pred(rec)
        if p is None:
            r = "red" if d >= 180 else "yellow" if d >= 60 else "green"
        else:
            r = "red" if p >= RED_P else "yellow" if (p >= YELLOW_P or d >= 90) else "green"
        return "early" if r == "green" and d <= EARLY_S else r  # прибудет раньше плана больше чем на минуту

    def vehicle_view(self, tr: int, v) -> dict | None:
        if not v.et:
            return None
        now = self.engine.max_ts
        lon, lat, speed, heading, jumps = self._display_position(v)
        rec = self.engine.latest.get(tr)
        if rec and rec["target_time_begin"] < now:
            rec = None  # событие уже наступило — алерт не показываем (никаких прогнозов задним числом)
        age = int(now - v.et[-1])
        out = {"tr_id": tr, "unit_id": self.tr_to_unit.get(tr), "lat": lat, "lon": lon, "speed": speed, "heading": heading,
               "gps_unstable": jumps >= 3, "pattern": self.current_pattern(tr, now),
               "age_s": age, "stale": age > STALE_S, "risk": self.risk(rec), "incident": None}
        risky = out["risk"] in ("red", "yellow", "early")
        if not risky:
            self.acks.pop(tr, None)  # событие завершилось — принятие снимается
        out["ack"] = self.acks.get(tr) if risky else None
        f = self.engine.latest_features.get(tr) or {}
        dev_now = f.get("gps_dev_med3")
        out["now"] = {  # производные признаки на момент последнего прогноза
            "deviation_s": float(dev_now) if dev_now is not None and np.isfinite(dev_now) else
            (float(rec["cur_dev_s"]) if rec else None),
            "segment_speed_kmh": _finite(f.get("speed_mean_300")),
            "moving_speed_kmh": _finite(f.get("moving_speed_30min")),
            "dwell_min_of_10": _finite(f.get("stopped_share_600"), scale=10),
        }
        if rec:
            k = int(np.flatnonzero(v.context().plan_id == rec["target_stop_id"])[0])
            prev_name = self._name(int(v.context().plan_id[k - 1])) if k > 0 else "начало рейса"
            out["incident"] = {
                "sample_id": rec["sample_id"], "T": rec["T"], "prediction_s": ops_pred(rec),
                "p_late": ops_p_late(rec), "ops_basis": rec.get("ops_basis"),
                "interval_s": None if rec.get("ops_basis") == "gps" else (rec.get("ops_interval_s") or rec.get("interval_s")),
                "manual_target": bool(rec.get("manual_target")), "score_prediction_s": rec["prediction"],
                "expected_error_s": rec.get("expected_error_s"), "causes": rec.get("causes", []),
                "degraded": bool(rec.get("degraded")), "cur_dev_s": rec["cur_dev_s"],
                "target_stop_id": rec["target_stop_id"], "target_time_begin": rec["target_time_begin"],
                "eta_min": max(0, round((rec["target_time_begin"] - now) / 60)),
                "issued_at": rec["T"], "lead_min": round((rec["target_time_begin"] - rec["T"]) / 60, 1),
                "forecast_age_s": int(now - rec["T"]), "target_name": self._name(int(rec["target_stop_id"])),
                "forecast_arrival": rec["target_time_begin"] + ops_pred(rec),
                "segment": {"from": prev_name, "to": self._name(int(rec["target_stop_id"]))},
                "target_lat": float(v.context().plan_lat[k]), "target_lon": float(v.context().plan_lon[k]),
            }
        return out

    @staticmethod
    def _display_position(v, window: int = 80, max_kmh: float = 150.0):
        """Позиция для карты без GPS-выбросов: точки, «перелёт» в которые быстрее max_kmh, отбрасываем.
        Модель при этом получает исходную телеметрию (как в офлайн-обучении) — это только отображение."""
        et, rows = v.et[-window:], v.rows[-window:]
        acc = [0]
        jumps = 0
        for j in range(1, len(rows)):
            t0, (lo0, la0, _) = et[acc[-1]], rows[acc[-1]]
            lo1, la1, _ = rows[j]
            d = np.hypot((lo1 - lo0) * 111_320 * np.cos(np.radians(la0)), (la1 - la0) * 110_540)
            if d > 150 and d / max(1, et[j] - t0) * 3.6 > max_kmh:  # телепортация, а не шум GPS за 1 с
                jumps += 1
                continue
            acc.append(j)
        lon, lat, _ = rows[acc[-1]]
        # скорость для диспетчера — средняя за последнюю минуту (терминал шлёт мгновенную: 0 на остановке, 60 на перегоне)
        t_last = et[acc[-1]]
        recent = [rows[j][2] for j in acc if t_last - et[j] <= 60]
        speed = float(np.mean(recent)) if recent else float(rows[acc[-1]][2])
        heading = None
        for j in reversed(acc[:-1]):  # курс по последнему заметному смещению среди «чистых» точек
            lo0, la0, _ = rows[j]
            dx, dy = (lon - lo0) * 111_320 * np.cos(np.radians(lat)), (lat - la0) * 110_540
            if dx * dx + dy * dy > 15 ** 2:
                heading = float((np.degrees(np.arctan2(dx, dy)) + 360) % 360)
                break
        return lon, lat, speed, heading, jumps

    def load_patterns(self):
        """Линии вариантов рейсов из кэша; нет в кэше — временно линия по остановкам (достроится в фоне)."""
        def _load(path):
            try:
                return json.loads(path.read_text())
            except (FileNotFoundError, json.JSONDecodeError):
                return {}
        self.pattern_lines = _load(PATTERNS_CACHE)          # готовые (все перегоны построены, с направлением)
        old = _load(PATTERNS_CACHE_V1)                        # прежние — показываем, пока не пересчитаны
        for tr, d in self.trips.items():
            self.routes.setdefault(tr, {"track": [], "stops": []})
            self.routes[tr]["patterns"] = {pid: self.pattern_lines.get(pid) or old.get(pid) or stops
                                           for pid, stops in d["patterns"].items()}
        self.routes_json = json.dumps(_sanitize(self.routes), ensure_ascii=False).encode()
        self.routes_version = hashlib.sha1(self.routes_json).hexdigest()[:10]

    def missing_patterns(self) -> list:
        """Варианты рейсов без пути по дорогам (новые в расписании или не достроенные раньше)."""
        out = []
        for tr, d in self.trips.items():
            for pid, stops in d["patterns"].items():
                if pid not in self.pattern_lines:
                    out.append((tr, pid, stops))
        return out

    def current_pattern(self, tr: int, now: int) -> str | None:
        """Рейс по плану на момент now: идущий сейчас, иначе ближайший следующий (до часа), иначе последний."""
        spans = self.trips.get(tr, {}).get("spans", [])
        for a, b, pid in spans:
            if a - 300 <= now <= b + 300:
                return pid
        nxt = [(a, pid) for a, b, pid in spans if 0 < a - now <= 3600]
        if nxt:
            return min(nxt)[1]
        past = [(b, pid) for a, b, pid in spans if b < now]
        return max(past)[1] if past else (spans[0][2] if spans else None)

    # --- ленты маршрутов (второй вид диспетчера) ----------------------------------
    def _trip(self, tr: int, v, now: int, any_trip: bool = False):
        """Текущий рейс ТС по плану: индексы остановок в плане, их время и путь от начала рейса (км).
        any_trip — если сейчас рейса нет, взять ближайший следующий (иначе последний): для лент «вне рейса»."""
        spans = self.trips.get(tr, {}).get("spans", [])
        cur = next(((a, b, pid) for a, b, pid in spans if a - 300 <= now <= b + 300), None)
        if not cur and any_trip and spans:
            nxt = [sp for sp in spans if sp[0] > now]
            cur = min(nxt) if nxt else max(spans, key=lambda sp: sp[1])
        if not cur:
            return None
        a, b, pid = cur
        ctx = v.context()
        ks = np.flatnonzero((ctx.plan_t >= a) & (ctx.plan_t <= b))
        if len(ks) < 2:
            return None
        d = self._road_d(pid, ctx.plan_lat[ks], ctx.plan_lon[ks])
        if d is None:  # пути по дорогам ещё нет — по прямой между остановками
            d = ctx.plan_cumdist[ks] - ctx.plan_cumdist[ks[0]]
        return pid, ctx, ks, ctx.plan_t[ks].astype(float), d.astype(float)

    def _road_d(self, pid: str, la, lo):
        """Путь от начала рейса до каждой остановки по линии на дорогах, км (кэш по варианту рейса)."""
        cache = self.__dict__.setdefault("_road_cache", {})
        line = self.pattern_lines.get(pid)
        if not line or len(line) < 2:
            return None
        if cache.get(pid, (None,))[0] is not line:
            L = np.asarray(line, dtype=float)
            kx = 111.32 * np.cos(np.radians(L[:, 0].mean()))
            xy = np.c_[L[:, 1] * kx, L[:, 0] * 110.54]
            cum = np.r_[0.0, np.cumsum(np.hypot(*np.diff(xy, axis=0).T))]
            out, j0 = [], 0
            for a, b in zip(la, lo):  # ближайшая точка линии не раньше предыдущей остановки (кольца, повторы)
                dd = np.hypot(xy[j0:, 0] - b * kx, xy[j0:, 1] - a * 110.54) + 0.02 * (cum[j0:] - cum[j0])
                j0 += int(np.argmin(dd))
                out.append(cum[j0])
            cache[pid] = (line, np.asarray(out) if len(out) == len(la) else None)
        d = cache[pid][1]
        return d if d is not None and len(d) == len(la) and d[-1] > 0 else None

    @staticmethod
    def _project(ctx, ks, d, lat, lon, near_km: float) -> float | None:
        """Положение ТС вдоль рейса (км): проекция на цепочку остановок; на кольцевых рейсах из двух
        одинаково близких участков берём тот, что ближе к плановому положению near_km."""
        la, lo = ctx.plan_lat[ks], ctx.plan_lon[ks]
        kx = 111.32 * np.cos(np.radians(lat))
        ax, ay = (lo[:-1] - lon) * kx, (la[:-1] - lat) * 110.54
        bx, by = (lo[1:] - lon) * kx, (la[1:] - lat) * 110.54
        vx, vy = bx - ax, by - ay
        f = np.clip(-(ax * vx + ay * vy) / np.maximum(vx * vx + vy * vy, 1e-9), 0, 1)
        dist = np.hypot(ax + f * vx, ay + f * vy)
        along = d[:-1] + f * (d[1:] - d[:-1])
        j = int(np.argmin(dist + 0.05 * np.abs(along - near_km)))
        return float(along[j]) if dist[j] < 0.6 else None

    def ribbons(self) -> dict:
        """Варианты рейсов с ТС на линии: остановки по длине рейса, факт и «где должен быть» по графику."""
        now = self.engine.max_ts
        if now is None:
            return {"now": None, "rows": []}
        rows, off = {}, 0
        for tr, v in self.engine.vehicles.items():
            view = self.vehicle_view(tr, v)  # None — телеметрии не было вообще
            trip = self._trip(tr, v, now, any_trip=True)  # показываем все ТС: вне рейса — ближайший рейс
            if not trip:
                off += 1
                continue
            pid, ctx, ks, t, d = trip
            L = max(float(d[-1]), 0.1)
            state = "before" if now < t[0] - 300 else "after" if now > t[-1] + 300 else "on"
            ghost = float(np.interp(now, t, d))
            no_gps = view is None or view["stale"]
            dev = (view.get("now") or {}).get("deviation_s") if view and not no_gps and state == "on" else None
            pos = None
            if not no_gps:
                pos = self._project(ctx, ks, d, view["lat"], view["lon"], ghost) if view["lat"] is not None else None
                if pos is None and dev is not None:
                    pos = float(np.interp(now - dev, t, d))
                if pos is None:  # далеко от линии (депо, отстой) — у начала/конца ленты
                    pos = 0.0 if state != "after" else L
            if state != "on":
                off += 1
            view = view or {"unit_id": self.tr_to_unit.get(tr), "risk": "none", "stale": True, "age_s": None}
            row = rows.setdefault(pid, {"pid": pid, "from": self._name(int(ctx.plan_id[ks[0]])),
                                        "to": self._name(int(ctx.plan_id[ks[-1]])), "length_km": round(L, 1),
                                        "ticks": [round(float(x) / L, 4) for x in d], "vehicles": []})
            i = view.get("incident")
            row["vehicles"].append({
                "tr_id": tr, "unit_id": view.get("unit_id"), "risk": view["risk"], "stale": view["stale"],
                "no_gps": no_gps, "gps_age_s": view.get("age_s"), "trip_state": state,
                "trip_start_min": round((t[0] - now) / 60) if state == "before" else None,
                "trip_end_ago_min": round((now - t[-1]) / 60) if state == "after" else None,  # нет GPS: положение неизвестно, показываем только план
                "x": round(pos / L, 4) if pos is not None else None, "gx": round(ghost / L, 4), "deviation_s": dev,
                "incident": i and {k: i[k] for k in ("prediction_s", "p_late", "eta_min", "causes", "segment",
                                                     "interval_s", "target_time_begin")}})
        order = {"red": 0, "yellow": 1, "early": 2, "green": 3, "none": 4}
        for pid, row in rows.items():  # плановый интервал: старты этого варианта рейса у всех ТС в ±2 ч
            starts = sorted(a for d in self.trips.values() for a, b, p in d["spans"] if p == pid and abs(a - now) < 7200)
            gaps = np.diff(starts)
            row["headway_min"] = round(float(np.median(gaps)) / 60) if len(gaps) else None
            row["vehicles"].sort(key=lambda x: x["x"] if x["x"] is not None else x["gx"])
            row["worst"] = min((order[x["risk"]] for x in row["vehicles"]), default=3)
        for row in rows.values():
            row["on_trip"] = sum(1 for x in row["vehicles"] if x["trip_state"] == "on")
        out = sorted(rows.values(), key=lambda r: (r["on_trip"] == 0, r["worst"], -len(r["vehicles"]), r["from"]))
        return {"now": now, "rows": out, "off_trip": off}

    def thread(self, tr: int) -> dict | None:
        """«Нитка графика» ТС: план, фактические проезды остановок и прогноз на текущем рейсе."""
        v, now = self.engine.vehicles.get(tr), self.engine.max_ts
        trip = self._trip(tr, v, now) if v and now else None
        if not trip:
            return None
        pid, ctx, ks, t, d = trip
        dev = ctx.day_pass_dev()[ks]
        fact = [float(t[j] + dev[j]) if np.isfinite(dev[j]) and t[j] + dev[j] <= now else None for j in range(len(ks))]
        view = self.vehicle_view(tr, v) or {}
        i = view.get("incident")
        target = None
        if i:
            j = np.flatnonzero(ctx.plan_id[ks] == i["target_stop_id"])
            if j.size:
                target = {"t": float(t[j[0]]), "d": float(d[j[0]]), "prediction_s": i["prediction_s"],
                          "interval_s": i.get("interval_s"), "issued_at": i["issued_at"]}
        return {"now": now, "pid": pid, "from": self._name(int(ctx.plan_id[ks[0]])), "to": self._name(int(ctx.plan_id[ks[-1]])),
                "deviation_s": (view.get("now") or {}).get("deviation_s"),
                "stops": [{"t": float(t[j]), "d": round(float(d[j]), 3), "fact": fact[j]} for j in range(len(ks))],
                "target": target}

    def _name(self, stop_id: int) -> str:
        n = self.names.get(stop_id)
        return n if isinstance(n, str) and n.strip() else f"остановка №{stop_id}"

    def state(self) -> dict:
        e = self.engine
        vehicles = [x for tr, v in e.vehicles.items() if (x := self.vehicle_view(tr, v))]
        if e.max_ts is not None:  # в плане есть, но валидной телеметрии нет вообще — тоже «без связи»
            seen = {x["tr_id"] for x in vehicles}
            vehicles += [{"tr_id": tr, "unit_id": self.tr_to_unit.get(tr), "lat": None, "lon": None, "speed": 0,
                          "heading": None, "gps_unstable": False, "age_s": None, "stale": True, "no_data": True,
                          "risk": "none", "incident": None, "now": {}} for tr in e.vehicles if tr not in seen]
        order = {"red": 0, "yellow": 1, "early": 2, "green": 3, "none": 4}
        incidents = sorted([x for x in vehicles if x["incident"] and x["risk"] in ("red", "yellow", "early")],
                           key=lambda x: (order[x["risk"]], -(x["incident"]["p_late"] or 0),
                                          -x["incident"]["prediction_s"]))
        errs = [x["err"] for x in self.evaluated.values()]
        wall_age = None if self.last_packet_wall is None else time.time() - self.last_packet_wall
        replaying = self.replay_task is not None and not self.replay_task.done()
        status = ("idle" if self.last_packet_wall is None else
                  "live" if wall_age < OFFLINE_WALL_S else "offline")
        return {
            "stream": {"status": status, "stream_time": e.max_ts, "last_packet_age_s": wall_age,
                       "packets_per_s": round(self.rate["pps"], 1), "points_total": e.stats["points"],
                       "late_points": e.stats["late"], "predictions_total": e.stats["predictions"],
                       "infer_ms_per_T": round(e.stats["infer_ms_total"] / max(1, len({p["T"] for p in e.predictions[-2000:]}) or 1), 1),
                       "replay": {**self.replay_info, "running": replaying},
                       "source": "demo" if (replaying or self.replay_info) else "live",
                       "dataset": self.split,
                       "connections": self.server.conn["active"], "crc_errors": self.server.conn["crc_errors"]},
            "kpi": {"on_line": sum(1 for x in vehicles if not x["stale"]), "total": len(e.vehicles),
                    "at_risk": sum(1 for x in vehicles if x["risk"] in ("red", "yellow")),
                    "early": sum(1 for x in vehicles if x["risk"] == "early"),
                    "stale": sum(1 for x in vehicles if x["stale"]),
                    "no_forecast": sum(1 for x in vehicles if not x["incident"]),
                    "online_mae_s": round(float(np.mean(errs)), 1) if errs else None, "n_evaluated": len(errs)},
            "ml": self.ml.stats(),
            "routes_version": self.routes_version,
            "vehicles": vehicles,
            "incidents": [x["tr_id"] for x in incidents],
        }


hub: Hub | None = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global hub
    hub = Hub()
    orig = hub.server.handle

    async def handle(reader, writer):  # отмечаем «живость» потока по факту приёма байт
        class R:
            async def read(self, n):
                data = await reader.read(n)
                if data:
                    hub.last_packet_wall = time.time()
                return data
        await orig(R(), writer)

    srv = await asyncio.start_server(handle, "0.0.0.0", NDTP_PORT)
    ticker = asyncio.create_task(_ticker())
    hub.routes_task = asyncio.create_task(_routes_builder())
    if AUTO_REPLAY:
        hub.replay_task = asyncio.create_task(_replay_loop(AUTO_REPLAY_SPEED, AUTO_REPLAY_START, AUTO_REPLAY_HOURS))
    log.info("NDTP on :%d, ML %s, ТС в плане %d", NDTP_PORT, ML_URL, len(hub.engine.vehicles))
    async with srv:
        yield
    ticker.cancel()


async def _replay_loop(speed: float, start: str | None, hours: float | None = None):
    """Архив validate как непрерывный NDTP-поток: проигрываем окно [start, start+hours) и начинаем заново."""
    hub.replay_args = type("A", (), dict(host="127.0.0.1", port=NDTP_PORT, split=hub.split, speed=speed, start=start, hours=hours))()
    while True:
        hub.reset()
        hub.replay_info = {"speed": hub.replay_args.speed, "start": hub.replay_args.start, "split": hub.split,
                           "started_wall": time.time(), "loop": True}
        try:
            await replay_mod.run(hub.replay_args)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 — обрыв/ошибка: пауза и заново, сервис не падает
            log.exception("replay")
            await asyncio.sleep(5)
        if not hub.replay_args.hours:
            hub.replay_args.start = None  # без окна — следующий круг с начала суток


async def _routes_builder():
    """Фон: пути по дорогам для рейсов без готовой линии — из кэша перегонов, недостающие перегоны параллельно.
    Рейс принимается, только если построены все его перегоны; неудачные повторяются через 10 минут."""
    while True:
        todo = {pid: stops for _, pid, stops in hub.missing_patterns()}
        if not todo:
            return
        t0 = time.time()
        lines, st = await asyncio.to_thread(build_pattern_lines, todo)
        hub.pattern_lines.update(lines)
        PATTERNS_CACHE.parent.mkdir(parents=True, exist_ok=True)
        PATTERNS_CACHE.write_text(json.dumps(hub.pattern_lines))
        hub.load_patterns()
        log.info("маршруты: %s за %.0f с", st, time.time() - t0)
        if len(lines) < len(todo):
            await asyncio.sleep(600)


async def _ticker():
    while True:
        await asyncio.sleep(2)
        try:
            hub.tick_rate()
            hub.evaluate_pending()
        except Exception:  # noqa: BLE001
            log.exception("evaluate_pending")


# Эндпоинты, читающие состояние движка, — async: выполняются в том же цикле событий, что и приём
# NDTP, поэтому не видят буфер ТС в середине записи.
app = FastAPI(title="Пульс маршрута — Backend", version="1.0.0", lifespan=lifespan,
              description="Приём NDTP-телеметрии, прогноз опозданий на 10–15 мин, API диспетчерского дашборда.")


DASH_USER = os.getenv("DASH_USER", "")
DASH_PASSWORD = os.getenv("DASH_PASSWORD", "")
# учётки: основная + дополнительные DASH_EXTRA_USERS="login:password,login2:password2"
USERS = {u: p for u, _, p in (x.partition(":") for x in os.getenv("DASH_EXTRA_USERS", "").split(",") if ":" in x)}
if DASH_USER:
    USERS[DASH_USER] = DASH_PASSWORD


def _password_ok(user: str, pwd: str) -> bool:
    ok = False
    for u, p in USERS.items():  # сравниваем со всеми, чтобы время ответа не выдавало существующие логины
        ok |= secrets.compare_digest(user, u) & secrets.compare_digest(pwd, p)
    return ok
SESSION_SECRET = (os.getenv("SESSION_SECRET") or secrets.token_hex(32)).encode()
SESSION_TTL_S = 12 * 3600
PUBLIC_PATHS = {"/health", "/login", "/api/login", "/static/login.html", "/static/logo-pulse.svg", "/docs", "/docs/oauth2-redirect", "/redoc", "/openapi.json"}
PUBLIC_PREFIXES = ("/code-docs",)  # документация и спецификация API — открыты; данные API — только со входом
_failed: dict[str, list[float]] = {}  # ip → времена неудачных попыток входа


def _sign(payload: str) -> str:
    return hmac.new(SESSION_SECRET, payload.encode(), hashlib.sha256).hexdigest()


def make_session(user: str) -> str:
    payload = f"{user}|{int(time.time()) + SESSION_TTL_S}"
    return base64.urlsafe_b64encode(payload.encode()).decode() + "." + _sign(payload)


def check_session(token: str | None) -> bool:
    if not token or "." not in token:
        return False
    b64, sig = token.rsplit(".", 1)
    try:
        payload = base64.urlsafe_b64decode(b64.encode()).decode()
    except Exception:  # noqa: BLE001
        return False
    if not hmac.compare_digest(sig, _sign(payload)):
        return False
    user, _, exp = payload.partition("|")
    return user in USERS and exp.isdigit() and int(exp) > time.time()


def current_user(request: Request) -> str:
    """Логин из cookie сессии (или Basic auth)."""
    tok = request.cookies.get("session") or ""
    if "." in tok:
        try:
            return base64.urlsafe_b64decode(tok.rsplit(".", 1)[0].encode()).decode().partition("|")[0]
        except Exception:  # noqa: BLE001
            pass
    h = request.headers.get("authorization", "")
    if h.startswith("Basic "):
        try:
            return base64.b64decode(h[6:]).decode().partition(":")[0]
        except Exception:  # noqa: BLE001
            pass
    return "default"


def _check_basic(header: str) -> bool:
    """Basic auth — для API-клиентов и скриптов (браузер ходит по cookie)."""
    if not header.startswith("Basic "):
        return False
    try:
        user, _, pwd = base64.b64decode(header[6:]).decode().partition(":")
    except Exception:  # noqa: BLE001
        return False
    return _password_ok(user, pwd)


def _client_ip(request: Request) -> str:
    return request.headers.get("x-real-ip") or (request.client.host if request.client else "?")


@app.middleware("http")
async def auth(request: Request, call_next):
    """Вход обязателен, если задан DASH_PASSWORD. Страница /login — без входа."""
    path = request.url.path
    if not USERS or path in PUBLIC_PATHS or path.startswith(PUBLIC_PREFIXES):
        return await call_next(request)
    if check_session(request.cookies.get("session")) or _check_basic(request.headers.get("authorization", "")):
        return await call_next(request)
    if path.startswith("/api/") or path == "/openapi.json":
        return JSONResponse({"detail": "Требуется вход"}, status_code=401)
    return RedirectResponse("/login", status_code=303)


def _openapi_with_auth():
    if app.openapi_schema:
        return app.openapi_schema
    from fastapi.openapi.utils import get_openapi
    schema = get_openapi(title=app.title, version=app.version, description=app.description, routes=app.routes)
    schema.setdefault("components", {})["securitySchemes"] = {
        "basic": {"type": "http", "scheme": "basic", "description": "Логин и пароль дашборда"}}
    schema["security"] = [{"basic": []}]
    app.openapi_schema = schema
    return schema


app.openapi = _openapi_with_auth


@app.get("/login", include_in_schema=False)
async def login_page():
    return FileResponse(STATIC / "login.html")


class LoginReq(BaseModel):
    user: str
    password: str


@app.post("/api/login", summary="Вход диспетчера: ставит cookie сессии")
async def api_login(req: LoginReq, request: Request):
    ip, now = _client_ip(request), time.time()
    recent = [t for t in _failed.get(ip, []) if now - t < 600]
    if len(recent) >= 10:
        return JSONResponse({"detail": "Слишком много попыток. Попробуйте через 10 минут"}, status_code=429)
    if not _password_ok(req.user, req.password):
        _failed[ip] = recent + [now]
        await asyncio.sleep(0.5)
        return JSONResponse({"detail": "Неверный логин или пароль"}, status_code=401)
    _failed.pop(ip, None)
    resp = JSONResponse({"status": "ok"})
    secure = request.headers.get("x-forwarded-proto") == "https"
    resp.set_cookie("session", make_session(req.user), max_age=SESSION_TTL_S, httponly=True,
                    samesite="lax", secure=secure)
    return resp


@app.post("/api/logout", summary="Выход")
async def api_logout():
    resp = JSONResponse({"status": "ok"})
    resp.delete_cookie("session")
    return resp


def _sanitize(o):
    if isinstance(o, dict):
        return {str(k): _sanitize(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_sanitize(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, np.integer):
        return int(o)
    return o


def J(o) -> JSONResponse:
    """JSON-ответ: numpy-типы → python, NaN/inf → null."""
    return JSONResponse(_sanitize(o))


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")
_CODE_DOCS = Path(__file__).resolve().parents[2] / "docs" / "html"
if _CODE_DOCS.exists():  # документация кода (Sphinx) — python -m sphinx docs/source docs/html
    app.mount("/code-docs", StaticFiles(directory=_CODE_DOCS, html=True), name="code-docs")


@app.get("/health")
async def health():
    return {"status": "ok", "ml": hub.ml.stats()["status"]}


@app.get("/api/state", summary="Снимок для дашборда: поток, KPI, ТС с прогнозами, список инцидентов")
async def api_state():
    return J(hub.state())


@app.get("/api/routes", summary="Геометрия маршрутов: GPS-треки (по дорогам) и остановки")
async def api_routes():
    return Response(hub.routes_json, media_type="application/json", headers={"Cache-Control": "no-store"})


@app.get("/api/ribbons", summary="Ленты маршрутов: ТС на линии вдоль рейса — факт и положение по графику")
async def api_ribbons():
    return J(hub.ribbons())


@app.get("/api/vehicles/{tr_id}/thread", summary="Нитка графика ТС: план, факт проездов и прогноз на текущем рейсе")
async def api_thread(tr_id: int):
    out = hub.thread(tr_id)
    if out is None:
        raise HTTPException(404, "ТС сейчас не в рейсе")
    return J(out)


@app.get("/api/vehicles/{tr_id}/history", summary="История прогнозов ТС и их онлайн-проверка")
async def api_history(tr_id: int, limit: int = 48):
    preds = [p for p in hub.engine.predictions if p["tr_id"] == tr_id][-limit:]
    return J([{"T": p["T"], "prediction_s": ops_pred(p), "p_late": ops_p_late(p),
               "fact_gps_s": hub.evaluated.get(p["sample_id"], {}).get("fact_gps")} for p in preds])


class WhatIf(BaseModel):
    scenarios: list[str] = Field(default=["none", "signal_priority", "short_dwell"])


SCENARIOS = {
    "none": ("Ничего не делать", {}),
    "signal_priority": ("Приоритет на светофорах", {"speed": 1.25, "stopped": 0.7}),
    "short_dwell": ("Сократить стоянки на остановках", {"stopped": 0.5}),
}


@app.post("/api/vehicles/{tr_id}/whatif",
          summary="What-if: пересчёт прогноза моделью при изменённых условиях движения")
async def api_whatif(tr_id: int, body: WhatIf):
    row = hub.engine.latest_features.get(tr_id)
    if row is None:
        raise HTTPException(404, "нет прогноза для этого ТС")
    rows, labels = [], []
    for key in body.scenarios:
        if key not in SCENARIOS:
            continue
        label, mod = SCENARIOS[key]
        r = dict(row)
        for f in list(r):
            if f.startswith(("speed_mean_", "moving_speed", "route_speed")) and r[f] is not None:
                r[f] = r[f] * mod.get("speed", 1.0)
            if f.startswith("stopped_share_") and r[f] is not None:
                r[f] = r[f] * mod.get("stopped", 1.0)
        rows.append(r)
        labels.append((key, label))
    res = hub.ml.predict_full(pd.DataFrame(rows), explain=False)
    return J([{"scenario": k, "label": lab, "prediction_s": ops_pred({"prediction": x["prediction_s"], **x}), "p_late": ops_p_late(x)}
              for (k, lab), x in zip(labels, res)])


@app.get("/api/analytics", summary="BI-аналитика: риски во времени, проблемные ТС, причины, точность, производительность")
async def api_analytics(hours: float | None = None, bin_min: int = 15):
    """hours — окно по времени потока (None — весь день); bin_min — шаг графика «когда были риски»."""
    now = hub.engine.max_ts or 0
    preds = [p for p in hub.engine.predictions if hours is None or p["T"] >= now - hours * 3600]
    ev = hub.evaluated
    step = bin_min * 60
    bins: dict[int, dict] = {}
    for p in preds:
        b = bins.setdefault(p["T"] // step * step, {"t": p["T"] // step * step, "n": 0, "red": 0, "yellow": 0,
                                                   "err": [], "trs": set()})
        b["n"] += 1
        r = p.get("risk")
        if r in ("red", "yellow"):
            b[r] += 1
            b["trs"].add(p["tr_id"])
        if p["sample_id"] in ev:
            b["err"].append(ev[p["sample_id"]]["err"])
    timeline = [{"t": b["t"], "n": b["n"], "red": b["red"], "yellow": b["yellow"], "trs": sorted(b["trs"]),
                 "mae_s": round(float(np.mean(b["err"])), 1) if b["err"] else None, "n_err": len(b["err"])}
                for b in sorted(bins.values(), key=lambda x: x["t"])]
    # ожидаемые опоздания — понятные категории
    cats = [("раньше графика", -1e9, -60), ("вовремя (±1 мин)", -60, 60), ("до 2 мин", 60, 120),
            ("2–5 мин", 120, 300), ("больше 5 мин", 300, 1e9)]
    dist = [{"label": lab, "n": sum(1 for p in preds if lo <= ops_pred(p) < hi)} for lab, lo, hi in cats]
    # главные причины рисков (первая причина SHAP у алертов)
    causes: dict[str, int] = {}
    for p in preds:
        if p.get("risk") in ("red", "yellow") and p.get("causes"):
            f = p["causes"][0]["factor"]
            causes[f] = causes.get(f, 0) + 1
    # по ТС
    per_tr: dict[int, dict] = {}
    for p in preds:
        d = per_tr.setdefault(p["tr_id"], {"tr_id": p["tr_id"], "n": 0, "alerts": 0, "red": 0, "preds": [], "err": []})
        d["n"] += 1
        d["preds"].append(ops_pred(p))
        if p.get("risk") in ("red", "yellow"):
            d["alerts"] += 1
            d["red"] += p.get("risk") == "red"
        if p["sample_id"] in ev:
            d["err"].append(ev[p["sample_id"]]["err"])
    vehicles = [{"tr_id": d["tr_id"], "unit_id": hub.tr_to_unit.get(d["tr_id"]), "n": d["n"], "alerts": d["alerts"],
                 "red": d["red"], "mean_pred_s": round(float(np.mean(d["preds"])), 1),
                 "max_pred_s": round(float(np.max(d["preds"])), 1),
                 "mae_s": round(float(np.mean(d["err"])), 1) if d["err"] else None,
                 "risk_now": hub.risk(hub.engine.latest.get(d["tr_id"]))} for d in per_tr.values()]
    vehicles.sort(key=lambda x: (-x["alerts"], -x["red"], -x["mean_pred_s"]))
    errs = [ev[p["sample_id"]]["err"] for p in preds if p["sample_id"] in ev]
    leads = [(p["target_time_begin"] - p["T"]) / 60 for p in preds]
    st = hub.state()
    return J({
        "window": {"hours": hours, "from": min((p["T"] for p in preds), default=None), "to": now or None, "bin_min": bin_min},
        "totals": {"predictions": len(preds), "alerts": sum(1 for p in preds if p.get("risk") in ("red", "yellow")),
                   "red": sum(1 for p in preds if p.get("risk") == "red"),
                   "at_risk_now": st["kpi"]["at_risk"], "on_line": st["kpi"]["on_line"], "total_vehicles": st["kpi"]["total"],
                   "evaluated": len(errs),
                   "online_mae_s": round(float(np.mean(errs)), 1) if errs else None,
                   "within_1min": round(float(np.mean([e <= 60 for e in errs])) * 100) if errs else None,
                   "within_2min": round(float(np.mean([e <= 120 for e in errs])) * 100) if errs else None,
                   "lead_min_min": round(min(leads), 1) if leads else None,
                   "lead_min_max": round(max(leads), 1) if leads else None},
        "timeline": timeline, "dist": dist,
        "causes": sorted(({"factor": k, "n": v} for k, v in causes.items()), key=lambda x: -x["n"]),
        "vehicles": vehicles, "perf": hub.perf[-300:],
    })


SETTINGS_FILE = OUT / "user_settings.json"
DEFAULT_SETTINGS = {"map_provider": "osm", "muted": True, "demo_speed": 1, "brand_font": "Tektur", "sound": True}
YANDEX_MAPS_KEY = os.getenv("YANDEX_MAPS_KEY", "")


def _load_settings() -> dict:
    try:
        return json.loads(SETTINGS_FILE.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


class Settings(BaseModel):
    map_provider: str = Field("osm", pattern="^(osm|yandex)$", description="Подложка: OpenStreetMap или Яндекс Карты")
    muted: bool = Field(True, description="Приглушённая подложка")
    demo_speed: int = Field(5, ge=1, le=120, description="Скорость демо-потока (1 — реальное время)")
    brand_font: str = Field("Tektur", max_length=40, description="Шрифт названия «Пульс маршрута»")
    sound: bool = Field(True, description="Звуковое оповещение о новых алертах высокого риска")


@app.get("/api/me", summary="Текущий пользователь, его настройки и конфигурация клиента")
async def api_me(request: Request):
    user = current_user(request)
    st = {**DEFAULT_SETTINGS, **_load_settings().get(user, {})}
    return {"user": user, "settings": st,
            "yandex": {"key": YANDEX_MAPS_KEY, "has_key": bool(YANDEX_MAPS_KEY)}}


@app.put("/api/me/settings", summary="Сохранить настройки аккаунта")
async def api_save_settings(body: Settings, request: Request):
    data = _load_settings()
    data[current_user(request)] = body.model_dump()
    SETTINGS_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=1))
    return {"status": "ok", "settings": body.model_dump()}



class Action(BaseModel):
    tr_id: int
    kind: str = Field(..., description="measure | follow | wrong_forecast")
    detail: dict[str, Any] = {}


@app.post("/api/actions", summary="Журнал действий диспетчера (меры, отметка «прогноз неверен» → датасет дообучения)")
async def api_action(a: Action, request: Request):
    rec = {"wall_time": time.time(), "stream_time": hub.engine.max_ts, "user": current_user(request), **a.model_dump(),
           "prediction": hub.engine.latest.get(a.tr_id, {}).get("sample_id")}
    hub.actions.append(rec)
    if a.kind == "ack":
        hub.acks[a.tr_id] = {"user": rec["user"], "at": rec["stream_time"], "signal_at": a.detail.get("signal_at")}
    elif a.kind == "ack_cancel":
        hub.acks.pop(a.tr_id, None)
    with open(OUT / "dispatcher_actions.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False, default=_clean) + "\n")
    return {"status": "ok"}


ACTION_LABEL = {"ack": "Принял в работу", "ack_cancel": "Отменил принятие", "wrong_forecast": "Отметил неверный прогноз",
                "follow": "Слежение за ТС", "measure": "Мера (модельная оценка)"}


def _episodes(preds: list) -> list:
    """События в прогнозах: подряд идущие прогнозы «под риском» по одной машине = одно событие."""
    out, cur = [], {}
    for p in sorted(preds, key=lambda x: (x["tr_id"], x["T"])):
        risky = p.get("risk") in ("red", "yellow", "early")
        e = cur.get(p["tr_id"])
        if risky and (e is None or p["T"] - e["last"] > STEP_GAP_S):
            e = cur[p["tr_id"]] = {"tr_id": p["tr_id"], "start": p["T"], "last": p["T"], "early": 0, "late": 0, "red": False}
            out.append(e)
        if risky:
            e["last"] = p["T"]
            e["early" if p["risk"] == "early" else "late"] += 1
            e["red"] |= p["risk"] == "red"
        elif e is not None:
            cur.pop(p["tr_id"], None)
    return out


STEP_GAP_S = 600  # разрыв больше двух шагов прогноза — уже другое событие


def dispatcher_report(hours: float | None) -> dict:
    now = hub.engine.max_ts or 0
    lo = now - hours * 3600 if hours else None
    t0 = min((p["T"] for p in hub.engine.predictions), default=None)
    if t0 is not None and (lo is None or lo < t0):
        lo = t0  # окно не раньше начала потока
    inwin = lambda t: t is not None and (lo is None or t >= lo)  # noqa: E731
    acts = [a for a in hub.actions if inwin(a.get("stream_time"))]
    eps = _episodes([p for p in hub.engine.predictions if inwin(p["T"])])
    acks = [a for a in acts if a["kind"] == "ack"]
    react = [a["stream_time"] - a["detail"]["signal_at"] for a in acks
             if isinstance(a.get("detail", {}).get("signal_at"), (int, float))]
    acked_trs = {(a["tr_id"], a["detail"].get("signal_at")) for a in acks}
    users = {}
    for a in acts:
        u = users.setdefault(a.get("user") or "—", {"user": a.get("user") or "—", "ack": 0, "ack_cancel": 0, "wrong_forecast": 0,
                                                    "follow": 0, "react": [], "first": a["stream_time"], "last": a["stream_time"]})
        if a["kind"] in u:
            u[a["kind"]] += 1
        if a["kind"] == "ack" and isinstance(a.get("detail", {}).get("signal_at"), (int, float)):
            u["react"].append(a["stream_time"] - a["detail"]["signal_at"])
        u["first"], u["last"] = min(u["first"], a["stream_time"]), max(u["last"], a["stream_time"])
    med = lambda xs: round(float(np.median(xs)) / 60, 1) if xs else None  # noqa: E731
    errs = [x["err"] for sid, x in hub.evaluated.items()]
    return {
        "generated_stream_time": now, "window": {"from": lo, "to": now, "hours": hours},
        "summary": {
            "events": len(eps), "events_late": sum(1 for e in eps if e["late"]), "events_early": sum(1 for e in eps if not e["late"]),
            "events_high_risk": sum(1 for e in eps if e["red"]), "acked": len(acks), "acked_unique": len(acked_trs),
            "ack_share": round(len(acked_trs) / len(eps), 3) if eps else None,
            "reaction_median_min": med(react), "reaction_max_min": round(max(react) / 60, 1) if react else None,
            "ack_cancel": sum(1 for a in acts if a["kind"] == "ack_cancel"),
            "wrong_forecast": sum(1 for a in acts if a["kind"] == "wrong_forecast"),
            "online_mae_s": round(float(np.mean(errs)), 1) if errs else None,
        },
        "dispatchers": [{**{k: v for k, v in u.items() if k != "react"}, "reaction_median_min": med(u["react"])}
                        for u in sorted(users.values(), key=lambda x: -x["ack"])],
        "journal": [{"stream_time": a["stream_time"], "user": a.get("user") or "—", "tr_id": a["tr_id"],
                     "action": ACTION_LABEL.get(a["kind"], a["kind"]), "stop": (a.get("detail") or {}).get("target_name", ""),
                     "risk": (a.get("detail") or {}).get("risk", "")} for a in sorted(acts, key=lambda x: x["stream_time"])],
    }


def _hm(ts) -> str:
    return pd.Timestamp(ts, unit="s").strftime("%H:%M") if ts else "—"


@app.get("/api/report/dispatcher", summary="Отчёт о работе диспетчеров за период: CSV (Excel) или страница для печати/PDF")
async def api_report(hours: float | None = None, fmt: str = "csv"):
    r = dispatcher_report(hours)
    sm, w = r["summary"], r["window"]
    day = pd.Timestamp(r["generated_stream_time"] or 0, unit="s").strftime("%d.%m.%Y")
    period = f"{day}, {_hm(w['from']) if w['from'] else 'начало потока'}–{_hm(w['to'])} (время данных)"
    rows = [("Сигналов (событий)", sm["events"]), ("  из них опоздание", sm["events_late"]), ("  из них раньше плана", sm["events_early"]),
            ("  с высоким риском", sm["events_high_risk"]), ("Принято в работу", sm["acked"]),
            ("Доля событий, взятых в работу", "—" if sm["ack_share"] is None else f"{round(sm['ack_share'] * 100)}%"),
            ("Время реакции, медиана, мин", sm["reaction_median_min"] if sm["reaction_median_min"] is not None else "—"),
            ("Время реакции, максимум, мин", sm["reaction_max_min"] if sm["reaction_max_min"] is not None else "—"),
            ("Отмен принятия", sm["ack_cancel"]), ("Отметок «прогноз неверен»", sm["wrong_forecast"]),
            ("Средняя ошибка прогноза (онлайн), с", sm["online_mae_s"] if sm["online_mae_s"] is not None else "—")]
    ds = r["dispatchers"]
    if fmt == "html":
        esc = lambda x: str(x).replace("&", "&amp;").replace("<", "&lt;")  # noqa: E731
        tr = lambda cells, th=False: "<tr>" + "".join(f"<{'th' if th else 'td'}>{esc(c)}</{'th' if th else 'td'}>" for c in cells) + "</tr>"  # noqa: E731
        html = f"""<!doctype html><html lang="ru"><head><meta charset="utf-8"><title>Отчёт диспетчера — {day}</title>
<style>body{{font:14px -apple-system,Segoe UI,Roboto,sans-serif;color:#1D1D1F;max-width:900px;margin:32px auto;padding:0 16px}}
h1{{font-size:24px;margin:0 0 4px}}.sub{{color:#6E6E73;margin:0 0 20px}}h2{{font-size:16px;margin:24px 0 8px}}
table{{border-collapse:collapse;width:100%}}td,th{{text-align:left;padding:6px 8px;border-bottom:1px solid #E5E5EA}}th{{color:#6E6E73;font-weight:600}}
button{{font:inherit;padding:8px 14px;border-radius:10px;border:none;background:#0071E3;color:#fff;cursor:pointer}}@media print{{button{{display:none}}}}
.note{{color:#86868B;font-size:12px;margin-top:20px}}</style></head><body>
<button onclick="print()">Сохранить в PDF / печать</button>
<h1>Отчёт о работе диспетчеров</h1><p class="sub">Пульс маршрута · {esc(period)}</p>
<h2>Итоги</h2><table>{''.join(tr(x) for x in rows)}</table>
<h2>По диспетчерам</h2><table>{tr(['Диспетчер', 'Принято', 'Отмен', '«Неверен»', 'Слежение', 'Реакция, мин (медиана)', 'Первое действие', 'Последнее'], True)}
{''.join(tr([d['user'], d['ack'], d['ack_cancel'], d['wrong_forecast'], d['follow'], d['reaction_median_min'] if d['reaction_median_min'] is not None else '—', _hm(d['first']), _hm(d['last'])]) for d in ds) or tr(['Действий за период нет'])}</table>
<h2>Журнал действий</h2><table>{tr(['Время', 'Диспетчер', 'ТС', 'Действие', 'Остановка'], True)}
{''.join(tr([_hm(j['stream_time']), j['user'], j['tr_id'], j['action'], j['stop']]) for j in r['journal']) or tr(['Пусто'])}</table>
<p class="note">Событие — период, когда по машине подряд выдавались прогнозы опоздания или прибытия раньше плана. Время реакции — от появления
сигнала на экране до «Принять в работу», по времени данных. Принятие в работу не означает устранения.</p></body></html>"""
        return Response(html, media_type="text/html; charset=utf-8")
    import csv
    import io
    buf = io.StringIO()
    wcsv = csv.writer(buf, delimiter=";")
    wcsv.writerow(["Отчёт о работе диспетчеров", period])
    wcsv.writerow([])
    wcsv.writerow(["Показатель", "Значение"])
    wcsv.writerows([(k, str(v).replace(".", ",") if isinstance(v, float) else v) for k, v in rows])  # Excel в русской локали
    wcsv.writerow([])
    wcsv.writerow(["Диспетчер", "Принято в работу", "Отмен", "Отметок «неверен»", "Слежение", "Реакция, мин (медиана)", "Первое действие", "Последнее действие"])
    wcsv.writerows([[d["user"], d["ack"], d["ack_cancel"], d["wrong_forecast"], d["follow"],
                     str(d["reaction_median_min"]).replace(".", ",") if d["reaction_median_min"] is not None else "",
                     _hm(d["first"]), _hm(d["last"])] for d in ds])
    wcsv.writerow([])
    wcsv.writerow(["Время", "Диспетчер", "ТС", "Действие", "Остановка", "Тип события"])
    wcsv.writerows([[_hm(j["stream_time"]), j["user"], j["tr_id"], j["action"], j["stop"], j["risk"]] for j in r["journal"]])
    name = f"dispatcher_report_{day.replace('.', '-')}.csv"
    return Response("\ufeff" + buf.getvalue(), media_type="text/csv; charset=utf-8",
                    headers={"Content-Disposition": f'attachment; filename="{name}"'})


class ReplayReq(BaseModel):
    speed: float = Field(5, description="ускорение времени потока (1 — реальное время)")
    start: str | None = Field("2026-01-06 06:00", description="с какого момента проигрывать")
    hours: float | None = None
    split: str = "validate"


@app.post("/api/replay/start", summary="Проиграть архив как живой NDTP-поток (TCP на собственный порт)")
async def replay_start(req: ReplayReq):
    await replay_stop()
    hub.replay_task = asyncio.create_task(_replay_loop(req.speed, req.start, req.hours))
    await asyncio.sleep(0.05)
    return {"status": "started", "speed": req.speed, "start": req.start}


@app.post("/api/routes/rebuild", summary="Перечитать расписание и достроить пути по дорогам для новых вариантов рейсов")
async def routes_rebuild():
    hub.plan = plan_table(hub.split)
    hub.trips = trip_patterns(hub.plan)
    hub.load_patterns()
    todo = len(hub.missing_patterns())
    if todo and (getattr(hub, "routes_task", None) is None or hub.routes_task.done()):
        hub.routes_task = asyncio.create_task(_routes_builder())
    return {"status": "ok", "patterns_total": sum(len(d["patterns"]) for d in hub.trips.values()), "to_build": todo,
            "routes_version": hub.routes_version}


DATASETS = {
    "validate": ("Архив 06.01.2026", "Реальная телеметрия датасета хакатона: 13 ТС, у каждого свой маршрут"),
    "synthetic": ("Синтетика: 3 машины на маршруте", "Сгенерированная телеметрия по реальным дорогам и расписанию: "
                                                     "на каждом из 13 маршрутов 3 автобуса с интервалом 12 мин"),
}


def _datasets() -> list:
    meta = {}
    if (SYNTH_DIR / "meta.json").exists():
        meta = json.loads((SYNTH_DIR / "meta.json").read_text())
    return [{"id": k, "title": t, "description": d, "current": hub.split == k,
             "ready": k != "synthetic" or (SYNTH_DIR / "traffic.csv").exists(),
             **({"vehicles": meta.get("vehicles"), "points": meta.get("points")} if k == "synthetic" and meta else {})}
            for k, (t, d) in DATASETS.items()]


@app.get("/api/datasets", summary="Доступные датасеты для воспроизведения и текущий")
async def api_datasets():
    return {"current": hub.split, "datasets": _datasets()}


class DatasetReq(BaseModel):
    dataset: str = Field(..., description="validate — архив хакатона, synthetic — синтетика «несколько машин на маршруте»")


@app.post("/api/dataset", summary="Переключить датасет: поток перезапускается с 06:00 на текущей скорости")
async def api_dataset(req: DatasetReq):
    if req.dataset not in DATASETS:
        raise HTTPException(400, "неизвестный датасет")
    if req.dataset == "synthetic" and not (SYNTH_DIR / "traffic.csv").exists():
        from src import synth
        await asyncio.to_thread(synth.main)  # первый раз — сгенерировать (~1 мин)
    speed = (hub.replay_info or {}).get("speed", AUTO_REPLAY_SPEED)
    await replay_stop()
    await asyncio.to_thread(hub.load_dataset, req.dataset)
    DATASET_FILE.write_text(req.dataset)
    hub.reset()
    hub.replay_task = asyncio.create_task(_replay_loop(speed, "2026-01-06 06:00", None))
    todo = len(hub.missing_patterns())
    if todo and (getattr(hub, "routes_task", None) is None or hub.routes_task.done()):
        hub.routes_task = asyncio.create_task(_routes_builder())
    return {"status": "ok", "dataset": hub.split, "vehicles": len(hub.plan.tr_id.unique()), "routes_version": hub.routes_version}


class SpeedReq(BaseModel):
    speed: float = Field(..., ge=0.5, le=120, description="1 — реальное время")


@app.post("/api/replay/speed", summary="Изменить скорость воспроизведения архива на лету (без перезапуска)")
async def replay_speed(req: SpeedReq):
    if getattr(hub, "replay_args", None) is None:
        raise HTTPException(409, "воспроизведение не запущено")
    hub.replay_args.speed = req.speed
    hub.replay_info["speed"] = req.speed
    return {"status": "ok", "speed": req.speed}


@app.post("/api/replay/stop", summary="Остановить replay (поток оборвётся — дашборд покажет «нет связи»)")
async def replay_stop():
    if hub.replay_task and not hub.replay_task.done():
        hub.replay_task.cancel()
        try:
            await hub.replay_task
        except (asyncio.CancelledError, Exception):  # noqa: BLE001
            pass
    return {"status": "stopped"}


@app.get("/api/stats", summary="Технические метрики: поток, инференс, ML")
async def api_stats():
    return J({"ndtp": hub.server.stats(), "ml_client": hub.ml.stats(), "ml_model": hub.ml.info(),
              "routes": {"patterns_total": sum(len(d["patterns"]) for d in hub.trips.values()),
                         "patterns_on_roads": sum(len(d["patterns"]) for d in hub.trips.values()) - len(hub.missing_patterns())}})


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
