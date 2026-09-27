"""Онлайн-инференс: буфер телеметрии по ТС → прогноз каждые 5 минут времени потока.

Момент прогноза T — граница 5 минут по времени телеметрии (как в разметке). Прогноз по T
выдаётся, когда «водяной знак» потока (макс. время − LAG) перешёл T: к этому моменту все точки
≤ T от всех ТС успели прийти. Точки ≤ T, пришедшие позже, считаются опоздавшими.
"""
import bisect
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.features import PLAN_COLS, TrContext, features_at

STEP_S = 300
HORIZON = (600, 900)   # цель: первая остановка с планом в (T+10 мин, T+15 мин]
WATERMARK_LAG_S = 30


@dataclass
class Vehicle:
    tr_id: int
    plan: pd.DataFrame
    et: list = field(default_factory=list)
    rows: list = field(default_factory=list)  # (lon, lat, speed)
    _ctx: TrContext = None
    _ctx_n: int = -1

    def add(self, ts: int, lon: float, lat: float, speed: float) -> bool:
        i = bisect.bisect_left(self.et, ts)
        if i < len(self.et) and self.et[i] == ts:
            return False  # дубль по времени — как drop_duplicates офлайн
        self.et.insert(i, ts)
        self.rows.insert(i, (lon, lat, speed))
        return True

    def context(self) -> TrContext:
        if self._ctx_n != len(self.et):
            n = min(len(self.et), len(self.rows))
            r = np.array(self.rows[:n], dtype=float).reshape(-1, 3)
            traffic = pd.DataFrame({"event_time": pd.to_datetime(np.array(self.et[:n], dtype="int64"), unit="s"),
                                    "lon": r[:, 0], "lat": r[:, 1], "speed": r[:, 2]})
            self._ctx, self._ctx_n = TrContext.build(self.plan, traffic), len(self.et)
            self._ctx.tr_key = str(self.tr_id)
        return self._ctx


class HintProvider:
    """Внешний поток запросов/подсказок для (tr_id, T): cur_dev_s и, если задана, целевая остановка.

    В проде cur_dev_s приходит из АСУ диспетчеризации. При проверке организаторами — из points.csv:
    там же заданы target_stop_id и sample_id, и потоковый контур отвечает ровно на эти запросы.
    """

    def __init__(self, table: pd.DataFrame | None = None):
        self.table, self.targets, self.sample_ids = {}, {}, {}
        if table is not None:
            ts = table["T"].values.astype("datetime64[s]").astype(np.int64)
            keys = list(zip(table.tr_id, ts))
            self.table = dict(zip(keys, table.cur_dev_s.astype(float)))
            if "target_stop_id" in table:
                self.targets = dict(zip(keys, table.target_stop_id.astype(np.int64)))
            if "sample_id" in table:
                self.sample_ids = dict(zip(keys, table.sample_id.astype(str)))

    def target(self, tr_id: int, T: int):
        return self.targets.get((tr_id, T))

    def sample_id(self, tr_id: int, T: int) -> str:
        return self.sample_ids.get((tr_id, T), f"{tr_id}_{T}")

    def get(self, tr_id: int, T: int, ctx: TrContext) -> tuple[float, str]:
        if (tr_id, T) in self.table:
            return self.table[(tr_id, T)], "feed"
        # fallback: оценка по GPS — медиана задержек последних проездов
        f = features_at(ctx, T, -1, T + 720, 0.0)
        v = f.get("gps_dev_med3")
        return (float(v), "gps") if v is not None and np.isfinite(v) else (0.0, "none")


class OnlineEngine:
    def __init__(self, plan: pd.DataFrame, unit_to_tr: dict, predictor, hints: HintProvider,
                 on_prediction=None, watermark_lag_s: int = WATERMARK_LAG_S):
        self.vehicles = {tr: Vehicle(tr, p[PLAN_COLS].reset_index(drop=True)) for tr, p in plan.groupby("tr_id")}
        self.unit_to_tr = unit_to_tr
        self.predictor = predictor
        self.hints = hints
        self.on_prediction = on_prediction or (lambda rec: None)
        self.lag = watermark_lag_s
        self.max_ts = None
        self.next_T = None
        self.predictions: list[dict] = []
        self.latest: dict[int, dict] = {}           # tr_id → последний прогноз (с деталями)
        self.latest_features: dict[int, dict] = {}  # tr_id → признаки последнего прогноза (для what-if)
        self.stats = dict(points=0, invalid=0, unknown_unit=0, duplicates=0, late=0, predictions=0,
                          infer_ms_total=0.0)

    def on_point(self, unit_id: int, ts: int, lon: float, lat: float, valid: bool, speed: float):
        self.stats["points"] += 1
        if not valid:
            self.stats["invalid"] += 1
            return
        tr = self.unit_to_tr.get(unit_id)
        v = self.vehicles.get(tr)
        if v is None:
            self.stats["unknown_unit"] += 1
            return
        if self.next_T is not None and ts <= self.next_T - STEP_S:
            self.stats["late"] += 1  # прогноз по этому T уже выдан
        if not v.add(ts, lon, lat, speed):
            self.stats["duplicates"] += 1
        self.max_ts = ts if self.max_ts is None else max(self.max_ts, ts)
        if self.next_T is None:
            self.next_T = (ts // STEP_S + 1) * STEP_S
        self._flush(self.max_ts - self.lag)

    def _flush(self, watermark: int):
        while self.next_T is not None and self.next_T <= watermark:
            self.predict_at(self.next_T)
            self.next_T += STEP_S

    def finish(self):
        """Конец потока: выдать прогнозы по оставшимся T."""
        if self.max_ts is not None:
            self._flush(self.max_ts)

    def predict_at(self, T: int) -> list[dict]:
        t0 = time.perf_counter()
        rows, meta = [], []
        for tr, v in self.vehicles.items():
            if not v.et:
                continue
            ctx = v.context()
            req = self.hints.target(tr, T)
            hit = np.flatnonzero(ctx.plan_id == req) if req is not None else np.array([], int)
            if hit.size:  # запрошенная цель (points.csv) — отвечаем ровно на неё
                k = int(hit[0])
            else:  # своя цель: первая остановка с планом в (T+10, T+15]
                k = int(np.searchsorted(ctx.plan_t, T + HORIZON[0], side="right"))
                if k >= len(ctx.plan_t) or ctx.plan_t[k] > T + HORIZON[1]:
                    continue
            cur, src = self.hints.get(tr, T, ctx)
            rows.append(features_at(ctx, T, ctx.plan_id[k], int(ctx.plan_t[k]), cur))
            meta.append((tr, int(ctx.plan_id[k]), int(ctx.plan_t[k]), cur, src))
        out = []
        if rows:
            X = pd.DataFrame(rows)
            if hasattr(self.predictor, "predict_full"):
                full = self.predictor.predict_full(X)
            else:
                full = [{"prediction_s": float(p)} for p in self.predictor.predict(X)]
            for (tr, stop, tt, cur, src), res, row in zip(meta, full, rows):
                rec = dict(sample_id=self.hints.sample_id(tr, T), tr_id=tr, T=T, target_stop_id=stop,
                           target_time_begin=tt, cur_dev_s=cur, hint_source=src,
                           prediction=round(float(res["prediction_s"]), 1),
                           **{k: v for k, v in res.items() if k != "prediction_s"})
                out.append(rec)
                self.latest[tr] = rec
                self.latest_features[tr] = row
        dt = (time.perf_counter() - t0) * 1000
        self.stats["predictions"] += len(out)
        self.stats["infer_ms_total"] += dt
        for rec in out:
            rec["infer_ms_batch"] = round(dt, 1)
            self.on_prediction(rec)
        self.predictions.extend(out)
        return out
