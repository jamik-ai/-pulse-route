"""Сабмит из потокового режима — для перепроверки результата организаторами.

Телеметрия набора кодируется в NDTP-кадры и проходит тот же путь, что на сервере:
FrameReader (разбор, CRC) → OnlineEngine (буфер ТС, признаки строго по данным ≤ T) → Predictor.
Запросы (sample_id, T, target_stop_id, cur_dev_s) берутся из points.csv как внешний поток.

Запуск::

    python -m src.realtime.stream_submission            # validate → output/submission_stream.csv
    python -m src.realtime.stream_submission --from-jsonl output/realtime_predictions.jsonl
    # то же из журнала работающего сервера (после docker compose run replay)
"""
import argparse
import time

import numpy as np
import pandas as pd

from src.dataio import CACHE, DATA, _ts, load_points, load_schedule
from src.make_submission import check
from src.predict import Predictor
from src.realtime import ndtp
from src.realtime.online import HintProvider, OnlineEngine

OUT = CACHE.parent


def replay_through_engine(split: str, hints: pd.DataFrame, predictor=None) -> OnlineEngine:
    """Проигрывает traffic.csv набора как NDTP-поток в OnlineEngine (в порядке времени событий)."""
    raw = pd.read_csv(DATA / split / "traffic.csv",
                      usecols=["tr_id", "unit_id", "event_time", "location_valid", "lon", "lat", "speed", "heading"])
    raw["ts"] = _ts(raw["event_time"]).values.astype("datetime64[s]").astype(np.int64)
    raw = raw.sort_values("ts", kind="stable")
    plan = load_schedule(split).drop(columns=["time_fact_begin"])  # в поток факт не попадает
    unit_to_tr = dict(zip(raw.unit_id.astype(int), raw.tr_id.astype(int)))
    eng = OnlineEngine(plan, unit_to_tr, predictor or Predictor(), HintProvider(hints), watermark_lag_s=0)
    reader = ndtp.FrameReader()
    req = {}
    for r in raw.itertuples(index=False):
        valid = str(r.location_valid) == "True" and not np.isnan(r.lon)
        cell = ndtp.build_nav_cell(r.ts, r.lon if valid else 0, r.lat if valid else 0, valid,
                                   0 if np.isnan(r.speed) else r.speed, 0 if np.isnan(r.heading) else r.heading)
        u = int(r.unit_id)
        req[u] = req.get(u, 1) + 1
        for f in reader.feed(ndtp.build_realtime(u, req[u], cell)):
            for p in ndtp.frame_nav_points(f):
                eng.on_point(p.unit_id, p.timestamp, p.lon, p.lat, p.valid, p.speed)
    eng.finish()
    assert reader.crc_errors == 0
    return eng


def to_submission(online: pd.DataFrame, points: pd.DataFrame) -> pd.DataFrame:
    online = online.drop_duplicates("sample_id", keep="last")
    sub = points[["sample_id"]].merge(online[["sample_id", "prediction"]], on="sample_id", how="left")
    missing = sub.prediction.isna().sum()
    if missing:
        print(f"⚠️ нет потокового прогноза для {missing} точек — ставим cur_dev_s")
        sub["prediction"] = sub.prediction.fillna(points.set_index("sample_id").loc[sub.sample_id, "cur_dev_s"].values)
    return sub


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="validate")
    ap.add_argument("--from-jsonl", default=None)
    a = ap.parse_args()
    points = pd.read_csv(DATA / "validate" / "points.csv") if a.split == "validate" else \
        pd.read_csv(DATA / "labels" / f"labels_{a.split}.csv")
    t0 = time.time()
    if a.from_jsonl:
        online = pd.read_json(a.from_jsonl, lines=True, dtype={"sample_id": str})
    else:
        eng = replay_through_engine(a.split, load_points(a.split))
        online = pd.DataFrame(eng.predictions)
        s = eng.stats
        print(f"поток: {s['points']} NDTP-точек, прогнозов {s['predictions']}, "
              f"инференс {s['infer_ms_total'] / max(s['predictions'], 1):.1f} мс/прогноз, {time.time() - t0:.0f} c")
    sub = to_submission(online, points)
    tpl = pd.read_csv(DATA / "sample_submission.csv", sep=";") if a.split == "validate" else points[["sample_id"]]
    if a.split == "validate":
        check(sub, tpl)
    sub.to_csv(OUT / f"submission_stream_{a.split}.csv", sep=";", index=False)
    ref = OUT / ("submission.csv" if a.split == "validate" else f"pred_{a.split}.csv")
    if ref.exists():
        off = pd.read_csv(ref, sep=";" if a.split == "validate" else ",")
        d = (sub.merge(off, on="sample_id", suffixes=("_stream", "_off")).eval("prediction_stream - prediction_off").abs())
        print(f"поток vs офлайн: совпало (|Δ|<0.1 с) {np.mean(d < 0.1) * 100:.1f}% точек, max |Δ| {d.max():.2f} с")
    if "target_delay_s" in points:
        print(f"MAE потокового режима: {np.abs(points.target_delay_s.values - sub.prediction.values).mean():.1f}")
    print(f"→ output/submission_stream_{a.split}.csv")
