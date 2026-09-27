"""Сверка онлайн-прогнозов (jsonl сервера) с офлайн-сабмитом и, если есть, с разметкой.

Запуск::

    python -m src.realtime.check [output/realtime_predictions.jsonl] [--split validate]
"""
import argparse

import numpy as np
import pandas as pd

from src.dataio import CACHE, DATA

OUT = CACHE.parent

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default=str(OUT / "realtime_predictions.jsonl"))
    ap.add_argument("--split", default="validate")
    a = ap.parse_args()
    on = pd.read_json(a.path, lines=True, dtype={"sample_id": str}).drop_duplicates("sample_id", keep="last")
    print(f"онлайн-прогнозов: {len(on)}, источники подсказки: {on.hint_source.value_counts().to_dict()}")
    off = pd.read_csv(OUT / f"pred_{a.split}.csv")
    pts = pd.read_csv(DATA / ("validate/points.csv" if a.split == "validate" else f"labels/labels_{a.split}.csv"))
    m = pts.merge(off, on="sample_id").merge(on[["sample_id", "target_stop_id", "prediction"]],
                                               on="sample_id", how="left", suffixes=("_off", "_on"))
    got = m.prediction_on.notna()
    same = got & (m.target_stop_id_off == m.target_stop_id_on)
    d = (m.prediction_off - m.prediction_on).abs()[same]
    print(f"точек {a.split}: {len(m)}; онлайн выдал прогноз: {got.sum()}; та же целевая остановка: {same.sum()}")
    print(f"|онлайн − офлайн|: median {d.median():.2f} c, p95 {d.quantile(.95):.2f} c, max {d.max():.2f} c")
    if "target_delay_s" in m:
        y = m.target_delay_s[same]
        print(f"MAE онлайн {np.abs(y - m.prediction_on[same]).mean():.1f}  офлайн {np.abs(y - m.prediction_off[same]).mean():.1f}")
