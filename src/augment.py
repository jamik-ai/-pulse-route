"""Аугментация обучающей выборки из фактов расписания реальных ТС.

README разрешает обучаться на train/test (факт в schedule.csv — источник target). Размеченных точек
мало (1494), а фактов — на каждой остановке. Генерируем точки (tr, T) на сетке 1 мин по тем же правилам,
что у организаторов:

* цель — первая остановка с планом в (T+10 мин, T+15 мин] (при равенстве — меньший stop_id);
* cur_dev_s — задержка на последней остановке с плановым временем ≤ T (0, если такой нет).

Честность по отношению к validate:

* выкидываем точки ближе ±VAL_GAP к любой точке validate того же ТС;
* выкидываем точки, чья цель — целевая остановка validate.

python -m src.augment  → output/features_aug.parquet
"""
import time

import numpy as np
import pandas as pd

from src.dataio import CACHE, load_points, load_schedule, load_traffic
from src.features import build_features

OUT = CACHE.parent
STEP = pd.Timedelta("1min")
VAL_GAP = pd.Timedelta("20min")


def generate_points(sched: pd.DataFrame, val: pd.DataFrame) -> pd.DataFrame:
    sched = sched.sort_values(["tr_id", "time_begin", "stop_id"])
    val_T = val.groupby("tr_id")["T"].apply(lambda s: s.values.astype("datetime64[s]").astype(np.int64))
    val_stops = set(val.target_stop_id)
    rows = []
    for tr, s in sched.groupby("tr_id"):
        pt = s.time_begin.values.astype("datetime64[s]").astype(np.int64)
        dev = (s.time_fact_begin - s.time_begin).dt.total_seconds().values
        ids = s.stop_id.values
        grid = np.arange((pt[0] // 60 - 16) * 60, pt[-1], 60)
        vt = val_T.get(tr, np.array([], dtype=np.int64))
        for T in grid:
            if vt.size and np.abs(vt - T).min() <= VAL_GAP.total_seconds():
                continue
            k = np.searchsorted(pt, T + 600, side="right")
            if k >= len(pt) or pt[k] > T + 900 or ids[k] in val_stops:
                continue
            kl = np.searchsorted(pt, T, side="right") - 1
            rows.append((tr, T, ids[k], pt[k], dev[kl] if kl >= 0 else 0.0, dev[k]))
    df = pd.DataFrame(rows, columns=["tr_id", "T", "target_stop_id", "target_time_begin", "cur_dev_s", "target_delay_s"])
    df["T"] = pd.to_datetime(df["T"], unit="s")
    df["target_time_begin"] = pd.to_datetime(df["target_time_begin"], unit="s")
    df["sample_id"] = df.tr_id.astype(str) + "_" + (df["T"].values.astype("datetime64[s]").astype(np.int64)).astype(str)
    df["is_synthetic"] = False
    return df


def check_rule_matches_labels(sched):
    """Санити-чек: наши правила воспроизводят разметку организаторов."""
    lab = pd.concat([load_points("train"), load_points("test")])
    lab = lab[~lab.is_synthetic]
    gen = generate_points(sched, lab.iloc[:0].assign(T=pd.to_datetime([])))
    m = lab.merge(gen, on=["tr_id", "T"], suffixes=("", "_g"))
    print(f"санити: размеченных {len(lab)}, найдено в сетке {len(m)}; цель совпала {np.mean(m.target_stop_id == m.target_stop_id_g):.3f}, "
          f"target совпал {np.mean(np.isclose(m.target_delay_s, m.target_delay_s_g)):.3f}, cur_dev совпал {np.mean(np.isclose(m.cur_dev_s, m.cur_dev_s_g)):.3f}")


if __name__ == "__main__":
    t0 = time.time()
    sched = load_schedule("test")  # расписание реальных ТС (в train те же 13 ТС + синтетика)
    check_rule_matches_labels(sched)
    val = load_points("validate")
    pts = generate_points(sched, val)
    plan = sched.drop(columns=["time_fact_begin"])
    X = build_features(pts, plan, load_traffic("test"))
    df = pts.join(X.drop(columns=["cur_dev_s"]))
    df.to_parquet(OUT / "features_aug.parquet")
    print(f"аугментация: {len(df)} точек, {df.tr_id.nunique()} ТС, за {time.time() - t0:.0f} c")
