"""Простые правила-прогнозы: MAE на реальных точках test и train."""
import numpy as np
import pandas as pd

from src.dataio import CACHE


def rules(d: pd.DataFrame) -> dict:
    gps = d.gps_dev_last.fillna(d.cur_dev_s)
    man = d.tgt_manual == 1
    return {
        "zero": np.zeros(len(d)),
        "cur_dev_s": d.cur_dev_s,
        "gps_dev_last": gps,
        "gps_dev_med3": d.gps_dev_med3.fillna(d.cur_dev_s),
        "lower_bound_tgt": d.dev_lower_bound_tgt.fillna(d.cur_dev_s),
        "eta_gap": d.eta_gap_s.fillna(gps).clip(-400, 700),
        "cur_dev, manual→0": np.where(man, 0, d.cur_dev_s),
        "gps_last, manual→0": np.where(man, 0, gps),
        "mean(cur,gps), manual→0": np.where(man, 0, (gps + d.cur_dev_s) / 2),
    }


if __name__ == "__main__":
    out = CACHE.parent
    for split in ["train", "test"]:
        d = pd.read_parquet(out / f"features_{split}.parquet")
        d = d[~d.is_synthetic]
        y = d.target_delay_s
        print(f"== {split} (реальные ТС, n={len(d)})")
        for name, p in rules(d).items():
            print(f"  {name:28s} MAE={np.abs(y - p).mean():6.1f}")
