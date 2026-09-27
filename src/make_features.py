"""Считает признаки для train/test/validate и кладёт в output/features_{split}.parquet."""
import sys
import time

from src.dataio import CACHE, load_points, load_schedule, load_traffic
from src.features import build_features

OUT = CACHE.parent


def main(splits):
    for split in splits:
        t0 = time.time()
        pts = load_points(split)
        plan = load_schedule(split).drop(columns=["time_fact_begin"])  # факт в признаки не идёт
        X = build_features(pts, plan, load_traffic(split))
        df = pts.join(X.drop(columns=["cur_dev_s"]))
        df.to_parquet(OUT / f"features_{split}.parquet")
        print(f"{split}: {df.shape} за {time.time() - t0:.0f} c; нет GPS-проезда у {df['gps_dev_last'].isna().mean():.1%}")


if __name__ == "__main__":
    main(sys.argv[1:] or ["train", "test", "validate"])
