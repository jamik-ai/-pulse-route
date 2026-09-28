"""output/pred_validate.csv → output/submission.csv (формат sample_submission) + проверка формата."""
import numpy as np
import pandas as pd

from src.dataio import CACHE, DATA

OUT = CACHE.parent


def check(sub: pd.DataFrame, template: pd.DataFrame):
    assert list(sub.columns) == ["sample_id", "prediction"], sub.columns
    assert len(sub) == len(template), (len(sub), len(template))
    assert sub.sample_id.is_unique, "дубли sample_id"
    assert set(sub.sample_id) == set(template.sample_id), "набор sample_id не совпадает"
    assert np.isfinite(sub.prediction).all(), "NaN/inf в prediction"


if __name__ == "__main__":
    tpl = pd.read_csv(DATA / "sample_submission.csv", sep=";")
    pred = pd.read_csv(OUT / "pred_validate.csv")
    sub = tpl[["sample_id"]].merge(pred, on="sample_id", how="left")
    check(sub, tpl)
    sub.to_csv(OUT / "submission.csv", sep=";", index=False)
    print(f"submission.csv: {len(sub)} строк; prediction: median {sub.prediction.median():.0f}, "
          f"min {sub.prediction.min():.0f}, max {sub.prediction.max():.0f}; "
          f"baseline(cur_dev_s) median {tpl.prediction.median():.0f}")
