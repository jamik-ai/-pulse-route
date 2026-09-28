"""Инференс сохранённого ансамбля — общий для офлайн-сабмита и real-time.

python -m src.predict [split]  → output/pred_{split}.csv (по умолчанию validate)
"""
import json
import sys

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor, Pool

from src.dataio import CACHE
from src.train import CAT_COLS

OUT = CACHE.parent
MODELS = OUT.parent / "models"


class Predictor:
    def __init__(self, models_dir=MODELS):
        meta = json.loads((models_dir / "meta.json").read_text())
        self.meta = meta
        self.features = meta["features"]
        self.w_delta = meta["w_delta"]
        self.clip = tuple(meta["clip"])
        self.models = {}
        for kind, names in meta["models"].items():
            self.models[kind] = []
            for n in names:
                m = CatBoostClassifier() if kind == "late" else CatBoostRegressor()
                m.load_model(str(models_dir / n))
                self.models[kind].append(m)

    def _prepare(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.reindex(columns=self.features)  # недостающие признаки → NaN (CatBoost умеет)
        for c in X.columns:
            X[c] = X[c].fillna("nan").astype(str) if c in CAT_COLS else X[c].astype(float)
        return X

    def predict_raw(self, X: pd.DataFrame) -> np.ndarray:
        """Прогноз ансамбля без правила manual_fill → 0 — то, что видит диспетчер."""
        X = self._prepare(X)
        a = np.mean([m.predict(X) for m in self.models["delta"]], axis=0) + X["cur_dev_s"].values
        b = np.mean([m.predict(X) for m in self.models["direct"]], axis=0)
        return self.w_delta * np.clip(a, *self.clip) + (1 - self.w_delta) * np.clip(b, *self.clip)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Задержка на целевой остановке, сек (то, что идёт в submission).

        Для целей с manual_fill факт в разметке почти всегда 0 (время внесено вручную), поэтому прогноз 0."""
        raw = self.predict_raw(X)
        return np.where(X["tgt_manual"].astype(float).values == 1, 0.0, raw)


    def predict_full(self, X: pd.DataFrame, explain: bool = True) -> list[dict]:
        """Для диспетчера: задержка, P(опоздание > порога), интервал ~80%, причины."""
        raw = X.to_dict("records")
        Xp = self._prepare(X)
        raw_pred = self.predict_raw(X)
        manual = X["tgt_manual"].astype(float).values == 1
        pred = np.where(manual, 0.0, raw_pred)
        # диспетчеру — тот же прогноз, что в сабмите: на целях с manual_fill официальный факт почти всегда 0
        # (проверка по labels_test: 95% нулей, опозданий > 2 мин нет); «сырой» выход модели — для справки
        out = [{"prediction_s": round(float(p), 1), "ops_prediction_s": round(float(p), 1), "model_prediction_s": round(float(r), 1),
                "manual_target": bool(m)}
               for p, r, m in zip(pred, raw_pred, manual)]
        if self.models.get("late"):
            p_late = np.mean([m.predict_proba(Xp)[:, 1] for m in self.models["late"]], axis=0)
            q = np.mean([m.predict(Xp) for m in self.models["quantile"]], axis=0)
            mg = self.meta.get("interval_margin_s", 0.0)
            for o, pl, qq, man in zip(out, p_late, q, Xp["tgt_manual"].values):
                lo, hi = qq[0] - mg, qq[-1] + mg
                if man == 1:  # ручное заполнение: в разметке задержка практически всегда 0
                    pl, lo, hi = 0.0, min(lo, 0.0), max(hi, 0.0)
                o.update(p_late=round(float(pl), 3), interval_s=[round(float(lo)), round(float(hi))],
                         expected_error_s=round(float((hi - lo) / 2 * 0.6)))
        if explain:
            from src.explain import top_causes
            m = self.models["direct"][0]  # объясняем модель прямой задержки: вклады = «почему столько секунд»
            cat_idx = [i for i, c in enumerate(self.features) if c in CAT_COLS]
            shap = m.get_feature_importance(Pool(Xp, cat_features=cat_idx), type="ShapValues")[:, :-1]
            for o, s, r in zip(out, shap, raw):
                o["causes"] = top_causes(s, self.features, r, direction=np.sign(o["ops_prediction_s"]) or 1)
        return out


if __name__ == "__main__":
    split = sys.argv[1] if len(sys.argv) > 1 else "validate"
    df = pd.read_parquet(OUT / f"features_{split}.parquet")
    p = Predictor().predict(df)
    pd.DataFrame({"sample_id": df.sample_id, "prediction": np.round(p, 1)}).to_csv(OUT / f"pred_{split}.csv", index=False)
    if "target_delay_s" in df:
        print(f"{split}: MAE {np.abs(df.target_delay_s - p).mean():.1f}")
    print(f"→ output/pred_{split}.csv")
