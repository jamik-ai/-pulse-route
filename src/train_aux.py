"""Дополнительные модели для диспетчера (не влияют на submission):
  * классификатор P(опоздание > 120 с) — порог «late» из разметки;
  * квантильная модель q10/q50/q90 → интервал прогноза и ожидаемая ошибка.

python -m src.train_aux          # CV-метрики на реальных train+test
python -m src.train_aux --final  # обучить на всём и сохранить в models/
"""
import argparse
import json

import numpy as np
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold

from src.evaluate import real_pool
from src.train import MODELS, feature_cols

LATE_S = 120
QUANTILES = (0.1, 0.5, 0.9)


def make_clf(seed=0):
    return CatBoostClassifier(loss_function="Logloss", iterations=800, learning_rate=0.03, depth=5,
                              l2_leaf_reg=10, random_seed=seed, verbose=False, thread_count=2)


def make_q(seed=0):
    return CatBoostRegressor(loss_function=f"MultiQuantile:alpha={','.join(map(str, QUANTILES))}",
                             iterations=1000, learning_rate=0.03, depth=5, l2_leaf_reg=10,
                             random_seed=seed, verbose=False, thread_count=2)


def conformal_margin(y, q, coverage=0.8) -> float:
    """CQR: на сколько расширить [q10, q90], чтобы накрыть `coverage` out-of-fold наблюдений."""
    score = np.maximum(q[:, 0] - y, y - q[:, 2])
    return float(np.quantile(score, coverage))


def cv():
    d = real_pool()
    cols = feature_cols(d)
    y = d.target_delay_s.values
    late = (y > LATE_S).astype(int)
    prob = np.zeros(len(d))
    q = np.zeros((len(d), len(QUANTILES)))
    for tr_i, va_i in GroupKFold(6).split(d, groups=d["T"].dt.hour // 2):
        c = make_clf().fit(d.iloc[tr_i][cols], late[tr_i])
        prob[va_i] = c.predict_proba(d.iloc[va_i][cols])[:, 1]
        m = make_q().fit(d.iloc[tr_i][cols], y[tr_i])
        q[va_i] = m.predict(d.iloc[va_i][cols])
    base_rate = late.mean()
    print(f"P(late>{LATE_S}s): доля late {base_rate:.2f}; AUC {roc_auc_score(late, prob):.3f}; "
          f"Brier {brier_score_loss(late, prob):.3f} (константа {brier_score_loss(late, np.full(len(d), base_rate)):.3f})")
    for lo, hi in ((0.0, 0.3), (0.3, 0.6), (0.6, 1.01)):
        m = (prob >= lo) & (prob < hi)
        print(f"  P в [{lo:.1f},{hi:.1f}): n={m.sum():4d}, фактически late {late[m].mean():.2f}")
    inside = (y >= q[:, 0]) & (y <= q[:, 2])
    print(f"интервал q10–q90: покрытие {inside.mean():.2f} (цель 0.80), средняя ширина {np.mean(q[:, 2] - q[:, 0]):.0f} с; "
          f"MAE медианы {np.abs(y - q[:, 1]).mean():.1f}")
    # конформная калибровка: margin подбираем на одной половине блоков, проверяем на другой
    blocks = (d["T"].dt.hour // 2).values
    a = blocks % 2 == 0
    for fit_m, chk in ((a, ~a), (~a, a)):
        mg = conformal_margin(y[fit_m], q[fit_m])
        cov = ((y[chk] >= q[chk, 0] - mg) & (y[chk] <= q[chk, 2] + mg)).mean()
        print(f"  CQR: margin {mg:.0f} с → покрытие на отложенной половине {cov:.2f}")
    margin = conformal_margin(y, q)
    print(f"  CQR margin по всем OOF: {margin:.0f} с")
    # ожидаемая ошибка ≈ половина ширины интервала — согласуется ли с фактической?
    half = (q[:, 2] - q[:, 0]) / 2
    for lo, hi in ((0, 40), (40, 80), (80, 1e9)):
        m = (half >= lo) & (half < hi)
        print(f"  полуширина {lo}-{hi if hi < 1e9 else '∞'} с: n={m.sum():4d}, фактическая MAE медианы {np.abs(y - q[:, 1])[m].mean():.0f} с")
    return margin


def final(seeds=(0, 1, 2)):
    margin = cv()  # margin конформной калибровки берём из out-of-fold
    d = real_pool()
    cols = feature_cols(d)
    y = d.target_delay_s.values
    late = (y > LATE_S).astype(int)
    meta_path = MODELS / "meta.json"
    meta = json.loads(meta_path.read_text())
    assert meta["features"] == cols, "признаки основного ансамбля и доп. моделей разошлись"
    meta["models"]["late"], meta["models"]["quantile"] = [], []
    for s in seeds:
        make_clf(s).fit(d[cols], late).save_model(str(MODELS / f"late_{s}.cbm"))
        make_q(s).fit(d[cols], y).save_model(str(MODELS / f"quantile_{s}.cbm"))
        meta["models"]["late"].append(f"late_{s}.cbm")
        meta["models"]["quantile"].append(f"quantile_{s}.cbm")
    meta["late_threshold_s"] = LATE_S
    meta["quantiles"] = list(QUANTILES)
    meta["interval_margin_s"] = margin
    meta["interval_coverage"] = 0.8
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    print("сохранено:", meta["models"]["late"], meta["models"]["quantile"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", action="store_true")
    final() if ap.parse_args().final else cv()
