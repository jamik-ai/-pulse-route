"""Обучение: эксперименты (--exp) и финальная модель (--final).

Валидация:
  * holdout = labels_test (только реальные ТС) — основная цифра;
  * CV на train: GroupKFold по часовым блокам T, MAE считается только по реальным ТС.
"""
import argparse
import json

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import GroupKFold

from src.dataio import CACHE

OUT = CACHE.parent
MODELS = OUT.parent / "models"
NON_FEATURES = {"sample_id", "tr_id", "T", "target_stop_id", "target_time_begin",
                "target_delay_s", "target_class", "is_synthetic", "src",
                "hour"}  # hour: те же ТС в тот же день во всех наборах — риск запоминания
CLIP = (-400.0, 700.0)
CAT_COLS = ("tgt_stop_key", "tr_cat")  # категориальные признаки (строки)


def load(split):
    return pd.read_parquet(OUT / f"features_{split}.parquet")


def feature_cols(df):
    return [c for c in df.columns if c not in NON_FEATURES]


def make_model(seed=0, **kw):
    params = dict(loss_function="MAE", iterations=1500, learning_rate=0.03, depth=5,
                  l2_leaf_reg=10, random_seed=seed, verbose=False, thread_count=2)
    params.update(kw)
    return CatBoostRegressor(**params)


def fit_predict(tr, te, cols, target="delta", syn_w=1.0, seeds=(0,), **kw):
    kw.setdefault("cat_features", [c for c in cols if c in CAT_COLS] or None)
    y = tr.target_delay_s - (tr.cur_dev_s if target == "delta" else 0)
    w = np.where(tr.is_synthetic, syn_w, 1.0)
    keep = w > 0
    preds, models = [], []
    for s in seeds:
        m = make_model(seed=s, **kw)
        m.fit(tr.loc[keep, cols], y[keep], sample_weight=w[keep])
        preds.append(m.predict(te[cols]))
        models.append(m)
    p = np.mean(preds, axis=0) + (te.cur_dev_s.values if target == "delta" else 0)
    return np.clip(p, *CLIP), models


def mae(y, p):
    return float(np.abs(np.asarray(y) - np.asarray(p)).mean())


def cv_mae(df, cols, n_splits=5, **kw):
    groups = df["T"].dt.hour // 3  # блоки по 3 часа
    oof = np.full(len(df), np.nan)
    for tr_i, va_i in GroupKFold(n_splits).split(df, groups=groups):
        oof[va_i], _ = fit_predict(df.iloc[tr_i], df.iloc[va_i], cols, **kw)
    real = ~df.is_synthetic.values
    return mae(df.target_delay_s.values[real], oof[real])


def experiments():
    tr, te = load("train"), load("test")
    cols = feature_cols(tr)
    print(f"признаков: {len(cols)}; train {len(tr)} (реальных {(~tr.is_synthetic).sum()}), test {len(te)}")
    print(f"baseline cur_dev_s: test {mae(te.target_delay_s, te.cur_dev_s):.1f}")
    rows = []
    for target in ["delta", "direct"]:
        for syn_w in [0.0, 0.3, 1.0]:
            p, _ = fit_predict(tr, te, cols, target=target, syn_w=syn_w)
            cv = cv_mae(tr, cols, target=target, syn_w=syn_w)
            rows.append((target, syn_w, cv, mae(te.target_delay_s, p)))
            print(f"  target={target:6s} syn_w={syn_w:.1f}  CV(real)={cv:6.1f}  test={rows[-1][-1]:6.1f}", flush=True)
    return rows


def predict_ensemble(train_df, pred_df, cols, seeds=(0, 1, 2, 3, 4), w_delta=0.5):
    """Итоговая схема (CV 64.5 на реальных train+test):
    A — поправка к cur_dev_s, обучение на всех целях;
    B — прямая задержка, обучение только на целях без manual_fill;
    прогноз = w·A + (1−w)·B, а для целей с manual_fill — 0 (там задержка почти всегда 0)."""
    a, ma = fit_predict(train_df, pred_df, cols, target="delta", syn_w=0, seeds=seeds)
    b, mb = fit_predict(train_df[train_df.tgt_manual == 0], pred_df, cols, target="direct", syn_w=0, seeds=seeds)
    p = w_delta * a + (1 - w_delta) * b
    return np.where(pred_df.tgt_manual.values == 1, 0.0, p), (ma, mb)


def final(seeds=(0, 1, 2, 3, 4)):
    tr, te, va = load("train"), load("test"), load("validate")
    tr = tr[~tr.is_synthetic]  # синтетика = копии ТС test/validate → утечка
    cols = feature_cols(tr)
    p_te, (ma, _) = predict_ensemble(tr, te, cols, seeds)
    print(f"test MAE (обучение только на train): {mae(te.target_delay_s, p_te):.1f}  "
          f"[baseline cur_dev_s {mae(te.target_delay_s, te.cur_dev_s):.1f}]")
    imp = pd.Series(ma[0].get_feature_importance(), cols).sort_values(ascending=False)
    print("топ-признаки (модель A):\n" + imp.head(12).round(1).to_string())
    full = pd.concat([tr, te], ignore_index=True)
    p_va, (ma, mb) = predict_ensemble(full, va, cols, seeds)
    MODELS.mkdir(exist_ok=True)
    for f in MODELS.glob("*.cbm"):
        f.unlink()
    files = {"delta": [], "direct": []}
    for kind, ms in (("delta", ma), ("direct", mb)):
        for i, m in enumerate(ms):
            name = f"{kind}_{i}.cbm"
            m.save_model(str(MODELS / name))
            files[kind].append(name)
    (MODELS / "meta.json").write_text(json.dumps({
        "features": cols, "w_delta": 0.5, "manual_rule": "tgt_manual==1 -> 0", "clip": CLIP,
        "models": files}, ensure_ascii=False, indent=1))
    pd.DataFrame({"sample_id": va.sample_id, "prediction": np.round(p_va, 1)}).to_csv(
        OUT / "pred_validate.csv", index=False)
    print("validate прогноз сохранён")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--target", default="delta")
    # синтетика = копии ТС из test/validate с их задержками → утечка, по умолчанию не используем
    ap.add_argument("--syn-w", type=float, default=0.0)
    a = ap.parse_args()
    if a.final:
        final()
    else:
        experiments()
