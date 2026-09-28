"""Надёжная оценка: все реальные размеченные точки (train+test), GroupKFold по 2-часовым блокам T,
повтор на нескольких сидах. Используется для сравнения вариантов признаков/моделей."""
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from src.train import feature_cols, fit_predict, load, mae


def real_pool() -> pd.DataFrame:
    tr, te = load("train"), load("test")
    tr = tr[~tr.is_synthetic]
    return pd.concat([tr.assign(src="train"), te.assign(src="test")], ignore_index=True)


def oof_predict(df, cols, n_splits=6, **kw) -> np.ndarray:
    groups = df["T"].dt.hour // 2
    oof = np.full(len(df), np.nan)
    for tr_i, va_i in GroupKFold(n_splits).split(df, groups=groups):
        oof[va_i], _ = fit_predict(df.iloc[tr_i], df.iloc[va_i], cols, syn_w=0, **kw)
    return oof


def report(name, df, oof):
    y = df.target_delay_s.values
    z = df.cur_dev_s.values == 0
    t = (df.src == "test").values
    print(f"{name:40s} CV={mae(y, oof):5.1f}  [test-часть {mae(y[t], oof[t]):5.1f}]  "
          f"cur_dev=0: {mae(y[z], oof[z]):5.1f}  иначе: {mae(y[~z], oof[~z]):5.1f}", flush=True)
    return mae(y, oof)


def oof_ensemble(df, cols, seeds=(0,), n_splits=6, **kw) -> np.ndarray:
    """OOF полного итогового пайплайна: 0.5·A(delta) + 0.5·B(direct, без manual) + правило manual→0."""
    groups = df["T"].dt.hour // 2
    A = np.zeros(len(df)); B = np.zeros(len(df))
    for tr_i, va_i in GroupKFold(n_splits).split(df, groups=groups):
        tr, va = df.iloc[tr_i], df.iloc[va_i]
        A[va_i], _ = fit_predict(tr, va, cols, target="delta", syn_w=0, seeds=seeds, **kw)
        B[va_i], _ = fit_predict(tr[tr.tgt_manual == 0], va, cols, target="direct", syn_w=0, seeds=seeds, **kw)
    return np.where(df.tgt_manual.values == 1, 0.0, 0.5 * A + 0.5 * B)


if __name__ == "__main__":
    df = real_pool()
    y = df.target_delay_s.values
    z = df.cur_dev_s.values == 0
    print(f"пул: {len(df)} точек; baseline cur_dev_s CV={mae(y, df.cur_dev_s):.1f} "
          f"(cur_dev=0: {mae(y[z], 0):.1f}, иначе {mae(y[~z], df.cur_dev_s[~z]):.1f})")
    cols = feature_cols(df)
    report("текущая модель", df, oof_predict(df, cols))

