"""Проверка онлайн-прогнозов по всем ТС за день: поток NDTP → OnlineEngine (как на сервере) → сверка с фактом.

Телеметрия validate совпадает с test, поэтому факты берём из labels_test: те же ТС, те же моменты T и целевые
остановки. Подсказки АСУ — из validate/points.csv (как у работающего сервиса); в остальные моменты движок сам
оценивает отклонение по GPS.

Запуск::

    ./run.sh python -m src.realtime.check_all            # онлайн, ~5–10 мин
    ./run.sh python -m src.realtime.check_all --offline  # то же на output/features_test.parquet
"""
import numpy as np
import pandas as pd

from src.dataio import DATA, _ts
from src.realtime.stream_submission import replay_through_engine

RED_P, YELLOW_P, EARLY_S = 0.7, 0.4, -60


def risk(r) -> str:  # та же логика, что в backend (ops-прогноз для диспетчера)
    d = r.ops if pd.notna(r.ops) else r.prediction
    p = None if r.get("ops_basis") == "gps" else (r.ops_p if pd.notna(r.ops_p) else r.p_late)
    if p is None or pd.isna(p):
        x = "red" if d >= 180 else "yellow" if d >= 60 else "green"
    else:
        x = "red" if p >= RED_P else "yellow" if (p >= YELLOW_P or d >= 90) else "green"
    return "early" if x == "green" and d <= EARLY_S else x


P_BINS = [0, 0.1, 0.3, 0.5, 0.7, 1.0001]


def auc(y, p) -> float:
    """ROC AUC через ранги (Манн — Уитни), связи — средним рангом."""
    y, r = np.asarray(y, bool), pd.Series(p).rank().values
    n1, n0 = y.sum(), (~y).sum()
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float("nan")


def calibration(fact, p, iv, title: str):
    """Покрытие интервала ~80% и калибровка P(опоздание > 2 мин) — то, что показывает карточка события."""
    fact = np.asarray(fact, float)
    print(f"\n{title}")
    ok = np.array([x is not None and not (isinstance(x, float) and np.isnan(x)) for x in iv])
    if ok.any():
        lo = np.array([x[0] for x in np.asarray(iv, object)[ok]], float)
        hi = np.array([x[1] for x in np.asarray(iv, object)[ok]], float)
        f = fact[ok]
        inside = (f >= lo) & (f <= hi)
        print(f"  интервал ~80%: n={ok.sum()}, факт внутри {inside.mean():.1%}, ниже {(f < lo).mean():.1%}, "
              f"выше {(f > hi).mean():.1%}; медианная ширина {np.median(hi - lo) / 60:.1f} мин")
    p = np.asarray(p, float)
    k = ~np.isnan(p)
    if k.any():
        y, pp = fact[k] > 120, p[k]
        print(f"  P(опоздание > 2 мин): n={k.sum()}, доля опозданий {y.mean():.1%}, средняя P {pp.mean():.3f}, "
              f"Brier {np.mean((pp - y) ** 2):.3f} (у константы {y.mean() * (1 - y.mean()):.3f}), AUC {auc(y, pp):.3f}")
        b = pd.cut(pp, P_BINS, right=False, labels=["0–0,1", "0,1–0,3", "0,3–0,5", "0,5–0,7", "0,7–1"])
        t = pd.DataFrame(dict(бин=b, p=pp, y=y)).groupby("бин", observed=False).agg(n=("y", "size"), P_средняя=("p", "mean"),
                                                                                    доля_опозданий=("y", "mean"))
        print(t.round(3).to_string())


def main():
    hints = pd.read_csv(DATA / "validate" / "points.csv")
    hints["T"] = _ts(hints["T"])
    eng = replay_through_engine("validate", hints)
    on = pd.DataFrame(eng.predictions)
    on["ops"] = on.get("ops_prediction_s", on.prediction)
    on["ops_p"] = on.get("ops_p_late", np.nan)
    lab = pd.read_csv(DATA / "labels" / "labels_test.csv")
    lab["T"] = _ts(lab["T"]).values.astype("datetime64[s]").astype(np.int64)
    m = on.merge(lab[["tr_id", "T", "target_stop_id", "target_delay_s", "cur_dev_s"]].rename(columns={"cur_dev_s": "asu_dev"}),
                 on=["tr_id", "T", "target_stop_id"], how="inner")
    m["risk"] = m.apply(risk, axis=1)
    m["lead_min"] = (m.target_time_begin - m["T"]) / 60
    print(f"прогнозов всего: {len(on)}, по {on.tr_id.nunique()} ТС; сверено с фактом: {len(m)} "
          f"(цель совпала с разметкой; остальные моменты разметкой не покрыты)")
    print(f"горизонт: {m.lead_min.min():.0f}–{m.lead_min.max():.0f} мин; источник отклонения: "
          f"{m.hint_source.value_counts().to_dict()}")
    rows = []
    for tr, g in m.groupby("tr_id"):
        late = g.target_delay_s > 120
        alert = g.risk.isin(["red", "yellow"])
        rows.append(dict(ТС=tr, n=len(g), MAE_диспетчер=round((g.ops - g.target_delay_s).abs().mean()),
                         MAE_сабмит=round((g.prediction - g.target_delay_s).abs().mean()),
                         MAE_baseline=round((g.asu_dev - g.target_delay_s).abs().mean()),
                         смещение=round((g.ops - g.target_delay_s).mean()),
                         опозданий=int(late.sum()), поймано=int((late & alert).sum()),
                         ложных=int((alert & ~late).sum()), подсказка_АСУ=f"{(g.hint_source == 'feed').mean():.0%}"))
    t = pd.DataFrame(rows).sort_values("MAE_диспетчер", ascending=False)
    print(t.to_string(index=False))
    late, alert = m.target_delay_s > 120, m.risk.isin(["red", "yellow"])
    early, ealert = m.target_delay_s < -60, m.risk == "early"
    print(f"\nИТОГО: MAE для диспетчера {(m.ops - m.target_delay_s).abs().mean():.1f} с, сабмит-логика "
          f"{(m.prediction - m.target_delay_s).abs().mean():.1f} с, baseline cur_dev {(m.asu_dev - m.target_delay_s).abs().mean():.1f} с, "
          f"«ноль» {m.target_delay_s.abs().mean():.1f} с")
    print(f"опоздания > 2 мин: {int(late.sum())}, предупреждений (жёлтый/красный) по ним {int((late & alert).sum())} "
          f"({(late & alert).sum() / max(1, late.sum()):.0%}); ложных предупреждений {int((alert & ~late).sum())} из {int(alert.sum())}")
    print(f"ранние < −1 мин: {int(early.sum())}, помечено «раньше плана» {int((early & ealert).sum())}; "
          f"ложных «раньше» {int((ealert & ~early).sum())} из {int(ealert.sum())}")
    mm = m[m.manual_target.astype(bool)]
    if len(mm):
        phys_late = mm.target_delay_s > 120
        print(f"\nцели с manual_fill: {len(mm)} (ТС {sorted(mm.tr_id.unique().tolist())}); факт: медиана {mm.target_delay_s.median():.0f} с, "
              f"доля 0 {(mm.target_delay_s == 0).mean():.0%}, опозданий > 2 мин {int(phys_late.sum())}")
        for name, col in [("0 (сабмит)", "prediction"), ("модель без правила", "model_prediction_s"), ("GPS-отставание (дашборд)", "ops")]:
            print(f"  {name}: MAE {(mm[col] - mm.target_delay_s).abs().mean():.0f} с, предупреждений {int((mm[col] >= 90).sum())}, "
                  f"верных {int(((mm[col] >= 90) & phys_late).sum())}")
    for src, g in m.groupby("hint_source"):
        print(f"  источник {src}: n={len(g)}, MAE {(g.ops - g.target_delay_s).abs().mean():.1f} с")

    # как на дашборде (backend/app.py): интервал ops_interval_s, иначе interval_s; P — ops_p_late, иначе p_late;
    # при ops_basis == "gps" ни P, ни интервал не показываются
    gps = m["ops_basis"].eq("gps") if "ops_basis" in m else pd.Series(False, index=m.index)
    iv = [None if g else (o if isinstance(o, (list, tuple)) else i)
          for g, o, i in zip(gps, m.get("ops_interval_s", pd.Series([None] * len(m), index=m.index)), m["interval_s"])]
    p = m.ops_p.where(m.ops_p.notna(), m.p_late).where(~gps)
    calibration(m.target_delay_s, p, iv, f"КАЛИБРОВКА (онлайн, все точки, сверенные с labels_test: {len(m)})")
    nm = ~m.manual_target.astype(bool).values
    calibration(m.target_delay_s[nm], p[nm], [x for x, k in zip(iv, nm) if k], f"  без целей manual_fill ({int(nm.sum())})")
    risky = m.risk.isin(["red", "yellow"]).values
    calibration(m.target_delay_s[risky], p[risky], [x for x, k in zip(iv, risky) if k],
                f"  только события на карточке (жёлтый/красный, {int(risky.sum())})")


def offline():
    """То же на офлайн-тесте (output/features_test.parquet), только реальные ТС."""
    from src.predict import Predictor
    df = pd.read_parquet("output/features_test.parquet")
    df = df[(df.tr_id < 9_000_000) & df.target_delay_s.notna()].reset_index(drop=True)
    res = pd.DataFrame(Predictor().predict_full(df, explain=False))
    iv = res.get("ops_interval_s", pd.Series([None] * len(res))).where(lambda x: x.notna(), res["interval_s"]).tolist()
    p = res.get("ops_p_late", pd.Series(np.nan, index=res.index)).fillna(res["p_late"])
    calibration(df.target_delay_s, p, iv, f"КАЛИБРОВКА (офлайн, features_test, реальные ТС: {len(df)})")
    nm = ~res.manual_target.astype(bool).values
    calibration(df.target_delay_s[nm], p[nm], [x for x, k in zip(iv, nm) if k], f"  без целей manual_fill ({int(nm.sum())})")


if __name__ == "__main__":
    import sys
    offline() if "--offline" in sys.argv else main()
