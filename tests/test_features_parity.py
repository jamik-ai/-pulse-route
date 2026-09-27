"""Потоковый контур (NDTP-кодек → OnlineEngine) даёт те же прогнозы, что офлайн-пайплайн."""
import pandas as pd
import pytest

from src.dataio import load_points
from src.predict import OUT
from src.realtime.stream_submission import replay_through_engine


def test_online_matches_offline():
    if not (OUT / "pred_test.csv").exists():
        pytest.skip("нет output/pred_test.csv — запустите python -m src.predict test")
    labels = load_points("test")
    eng = replay_through_engine("test", labels)
    online = pd.DataFrame(eng.predictions)
    offline = pd.read_csv(OUT / "pred_test.csv")
    m = labels[["sample_id", "target_stop_id"]].merge(offline, on="sample_id").merge(
        online[["sample_id", "target_stop_id", "prediction"]], on="sample_id", how="left",
        suffixes=("_lbl", "_on")).rename(columns={"prediction_lbl": "prediction_off"})
    assert m.prediction_on.notna().all(), "поток не выдал прогноз для части размеченных точек"
    assert (m.target_stop_id_lbl == m.target_stop_id_on).all(), "поток ответил не на ту целевую остановку"
    diff = (m.prediction_off - m.prediction_on).abs()
    print(f"\nпаритет: n={len(m)}, median |Δ|={diff.median():.2f} c, p95={diff.quantile(.95):.2f}, max={diff.max():.2f}")
    assert diff.median() < 1.0 and diff.quantile(0.95) < 10
