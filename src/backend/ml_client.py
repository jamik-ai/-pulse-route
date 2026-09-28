"""Клиент ML-сервиса с деградацией: при недоступности ML — правило (cur_dev_s, manual→0)."""
import json
import logging
import time

import httpx
import numpy as np
import pandas as pd

log = logging.getLogger("ml-client")


def _clean(v):
    if isinstance(v, (np.floating, float)):
        return None if not np.isfinite(v) else float(v)
    if isinstance(v, np.integer):
        return int(v)
    return v


class MLClient:
    def __init__(self, url: str, timeout_s: float = 3.0):
        self.url = url.rstrip("/")
        self.http = httpx.Client(timeout=timeout_s)
        self.degraded = False
        self.last_error: str | None = None
        self.latency_ms: list[float] = []
        self.fallbacks = 0

    def _call(self, items: list[dict], explain: bool) -> list[dict]:
        t0 = time.perf_counter()
        r = self.http.post(f"{self.url}/v1/predict", json={"items": items, "explain": explain})
        r.raise_for_status()
        self.latency_ms.append((time.perf_counter() - t0) * 1000)
        del self.latency_ms[:-1000]
        return r.json()["results"]

    def predict_full(self, X: pd.DataFrame, explain: bool = True) -> list[dict]:
        items = [{k: _clean(v) for k, v in row.items()} for row in X.to_dict("records")]
        try:
            res = self._call(items, explain)
            if self.degraded:
                log.info("ML-сервис снова доступен")
            self.degraded, self.last_error = False, None
            return res
        except Exception as e:  # noqa: BLE001 — любая ошибка ML не должна ронять приём потока
            if not self.degraded:
                log.warning("ML-сервис недоступен (%s) — деградация на правило", e)
            self.degraded, self.last_error = True, str(e)[:200]
            self.fallbacks += len(items)
            return [self.rule(r) for r in items]

    @staticmethod
    def rule(row: dict) -> dict:
        """Запасной прогноз без ML: подсказка АСУ, для ручных участков — 0."""
        cur = float(row.get("cur_dev_s") or 0.0)
        gps = row.get("gps_dev_med3")  # для диспетчера — фактическое отклонение по GPS, если подсказка АСУ пустая
        ops = float(gps) if gps is not None and np.isfinite(gps) else cur
        p = 0.0 if row.get("tgt_manual") == 1 else cur
        return {"prediction_s": p, "ops_prediction_s": ops, "manual_target": row.get("tgt_manual") == 1,
                "p_late": None, "interval_s": None, "expected_error_s": None,
                "causes": [{"factor": "fallback", "contribution_s": 0.0,
                            "text": "ML-модуль недоступен: прогноз по последнему известному отклонению"}],
                "degraded": True}

    def info(self) -> dict:
        try:
            return self.http.get(f"{self.url}/v1/model", timeout=1.0).json()
        except Exception as e:  # noqa: BLE001
            return {"error": str(e)[:200]}

    def stats(self) -> dict:
        lat = self.latency_ms[-300:]
        return {"status": "degraded" if self.degraded else "ok", "last_error": self.last_error,
                "fallback_items": self.fallbacks,
                "latency_ms_p50": round(float(np.percentile(lat, 50)), 1) if lat else None,
                "latency_ms_p95": round(float(np.percentile(lat, 95)), 1) if lat else None}


def dumps(o) -> str:
    return json.dumps(o, ensure_ascii=False, default=_clean)
