"""ML-сервис: инференс и управление моделью. Отдельный контейнер, stateless.

uvicorn src.ml_service.app:app --port 8001      → Swagger: /docs, спецификация: /openapi.json

Backend присылает батч признаков (посчитанных строго по данным ≤ T), сервис возвращает
прогноз задержки, P(опоздание), интервал и причины. Горизонтально масштабируется репликами.
"""
import threading
import time
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.predict import MODELS, Predictor

state: dict[str, Any] = {"predictor": None, "loaded_at": None, "requests": 0, "items": 0, "latency_ms": []}
_lock = threading.Lock()


def load_model():
    p = Predictor(MODELS)
    with _lock:
        state["predictor"], state["loaded_at"] = p, time.time()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    load_model()
    yield


app = FastAPI(title="Пульс маршрута — ML service", version="1.0.0", lifespan=lifespan,
              description="Инференс ансамбля CatBoost: задержка на целевой остановке через 10–15 мин, "
                          "вероятность опоздания, интервал ~80% и причины (SHAP).")


class PredictRequest(BaseModel):
    items: list[dict[str, Any]] = Field(..., description="Признаки прогнозных точек (выход src.features.features_at)")
    explain: bool = Field(True, description="Считать причины (SHAP), +~2 мс на точку")


class Cause(BaseModel):
    factor: str
    contribution_s: float
    text: str


class PredictItem(BaseModel):
    prediction_s: float = Field(..., description="Прогноз задержки, сек (+ опоздание, − опережение)")
    p_late: float | None = Field(None, description="P(опоздание > late_threshold_s)")
    interval_s: list[float] | None = Field(None, description="Интервал ~80%, сек")
    expected_error_s: float | None = None
    causes: list[Cause] = []
    ops_prediction_s: float | None = Field(None, description="Прогноз для диспетчера (совпадает с prediction_s)")
    ops_p_late: float | None = Field(None, description="P(опоздание) для диспетчера: без правила manual_fill")
    ops_interval_s: list[float] | None = Field(None, description="Интервал ~80% для диспетчера")
    manual_target: bool = Field(False, description="Плановое время цели заполнено вручную (manual_fill)")
    model_prediction_s: float | None = Field(None, description="Выход ансамбля без правила manual_fill → 0 (для справки)")


class PredictResponse(BaseModel):
    results: list[PredictItem]
    model_version: str
    latency_ms: float


@app.get("/health")
def health():
    return {"status": "ok" if state["predictor"] else "loading"}


@app.get("/v1/model")
def model_info():
    p: Predictor = state["predictor"]
    lat = state["latency_ms"][-500:]
    return {
        "version": _version(), "loaded_at": state["loaded_at"], "n_features": len(p.features),
        "ensemble": {k: len(v) for k, v in p.models.items()},
        "late_threshold_s": p.meta.get("late_threshold_s"), "interval_coverage": p.meta.get("interval_coverage"),
        "requests": state["requests"], "items": state["items"],
        "latency_ms_p50": float(np.percentile(lat, 50)) if lat else None,
        "latency_ms_p95": float(np.percentile(lat, 95)) if lat else None,
    }


@app.post("/v1/model/reload", summary="Перечитать модели из models/ (после дообучения)")
def reload_model():
    load_model()
    return {"status": "reloaded", "version": _version()}


@app.post("/v1/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if not req.items:
        return PredictResponse(results=[], model_version=_version(), latency_ms=0.0)
    p: Predictor = state["predictor"]
    if p is None:
        raise HTTPException(503, "model not loaded")
    t0 = time.perf_counter()
    res = p.predict_full(pd.DataFrame(req.items), explain=req.explain)
    dt = (time.perf_counter() - t0) * 1000
    state["requests"] += 1
    state["items"] += len(req.items)
    state["latency_ms"].append(dt)
    del state["latency_ms"][:-2000]
    return PredictResponse(results=res, model_version=_version(), latency_ms=round(dt, 1))


def _version() -> str:
    f = MODELS / "meta.json"
    return time.strftime("%Y%m%d-%H%M%S", time.localtime(f.stat().st_mtime)) if f.exists() else "none"
