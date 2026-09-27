"""Загрузка сырых CSV в нормализованные DataFrame."""
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"
CACHE = Path(__file__).resolve().parent.parent / "output" / "cache"
SYNTHETIC_MIN_ID = 9_000_000
SYNTH_DIR = Path(__file__).resolve().parent.parent / "output" / "synthetic"  # демо-датасет src/synth.py


def split_dir(split: str) -> Path:
    """Каталог сплита: датасет хакатона или синтетика «несколько машин на маршруте»."""
    return SYNTH_DIR if split == "synthetic" else DATA / split


def _ts(s: pd.Series) -> pd.Series:
    # в файлах встречаются форматы с наносекундами и без — приводим к секундам
    return pd.to_datetime(s, format="mixed").dt.floor("s")


def load_traffic(split: str) -> pd.DataFrame:
    """Телеметрия: только валидные координаты, отсортирована по (tr_id, event_time)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    cache = CACHE / f"traffic_{split}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    df = pd.read_csv(
        split_dir(split) / "traffic.csv",
        usecols=["tr_id", "unit_id", "event_time", "location_valid", "lon", "lat", "speed", "heading"],
    )
    df = df[df["location_valid"].astype(str) == "True"].drop(columns="location_valid")
    df["event_time"] = _ts(df["event_time"])
    df = df.dropna(subset=["lon", "lat"])
    df = df.sort_values(["tr_id", "event_time"]).drop_duplicates(["tr_id", "event_time"])
    df = df.reset_index(drop=True)
    df.to_parquet(cache)
    return df


def load_schedule(split: str) -> pd.DataFrame:
    """Расписание. В validate факта нет — колонка time_fact_begin будет NaT."""
    path = split_dir(split) / ("schedule_plan.csv" if split == "validate" else "schedule.csv")
    df = pd.read_csv(path)
    df = df.rename(columns={"tt_action_item_id": "stop_id"})
    df["time_begin"] = _ts(df["time_begin"])
    if "time_fact_begin" in df:
        df["time_fact_begin"] = _ts(df["time_fact_begin"])
    else:
        df["time_fact_begin"] = pd.NaT
    xy = df["geom"].str.extract(r"POINT \(([-\d.]+) ([-\d.]+)\)").astype(float)
    df["stop_lon"], df["stop_lat"] = xy[0], xy[1]
    df["manual_fill"] = df["manual_fill"].astype(str) == "True"
    df = df.drop(columns=["geom", "order_date"])
    return df.sort_values(["tr_id", "time_begin", "stop_id"]).reset_index(drop=True)


def load_points(split: str) -> pd.DataFrame:
    """Прогнозные точки: labels для train/test, points.csv для validate."""
    path = DATA / ("validate/points.csv" if split == "validate" else f"labels/labels_{split}.csv")
    df = pd.read_csv(path)
    df["T"] = _ts(df["T"])
    df["target_time_begin"] = _ts(df["target_time_begin"])
    df["is_synthetic"] = df["tr_id"] >= SYNTHETIC_MIN_ID
    return df


def haversine_km(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))
