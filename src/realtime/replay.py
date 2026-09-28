"""Replay архивной телеметрии как живого NDTP-потока (по TCP-соединению на терминал, как эмулятор).

Эмулятор с autoGenerate шлёт случайные точки около Москвы, не по маршрутам — для проверки
качества онлайн-прогноза нужен поток реальных треков. Этот клиент проигрывает traffic.csv
в ускоренном времени в тех же NDTP-кадрах.

python -m src.realtime.replay --host server --port 9201 --split validate --speed 600
"""
import argparse
import asyncio
import logging
import time

import numpy as np
import pandas as pd

from src.dataio import _ts, split_dir
from src.realtime import ndtp

log = logging.getLogger("replay")


async def drain_acks(reader: asyncio.StreamReader, counter: dict):
    fr = ndtp.FrameReader()
    while data := await reader.read(65536):
        counter["acks"] += len(fr.feed(data))


def _load(split: str) -> pd.DataFrame:
    raw = pd.read_csv(split_dir(split) / "traffic.csv",
                      usecols=["unit_id", "event_time", "location_valid", "lon", "lat", "speed", "heading", "alt"])
    raw["ts"] = _ts(raw["event_time"]).values.astype("datetime64[s]").astype(np.int64)
    return raw.sort_values("ts", kind="stable")


async def run(a):
    raw = await asyncio.to_thread(_load, a.split)  # не блокируем цикл событий (replay внутри backend)
    if a.start:
        raw = raw[raw.ts >= pd.Timestamp(a.start).value // 10**9]
    if a.hours:
        raw = raw[raw.ts < raw.ts.iloc[0] + a.hours * 3600]
    units = sorted(raw.unit_id.astype(int).unique())
    counter = {"acks": 0}
    conns, req = {}, {}
    for u in units:
        r, w = await asyncio.open_connection(a.host, a.port)
        w.write(ndtp.build_handshake(u))
        conns[u] = w
        req[u] = 2
        asyncio.create_task(drain_acks(r, counter))
    await asyncio.sleep(0.2)  # как у эмулятора: пауза после handshake
    log.info("открыто %d соединений, точек %d, поток %s … %s, ускорение ×%s", len(conns), len(raw),
             pd.Timestamp(raw.ts.iloc[0], unit="s"), pd.Timestamp(raw.ts.iloc[-1], unit="s"), a.speed)
    t_wall0, t_stream0 = time.monotonic(), raw.ts.iloc[0]
    speed0 = a.speed
    sent = 0
    for r in raw.itertuples(index=False):
        if a.speed != speed0:  # скорость меняют на лету (настройки дашборда) — перепривязываем «часы»
            t_wall0, t_stream0, speed0 = time.monotonic(), r.ts, a.speed
        if a.speed > 0:
            wait = (r.ts - t_stream0) / a.speed - (time.monotonic() - t_wall0)
            if wait > 0:
                await asyncio.sleep(wait)
        valid = str(r.location_valid) == "True" and not np.isnan(r.lon)
        cell = ndtp.build_nav_cell(r.ts, r.lon if valid else 0, r.lat if valid else 0, valid,
                                   0 if np.isnan(r.speed) else r.speed, 0 if np.isnan(r.heading) else r.heading,
                                   0 if np.isnan(r.alt) else r.alt)
        u = int(r.unit_id)
        conns[u].write(ndtp.build_realtime(u, req[u], cell))
        req[u] += 1
        sent += 1
        if sent % 2000 == 0:
            await asyncio.gather(*(w.drain() for w in conns.values()))
            log.info("отправлено %d/%d, время потока %s, подтверждений %d", sent, len(raw),
                     pd.Timestamp(r.ts, unit="s"), counter["acks"])
    await asyncio.gather(*(w.drain() for w in conns.values()))
    await asyncio.sleep(1)
    for w in conns.values():
        w.close()
    log.info("готово: отправлено %d пакетов за %.1f c, подтверждений %d", sent, time.monotonic() - t_wall0, counter["acks"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9201)
    ap.add_argument("--split", default="validate")
    ap.add_argument("--speed", type=float, default=600, help="ускорение времени; 0 — без пауз")
    ap.add_argument("--start", default=None, help="начать с момента, напр. '2026-01-06 06:00'")
    ap.add_argument("--hours", type=float, default=None, help="сколько часов потока проиграть")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    asyncio.run(run(ap.parse_args()))
