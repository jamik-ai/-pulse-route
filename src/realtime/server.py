"""NDTP-сервер приёма телеметрии + онлайн-прогноз + HTTP API.

python -m src.realtime.server --ndtp-port 9201 --http-port 8080 \
    --plan validate [--hints data/validate/points.csv]

HTTP:  GET /health   GET /stats   GET /predictions?limit=100&tr_id=...
"""
import argparse
import asyncio
import json
import logging
import time
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pandas as pd

from src.dataio import DATA, SYNTH_DIR, _ts, load_schedule
from src.predict import Predictor
from src.realtime import ndtp
from src.realtime.online import HintProvider, OnlineEngine

log = logging.getLogger("ndtp")


def unit_registry() -> dict:
    """unit_id (NDTP peerAddress) → tr_id. Справочник бортовых терминалов берём из архивной телеметрии."""
    reg = {}
    files = [DATA / split / "traffic.csv" for split in ("train", "test", "validate")]
    files += [SYNTH_DIR / "traffic.csv"] if (SYNTH_DIR / "traffic.csv").exists() else []  # терминалы синтетики
    for f in files:
        df = pd.read_csv(f, usecols=["unit_id", "tr_id"]).drop_duplicates()
        reg.update(dict(zip(df.unit_id.astype(int), df.tr_id.astype(int))))
    return reg


def load_plan(split: str) -> pd.DataFrame:
    return load_schedule(split).drop(columns=["time_fact_begin"])  # факт в онлайн не попадает


def load_hints(path: Path | None) -> HintProvider:
    if not path:
        return HintProvider()
    df = pd.read_csv(path)
    df["T"] = _ts(df["T"])
    return HintProvider(df)


class NdtpServer:
    def __init__(self, engine: OnlineEngine, ack: bool = True):
        self.engine = engine
        self.ack = ack
        self.conn = dict(active=0, total=0, frames=0, handshakes=0, crc_errors=0, bytes=0)
        self.started = time.time()

    async def handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        peer = writer.get_extra_info("peername")
        self.conn["active"] += 1
        self.conn["total"] += 1
        fr = ndtp.FrameReader()
        unit = None
        try:
            while data := await reader.read(65536):
                self.conn["bytes"] += len(data)
                for f in fr.feed(data):
                    self.conn["frames"] += 1
                    unit = f.peer
                    if (f.service, f.type) == (ndtp.SERVICE_GENERIC, ndtp.TYPE_CONN_REQUEST):
                        self.conn["handshakes"] += 1
                        log.info("handshake unit=%s from %s", f.peer, peer)
                    for p in ndtp.frame_nav_points(f):
                        self.engine.on_point(p.unit_id, p.timestamp, p.lon, p.lat, p.valid, p.speed)
                    if self.ack:
                        writer.write(ndtp.build_result(f.peer, f.request_id, f.service))
                if self.ack:
                    await writer.drain()
        except (ConnectionResetError, BrokenPipeError):
            pass
        finally:
            self.conn["crc_errors"] += fr.crc_errors
            self.conn["active"] -= 1
            writer.close()
            log.info("disconnect unit=%s", unit)

    def stats(self) -> dict:
        e = self.engine.stats
        return {**self.conn, **e, "uptime_s": round(time.time() - self.started, 1),
                "avg_infer_ms_per_T": round(e["infer_ms_total"] / max(len({p["T"] for p in self.engine.predictions}), 1), 1),
                "stream_time": pd.Timestamp(self.engine.max_ts, unit="s").isoformat() if self.engine.max_ts else None,
                "vehicles_with_data": sum(1 for v in self.engine.vehicles.values() if v.et)}


async def http_handler(server: NdtpServer, reader, writer):
    try:
        line = (await reader.readline()).decode(errors="replace")
        while (await reader.readline()) not in (b"\r\n", b"\n", b""):
            pass
        url = urlparse(line.split(" ")[1] if " " in line else "/")
        q = parse_qs(url.query)
        if url.path == "/health":
            body, code = {"status": "ok"}, 200
        elif url.path == "/stats":
            body, code = server.stats(), 200
        elif url.path == "/predictions":
            preds = server.engine.predictions
            if "tr_id" in q:
                preds = [p for p in preds if str(p["tr_id"]) == q["tr_id"][0]]
            body, code = preds[-int(q.get("limit", ["100"])[0]):], 200
        else:
            body, code = {"error": "not found"}, 404
        raw = json.dumps(body, ensure_ascii=False, default=str).encode()
        writer.write(f"HTTP/1.1 {code} OK\r\nContent-Type: application/json; charset=utf-8\r\n"
                     f"Content-Length: {len(raw)}\r\nConnection: close\r\n\r\n".encode() + raw)
        await writer.drain()
    finally:
        writer.close()


async def main(a):
    predictor = Predictor()
    out = open(a.out, "a", encoding="utf-8") if a.out else None

    def on_pred(rec):
        if out:
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
        log.info("prediction %s stop=%s cur_dev=%s(%s) → %+.0f s", rec["sample_id"], rec["target_stop_id"],
                 rec["cur_dev_s"], rec["hint_source"], rec["prediction"])

    engine = OnlineEngine(load_plan(a.plan), unit_registry(), predictor, load_hints(a.hints), on_pred,
                          watermark_lag_s=a.watermark_lag)
    srv = NdtpServer(engine, ack=not a.no_ack)
    ndtp_srv = await asyncio.start_server(srv.handle, a.host, a.ndtp_port)
    http_srv = await asyncio.start_server(lambda r, w: http_handler(srv, r, w), a.host, a.http_port)
    log.info("NDTP on %s:%s, HTTP on %s:%s, ТС в плане: %d, терминалов в справочнике: %d",
             a.host, a.ndtp_port, a.host, a.http_port, len(engine.vehicles), len(engine.unit_to_tr))
    async with ndtp_srv, http_srv:
        await asyncio.gather(ndtp_srv.serve_forever(), http_srv.serve_forever())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--ndtp-port", type=int, default=9201)
    ap.add_argument("--http-port", type=int, default=8080)
    ap.add_argument("--plan", default="validate", choices=["validate", "test", "train"],
                    help="плановое расписание какого набора загрузить")
    ap.add_argument("--hints", default=None, help="CSV с колонками tr_id,T,cur_dev_s (поток подсказок АСУ)")
    ap.add_argument("--out", default="output/realtime_predictions.jsonl")
    ap.add_argument("--watermark-lag", type=int, default=30)
    ap.add_argument("--no-ack", action="store_true")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    asyncio.run(main(ap.parse_args()))
