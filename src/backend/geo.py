"""Геометрия маршрутов для карты: реальные GPS-треки ТС (исторические) + остановки из расписания.

Треки из архива телеметрии идут по реальной улично-дорожной сети, упрощаются алгоритмом
Рамера–Дугласа–Пекера и режутся на куски по разрывам (депо, пропуски связи).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.dataio import load_schedule, load_traffic


def rdp(xy: np.ndarray, eps: float) -> np.ndarray:
    """Индексы точек, оставшихся после упрощения (итеративный RDP), xy — метры."""
    keep = np.zeros(len(xy), bool)
    keep[[0, -1]] = True
    stack = [(0, len(xy) - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        a, b = xy[i], xy[j]
        ab = b - a
        L = np.hypot(*ab)
        seg = xy[i + 1:j] - a
        d = np.abs(ab[0] * seg[:, 1] - ab[1] * seg[:, 0]) / L if L > 0 else np.hypot(seg[:, 0], seg[:, 1])
        k = int(d.argmax())
        if d[k] > eps:
            m = i + 1 + k
            keep[m] = True
            stack += [(i, m), (m, j)]
    return np.flatnonzero(keep)


def _xy(lon, lat, lon0, lat0):
    return (lon - lon0) * 111_320 * np.cos(np.radians(lat0)), (lat - lat0) * 110_540


def build_routes(split: str = "validate", eps_m: float = 8.0, jump_m: float = 400) -> dict:
    """Нитка маршрута каждого ТС: по одному реальному проезду рейса в каждом направлении.

    Рейсы берём из расписания (разрыв плана > 5 мин — новый рейс), для рейса — GPS-точки
    от первой до последней остановки. Из рейсов с одинаковыми конечными выбираем один с
    наилучшим покрытием GPS. Получается чистая линия по дорогам без кругов, депо и наложений.
    """
    t = load_traffic(split)
    plan = load_schedule(split)
    routes = {}
    for tr, g in plan.groupby("tr_id"):
        g = g.sort_values("time_begin")
        tt = t[t.tr_id == tr]
        et = tt.event_time.values.astype("datetime64[s]").astype(np.int64)
        lon, lat = tt.lon.values, tt.lat.values
        pt = g.time_begin.values.astype("datetime64[s]").astype(np.int64)
        brk = np.flatnonzero(np.diff(pt) > 300) + 1
        best: dict[tuple, tuple] = {}  # (первая, последняя остановка) → (оценка, индексы GPS)
        for s0, e0 in zip(np.r_[0, brk], np.r_[brk, len(g)]):
            if e0 - s0 < 5 or len(et) < 10:
                continue
            i0, i1 = np.searchsorted(et, [pt[s0] - 120, pt[e0 - 1] + 600])
            if i1 - i0 < 10:
                continue
            x, y = _xy(lon[i0:i1], lat[i0:i1], lon.mean(), lat.mean())
            step = np.hypot(np.diff(x), np.diff(y))
            bad = int(((step > jump_m) | (np.diff(et[i0:i1]) > 180)).sum())  # провалы связи/скачки
            key = (round(g.stop_lon.values[s0], 3), round(g.stop_lat.values[s0], 3),
                   round(g.stop_lon.values[e0 - 1], 3), round(g.stop_lat.values[e0 - 1], 3))
            score = (-bad, (i1 - i0) / max(1, (et[i1 - 1] - et[i0])))  # меньше провалов, плотнее точки
            if key not in best or score > best[key][0]:
                best[key] = (score, i0, i1)
        pieces = []
        for _, i0, i1 in sorted(best.values(), reverse=True)[:3]:  # до 3 уникальных вариантов рейса
            x, y = _xy(lon[i0:i1], lat[i0:i1], lon.mean(), lat.mean())
            cut = np.flatnonzero((np.hypot(np.diff(x), np.diff(y)) > jump_m) | (np.diff(et[i0:i1]) > 180)) + 1
            for a, b in zip(np.r_[0, cut], np.r_[cut, i1 - i0]):  # провалы не соединяем прямой
                if b - a < 3:
                    continue
                idx = i0 + a + rdp(np.c_[x[a:b], y[a:b]], eps_m)
                pieces.append([[round(float(lat[i]), 6), round(float(lon[i]), 6)] for i in idx])
        stops = (g.assign(key=g.stop_lon.round(4).astype(str) + g.stop_lat.round(4).astype(str))
                 .drop_duplicates("key"))
        routes[int(tr)] = {
            "track": pieces,
            "stops": [{"lat": float(r.stop_lat), "lon": float(r.stop_lon),
                       "name": r.building_address if isinstance(r.building_address, str) else ""}
                      for r in stops.itertuples()],
        }
    return routes


def stop_routes(split: str) -> dict:
    """Маршруты без GPS-треков — только остановки (для синтетики: линии рейсов берутся из кэша путей)."""
    plan = load_schedule(split)
    out = {}
    for tr, g in plan.groupby("tr_id"):
        g = g.assign(key=g.stop_lon.round(4).astype(str) + g.stop_lat.round(4).astype(str)).drop_duplicates("key")
        out[int(tr)] = {"track": [], "stops": [{"lat": float(r.stop_lat), "lon": float(r.stop_lon),
                                                "name": r.building_address if isinstance(r.building_address, str) else ""}
                                               for r in g.itertuples()]}
    return out


def stop_names(split: str = "validate") -> dict:
    p = load_schedule(split)
    return dict(zip(p.stop_id.astype(np.int64), p.building_address.fillna("")))


def plan_table(split: str = "validate") -> pd.DataFrame:
    return load_schedule(split).drop(columns=["time_fact_begin"])


# ---------------- привязка к дорогам (map matching) ----------------
MATCHED_CACHE = Path(__file__).resolve().parents[2] / "output" / "cache" / "routes_matched.json"


def smooth(line: list, win: int = 5, eps_m: float = 18.0) -> list:
    """Запасной вариант без OSRM: скользящее среднее (убирает GPS-зигзаги) + упрощение RDP."""
    if len(line) < win + 2:
        return line
    a = np.array(line)
    k = np.ones(win) / win
    sm = np.c_[np.convolve(a[:, 0], k, "valid"), np.convolve(a[:, 1], k, "valid")]
    sm = np.vstack([a[:1], sm, a[-1:]])
    x, y = _xy(sm[:, 1], sm[:, 0], sm[:, 1].mean(), sm[:, 0].mean())
    idx = rdp(np.c_[x, y], eps_m)
    return [[round(float(sm[i, 0]), 6), round(float(sm[i, 1]), 6)] for i in idx]


def osrm_match(line: list, chunk: int = 25, overlap: int = 3, pause_s: float = 1.2, tries: int = 4) -> list | None:
    """Привязка трека к дорожному графу OSM через OSRM /match (публичный демо-сервер, кусками)."""
    import time

    import httpx
    step = max(1, len(line) // 120)
    pts = line[::step] + ([line[-1]] if (len(line) - 1) % step else [])
    out: list = []
    i = 0
    with httpx.Client(timeout=20, headers={"User-Agent": "PulseRoute-hackathon/1.0"}) as cli:
        while i < len(pts) - 1:
            part = pts[i:i + chunk]
            coords = ";".join(f"{lo},{la}" for la, lo in part)
            geo = None
            for t in range(tries):
                try:
                    r = cli.get(f"https://router.project-osrm.org/match/v1/driving/{coords}",
                                params={"geometries": "geojson", "overview": "full", "gaps": "ignore", "tidy": "true",
                                        "radiuses": ";".join(["40"] * len(part))})
                    j = r.json()
                    if j.get("code") == "Ok":
                        geo = [[c[1], c[0]] for m in j["matchings"] for c in m["geometry"]["coordinates"]]
                        break
                except Exception:  # noqa: BLE001
                    pass
                time.sleep(pause_s * (t + 2))
            if geo is None:
                return None
            out.extend(geo if not out else geo[1:])
            i += chunk - overlap
            time.sleep(pause_s)
    return [[round(a, 6), round(b, 6)] for a, b in out]


def _decode_polyline6(enc: str) -> list:
    """Декодер Google encoded polyline с точностью 1e-6 (формат Valhalla)."""
    out, idx, lat, lon = [], 0, 0, 0
    while idx < len(enc):
        for coord in (0, 1):
            shift = result = 0
            while True:
                b = ord(enc[idx]) - 63
                idx += 1
                result |= (b & 0x1F) << shift
                shift += 5
                if b < 0x20:
                    break
            d = ~(result >> 1) if result & 1 else result >> 1
            if coord == 0:
                lat += d
            else:
                lon += d
        out.append([round(lat / 1e6, 6), round(lon / 1e6, 6)])
    return out


def valhalla_match(line: list, tries: int = 3) -> list | None:
    """Привязка трека к дорогам через Valhalla trace_route, профиль «автобус» (только дороги для автобусов)."""
    import time

    import httpx
    body = {"shape": [{"lat": a, "lon": b} for a, b in line], "costing": "bus", "shape_match": "map_snap",
            "trace_options": {"search_radius": 40, "gps_accuracy": 15}}
    for t in range(tries):
        try:
            r = httpx.post("https://valhalla1.openstreetmap.de/trace_route", json=body, timeout=60,
                           headers={"User-Agent": "PulseRoute-hackathon/1.0"})
            if r.status_code == 200:
                pts = []
                for leg in r.json()["trip"]["legs"]:
                    seg = _decode_polyline6(leg["shape"])
                    pts.extend(seg if not pts else seg[1:])
                return pts
        except Exception:  # noqa: BLE001
            pass
        time.sleep(2 * (t + 1))
    return None


def matched_routes(routes: dict) -> dict:
    """Маршруты «как в навигаторе»: из кэша привязки к дорогам, иначе сглаженные треки."""
    try:
        cache = json.loads(MATCHED_CACHE.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        cache = {}
    out = {}
    for tr, r in routes.items():
        c = cache.get(str(tr))
        pieces = c if c else r["track"]  # без сглаживания: оно срезает углы на поворотах
        # как в Яндекс Транспорте — одна линия на маршрут: берём самую длинную нитку
        best = max(pieces, key=_length_m, default=None)
        out[tr] = {**r, "track": [best] if best else [], "matched": bool(c)}
    return out


def _length_m(line: list) -> float:
    if len(line) < 2:
        return 0.0
    a = np.array(line)
    x, y = _xy(a[:, 1], a[:, 0], a[:, 1].mean(), a[:, 0].mean())
    return float(np.hypot(np.diff(x), np.diff(y)).sum())


if __name__ == "__main__":  # python -m src.backend.geo → посчитать привязку к дорогам (один раз, с паузами)
    import time
    base = build_routes()
    try:
        cache = json.loads(MATCHED_CACHE.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        cache = {}
    for tr, r in base.items():
        if str(tr) in cache:
            continue
        pieces = []
        longest = max(r["track"], key=_length_m, default=None)  # на карте показывается одна нитка — привязываем её
        for p in ([longest] if longest else []):
            m = valhalla_match(p)
            src = "valhalla"
            if not m:
                m, src = osrm_match(p), "osrm"
            pieces.append(m if m else p)
            print(tr, "нитка", len(p), "точек →", len(pieces[-1]), src if m else "исходный трек (сервисы не ответили)", flush=True)
            time.sleep(1.1)
        cache[str(tr)] = pieces
        MATCHED_CACHE.write_text(json.dumps(cache))
    print("готово:", len(cache))


# ---------------- варианты рейсов: путь по дорогам через остановки ----------------
PATTERNS_CACHE = Path(__file__).resolve().parents[2] / "output" / "cache" / "trip_patterns_v2.json"
PATTERNS_CACHE_V1 = Path(__file__).resolve().parents[2] / "output" / "cache" / "trip_patterns.json"  # до пересчёта


def trip_patterns(plan: pd.DataFrame, gap_s: int = 300) -> dict:
    """{tr: {"spans": [(начало, конец, pid)], "patterns": {pid: [[lat, lon] остановок]}}}; рейс — разрыв плана > gap_s."""
    import hashlib
    out = {}
    for tr, g in plan.sort_values(["tr_id", "time_begin"]).groupby("tr_id"):
        t = g.time_begin.values.astype("datetime64[s]").astype(np.int64)
        brk = np.flatnonzero(np.diff(t) > gap_s) + 1
        spans, pats = [], {}
        for a, b in zip(np.r_[0, brk], np.r_[brk, len(g)]):
            stops = [[round(float(la), 6), round(float(lo), 6)] for la, lo in zip(g.stop_lat.values[a:b], g.stop_lon.values[a:b])]
            if len(stops) < 2:
                continue
            pid = hashlib.sha1(json.dumps([[round(x, 4) for x in p] for p in stops]).encode()).hexdigest()[:10]
            pats[pid] = stops
            spans.append((int(t[a]), int(t[b - 1]), pid))
        out[int(tr)] = {"spans": spans, "patterns": pats}
    return out


def osrm_route(stops: list, chunk: int = 25, pause_s: float = 0.6) -> list | None:
    """Путь по дорогам через остановки: OSRM FOSSGIS (routing.openstreetmap.de), кусками."""
    import time

    import httpx
    out: list = []
    i = 0
    with httpx.Client(timeout=30, headers={"User-Agent": "PulseRoute-hackathon/1.0"}) as cli:
        while i < len(stops) - 1:
            part = stops[i:i + chunk]
            coords = ";".join(f"{lo},{la}" for la, lo in part)
            geo = None
            for t in range(3):
                try:
                    r = cli.get(f"https://routing.openstreetmap.de/routed-car/route/v1/driving/{coords}",
                                params={"overview": "full", "geometries": "geojson", "continue_straight": "false"})
                    j = r.json()
                    if j.get("code") == "Ok":
                        geo = [[round(c[1], 6), round(c[0], 6)] for c in j["routes"][0]["geometry"]["coordinates"]]
                        break
                except Exception:  # noqa: BLE001
                    pass
                time.sleep(1.5 * (t + 1))
            if geo is None:
                return None
            out.extend(geo if not out else geo[1:])
            i += chunk - 1
            time.sleep(pause_s)
    return out


def valhalla_route(stops: list, chunk: int = 20, pause_s: float = 1.1) -> list | None:
    """Путь по дорогам через последовательность остановок (Valhalla /route, профиль «автобус»)."""
    import time

    import httpx
    out: list = []
    i = 0
    with httpx.Client(timeout=60, headers={"User-Agent": "PulseRoute-hackathon/1.0"}) as cli:
        while i < len(stops) - 1:
            part = stops[i:i + chunk]
            body = {"locations": [{"lat": a, "lon": b, "type": "break"} for a, b in part], "costing": "bus",
                    "directions_type": "none"}
            geo = None
            for t in range(3):
                try:
                    r = cli.post("https://valhalla1.openstreetmap.de/route", json=body)
                    if r.status_code == 200:
                        geo = []
                        for leg in r.json()["trip"]["legs"]:
                            seg = _decode_polyline6(leg["shape"])
                            geo.extend(seg if not geo else seg[1:])
                        break
                except Exception:  # noqa: BLE001
                    pass
                time.sleep(2 * (t + 1))
            if geo is None:
                return None
            out.extend(geo if not out else geo[1:])
            i += chunk - 1
            time.sleep(pause_s)
    return out


def build_pattern_routes() -> None:
    """python -c 'from src.backend.geo import build_pattern_routes; build_pattern_routes()' — один раз, в кэш."""
    tp = trip_patterns(load_schedule("validate"))
    try:
        cache = json.loads(PATTERNS_CACHE.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        cache = {}
    for tr, d in tp.items():
        for pid, stops in d["patterns"].items():
            if pid in cache and len(cache[pid]) > len(stops) + 5:  # уже есть настоящий путь по дорогам
                continue
            line = osrm_route(stops) or valhalla_route(stops)
            cache[pid] = line if line else stops
            print(tr, pid, "остановок", len(stops), "→", len(cache[pid]), "по дорогам" if line else "по остановкам (сервис не ответил)", flush=True)
            PATTERNS_CACHE.write_text(json.dumps(cache))
    print("готово вариантов:", len(cache))


# ---------------- быстрое построение: кэш перегонов «остановка → остановка» + параллельно ----------------
import os as _os

SEGMENTS_CACHE = Path(__file__).resolve().parents[2] / "output" / "cache" / "route_segments_v2.json"  # v2: с направлением
# свой OSRM (docker osrm/osrm-backend с выгрузкой OSM Москвы) — миллисекунды на запрос, без лимитов публичного сервера
ROUTER_URL = _os.getenv("ROUTER_URL", "https://routing.openstreetmap.de/routed-car")


def _seg_key(a, b) -> str:
    return f"{a[0]:.5f},{a[1]:.5f}|{b[0]:.5f},{b[1]:.5f}"


def _bearing(a, b) -> int:
    y = np.sin(np.radians(b[1] - a[1])) * np.cos(np.radians(b[0]))
    x = np.cos(np.radians(a[0])) * np.sin(np.radians(b[0])) - np.sin(np.radians(a[0])) * np.cos(np.radians(b[0])) * np.cos(np.radians(b[1] - a[1]))
    return int((np.degrees(np.arctan2(y, x)) + 360) % 360)


FALLBACK_ROUTER_URL = "https://routing.openstreetmap.de/routed-car"
_fallback_lock = __import__("threading").Lock()
_fallback_last = [0.0]


def _osrm_pair(cli, url, a, b, brg, polite=False):
    """Один запрос перегона; polite — для публичного сервера: не чаще 1 запроса в секунду, повтор при 429."""
    import time
    for t in range(6 if polite else 3):
        try:
            if polite:
                with _fallback_lock:
                    time.sleep(max(0.0, _fallback_last[0] + 1.1 - time.time()))
                    _fallback_last[0] = time.time()
            r = cli.get(f"{url}/route/v1/driving/{a[1]},{a[0]};{b[1]},{b[0]}",
                        params={"overview": "full", "geometries": "geojson",
                                **({"bearings": f"{brg},80;{brg},80", "radiuses": "60;60"} if brg is not None else {})})
            if r.status_code == 429:
                time.sleep(3.0 * (t + 1))
                continue
            j = r.json()
            if j.get("code") == "Ok":
                return [[round(c[1], 6), round(c[0], 6)] for c in j["routes"][0]["geometry"]["coordinates"]]
            if j.get("code") in ("NoSegment", "NoRoute"):
                return None
        except Exception:  # noqa: BLE001
            pass
        time.sleep(1.0 * (t + 1))
    return None


def _route_pair(cli, a, b) -> list | None:
    # направление движения на обеих точках: остановка цепляется к полосе «туда», без петель и разворотов
    brg = _bearing(a, b)
    line = _osrm_pair(cli, ROUTER_URL, a, b, brg)
    if line is None and ROUTER_URL != FALLBACK_ROUTER_URL:  # перегон вне карты своего OSRM — публичный OSRM
        line = _osrm_pair(cli, FALLBACK_ROUTER_URL, a, b, brg, polite=True)
    if line is None:  # остановка далеко от дороги нужного направления — без ограничения по курсу
        line = _osrm_pair(cli, ROUTER_URL, a, b, None)
        if line is None or len(line) < 2 or line[0] == line[-1]:
            line = _osrm_pair(cli, FALLBACK_ROUTER_URL, a, b, None, polite=True)
    return line


def build_pattern_lines(patterns: dict, workers: int = 4) -> tuple[dict, dict]:
    """Пути по дорогам для вариантов рейсов из перегонов между соседними остановками.

    Каждый перегон строится один раз и кэшируется: новые рейсы/маршруты почти целиком собираются из уже известных
    перегонов. Недостающие перегоны запрашиваются параллельно. Возвращает ({pid: линия}, статистика).
    """
    from concurrent.futures import ThreadPoolExecutor

    import httpx
    try:
        seg = json.loads(SEGMENTS_CACHE.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        seg = {}
    need = {}
    for stops in patterns.values():
        for a, b in zip(stops[:-1], stops[1:]):
            if (a[0], a[1]) != (b[0], b[1]):
                k = _seg_key(a, b)
                if k not in seg:
                    need[k] = (a, b)
    fetched = 0
    if need:
        with httpx.Client(timeout=30, headers={"User-Agent": "PulseRoute-hackathon/1.0"},
                          limits=httpx.Limits(max_connections=workers)) as cli, ThreadPoolExecutor(workers) as ex:
            for k, line in zip(need, ex.map(lambda ab: _route_pair(cli, *ab), need.values())):
                if line:
                    seg[k] = line
                    fetched += 1
        SEGMENTS_CACHE.parent.mkdir(parents=True, exist_ok=True)
        SEGMENTS_CACHE.write_text(json.dumps(seg))
    lines, complete, ok_pids = {}, 0, set()
    for pid, stops in patterns.items():
        out, ok = [], True
        for a, b in zip(stops[:-1], stops[1:]):
            part = seg.get(_seg_key(a, b)) if (a[0], a[1]) != (b[0], b[1]) else [a]
            if not part:
                ok, part = False, [a, b]  # перегон не построился — пока прямая, достроится в следующий раз
            out.extend(part if not out else part[1:])
        lines[pid] = out
        complete += ok
        if ok:
            ok_pids.add(pid)
    return {p: l for p, l in lines.items() if p in ok_pids}, {"patterns": len(patterns), "complete": complete,
            "segments_requested": len(need), "segments_fetched": fetched, "segments_cached": len(seg)}
