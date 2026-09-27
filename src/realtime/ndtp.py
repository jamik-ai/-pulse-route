"""Кодек протокола NDTP (подмножество, достаточное для эмулятора).

Кадр: [NPL 15 байт][NPH 10 байт][тело], всё little-endian.
Проверено на живом потоке ndtp-telemetry-emulator:1.0 (tests/test_ndtp.py).
"""
import struct
from dataclasses import dataclass

NPL = struct.Struct("<HHHHBIH")   # signature, dataSize, flags, crc(swapped), type, peerAddress, requestId
NPH = struct.Struct("<HHHI")      # serviceId, type, flags, requestId
NAV = struct.Struct("<IIIBBHHHHHBB")  # G6CellNav00, 26 байт
SIGNATURE = 0x7E7E
NPL_TYPE_NPH = 0x02

SERVICE_GENERIC, TYPE_CONN_REQUEST = 0, 100
SERVICE_NAVDATA, TYPE_REALTIME = 1, 101
TYPE_RESULT = 0  # NPH_RESULT — ответ сервера

# размеры payload известных ячеек (без 2 байт [type][number])
CELL_SIZES = {0: 26, 2: 26, 8: 6, 10: 37, 15: 50, 16: 8}
MAX_DATA_SIZE = 65535


class NdtpError(ValueError):
    pass


def crc16_modbus(data: bytes) -> int:
    crc = 0xFFFF
    for b in data:
        crc ^= b
        for _ in range(8):
            crc = (crc >> 1) ^ 0xA001 if crc & 1 else crc >> 1
    return crc


def _swap16(x: int) -> int:
    return ((x & 0xFF) << 8) | (x >> 8)


@dataclass
class Frame:
    peer: int            # unitId
    service: int
    type: int
    request_id: int
    body: bytes


@dataclass
class NavPoint:
    unit_id: int
    timestamp: int       # unix-сек
    lon: float
    lat: float
    valid: bool
    speed: float         # км/ч (speedAvg)
    speed_max: float
    course: int
    altitude: int
    nsat: int


def parse_nav(unit_id: int, payload: bytes) -> NavPoint:
    ts, lon, lat, bits, _bat, spd, spd_max, course, _track, alt, nsat, _pdop = NAV.unpack(payload)
    lat_sign = 1 if bits & (1 << 5) else -1
    lon_sign = 1 if bits & (1 << 6) else -1
    return NavPoint(unit_id, ts, lon_sign * lon / 1e7, lat_sign * lat / 1e7, bool(bits & (1 << 7)),
                    float(spd), float(spd_max), course, alt, nsat)


def parse_cells(body: bytes) -> list[tuple[int, int, bytes]]:
    """Тело realtime-пакета → [(type, number, payload)]. На неизвестном типе разбор прекращается."""
    cells, i = [], 0
    while i + 2 <= len(body):
        ctype, num = body[i], body[i + 1]
        size = CELL_SIZES.get(ctype)
        if size is None or i + 2 + size > len(body):
            break  # длину неизвестной ячейки не знаем — дальше разбирать нельзя
        cells.append((ctype, num, body[i + 2:i + 2 + size]))
        i += 2 + size
    return cells


class FrameReader:
    """Потоковый разбор: TCP может резать и склеивать кадры как угодно."""

    def __init__(self, check_crc: bool = True):
        self.buf = bytearray()
        self.check_crc = check_crc
        self.crc_errors = 0
        self.resyncs = 0

    def feed(self, data: bytes) -> list[Frame]:
        self.buf += data
        out = []
        while True:
            start = self.buf.find(b"\x7e\x7e")
            if start < 0:
                del self.buf[:max(len(self.buf) - 1, 0)]
                return out
            if start:
                self.resyncs += 1
                del self.buf[:start]
            if len(self.buf) < NPL.size:
                return out
            sig, size, _flags, crc, ntype, peer, _rid = NPL.unpack_from(self.buf)
            if ntype != NPL_TYPE_NPH or size < NPH.size:
                del self.buf[:1]  # ложная сигнатура: сдвиг на 1 байт (кадр мог начаться с «~» мусора)
                self.resyncs += 1
                continue
            end = NPL.size + size
            if len(self.buf) < end:
                return out
            data_ = bytes(self.buf[NPL.size:end])
            if self.check_crc and crc16_modbus(data_) != _swap16(crc):
                self.crc_errors += 1
                del self.buf[:1]  # не доверяем заголовку — ищем следующую сигнатуру
                continue
            del self.buf[:end]
            service, ptype, _pflags, req = NPH.unpack_from(data_)
            out.append(Frame(peer, service, ptype, req, data_[NPH.size:]))


def frame_nav_points(frame: Frame) -> list[NavPoint]:
    if (frame.service, frame.type) != (SERVICE_NAVDATA, TYPE_REALTIME):
        return []
    return [parse_nav(frame.peer, p) for t, _n, p in parse_cells(frame.body) if t == 0]


# --- сборка пакетов (для replay-клиента, ответов сервера и тестов) --------------

def build_frame(peer: int, service: int, ptype: int, request_id: int, body: bytes, flags: int = 1) -> bytes:
    data = NPH.pack(service, ptype, flags, request_id) + body
    return NPL.pack(SIGNATURE, len(data), 0, _swap16(crc16_modbus(data)), NPL_TYPE_NPH, peer, 0) + data


def build_handshake(unit_id: int, request_id: int = 1) -> bytes:
    body = struct.pack("<HHHIII", 6, 2, 0, unit_id, 65535, 0)
    return build_frame(unit_id, SERVICE_GENERIC, TYPE_CONN_REQUEST, request_id, body)


def build_nav_cell(ts: int, lon: float, lat: float, valid: bool = True, speed: float = 0,
                   course: int = 0, altitude: int = 0, nsat: int = 10) -> bytes:
    bits = (1 << 5) * (lat >= 0) | (1 << 6) * (lon >= 0) | (1 << 7) * bool(valid)
    payload = NAV.pack(int(ts), round(abs(lon) * 1e7), round(abs(lat) * 1e7), bits, 200,
                       int(speed), int(speed), int(course) % 361, 0, int(altitude) & 0xFFFF, nsat, 1)
    return bytes([0, 0]) + payload


def build_realtime(unit_id: int, request_id: int, nav_cell: bytes) -> bytes:
    return build_frame(unit_id, SERVICE_NAVDATA, TYPE_REALTIME, request_id, nav_cell)


def build_result(unit_id: int, request_id: int, service: int, error: int = 0) -> bytes:
    """Подтверждение приёма (NPH_RESULT). Эмулятор ответы не разбирает, но реальные терминалы ждут."""
    return build_frame(unit_id, service, TYPE_RESULT, request_id, struct.pack("<I", error), flags=0)
