from pathlib import Path

from src.realtime import ndtp

CAPTURE = Path(__file__).parent / "emulator_capture.bin"


def test_spec_example_coordinates():
    cell = ndtp.build_nav_cell(1767665400, 37.6173210, 55.7551234, speed=42, course=90)
    frame = ndtp.build_realtime(1166336, 5, cell)
    [f] = ndtp.FrameReader().feed(frame)
    [p] = ndtp.frame_nav_points(f)
    assert (p.unit_id, p.timestamp, p.speed, p.course, p.valid) == (1166336, 1767665400, 42, 90, True)
    assert abs(p.lon - 37.617321) < 1e-7 and abs(p.lat - 55.7551234) < 1e-7


def test_live_emulator_capture():
    r = ndtp.FrameReader()
    frames = r.feed(CAPTURE.read_bytes())
    assert r.crc_errors == 0 and len(frames) >= 2
    assert (frames[0].service, frames[0].type) == (ndtp.SERVICE_GENERIC, ndtp.TYPE_CONN_REQUEST)
    cells = ndtp.parse_cells(frames[1].body)
    assert [c[0] for c in cells] == [0, 8, 16, 2, 10]  # Nav первой, остальные из autoGenerate
    p = ndtp.frame_nav_points(frames[1])[0]
    assert 55 < p.lat < 56 and 37 < p.lon < 38 and p.valid


def test_stream_split_at_every_byte_and_garbage():
    data = CAPTURE.read_bytes()
    r = ndtp.FrameReader()
    frames = r.feed(b"\x00\x01garbage~")
    for b in data:  # по одному байту
        frames += r.feed(bytes([b]))
    assert len(frames) == len(ndtp.FrameReader().feed(data))


def test_crc_corruption_is_rejected():
    frame = bytearray(ndtp.build_realtime(1, 1, ndtp.build_nav_cell(0, 37.5, 55.7)))
    frame[-3] ^= 0xFF
    r = ndtp.FrameReader()
    assert r.feed(bytes(frame)) == [] and r.crc_errors == 1


def test_southern_western_hemisphere():
    [f] = ndtp.FrameReader().feed(ndtp.build_realtime(1, 1, ndtp.build_nav_cell(0, -70.5, -33.4, valid=False)))
    p = ndtp.frame_nav_points(f)[0]
    assert p.lon < 0 and p.lat < 0 and not p.valid
