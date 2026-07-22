"""Unit and integration tests for the UDP receiver."""

from __future__ import annotations

import json
import socket
import threading
import time

from ingest.buffer import RingBuffer
from ingest.receiver import UDPReceiver


class FakePublisher:
    def __init__(self) -> None:
        self.published: list[dict] = []

    def publish(self, frame: dict) -> None:
        self.published.append(frame)


def make_frame(**overrides) -> dict:
    frame = {
        "schema_version": 1,
        "timestamp_us": 1_700_000_000_000_000,
        "source_mac": "24:6F:28:AB:CD:EF",
        "rssi": -52,
        "channel": 6,
        "subcarrier_count": 2,
        "amplitude": [12.3, 11.9],
        "phase": [0.12, -0.45],
        "sequence_number": 0,
    }
    frame.update(overrides)
    return frame


def make_receiver(publisher=None) -> UDPReceiver:
    # port=0 binds an ephemeral free port; these tests exercise
    # handle_datagram() directly, not the network socket.
    return UDPReceiver("127.0.0.1", 0, RingBuffer(capacity=10), publisher)


def test_valid_datagram_is_buffered_and_published():
    publisher = FakePublisher()
    receiver = make_receiver(publisher)
    frame = make_frame()

    accepted = receiver.handle_datagram(json.dumps(frame).encode("utf-8"))

    assert accepted is True
    assert receiver.stats.received == 1
    assert receiver.stats.dropped == 0
    assert publisher.published == [frame]
    receiver.close()


def test_malformed_json_is_dropped():
    receiver = make_receiver()
    accepted = receiver.handle_datagram(b"not json{{{")

    assert accepted is False
    assert receiver.stats.dropped == 1
    assert receiver.stats.received == 0
    assert len(receiver._buffer) == 0
    receiver.close()


def test_schema_invalid_frame_is_dropped():
    receiver = make_receiver()
    bad_frame = make_frame(source_mac="not-a-mac")
    accepted = receiver.handle_datagram(json.dumps(bad_frame).encode("utf-8"))

    assert accepted is False
    assert receiver.stats.dropped == 1
    receiver.close()


def test_length_mismatch_frame_is_dropped():
    receiver = make_receiver()
    bad_frame = make_frame(amplitude=[1.0])  # subcarrier_count says 2
    accepted = receiver.handle_datagram(json.dumps(bad_frame).encode("utf-8"))

    assert accepted is False
    assert receiver.stats.dropped == 1
    receiver.close()


def test_no_publisher_is_fine():
    receiver = make_receiver(publisher=None)
    accepted = receiver.handle_datagram(json.dumps(make_frame()).encode("utf-8"))
    assert accepted is True
    receiver.close()


def test_serve_forever_receives_real_udp_datagrams():
    buffer = RingBuffer(capacity=10)
    receiver = UDPReceiver("127.0.0.1", 0, buffer, publisher=None, socket_timeout_s=0.1)
    port = receiver._sock.getsockname()[1]

    thread = threading.Thread(target=receiver.serve_forever, daemon=True)
    thread.start()

    sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    frame = make_frame(sequence_number=7)
    sender.sendto(json.dumps(frame).encode("utf-8"), ("127.0.0.1", port))
    sender.close()

    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and len(buffer) == 0:
        time.sleep(0.02)

    receiver.stop()
    thread.join(timeout=2)
    receiver.close()

    snapshot = buffer.snapshot()
    assert len(snapshot) == 1
    assert snapshot[0]["sequence_number"] == 7
