"""Tests for the debug web waterfall server."""

from __future__ import annotations

import json
import urllib.error
import urllib.request

import pytest

from ingest.buffer import RingBuffer
from ingest.plot_web import start_web_plot_server


@pytest.fixture
def running_server():
    buffer = RingBuffer(capacity=5)
    server, thread = start_web_plot_server(buffer, host="127.0.0.1", port=0)
    port = server.server_address[1]
    yield buffer, f"http://127.0.0.1:{port}"
    server.shutdown()
    thread.join(timeout=2)


def test_index_page_served(running_server):
    _buffer, base_url = running_server
    with urllib.request.urlopen(f"{base_url}/") as resp:
        assert resp.status == 200
        assert "text/html" in resp.headers["Content-Type"]
        body = resp.read().decode("utf-8")
    assert "<canvas" in body


def test_data_json_reflects_buffer_contents(running_server):
    buffer, base_url = running_server
    buffer.append({"sequence_number": 1, "amplitude": [1.0, 2.0, 3.0]})
    buffer.append({"sequence_number": 2, "amplitude": [4.0, 5.0, 6.0]})

    with urllib.request.urlopen(f"{base_url}/data.json") as resp:
        assert resp.status == 200
        assert resp.headers["Content-Type"] == "application/json"
        data = json.loads(resp.read().decode("utf-8"))

    assert data["capacity"] == 5
    assert data["frame_count"] == 2
    assert data["subcarrier_count"] == 3
    assert data["amplitude_matrix"] == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    assert data["sequence_numbers"] == [1, 2]


def test_data_json_empty_buffer(running_server):
    _buffer, base_url = running_server
    with urllib.request.urlopen(f"{base_url}/data.json") as resp:
        data = json.loads(resp.read().decode("utf-8"))

    assert data["frame_count"] == 0
    assert data["amplitude_matrix"] == []
    assert data["subcarrier_count"] == 0


def test_unknown_path_is_404(running_server):
    _buffer, base_url = running_server
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(f"{base_url}/nope")
    assert exc_info.value.code == 404
