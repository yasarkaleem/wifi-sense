"""ZeroMQ PUB socket that republishes validated CSI frames to subscribers.

Downstream services (e.g. services/pipeline) connect a SUB socket to
tcp://<pub-host>:<pub-port> and subscribe to the `csi` topic to receive
every accepted frame in real time.
"""

from __future__ import annotations

import json

import zmq

TOPIC = b"csi"


class ZMQFramePublisher:
    """Publishes CSI frame dicts as JSON over a ZeroMQ PUB socket."""

    def __init__(self, host: str, port: int, *, topic: bytes = TOPIC) -> None:
        self._topic = topic
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.PUB)
        self._socket.bind(f"tcp://{host}:{port}")

    @property
    def endpoint(self) -> str:
        return self._socket.getsockopt(zmq.LAST_ENDPOINT).decode("utf-8")

    def publish(self, frame: dict) -> None:
        payload = json.dumps(frame).encode("utf-8")
        self._socket.send_multipart([self._topic, payload])

    def close(self) -> None:
        self._socket.close()
        self._context.term()
