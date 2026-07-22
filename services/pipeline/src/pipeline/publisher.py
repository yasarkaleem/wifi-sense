"""ZeroMQ PUB socket that publishes pipeline events.

Downstream services (e.g. services/api) connect a SUB socket to
tcp://<pub-host>:<pub-port> and subscribe to the topic they care about —
`presence` for PresenceEvent, `count` for CountEvent — to receive that
event type in real time. Both share one socket/port; ZeroMQ PUB/SUB
handles per-topic filtering on the subscriber side.
"""

from __future__ import annotations

import json

import zmq

TOPIC = b"presence"


class ZMQEventPublisher:
    """Publishes dict events as JSON over a ZeroMQ PUB socket."""

    def __init__(self, host: str, port: int, *, topic: bytes = TOPIC) -> None:
        self._default_topic = topic
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.PUB)
        self._socket.bind(f"tcp://{host}:{port}")

    @property
    def endpoint(self) -> str:
        return self._socket.getsockopt(zmq.LAST_ENDPOINT).decode("utf-8")

    def publish(self, event: dict, *, topic: bytes | None = None) -> None:
        self._socket.send_multipart([topic or self._default_topic, json.dumps(event).encode("utf-8")])

    def close(self) -> None:
        self._socket.close()
        self._context.term()
