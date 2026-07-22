"""UDP receiver: parses, validates, buffers, and republishes CSI frames."""

from __future__ import annotations

import json
import logging
import socket
import threading
from dataclasses import dataclass

import jsonschema

from ingest.buffer import RingBuffer
from ingest.publisher import ZMQFramePublisher
from ingest.schema import validate_csi_frame

logger = logging.getLogger("ingest.receiver")


@dataclass
class ReceiverStats:
    received: int = 0
    dropped: int = 0


class UDPReceiver:
    """Binds a UDP socket and routes valid CSI frames to a buffer and an
    optional publisher; invalid datagrams are logged and dropped."""

    def __init__(
        self,
        host: str,
        port: int,
        buffer: RingBuffer,
        publisher: ZMQFramePublisher | None = None,
        *,
        socket_timeout_s: float = 0.5,
    ) -> None:
        self._buffer = buffer
        self._publisher = publisher
        self.stats = ReceiverStats()
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind((host, port))
        self._sock.settimeout(socket_timeout_s)
        self._stop = threading.Event()

    def handle_datagram(self, payload: bytes) -> bool:
        """Parse, validate, and route a single UDP payload.

        Returns True if the frame was accepted (buffered + published).
        """
        try:
            frame = json.loads(payload.decode("utf-8"))
            validate_csi_frame(frame)
        except (
            UnicodeDecodeError,
            json.JSONDecodeError,
            jsonschema.ValidationError,
            ValueError,
        ) as exc:
            self.stats.dropped += 1
            logger.warning("dropping invalid CSI frame: %s", exc)
            return False

        self.stats.received += 1
        self._buffer.append(frame)
        if self._publisher is not None:
            self._publisher.publish(frame)
        return True

    def serve_forever(self) -> None:
        """Receive datagrams until `stop()` is called."""
        while not self._stop.is_set():
            try:
                payload, _addr = self._sock.recvfrom(65536)
            except socket.timeout:
                continue
            except OSError:
                break
            self.handle_datagram(payload)

    def stop(self) -> None:
        self._stop.set()

    def close(self) -> None:
        self._sock.close()
