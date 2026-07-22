"""Wires together the UDP receiver, ring buffer, publisher, and debug plot."""

from __future__ import annotations

import argparse
import logging
import sys
import threading

from ingest.buffer import RingBuffer
from ingest.publisher import ZMQFramePublisher
from ingest.receiver import UDPReceiver

logger = logging.getLogger("ingest.service")


def run_service(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )

    buffer = RingBuffer(capacity=args.buffer_size)
    publisher = ZMQFramePublisher(args.pub_host, args.pub_port) if args.pub_port else None
    receiver = UDPReceiver(args.udp_host, args.udp_port, buffer, publisher)

    logger.info("listening for CSI frames on udp://%s:%s", args.udp_host, args.udp_port)
    if publisher is not None:
        logger.info("publishing frames on tcp://%s:%s (topic=csi)", args.pub_host, args.pub_port)

    if args.plot == "web":
        from ingest.plot_web import start_web_plot_server

        server, _thread = start_web_plot_server(buffer, host=args.plot_host, port=args.plot_port)
        logger.info("debug waterfall at http://%s:%s/", args.plot_host, args.plot_port)
        try:
            receiver.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.shutdown()
            _shutdown(receiver, publisher)
        return

    if args.plot == "matplotlib":
        from ingest.plot_matplotlib import run_matplotlib_waterfall

        # matplotlib's GUI event loop wants the main thread, so the
        # receiver runs in the background here instead of the other way
        # around.
        receiver_thread = threading.Thread(target=receiver.serve_forever, daemon=True)
        receiver_thread.start()
        try:
            run_matplotlib_waterfall(buffer, refresh_hz=args.plot_refresh_hz)
        finally:
            _shutdown(receiver, publisher)
            receiver_thread.join(timeout=2)
        return

    try:
        receiver.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        _shutdown(receiver, publisher)


def _shutdown(receiver: UDPReceiver, publisher: ZMQFramePublisher | None) -> None:
    receiver.stop()
    receiver.close()
    if publisher is not None:
        publisher.close()
