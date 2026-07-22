"""Tests for the ZeroMQ frame publisher."""

from __future__ import annotations

import json

import zmq

from ingest.publisher import TOPIC, ZMQFramePublisher


def test_publish_is_received_by_a_subscriber():
    publisher = ZMQFramePublisher("127.0.0.1", 0)  # port 0 -> OS picks a free port

    context = zmq.Context()
    subscriber = context.socket(zmq.SUB)
    subscriber.connect(publisher.endpoint)
    subscriber.setsockopt(zmq.SUBSCRIBE, TOPIC)
    subscriber.setsockopt(zmq.RCVTIMEO, 200)

    frame = {"sequence_number": 42, "amplitude": [1.0, 2.0]}

    # ZeroMQ PUB/SUB has a "slow joiner" delay: the subscription may not
    # have propagated to the publisher yet, so retry publishing the same
    # (idempotent) frame until the subscriber catches one.
    received = None
    for _ in range(50):
        publisher.publish(frame)
        try:
            topic, payload = subscriber.recv_multipart()
            received = (topic, json.loads(payload.decode("utf-8")))
            break
        except zmq.Again:
            continue

    subscriber.close()
    context.term()
    publisher.close()

    assert received is not None
    topic, decoded_frame = received
    assert topic == TOPIC
    assert decoded_frame == frame


def test_endpoint_reports_the_bound_port():
    publisher = ZMQFramePublisher("127.0.0.1", 0)
    try:
        assert publisher.endpoint.startswith("tcp://")
        assert not publisher.endpoint.endswith(":0")
    finally:
        publisher.close()
