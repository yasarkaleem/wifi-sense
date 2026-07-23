"""Wires together: subscribe to ingest's CSI stream, preprocess each
window, run the presence detector (always), the ML people counter (if a
checkpoint is configured), and the zone localizer (if a checkpoint is
configured), publish detections, and log transitions."""

from __future__ import annotations

import argparse
import json
import logging
import sys

import numpy as np
import zmq

from pipeline.detectors.presence import PresenceDetector
from pipeline.preprocess import hampel_filter, savitzky_golay_smooth
from pipeline.publisher import ZMQEventPublisher
from pipeline.smoothing import ZoneEMASmoother
from pipeline.windowing import RollingWindower

logger = logging.getLogger("pipeline.service")

CSI_TOPIC = b"csi"
PRESENCE_TOPIC = b"presence"
COUNT_TOPIC = b"count"
ZONES_TOPIC = b"zones"


def _load_counter(args: argparse.Namespace):
    # Local import: torch (the "ml" extra) is only required when a
    # checkpoint is actually configured, keeping presence-only deployments
    # free of that dependency.
    from pipeline.models.counter_inference import CounterInference

    counter = CounterInference(
        args.counter_checkpoint,
        n_components=args.counter_n_components,
        sample_rate_hz=args.sample_rate_hz,
        nperseg=args.counter_nperseg,
        noverlap=args.counter_noverlap,
    )
    logger.info("loaded people-counter checkpoint from %s", args.counter_checkpoint)
    return counter


def _load_localizer(args: argparse.Namespace):
    # Local import: scikit-learn/joblib (the "localize" extra) are only
    # required when a checkpoint is actually configured.
    from pipeline.models.localizer import ZoneLocalizer

    localizer = ZoneLocalizer.load(args.localizer_checkpoint)
    logger.info(
        "loaded localizer checkpoint from %s (zones: %s)", args.localizer_checkpoint, localizer.zone_ids
    )
    return localizer


def run_service(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )

    context = zmq.Context()
    sub = context.socket(zmq.SUB)
    sub.connect(f"tcp://{args.sub_host}:{args.sub_port}")
    sub.setsockopt(zmq.SUBSCRIBE, CSI_TOPIC)
    sub.setsockopt(zmq.RCVTIMEO, 500)

    publisher = ZMQEventPublisher(args.pub_host, args.pub_port)

    logger.info("subscribing to CSI frames on tcp://%s:%s", args.sub_host, args.sub_port)
    logger.info(
        "publishing presence events on tcp://%s:%s (topic=presence)", args.pub_host, args.pub_port
    )

    window_size = max(1, round(args.window_s * args.sample_rate_hz))
    stride_frames = max(1, round(args.stride_s * args.sample_rate_hz))
    windower = RollingWindower(window_size=window_size, stride_frames=stride_frames)
    detector = PresenceDetector(
        calibration_s=args.calibration_s,
        n_sigmas=args.n_sigmas,
        n_components=args.pca_components,
    )
    logger.info(
        "window_size=%d frames (%.2fs) stride=%d frames (%.2fs) calibration=%.1fs",
        window_size,
        args.window_s,
        stride_frames,
        args.stride_s,
        args.calibration_s,
    )

    counter = None
    if args.counter_checkpoint:
        counter = _load_counter(args)
        logger.info(
            "publishing count events on tcp://%s:%s (topic=count)", args.pub_host, args.pub_port
        )
    else:
        logger.info(
            "people counter: not loaded (set --counter-checkpoint / PIPELINE_COUNTER_CHECKPOINT to enable)"
        )

    localizer = None
    zone_smoother = None
    if args.localizer_checkpoint:
        localizer = _load_localizer(args)
        zone_smoother = ZoneEMASmoother(span=3)
        logger.info(
            "publishing zone events on tcp://%s:%s (topic=zones)", args.pub_host, args.pub_port
        )
    else:
        logger.info(
            "zone localizer: not loaded (set --localizer-checkpoint / PIPELINE_LOCALIZER_CHECKPOINT to enable)"
        )

    logger.info(
        "startup summary: presence=always-on counter=%s localizer=%s",
        f"loaded ({args.counter_checkpoint})" if counter else "not loaded",
        f"loaded ({args.localizer_checkpoint}, zones={localizer.zone_ids})" if localizer else "not loaded",
    )

    last_presence: bool | None = None
    last_count: int | None = None
    last_zone: str | None = None

    try:
        while True:
            try:
                _topic, payload = sub.recv_multipart()
            except zmq.Again:
                continue

            frame = json.loads(payload.decode("utf-8"))
            ready = windower.add(np.array(frame["amplitude"], dtype=np.float64), frame["timestamp_us"])
            if ready is None:
                continue

            raw_window, timestamp_us = ready
            cleaned = savitzky_golay_smooth(
                hampel_filter(raw_window, window_size=args.hampel_window, n_sigmas=args.hampel_sigmas),
                window_length=args.savgol_window,
                polyorder=args.savgol_polyorder,
            )

            presence_event = detector.update(cleaned, timestamp_us)
            if presence_event is not None:
                publisher.publish(presence_event.to_dict(), topic=PRESENCE_TOPIC)
                if presence_event.presence != last_presence:
                    logger.info(
                        "presence -> %s (motion_intensity=%.2f, threshold=%.4g)",
                        presence_event.presence,
                        presence_event.motion_intensity,
                        detector.threshold,
                    )
                    last_presence = presence_event.presence

            if counter is not None:
                count_event = counter.predict(cleaned, timestamp_us)
                publisher.publish(count_event.to_dict(), topic=COUNT_TOPIC)
                if count_event.count != last_count:
                    logger.info(
                        "count -> %d (confidence=%.2f)", count_event.count, count_event.confidence
                    )
                    last_count = count_event.count

            if localizer is not None:
                from pipeline.models.localizer import predict_from_window

                zone_event = predict_from_window(
                    localizer,
                    cleaned,
                    timestamp_us,
                    n_components=args.localizer_n_components,
                    sample_rate_hz=args.sample_rate_hz,
                    nperseg=args.localizer_nperseg,
                    noverlap=args.localizer_noverlap,
                )
                zone_event = zone_smoother.update(zone_event)
                publisher.publish(zone_event.to_dict(), topic=ZONES_TOPIC)
                best = zone_event.best_zone
                if best.zone_id != last_zone:
                    logger.info(
                        "zone -> %s (occupancy_probability=%.2f)", best.zone_id, best.occupancy_probability
                    )
                    last_zone = best.zone_id
    except KeyboardInterrupt:
        pass
    finally:
        sub.close()
        publisher.close()
        context.term()
