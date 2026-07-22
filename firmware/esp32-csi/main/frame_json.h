#pragma once

#include <stddef.h>
#include <stdint.h>

#include "csi_capture.h"

/**
 * Large enough for the fixed fields plus two CSI_MAX_SUBCARRIERS-length
 * float arrays at up to ~9 chars/value ("-123.456,") — see frame_json.c.
 * Callers own this buffer (typically a stack buffer in the on_frame
 * callback); no heap allocation here, since this runs at up to
 * PING_RATE_HZ (default 100) times a second.
 */
#define FRAME_JSON_MAX_LEN 8192

/**
 * Serializes `frame` to JSON matching docs/csi-frame-schema.md exactly
 * (field order: schema_version, timestamp_us, source_mac, rssi, channel,
 * subcarrier_count, amplitude, phase, sequence_number).
 *
 * Returns the number of bytes written (excluding the NUL terminator), or 0
 * if `out_len` was too small to fit the frame (should not happen at
 * CSI_MAX_SUBCARRIERS with FRAME_JSON_MAX_LEN, but checked defensively
 * since a truncated JSON frame is worse than a dropped one).
 */
size_t frame_json_build(const csi_frame_t *frame, uint32_t sequence_number, char *out, size_t out_len);
