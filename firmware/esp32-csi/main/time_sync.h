#pragma once

#include <stdbool.h>
#include <stdint.h>

/**
 * docs/csi-frame-schema.md's `timestamp_us` is epoch microseconds (matching
 * services/replay, which uses `time.time_ns()`), but the ESP32 has no
 * battery-backed RTC — system time starts at the 1970 epoch on every boot.
 * This starts an SNTP client to correct it; csi_capture.c reads the result
 * via `gettimeofday()` once synced.
 *
 * Call once after WiFi connects.
 */
void time_sync_start(void);

/** Blocks (up to timeout_ms) until the first SNTP sync completes, or returns false. */
bool time_sync_wait(uint32_t timeout_ms);
