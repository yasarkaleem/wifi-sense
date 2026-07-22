#pragma once

#include <stdbool.h>
#include <stdint.h>

#include "esp_err.h"

/**
 * Largest subcarrier-pair count across every CSI buffer layout the legacy
 * (ESP32/S2/S3/C3) CSI hardware can produce: HT40 + STBC yields a 612-byte
 * raw buffer (306 int8 imaginary/real pairs) with `ltf_merge_en` on — see
 * the "CSI buffer layout" note in README.md. Sized with a small margin.
 */
#define CSI_MAX_SUBCARRIERS 320

typedef struct {
    uint64_t timestamp_us;
    uint8_t mac[6];
    int8_t rssi;
    uint8_t channel;
    uint16_t subcarrier_count;
    float amplitude[CSI_MAX_SUBCARRIERS];
    float phase[CSI_MAX_SUBCARRIERS];
} csi_frame_t;

/**
 * Enables promiscuous mode + CSI capture and starts a background task that
 * decodes each raw CSI callback into a `csi_frame_t` and invokes
 * `on_frame` (called from the csi_process task, NOT from the WiFi driver's
 * callback context — safe to do UDP sends, JSON building, etc. from it).
 *
 * `expected_mac` (6 bytes, or NULL to accept CSI triggered by any WiFi
 * traffic on the channel) filters which source MAC's packets we bother
 * decoding — see CONFIG_CSI_SENDER_MAC in Kconfig.projbuild.
 *
 * Must be called AFTER WiFi has connected to the AP (wifi_conn_is_connected()
 * true) — CSI config calls are only meaningful once associated.
 */
esp_err_t csi_capture_start(const uint8_t *expected_mac, void (*on_frame)(const csi_frame_t *frame));

/** Frames the WiFi-driver callback produced faster than csi_process could decode. */
uint32_t csi_capture_get_dropped_count(void);
