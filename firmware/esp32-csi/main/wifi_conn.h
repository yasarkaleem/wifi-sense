#pragma once

#include <stdbool.h>
#include <stdint.h>

#include "esp_err.h"

/**
 * Connects to WiFi (STA mode) using credentials from nvs_config (NVS
 * override, else Kconfig CONFIG_WIFI_SSID/CONFIG_WIFI_PASSWORD), and
 * installs event handlers that automatically reconnect — with capped
 * exponential backoff — on every subsequent disconnect for as long as the
 * device is running. Blocks until the first connection succeeds or
 * `timeout_ms` elapses (0 = wait forever).
 *
 * Call once at startup, after nvs_flash_init().
 */
esp_err_t wifi_conn_init_and_connect(uint32_t timeout_ms);

/** True once we have an IP address (false while disconnected/reconnecting). */
bool wifi_conn_is_connected(void);

/**
 * The WiFi channel actually in use, read back from the AP we're
 * associated with (0 if not connected yet). Both firmware roles must be
 * configured onto the same channel for CSI capture to see the sender's
 * traffic — see README.md.
 */
uint8_t wifi_conn_get_channel(void);
