#pragma once

#include "esp_err.h"

/**
 * Starts a background task that broadcasts a small UDP packet at
 * CONFIG_PING_RATE_HZ to 255.255.255.255:CONFIG_PING_BROADCAST_PORT,
 * controlling the CSI sample rate the receiver observes. Sender role only
 * — see README.md. Call after WiFi has connected.
 */
esp_err_t ping_sender_start(void);
