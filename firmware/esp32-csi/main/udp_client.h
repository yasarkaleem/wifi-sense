#pragma once

#include <stddef.h>
#include <stdbool.h>
#include <stdint.h>

#include "esp_err.h"

typedef struct {
    int sock;
} udp_client_t;

/**
 * Opens a UDP socket. `broadcast` must be true to send to a broadcast
 * address (e.g. "255.255.255.255") later — used by ping_sender.c, not by
 * the receiver's frame_json sends to a specific host.
 */
esp_err_t udp_client_open(udp_client_t *client, bool broadcast);

/** Sends one UDP datagram to `host` (IPv4 dotted-decimal, e.g. from nvs_config or "255.255.255.255") : `port`. */
esp_err_t udp_client_send_to(udp_client_t *client, const char *host, uint16_t port, const void *data, size_t len);

void udp_client_close(udp_client_t *client);
