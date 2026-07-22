#pragma once

#include <stdint.h>

#include "esp_err.h"

/**
 * Runtime configuration store (WiFi credentials, UDP target host/port) with
 * NVS as the source of truth and Kconfig values (CONFIG_WIFI_SSID,
 * CONFIG_WIFI_PASSWORD, CONFIG_CSI_TARGET_HOST, CONFIG_CSI_TARGET_PORT) as
 * the fallback when NVS has never been provisioned. See the
 * "set_wifi"/"set_target" console commands below and the "NVS provisioning"
 * section in README.md.
 */

/** Reads WiFi credentials, NVS override else Kconfig default. Always NUL-terminates. */
void nvs_config_get_wifi(char *ssid, size_t ssid_len, char *password, size_t password_len);

/** Reads the UDP target (receiver role only), NVS override else Kconfig default. */
void nvs_config_get_target(char *host, size_t host_len, uint16_t *port);

/** Persists WiFi credentials to NVS; takes effect after a restart. */
esp_err_t nvs_config_set_wifi(const char *ssid, const char *password);

/** Persists the UDP target to NVS; takes effect after a restart. */
esp_err_t nvs_config_set_target(const char *host, uint16_t port);

/**
 * Registers "set_wifi <ssid> <password>", "set_target <host> <port>", and
 * "restart" with the esp_console REPL started in main.c. Call once after
 * esp_console_new_repl_uart().
 */
void nvs_config_register_console_commands(void);
