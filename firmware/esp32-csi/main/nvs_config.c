#include "nvs_config.h"

#include <stdio.h>
#include <string.h>

#include "argtable3/argtable3.h"
#include "esp_console.h"
#include "esp_log.h"
#include "esp_system.h"
#include "nvs.h"
#include "sdkconfig.h"

static const char *TAG = "nvs_config";
#define NVS_NAMESPACE "csi_cfg"

void nvs_config_get_wifi(char *ssid, size_t ssid_len, char *password, size_t password_len)
{
    strlcpy(ssid, CONFIG_WIFI_SSID, ssid_len);
    strlcpy(password, CONFIG_WIFI_PASSWORD, password_len);

    nvs_handle_t handle;
    if (nvs_open(NVS_NAMESPACE, NVS_READONLY, &handle) != ESP_OK) {
        return; // never provisioned — Kconfig defaults stand
    }

    size_t len = ssid_len;
    nvs_get_str(handle, "ssid", ssid, &len);
    len = password_len;
    nvs_get_str(handle, "password", password, &len);

    nvs_close(handle);
}

void nvs_config_get_target(char *host, size_t host_len, uint16_t *port)
{
    strlcpy(host, CONFIG_CSI_TARGET_HOST, host_len);
    *port = CONFIG_CSI_TARGET_PORT;

    nvs_handle_t handle;
    if (nvs_open(NVS_NAMESPACE, NVS_READONLY, &handle) != ESP_OK) {
        return;
    }

    size_t len = host_len;
    nvs_get_str(handle, "host", host, &len);

    uint16_t stored_port;
    if (nvs_get_u16(handle, "port", &stored_port) == ESP_OK) {
        *port = stored_port;
    }

    nvs_close(handle);
}

esp_err_t nvs_config_set_wifi(const char *ssid, const char *password)
{
    nvs_handle_t handle;
    esp_err_t err = nvs_open(NVS_NAMESPACE, NVS_READWRITE, &handle);
    if (err != ESP_OK) {
        return err;
    }

    err = nvs_set_str(handle, "ssid", ssid);
    if (err == ESP_OK) {
        err = nvs_set_str(handle, "password", password);
    }
    if (err == ESP_OK) {
        err = nvs_commit(handle);
    }

    nvs_close(handle);
    return err;
}

esp_err_t nvs_config_set_target(const char *host, uint16_t port)
{
    nvs_handle_t handle;
    esp_err_t err = nvs_open(NVS_NAMESPACE, NVS_READWRITE, &handle);
    if (err != ESP_OK) {
        return err;
    }

    err = nvs_set_str(handle, "host", host);
    if (err == ESP_OK) {
        err = nvs_set_u16(handle, "port", port);
    }
    if (err == ESP_OK) {
        err = nvs_commit(handle);
    }

    nvs_close(handle);
    return err;
}

static struct {
    struct arg_str *ssid;
    struct arg_str *password;
    struct arg_end *end;
} set_wifi_args;

static int cmd_set_wifi(int argc, char **argv)
{
    int nerrors = arg_parse(argc, argv, (void **)&set_wifi_args);
    if (nerrors != 0) {
        arg_print_errors(stderr, set_wifi_args.end, argv[0]);
        return 1;
    }

    esp_err_t err = nvs_config_set_wifi(set_wifi_args.ssid->sval[0], set_wifi_args.password->sval[0]);
    if (err != ESP_OK) {
        printf("failed to save: %s\n", esp_err_to_name(err));
        return 1;
    }

    printf("saved. run 'restart' to reconnect with the new credentials.\n");
    return 0;
}

static struct {
    struct arg_str *host;
    struct arg_int *port;
    struct arg_end *end;
} set_target_args;

static int cmd_set_target(int argc, char **argv)
{
    int nerrors = arg_parse(argc, argv, (void **)&set_target_args);
    if (nerrors != 0) {
        arg_print_errors(stderr, set_target_args.end, argv[0]);
        return 1;
    }

    uint16_t port = (uint16_t)set_target_args.port->ival[0];
    esp_err_t err = nvs_config_set_target(set_target_args.host->sval[0], port);
    if (err != ESP_OK) {
        printf("failed to save: %s\n", esp_err_to_name(err));
        return 1;
    }

    printf("saved. run 'restart' to apply.\n");
    return 0;
}

static int cmd_restart(int argc, char **argv)
{
    printf("restarting...\n");
    esp_restart();
    return 0;
}

void nvs_config_register_console_commands(void)
{
    set_wifi_args.ssid = arg_str1(NULL, NULL, "<ssid>", "WiFi SSID");
    set_wifi_args.password = arg_str1(NULL, NULL, "<password>", "WiFi password (\"\" for open network)");
    set_wifi_args.end = arg_end(2);

    const esp_console_cmd_t set_wifi_cmd = {
        .command = "set_wifi",
        .help = "Save WiFi credentials to NVS (takes effect after 'restart')",
        .hint = NULL,
        .func = &cmd_set_wifi,
        .argtable = &set_wifi_args,
    };
    ESP_ERROR_CHECK(esp_console_cmd_register(&set_wifi_cmd));

    set_target_args.host = arg_str1(NULL, NULL, "<host>", "UDP target host (services/ingest)");
    set_target_args.port = arg_int1(NULL, NULL, "<port>", "UDP target port");
    set_target_args.end = arg_end(2);

    const esp_console_cmd_t set_target_cmd = {
        .command = "set_target",
        .help = "Save the UDP target host:port to NVS (receiver role, takes effect after 'restart')",
        .hint = NULL,
        .func = &cmd_set_target,
        .argtable = &set_target_args,
    };
    ESP_ERROR_CHECK(esp_console_cmd_register(&set_target_cmd));

    const esp_console_cmd_t restart_cmd = {
        .command = "restart",
        .help = "Restart the device",
        .hint = NULL,
        .func = &cmd_restart,
        .argtable = NULL,
    };
    ESP_ERROR_CHECK(esp_console_cmd_register(&restart_cmd));

    ESP_LOGI(TAG, "console commands registered: set_wifi, set_target, restart");
}
