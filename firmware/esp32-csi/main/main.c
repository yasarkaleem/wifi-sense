#include <stdio.h>
#include <string.h>

#include "esp_console.h"
#include "esp_log.h"
#include "nvs_flash.h"
#include "sdkconfig.h"

#include "csi_capture.h"
#include "frame_json.h"
#include "nvs_config.h"
#include "ping_sender.h"
#include "time_sync.h"
#include "udp_client.h"
#include "wifi_conn.h"

static const char *TAG = "main";

#if CONFIG_CSI_ROLE_RECEIVER

static udp_client_t s_target_client;
static char s_target_host[64];
static uint16_t s_target_port;
static uint32_t s_sequence_number;
static char s_json_buf[FRAME_JSON_MAX_LEN]; // static: too large for the on_frame callback's stack

static void on_csi_frame(const csi_frame_t *frame)
{
    size_t len = frame_json_build(frame, s_sequence_number, s_json_buf, sizeof(s_json_buf));
    if (len == 0) {
        ESP_LOGW(TAG, "frame_json_build failed (subcarrier_count=%u), dropping frame", frame->subcarrier_count);
        return;
    }

    udp_client_send_to(&s_target_client, s_target_host, s_target_port, s_json_buf, len);
    s_sequence_number++;
}

static bool parse_mac(const char *str, uint8_t *mac_out)
{
    if (!str || strlen(str) == 0) {
        return false;
    }
    unsigned int b[6];
    if (sscanf(str, "%x:%x:%x:%x:%x:%x", &b[0], &b[1], &b[2], &b[3], &b[4], &b[5]) != 6) {
        ESP_LOGW(TAG, "CONFIG_CSI_SENDER_MAC '%s' is not a valid xx:xx:xx:xx:xx:xx address, ignoring", str);
        return false;
    }
    for (int i = 0; i < 6; i++) {
        mac_out[i] = (uint8_t)b[i];
    }
    return true;
}

static void start_receiver(void)
{
    nvs_config_get_target(s_target_host, sizeof(s_target_host), &s_target_port);
    ESP_ERROR_CHECK(udp_client_open(&s_target_client, false));
    ESP_LOGI(TAG, "receiver: sending CSI frames to %s:%u", s_target_host, s_target_port);

    time_sync_start();
    if (!time_sync_wait(10000)) {
        ESP_LOGW(TAG, "SNTP sync did not complete in 10s — timestamp_us will read near-epoch-0 until it does");
    }

    uint8_t expected_mac[6];
    bool have_mac = parse_mac(CONFIG_CSI_SENDER_MAC, expected_mac);

    ESP_ERROR_CHECK(csi_capture_start(have_mac ? expected_mac : NULL, on_csi_frame));
}

#endif // CONFIG_CSI_ROLE_RECEIVER

static void start_console(void)
{
    esp_console_repl_t *repl = NULL;
    esp_console_repl_config_t repl_config = ESP_CONSOLE_REPL_CONFIG_DEFAULT();
    repl_config.prompt = "csi>";

    // Assumes the default UART console (true for every board this firmware
    // targets). If your sdkconfig switches CONFIG_ESP_CONSOLE to USB-CDC/JTAG,
    // swap this for esp_console_new_repl_usb_cdc()/_usb_serial_jtag().
    esp_console_dev_uart_config_t uart_config = ESP_CONSOLE_DEV_UART_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_console_new_repl_uart(&uart_config, &repl_config, &repl));

    esp_console_register_help_command();
    nvs_config_register_console_commands();
    ESP_ERROR_CHECK(esp_console_start_repl(repl));
}

void app_main(void)
{
    esp_err_t nvs_err = nvs_flash_init();
    if (nvs_err == ESP_ERR_NVS_NO_FREE_PAGES || nvs_err == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        nvs_err = nvs_flash_init();
    }
    ESP_ERROR_CHECK(nvs_err);

    start_console();

    ESP_LOGI(TAG, "connecting to WiFi...");
    ESP_ERROR_CHECK(wifi_conn_init_and_connect(0)); // 0 = wait forever for the first connection
    ESP_LOGI(TAG, "connected, channel %u", wifi_conn_get_channel());

#if CONFIG_CSI_ROLE_RECEIVER
    start_receiver();
#elif CONFIG_CSI_ROLE_SENDER
    ESP_ERROR_CHECK(ping_sender_start());
    ESP_LOGI(TAG, "sender: broadcasting pings at %dHz", CONFIG_PING_RATE_HZ);
#endif
}
