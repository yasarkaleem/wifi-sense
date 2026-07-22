/**
 * WiFi STA connect + automatic reconnect. Standard ESP-IDF event-driven
 * connection pattern (WIFI_EVENT_STA_START -> esp_wifi_connect(),
 * WIFI_EVENT_STA_DISCONNECTED -> retry, IP_EVENT_STA_GOT_IP -> done) with
 * a capped exponential backoff added on top so a persistently-unreachable
 * AP doesn't spin the radio at full tilt forever — see the "Handling
 * reconnects" section in README.md.
 */
#include "wifi_conn.h"

#include <string.h>

#include "esp_event.h"
#include "esp_log.h"
#include "esp_netif.h"
#include "esp_wifi.h"
#include "freertos/FreeRTOS.h"
#include "freertos/event_groups.h"
#include "freertos/task.h"
#include "sdkconfig.h"

#include "nvs_config.h"

static const char *TAG = "wifi_conn";

#define WIFI_CONNECTED_BIT BIT0
#define RECONNECT_BASE_DELAY_MS 1000

static EventGroupHandle_t s_wifi_event_group;
static uint32_t s_reconnect_delay_ms;
static uint32_t s_disconnect_count;

static void event_handler(void *arg, esp_event_base_t event_base, int32_t event_id, void *event_data)
{
    if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_START) {
        esp_wifi_connect();
    } else if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_DISCONNECTED) {
        xEventGroupClearBits(s_wifi_event_group, WIFI_CONNECTED_BIT);
        s_disconnect_count++;

        s_reconnect_delay_ms = (s_reconnect_delay_ms == 0) ? RECONNECT_BASE_DELAY_MS : s_reconnect_delay_ms * 2;
        if (s_reconnect_delay_ms > CONFIG_WIFI_MAX_RECONNECT_DELAY_MS) {
            s_reconnect_delay_ms = CONFIG_WIFI_MAX_RECONNECT_DELAY_MS;
        }

        ESP_LOGW(TAG, "disconnected (count=%lu), retrying in %lums", (unsigned long)s_disconnect_count,
                 (unsigned long)s_reconnect_delay_ms);
        vTaskDelay(pdMS_TO_TICKS(s_reconnect_delay_ms));
        esp_wifi_connect();
    } else if (event_base == IP_EVENT && event_id == IP_EVENT_STA_GOT_IP) {
        ip_event_got_ip_t *event = (ip_event_got_ip_t *)event_data;
        ESP_LOGI(TAG, "got IP: " IPSTR, IP2STR(&event->ip_info.ip));
        s_reconnect_delay_ms = 0; // back off resets after a successful connection
        xEventGroupSetBits(s_wifi_event_group, WIFI_CONNECTED_BIT);
    }
}

esp_err_t wifi_conn_init_and_connect(uint32_t timeout_ms)
{
    s_wifi_event_group = xEventGroupCreate();
    if (!s_wifi_event_group) {
        return ESP_ERR_NO_MEM;
    }

    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    esp_netif_create_default_wifi_sta();

    wifi_init_config_t init_cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&init_cfg));

    ESP_ERROR_CHECK(esp_event_handler_register(WIFI_EVENT, ESP_EVENT_ANY_ID, &event_handler, NULL));
    ESP_ERROR_CHECK(esp_event_handler_register(IP_EVENT, IP_EVENT_STA_GOT_IP, &event_handler, NULL));

    char ssid[33] = {0};
    char password[65] = {0};
    nvs_config_get_wifi(ssid, sizeof(ssid), password, sizeof(password));

    wifi_config_t wifi_config = {0};
    strlcpy((char *)wifi_config.sta.ssid, ssid, sizeof(wifi_config.sta.ssid));
    strlcpy((char *)wifi_config.sta.password, password, sizeof(wifi_config.sta.password));
    wifi_config.sta.threshold.authmode = strlen(password) ? WIFI_AUTH_WPA2_PSK : WIFI_AUTH_OPEN;

    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &wifi_config));
    // Modem/light sleep cycles the radio off between beacons, which
    // disrupts CSI capture timing on the receiver — disable power-save on
    // both roles.
    ESP_ERROR_CHECK(esp_wifi_set_ps(WIFI_PS_NONE));
    ESP_ERROR_CHECK(esp_wifi_start());

    ESP_LOGI(TAG, "connecting to SSID '%s'...", ssid);

    TickType_t wait_ticks = (timeout_ms == 0) ? portMAX_DELAY : pdMS_TO_TICKS(timeout_ms);
    EventBits_t bits = xEventGroupWaitBits(s_wifi_event_group, WIFI_CONNECTED_BIT, pdFALSE, pdTRUE, wait_ticks);
    return (bits & WIFI_CONNECTED_BIT) ? ESP_OK : ESP_ERR_TIMEOUT;
}

bool wifi_conn_is_connected(void)
{
    return s_wifi_event_group != NULL && (xEventGroupGetBits(s_wifi_event_group) & WIFI_CONNECTED_BIT);
}

uint8_t wifi_conn_get_channel(void)
{
    wifi_ap_record_t ap_info = {0};
    if (esp_wifi_sta_get_ap_info(&ap_info) != ESP_OK) {
        return 0;
    }
    return ap_info.primary;
}
