/**
 * CSI capture: promiscuous mode + esp_wifi_set_csi_rx_cb() driver callback,
 * decoupled from decoding by a FreeRTOS queue so a slow consumer drops
 * frames instead of blocking the WiFi driver task — see "rate limiting" in
 * README.md.
 *
 * Raw buffer format (verified against Espressif's esp-csi examples and
 * esp_wifi_types.h docs): a flat array of signed 8-bit (imaginary, real)
 * byte pairs, one pair per subcarrier — `buf[2*i]` = imaginary, `buf[2*i+1]`
 * = real. When `first_word_invalid` is set, the first 4 bytes (2 pairs) are
 * hardware-invalid and must be skipped.
 */
#include "csi_capture.h"

#include <math.h>
#include <string.h>
#include <sys/time.h>

#include "esp_log.h"
#include "esp_wifi.h"
#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"
#include "freertos/task.h"

#if CONFIG_IDF_TARGET_ESP32C5 || CONFIG_IDF_TARGET_ESP32C6 || CONFIG_IDF_TARGET_ESP32C61
#error \
    "csi_capture.c decodes the legacy int8 (imaginary, real) CSI buffer format used by ESP32/S2/S3/C3. \
C5/C6/C61 report CSI through a different wifi_csi_config_t/wifi_csi_info_t field set and are not supported here \
— see the ESP32 variant table in README.md."
#endif

static const char *TAG = "csi_capture";

#define CSI_QUEUE_LEN 16
#define CSI_PROCESS_TASK_STACK 8192
#define CSI_PROCESS_TASK_PRIORITY 5

typedef struct {
    uint64_t timestamp_us;
    uint8_t mac[6];
    int8_t rssi;
    uint8_t channel;
    uint16_t len;
    bool first_word_invalid;
    int8_t buf[CSI_MAX_SUBCARRIERS * 2];
} raw_csi_item_t;

static QueueHandle_t s_csi_queue;
static uint8_t s_expected_mac[6];
static bool s_filter_by_mac;
static void (*s_on_frame)(const csi_frame_t *frame);
static uint32_t s_dropped_count;

static void wifi_csi_rx_cb(void *ctx, wifi_csi_info_t *info)
{
    if (!info || !info->buf) {
        return;
    }

    if (s_filter_by_mac && memcmp(info->mac, s_expected_mac, 6) != 0) {
        return;
    }

    raw_csi_item_t item;
    struct timeval tv;
    gettimeofday(&tv, NULL); // epoch time (SNTP-corrected — see time_sync.h); pre-sync this reads as ~1970
    item.timestamp_us = (uint64_t)tv.tv_sec * 1000000ULL + (uint64_t)tv.tv_usec;
    memcpy(item.mac, info->mac, 6);
    item.rssi = info->rx_ctrl.rssi;
    item.channel = info->rx_ctrl.channel;
    item.first_word_invalid = info->first_word_invalid;

    uint16_t len = info->len;
    if (len > sizeof(item.buf)) {
        len = sizeof(item.buf); // clamp — shouldn't happen given CSI_MAX_SUBCARRIERS's margin
    }
    item.len = len;
    memcpy(item.buf, info->buf, len);

    if (xQueueSend(s_csi_queue, &item, 0) != pdTRUE) {
        s_dropped_count++;
    }
}

static void csi_process_task(void *arg)
{
    raw_csi_item_t item;
    csi_frame_t frame;

    while (1) {
        if (xQueueReceive(s_csi_queue, &item, portMAX_DELAY) != pdTRUE) {
            continue;
        }

        int start_pair = item.first_word_invalid ? 2 : 0;
        int n_pairs = item.len / 2;
        int count = n_pairs - start_pair;
        if (count < 0) {
            count = 0;
        }
        if (count > CSI_MAX_SUBCARRIERS) {
            count = CSI_MAX_SUBCARRIERS;
        }

        frame.timestamp_us = item.timestamp_us;
        memcpy(frame.mac, item.mac, 6);
        frame.rssi = item.rssi;
        frame.channel = item.channel;
        frame.subcarrier_count = (uint16_t)count;

        for (int i = 0; i < count; i++) {
            int pair = start_pair + i;
            float imag = (float)item.buf[2 * pair];
            float real = (float)item.buf[2 * pair + 1];
            frame.amplitude[i] = sqrtf(imag * imag + real * real);
            frame.phase[i] = atan2f(imag, real);
        }

        if (s_on_frame) {
            s_on_frame(&frame);
        }
    }
}

esp_err_t csi_capture_start(const uint8_t *expected_mac, void (*on_frame)(const csi_frame_t *frame))
{
    s_on_frame = on_frame;
    s_filter_by_mac = (expected_mac != NULL);
    if (s_filter_by_mac) {
        memcpy(s_expected_mac, expected_mac, 6);
    }

    s_csi_queue = xQueueCreate(CSI_QUEUE_LEN, sizeof(raw_csi_item_t));
    if (!s_csi_queue) {
        return ESP_ERR_NO_MEM;
    }

    if (xTaskCreate(csi_process_task, "csi_process", CSI_PROCESS_TASK_STACK, NULL, CSI_PROCESS_TASK_PRIORITY, NULL) !=
        pdPASS) {
        return ESP_ERR_NO_MEM;
    }

    ESP_ERROR_CHECK(esp_wifi_set_promiscuous(true));

    wifi_csi_config_t csi_config = {
        .lltf_en = true,
        .htltf_en = true,
        .stbc_htltf2_en = true,
        .ltf_merge_en = true, // merges antenna/spatial-stream LTFs into one buffer — see docs/csi-frame-schema.md
        .channel_filter_en = true,
        .manu_scale = false,
        .shift = false,
    };
    ESP_ERROR_CHECK(esp_wifi_set_csi_config(&csi_config));
    ESP_ERROR_CHECK(esp_wifi_set_csi_rx_cb(&wifi_csi_rx_cb, NULL));
    ESP_ERROR_CHECK(esp_wifi_set_csi(true));

    ESP_LOGI(TAG, "CSI capture started (mac filter %s)", s_filter_by_mac ? "on" : "off");
    return ESP_OK;
}

uint32_t csi_capture_get_dropped_count(void)
{
    return s_dropped_count;
}
