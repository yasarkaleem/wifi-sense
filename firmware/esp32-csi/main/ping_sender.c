#include "ping_sender.h"

#include <stdint.h>

#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "sdkconfig.h"

#include "udp_client.h"

static const char *TAG = "ping_sender";

#define BROADCAST_ADDR "255.255.255.255"
#define PING_TASK_STACK 4096
#define PING_TASK_PRIORITY 5

typedef struct {
    uint32_t counter;
} ping_payload_t;

static void ping_sender_task(void *arg)
{
    udp_client_t client;
    if (udp_client_open(&client, true) != ESP_OK) {
        ESP_LOGE(TAG, "failed to open broadcast socket, ping sender task exiting");
        vTaskDelete(NULL);
        return;
    }

    // vTaskDelayUntil (not vTaskDelay) so each send's own duration doesn't
    // accumulate drift into the period — actual achievable resolution is
    // capped at portTICK_PERIOD_MS (10ms by default => 100Hz), see the
    // "rate limiting" note in README.md.
    TickType_t period_ticks = pdMS_TO_TICKS(1000 / CONFIG_PING_RATE_HZ);
    if (period_ticks == 0) {
        period_ticks = 1;
    }

    ping_payload_t payload = {.counter = 0};
    TickType_t last_wake = xTaskGetTickCount();

    ESP_LOGI(TAG, "broadcasting to %s:%d at %dHz (period %lu ticks)", BROADCAST_ADDR, CONFIG_PING_BROADCAST_PORT,
             CONFIG_PING_RATE_HZ, (unsigned long)period_ticks);

    while (1) {
        udp_client_send_to(&client, BROADCAST_ADDR, CONFIG_PING_BROADCAST_PORT, &payload, sizeof(payload));
        payload.counter++;
        vTaskDelayUntil(&last_wake, period_ticks);
    }
}

esp_err_t ping_sender_start(void)
{
    if (xTaskCreate(ping_sender_task, "ping_sender", PING_TASK_STACK, NULL, PING_TASK_PRIORITY, NULL) != pdPASS) {
        return ESP_ERR_NO_MEM;
    }
    return ESP_OK;
}
