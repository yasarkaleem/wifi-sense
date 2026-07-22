#include "time_sync.h"

#include <string.h>
#include <sys/time.h>
#include <time.h>

#include "esp_log.h"
#include "esp_sntp.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

static const char *TAG = "time_sync";
static volatile bool s_synced;

static void on_sync(struct timeval *tv)
{
    (void)tv;
    s_synced = true;
    ESP_LOGI(TAG, "time synced");
}

void time_sync_start(void)
{
    sntp_setoperatingmode(SNTP_OPMODE_POLL);
    sntp_setservername(0, "pool.ntp.org");
    sntp_set_time_sync_notification_cb(on_sync);
    sntp_init();
}

bool time_sync_wait(uint32_t timeout_ms)
{
    uint32_t waited_ms = 0;
    const uint32_t step_ms = 200;
    while (!s_synced && waited_ms < timeout_ms) {
        vTaskDelay(pdMS_TO_TICKS(step_ms));
        waited_ms += step_ms;
    }
    return s_synced;
}
