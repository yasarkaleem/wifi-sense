#include "udp_client.h"

#include <errno.h>
#include <string.h>

#include "esp_log.h"
#include "lwip/sockets.h"

static const char *TAG = "udp_client";

esp_err_t udp_client_open(udp_client_t *client, bool broadcast)
{
    client->sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_IP);
    if (client->sock < 0) {
        ESP_LOGE(TAG, "socket() failed: errno %d", errno);
        return ESP_FAIL;
    }

    if (broadcast) {
        int enable = 1;
        if (setsockopt(client->sock, SOL_SOCKET, SO_BROADCAST, &enable, sizeof(enable)) < 0) {
            ESP_LOGE(TAG, "setsockopt(SO_BROADCAST) failed: errno %d", errno);
            close(client->sock);
            client->sock = -1;
            return ESP_FAIL;
        }
    }

    return ESP_OK;
}

esp_err_t udp_client_send_to(udp_client_t *client, const char *host, uint16_t port, const void *data, size_t len)
{
    struct sockaddr_in dest = {0};
    dest.sin_family = AF_INET;
    dest.sin_port = htons(port);
    if (inet_pton(AF_INET, host, &dest.sin_addr) != 1) {
        ESP_LOGE(TAG, "invalid IPv4 address: %s", host);
        return ESP_ERR_INVALID_ARG;
    }

    int sent = sendto(client->sock, data, len, 0, (struct sockaddr *)&dest, sizeof(dest));
    if (sent < 0) {
        ESP_LOGW(TAG, "sendto(%s:%u) failed: errno %d", host, port, errno);
        return ESP_FAIL;
    }

    return ESP_OK;
}

void udp_client_close(udp_client_t *client)
{
    if (client->sock >= 0) {
        close(client->sock);
        client->sock = -1;
    }
}
