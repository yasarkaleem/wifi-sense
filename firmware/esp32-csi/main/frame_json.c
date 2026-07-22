/**
 * Hand-built JSON (snprintf, no cJSON/heap allocation) since this runs at
 * up to PING_RATE_HZ (default 100) times a second on a microcontroller —
 * see frame_json.h.
 */
#include "frame_json.h"

#include <stdarg.h>
#include <stdio.h>

#define SCHEMA_VERSION 1

static int append(char *out, size_t out_len, size_t *offset, const char *fmt, ...)
{
    if (*offset >= out_len) {
        return -1;
    }

    va_list args;
    va_start(args, fmt);
    int n = vsnprintf(out + *offset, out_len - *offset, fmt, args);
    va_end(args);

    if (n < 0 || (size_t)n >= out_len - *offset) {
        return -1; // truncated — caller treats this as a failed build
    }

    *offset += (size_t)n;
    return 0;
}

size_t frame_json_build(const csi_frame_t *frame, uint32_t sequence_number, char *out, size_t out_len)
{
    size_t offset = 0;

    if (append(out, out_len, &offset,
               "{\"schema_version\":%d,\"timestamp_us\":%llu,\"source_mac\":\"%02X:%02X:%02X:%02X:%02X:%02X\","
               "\"rssi\":%d,\"channel\":%u,\"subcarrier_count\":%u,\"amplitude\":[",
               SCHEMA_VERSION, (unsigned long long)frame->timestamp_us, frame->mac[0], frame->mac[1],
               frame->mac[2], frame->mac[3], frame->mac[4], frame->mac[5], frame->rssi, frame->channel,
               frame->subcarrier_count) != 0) {
        return 0;
    }

    for (uint16_t i = 0; i < frame->subcarrier_count; i++) {
        if (append(out, out_len, &offset, i == 0 ? "%.3f" : ",%.3f", frame->amplitude[i]) != 0) {
            return 0;
        }
    }

    if (append(out, out_len, &offset, "],\"phase\":[") != 0) {
        return 0;
    }

    for (uint16_t i = 0; i < frame->subcarrier_count; i++) {
        if (append(out, out_len, &offset, i == 0 ? "%.4f" : ",%.4f", frame->phase[i]) != 0) {
            return 0;
        }
    }

    if (append(out, out_len, &offset, "],\"sequence_number\":%lu}", (unsigned long)sequence_number) != 0) {
        return 0;
    }

    return offset;
}
