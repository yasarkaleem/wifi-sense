# CSI Frame Schema

This is the canonical wire format for a single WiFi Channel State Information
(CSI) reading. Every producer (ESP32 firmware, the `replay` service) emits
this shape over UDP as JSON, and every consumer (`ingest`, `pipeline`) treats
it as the source of truth. If the shape needs to change, update this doc and
bump `schema_version` in the same PR.

## Fields

| Field              | Type            | Units / Range         | Description                                                                 |
|--------------------|-----------------|------------------------|-------------------------------------------------------------------------------|
| `schema_version`   | `integer`       | starts at `1`          | Version of this schema the frame was produced against.                       |
| `timestamp_us`     | `integer`       | microseconds (epoch)   | Capture time on the sender, microsecond resolution.                          |
| `source_mac`       | `string`        | `"AA:BB:CC:DD:EE:FF"`  | MAC address of the ESP32 (or emulated source) that captured the frame.       |
| `rssi`             | `integer`       | dBm (typically -100..0)| Received signal strength indicator for the packet the CSI was extracted from.|
| `channel`          | `integer`       | WiFi channel number    | 2.4GHz/5GHz channel the radio was tuned to when capturing.                   |
| `subcarrier_count` | `integer`       | count, `> 0`           | Number of subcarriers represented in `amplitude` / `phase`.                  |
| `amplitude`        | `array[number]` | length == `subcarrier_count` | Per-subcarrier amplitude values.                                       |
| `phase`            | `array[number]` | length == `subcarrier_count`, radians | Per-subcarrier phase values.                                    |
| `sequence_number`  | `integer`       | monotonically increasing per `source_mac`, wraps at implementation limit | Used to detect drops/reordering. |

## Example

```json
{
  "schema_version": 1,
  "timestamp_us": 1721600000123456,
  "source_mac": "24:6F:28:AB:CD:EF",
  "rssi": -52,
  "channel": 6,
  "subcarrier_count": 64,
  "amplitude": [12.3, 11.9, 13.4, "... 61 more values ..."],
  "phase": [0.12, -0.45, 1.02, "... 61 more values ..."],
  "sequence_number": 481932
}
```

## Notes / open questions

- `amplitude` and `phase` are parallel arrays indexed by subcarrier index
  (index `0` is the same physical subcarrier in both arrays).
- `subcarrier_count` is included explicitly (rather than inferred from array
  length) so `ingest` can validate frames cheaply before parsing the arrays.
- Transport is UDP, so frames may arrive out of order or be dropped —
  `sequence_number` exists for the `pipeline` service to detect and handle
  this; the wire format itself does not guarantee delivery.
- **Resolved:** raw ESP32 CSI is flattened to a single amplitude/phase pair
  per subcarrier at the firmware level, not exposed as a separate `antenna`
  dimension. `firmware/esp32-csi/main/csi_capture.c` sets
  `wifi_csi_config_t.ltf_merge_en = true`, which merges legacy/HT
  long-training-field (antenna/spatial-stream) data into one buffer in the
  ESP32's own CSI hardware, before this schema ever sees it — so there's
  nothing left to flatten downstream.
