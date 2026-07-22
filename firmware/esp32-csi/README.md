# esp32-csi

ESP-IDF firmware for the ESP32 that captures WiFi CSI (Channel State
Information) and streams it over UDP as JSON frames matching
[`docs/csi-frame-schema.md`](../../docs/csi-frame-schema.md).

One firmware image, two roles, chosen at build/config time:

- **Receiver** — joins WiFi, enables CSI capture, decodes each captured
  frame, and sends it as JSON to `services/ingest` (`host:5566` by
  default).
- **Sender** — joins the *same* WiFi network and broadcasts a small UDP
  packet at a fixed, configurable rate (default 100Hz). This is the
  "stimulus" traffic the receiver's radio sees and extracts CSI from —
  without it, the receiver only gets CSI from whatever other traffic
  happens to be on the channel (beacons, unrelated devices), at an
  uncontrolled rate.

Flash the same project onto two boards, choosing a different role for
each (`idf.py menuconfig` → "wifi-sense CSI firmware" → "Device role").

> **Verification status:** this firmware was written by carefully reading
> Espressif's own [esp-csi](https://github.com/espressif/esp-csi) example
> sources and current ESP-IDF API docs, but it has **not been compiled or
> run on real hardware** — no ESP-IDF toolchain was available in the
> environment this was written in. Treat it as a solid, research-grounded
> starting point, not as verified-working firmware. Please open an issue
> (or just fix it) if `idf.py build` turns up problems — a config struct
> field name drifting between ESP-IDF versions is the most likely
> culprit; see "ESP-IDF version" below.

## Which ESP32 variants support CSI

| Chip       | CSI capable | Supported by this firmware |
|------------|-------------|-----------------------------|
| ESP32      | Yes         | Yes |
| ESP32-S2   | Yes         | Yes |
| ESP32-S3   | Yes         | Yes |
| ESP32-C3   | Yes         | Yes |
| ESP32-C5   | Yes         | **No** — different CSI config/struct layout, see below |
| ESP32-C6   | Yes         | **No** — different CSI config/struct layout, see below |
| ESP32-C61  | Yes         | **No** — different CSI config/struct layout, see below |
| ESP32-S2 (non-CSI variants), ESP32-C2, ESP8266 | No | N/A |

`main/csi_capture.c` decodes the legacy CSI raw-buffer format (a flat
array of signed 8-bit `(imaginary, real)` byte pairs) used by
ESP32/S2/S3/C3. The C5/C6/C61 CSI hardware exposes a different
`wifi_csi_config_t`/`wifi_csi_info_t` field set entirely; supporting them
would need a second decode path. The build fails with a clear `#error` if
you target one of those chips, rather than silently producing garbage
data.

Both boards (receiver + sender) must use a chip from the "Yes" column —
they don't need to be the *same* chip, just both CSI-capable, and both
associated to the same AP on the same channel.

## ESP-IDF version

Developed against the ESP-IDF v5.x CSI API surface
(`esp_wifi_set_csi_config`/`esp_wifi_set_csi_rx_cb`/`esp_wifi_set_csi`,
`wifi_csi_config_t`, `wifi_csi_info_t`). If you're on a materially older
or newer ESP-IDF and `idf.py build` fails on `main/csi_capture.c`, check
the current field names in `wifi_csi_config_t` (`idf.py --idf-path
$IDF_PATH` → `components/esp_wifi/include/esp_wifi_types.h` or
`esp_wifi_he_types.h` depending on version) — this struct has grown
fields across releases as more chips gained CSI support.

## Flashing

Requires the ESP-IDF toolchain installed and sourced (`. $IDF_PATH/export.sh`
or the "ESP-IDF" terminal shortcut the installer sets up) — see
[Espressif's Get Started guide](https://docs.espressif.com/projects/esp-idf/en/stable/esp32/get-started/)
if you don't have it yet.

```bash
cd firmware/esp32-csi

idf.py set-target esp32        # or esp32s2 / esp32s3 / esp32c3
idf.py menuconfig              # see "Configuration" below — at minimum,
                                # set the device role and WiFi credentials
idf.py -p <PORT> flash monitor
```

Repeat on a second board with the *other* role selected in menuconfig.
`<PORT>` is the board's serial device — `/dev/ttyUSB0`/`/dev/ttyACM0` on
Linux, `/dev/cu.usbserial-*`/`/dev/cu.usbmodem*` on macOS, `COM<n>` on
Windows.

## Configuration

All of this lives under `idf.py menuconfig` → **"wifi-sense CSI
firmware"** (backed by `main/Kconfig.projbuild`):

| Option | Default | Applies to | Notes |
|--------|---------|------------|-------|
| Device role | Receiver | both | Receiver captures + forwards CSI; sender broadcasts pings. |
| WiFi SSID / password | `myssid` / `mypassword` | both | Both roles must join the same network. Leave password blank for an open network. |
| Max reconnect backoff (ms) | 30000 | both | See "Handling reconnects" below. |
| UDP target host | `192.168.1.100` | receiver | IPv4 of the machine running `services/ingest`. |
| UDP target port | 5566 | receiver | Matches `services/ingest`'s default `--udp-port`. |
| Expected sender MAC | *(blank)* | receiver | See "Filtering by sender MAC" below. |
| Ping send rate (Hz) | 100 | sender | The CSI sample rate you get on the receiver, assuming it isn't dropping frames. |
| UDP broadcast port | 3333 | sender | Arbitrary — nothing listens on it, these packets exist only to produce CSI-capturable traffic. |

WiFi SSID/password and the UDP target host/port can also be set **without
reflashing**, at runtime, over the serial console — see "NVS
provisioning" below. Kconfig values are the fallback used only if NVS has
never been provisioned.

### NVS provisioning (no reflash needed)

Connect to the board's serial console (`idf.py -p <PORT> monitor`, or any
serial terminal at 115200 8N1) and use:

```
csi> set_wifi "MyNetwork" "MyPassword"
csi> set_target 192.168.1.50 5566
csi> restart
```

Both take effect after `restart`. `set_wifi`/`set_target` persist to NVS
and override the Kconfig defaults on every subsequent boot; there's no
"unset" — reflash with `idf.py erase-flash` to go back to Kconfig
defaults.

### Filtering by sender MAC

If "Expected sender MAC" is blank, the receiver decodes CSI triggered by
*any* WiFi traffic it overhears on the channel — fine for a first test
with one sender and no other nearby devices, but noisy/ambiguous
otherwise (you'd be mixing CSI from the sender's pings with CSI from
unrelated beacons/traffic). Once you know the sender's MAC (printed in
its boot log — look for the STA MAC address line), set it via
menuconfig or NVS to filter to just that source.

### Channel coordination

Both boards must be associated to the **same WiFi channel** — CSI is only
extracted from packets the radio actually receives, and a station only
receives on the channel it's tuned to. In practice this means: both
boards join the same AP/SSID, and whatever channel that AP happens to be
using is what both boards get. Check a board's actual channel from its
boot log (`connected, channel N`) if you need to confirm they match.

## Handling reconnects

`main/wifi_conn.c` reconnects automatically on every disconnect, with
exponential backoff (starting at 1s, doubling, capped at "Max reconnect
backoff") that resets to 1s after each successful reconnection — so a
transient AP hiccup retries fast, but a genuinely down AP doesn't spin
the radio at full tilt indefinitely.

## Rate limiting

CSI frames are decoded off the WiFi driver's own callback context, which
must return quickly. `main/csi_capture.c` copies just the raw bytes there
and hands off to a separate FreeRTOS task (`csi_process`) over a
16-entry queue; if that task falls behind (slow UDP send, JSON building
taking too long, `PING_RATE_HZ` set higher than the receiver can keep up
with), new frames are **dropped, not queued indefinitely** —
`csi_capture_get_dropped_count()` reports how many. This bounds both
memory use and end-to-end latency at the cost of occasional gaps, which
`sequence_number` (below) makes visible to downstream consumers.

## Sequence numbers

Every JSON frame the receiver sends includes a `sequence_number` that
increments once per frame *actually sent* (starting at 0 on every boot,
per `docs/csi-frame-schema.md`). Gaps in the sequence — from queue drops
above, or from UDP packet loss on the wire — are for `services/ingest`/
`services/pipeline` to detect and handle; this firmware does not retry
or buffer.

## Timestamps

`timestamp_us` is epoch microseconds (matching `services/replay`), not
boot-relative. Since the ESP32 has no battery-backed RTC, the receiver
runs an SNTP client (`main/time_sync.c`, pool.ntp.org) at startup and
waits up to 10s for the first sync before starting CSI capture. If SNTP
can't reach the internet (isolated network, no outbound access), capture
still starts — `timestamp_us` will read near the 1970 epoch until/unless
a sync eventually succeeds in the background.

## Project layout

```
esp32-csi/
├── CMakeLists.txt          # top-level ESP-IDF project file
├── sdkconfig.defaults      # default build config (stack sizes, power-save)
└── main/
    ├── CMakeLists.txt
    ├── Kconfig.projbuild    # "wifi-sense CSI firmware" menuconfig options
    ├── main.c               # app_main: init, role dispatch
    ├── wifi_conn.{h,c}      # WiFi STA connect + auto-reconnect
    ├── nvs_config.{h,c}     # NVS-backed config + serial console commands
    ├── csi_capture.{h,c}    # CSI capture, decode, rate-limited queue (receiver)
    ├── frame_json.{h,c}     # JSON serialization matching the CSI frame schema
    ├── udp_client.{h,c}     # UDP socket send helper (both roles)
    ├── ping_sender.{h,c}    # fixed-rate UDP broadcast (sender)
    └── time_sync.{h,c}      # SNTP epoch time sync (receiver)
```

Docker isn't applicable to flashing physical hardware; the "runs via
docker compose" rule for this component instead applies to
`services/replay`, which emulates this firmware's UDP output on a host
machine so the rest of the pipeline can be developed without an ESP32.
