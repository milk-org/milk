# milk-fps-valkey — FPS Parameter Sync to Valkey

Bidirectional real-time sync of [milk](https://github.com/milk-org/milk) FPS (Function Parameter Structure) parameters to a [Valkey](https://valkey.io/) key-value store, enabling parameter sharing and remote control across multiple computers.

***

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Building](#building)
- [Usage](#usage)
- [Architecture](#architecture)
- [Valkey Key Schema](#valkey-key-schema)
- [Pub/Sub Protocol](#pubsub-protocol)
- [Configuration Examples](#configuration-examples)
- [Troubleshooting](#troubleshooting)
- [API Reference](#api-reference)

***

## Overview

`milk-fps-valkey` is a standalone executable that:

1. **Polls** local FPS shared-memory segments for parameter changes (push).
2. **Writes** changed parameters to a Valkey Hash (`HSET`).
3. **Publishes** change notifications via Valkey Pub/Sub (`PUBLISH`).
4. **Subscribes** to remote Pub/Sub notifications (`PSUBSCRIBE`) and applies incoming parameter changes to local shared memory (pull).

This enables a real-time, low-latency bridge between FPS instances on different machines, all mediated through a central Valkey server.

### Key Features

- **Bidirectional sync** — changes propagate in both directions.
- **Low-latency pull** — Pub/Sub notifications arrive in milliseconds (not polling-based).
- **Echo prevention** — messages from the local host are ignored on the pull path.
- **Conflict resolution** — last-writer-wins semantics.
- **Auto-reconnect** — the command connection recovers from transient Valkey failures.
- **Standalone build** — not compiled by default; independent CMake project.

***

## Prerequisites

### 1. milk (installed)

`milk-fps-valkey` links against the installed milk libraries (`milkfps`, `ImageStreamIO`, `milkprocessinfo`). Ensure milk is built and installed:

```bash
cd /path/to/milk/_build
cmake ..
make -j$(nproc)
sudo make install
```

Verify:

```bash
pkg-config --cflags --libs milk
## Should output include/lib paths
```

### 2. Valkey Server

Install one of:

```bash
## Ubuntu 24.04+
sudo apt install valkey-server

## Or Redis (wire-compatible)
sudo apt install redis-server

## Or from source
git clone https://github.com/valkey-io/valkey.git
cd valkey && make -j$(nproc) && sudo make install
```

### 3. libvalkey (C client library)

```bash
git clone https://github.com/valkey-io/libvalkey.git
cd libvalkey
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local
make -j$(nproc)
sudo make install
sudo ldconfig
```

Verify:

```bash
pkg-config --cflags --libs valkey
## Should output: -I/usr/local/include -L/usr/local/lib -lvalkey
```

***

## Building

The build uses milk **source tree** headers (the installed headers are
incomplete for out-of-tree consumers). By default, CMake auto-detects
`MILK_SRC` as two directories above `src/fpsvalkey/` and `MILK_BUILD`
as `MILK_SRC/_build`.

```bash
cd /path/to/milk/src/fpsvalkey
mkdir build && cd build
cmake ..
make -j$(nproc)
```

If libvalkey is not installed system-wide (e.g. built in `/tmp/libvalkey`):

```bash
PKG_CONFIG_PATH="/tmp/libvalkey/build:$PKG_CONFIG_PATH" \
cmake .. \
  -DCMAKE_C_FLAGS="-I/tmp/libvalkey/include" \
  -DCMAKE_EXE_LINKER_FLAGS="-L/tmp/libvalkey/build"
make -j$(nproc)
```

If milk source/build paths differ from the default:

```bash
cmake .. -DMILK_SRC=/path/to/milk -DMILK_BUILD=/path/to/milk/_build
```

### Installing

```bash
sudo make install
```

Installs `milk-fps-valkey` to `CMAKE_INSTALL_PREFIX/bin`.

***

## Usage

### Basic (single host)

```bash
## Start Valkey server
valkey-server --port 6379 &

## Start an FPS process
milk-fpsclitest ..confstart

## Start syncing all FPS instances to Valkey
milk-fps-valkey
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `-i, --interval SEC` | `0.1` | Polling interval in seconds |
| `-V, --valkey-host H` | `127.0.0.1` | Valkey server address |
| `-P, --valkey-port P` | `6379` | Valkey server port |
| `-h, --help` | — | Show help |
| `regex_pattern` | `.*` | Filter FPS names by regex |

### Examples

```bash
## Sync only FPS names matching "dmcomb"
milk-fps-valkey "dmcomb.*"

## Use remote Valkey server, faster polling
milk-fps-valkey -V 192.168.1.100 -P 6379 -i 0.05

## Sync all FPS to a specific Valkey instance
milk-fps-valkey -V valkey.cluster.local
```

### Multi-Host Setup

On each computer:

```bash
## Host A (IP: 192.168.1.10)
milk-fps-valkey -V 192.168.1.100

## Host B (IP: 192.168.1.20)
milk-fps-valkey -V 192.168.1.100
```

Both hosts connect to the same Valkey server at `192.168.1.100`. Parameter changes on host A appear on host B within milliseconds, and vice versa.

***

## Architecture

### Components

```text
┌─────────────────────────────────────────────┐
│              milk-fps-valkey                 │
│                                             │
│  ┌──────────────┐    ┌──────────────┐       │
│  │  Main Thread  │    │  Sub Thread   │      │
│  │              │    │              │       │
│  │  Poll SHM    │    │  Block on    │       │
│  │  Detect Δcnt0│    │  PSUBSCRIBE  │       │
│  │  HSET+PUBLISH│    │  Parse msg   │       │
│  │  (push)      │    │  Write SHM   │       │
│  │              │    │  (pull)      │       │
│  └──────┬───────┘    └──────┬───────┘       │
│         │                   │               │
│         │  cmd_ctx          │  sub_ctx       │
│         │  (sync)           │  (blocking)    │
└─────────┼───────────────────┼───────────────┘
          │                   │
          ▼                   ▼
    ┌─────────────────────────────┐
    │       Valkey Server          │
    │                             │
    │  Hash: fps:<host>:<name>    │
    │  Set:  fps_list:<host>      │
    │  PubSub: fps_update:<name>  │
    └─────────────────────────────┘
```

### Two Connections

| Connection | Type | Thread | Purpose |
|------------|------|--------|---------|
| `cmd_ctx` | Synchronous `valkeyContext` | Main | `HSET`, `PUBLISH`, `SADD`, `SREM`, `DEL` |
| `sub_ctx` | Synchronous `valkeyContext` (blocking) | Subscriber pthread | `PSUBSCRIBE fps_update:*` → `valkeyGetReply()` |

### Push Path (Main Thread)

1. Poll FPS shared memory at the configured interval.
2. For each parameter, compare `cnt0` with the last-known value.
3. On change, pipeline:
   - `HSET fps:<host>:<name> <keyword> <value>`
   - `HSET fps:<host>:<name> _type.<keyword> <typename>`
   - `HSET fps:<host>:<name> _cnt0.<keyword> <cnt0>`
   - `PUBLISH fps_update:<name> "<host> <name> <keyword> <value> <typename>"`

### Pull Path (Subscriber Thread)

1. At startup, sends `PSUBSCRIBE fps_update:*` on the `sub_ctx` connection.
2. A dedicated `pthread` blocks on `valkeyGetReply()`.
3. On message arrival:
   - Parse: `<source_host> <fpsname> <keyword> <value> <typename>`
   - **Skip** if `source_host` == local hostname (prevents echo).
   - Connect to local FPS SHM (`function_parameter_struct_connect`).
   - Find parameter by keyword (`functionparameter_GetParamIndex`).
   - Write value using type-aware parsing (same as `milk-fps-set`).
   - Increment `cnt0`, `value_cnt`, set `SIGNAL_UPDATE`.
   - Disconnect.

### Conflict Resolution

**Last-writer-wins**: each host pushes its changes independently. The Pub/Sub subscriber on the opposing host receives and applies the change. The `source_host` field prevents echo loops (a host never applies its own messages).

### FPS Lifecycle Events

| Event | Action |
|-------|--------|
| New FPS discovered | `SADD fps_list:<host> <name>`, push all params |
| FPS deleted | `SREM fps_list:<host> <name>`, `DEL fps:<host>:<name>` |
| Tracker shutdown | Subscriber thread stopped, connections freed |

***

## Valkey Key Schema

### FPS Hash

**Key**: `fps:<hostname>:<fpsname>`

| Field | Type | Description |
|-------|------|-------------|
| `<keyword>` | String | Parameter value |
| `_type.<keyword>` | String | Parameter type (e.g. `FLOAT64`, `UINT32`, `ONOFF`) |
| `_cnt0.<keyword>` | Integer | Change counter |
| `_status` | String | FPS status bitmask (hex) |
| `_confpid` | Integer | Configuration process PID |
| `_runpid` | Integer | Run process PID |
| `_lastsync` | String | Last metadata sync (UTC ISO-8601) |

Example:

```text
HGETALL fps:rtc1:dmcomb-00
 1) ".delayus"
 2) "200"
 3) "_type..delayus"
 4) "UINT32"
 5) "_cnt0..delayus"
 6) "42"
 7) ".voltmode"
 8) "ON"
 9) "_type..voltmode"
10) "ONOFF"
11) "_cnt0..voltmode"
12) "5"
13) "_status"
14) "0x0201"
15) "_confpid"
16) "12345"
17) "_runpid"
18) "12346"
19) "_lastsync"
20) "2026-03-05T08:30:00Z"
```

### FPS List Set

**Key**: `fps_list:<hostname>`

Contains the names of all active FPS instances on that host.

```text
SMEMBERS fps_list:rtc1
1) "dmcomb-00"
2) "acquireWFS-00"
3) "mlat-00"
```

### Pub/Sub Channel

**Pattern**: `fps_update:<fpsname>`

**Message format**: `<hostname> <fpsname> <keyword> <value> <typename>`

Example:

```text
PUBLISH fps_update:dmcomb-00 "rtc1 dmcomb-00 .delayus 200 UINT32"
```

***

## Pub/Sub Protocol

The Pub/Sub protocol enables low-latency propagation of parameter changes without polling. Each parameter change triggers a `PUBLISH` on the `fps_update:<fpsname>` channel.

### Message Format

Fields are space-separated:

```text
<source_hostname> <fpsname> <keyword> <value> <typename>
```

| Field | Description |
|-------|-------------|
| `source_hostname` | Hostname of the machine that made the change |
| `fpsname` | FPS instance name |
| `keyword` | Full parameter keyword (e.g. `.delayus`) |
| `value` | New value as a string |
| `typename` | FPTYPE name (e.g. `FLOAT64`, `ONOFF`) |

### Subscribing from External Tools

You can subscribe to FPS changes from any Valkey client:

```bash
## Watch all FPS changes
valkey-cli PSUBSCRIBE "fps_update:*"

## Watch a specific FPS
valkey-cli SUBSCRIBE "fps_update:dmcomb-00"
```

### Injecting Changes Remotely

You can set FPS parameters remotely using `valkey-cli`:

```bash
## This will be received by milk-fps-valkey on all hosts
valkey-cli PUBLISH fps_update:dmcomb-00 \
  "remote-host dmcomb-00 .delayus 500 UINT32"
```

Note: the `source_hostname` field should be different from the target host's hostname to avoid echo filtering.

***

## Configuration Examples

### Single-Host Development

```bash
## Terminal 1: Start Valkey
valkey-server

## Terminal 2: Start FPS
milk -e "exfpscli ..confstart"

## Terminal 3: Sync to Valkey
milk-fps-valkey -i 0.5

## Terminal 4: Monitor in Valkey
valkey-cli PSUBSCRIBE "fps_update:*"
```

### Two-Host AO System

```text
   ┌──────────┐          ┌──────────────┐          ┌──────────┐
   │  RTC Host │◄────────►│ Valkey Server │◄────────►│ GUI Host │
   │  (rtc1)   │          │ (192.168.1.1) │          │  (gui1)  │
   └──────────┘          └──────────────┘          └──────────┘
```

On the RTC host:

```bash
milk-fps-valkey -V 192.168.1.1
```

On the GUI host:

```bash
milk-fps-valkey -V 192.168.1.1
```

Changes made on either host propagate to the other in real time.

***

## Troubleshooting

### Connection Issues

```text
[fpsvalkey] cmd connect failed: Connection refused
```

→ Valkey server is not running or wrong host/port. Start with `valkey-server` or check firewall.

### No Parameters Syncing

- Verify FPS processes are running: `milk-fps-list`
- Check regex pattern matches: try `milk-fps-valkey ".*"`
- Verify Valkey connectivity: `valkey-cli ping` → should return `PONG`

### Pull Not Working

- Check subscriber thread started: look for `[fpsvalkey] Subscriber thread started`
- Verify channel: `valkey-cli PSUBSCRIBE "fps_update:*"` should show messages
- Check hostname: pull skips messages from the same hostname. Use different hostnames or test with `valkey-cli PUBLISH`.

### Build Errors

**`Package valkey was not found`**:

```bash
export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig:$PKG_CONFIG_PATH
```

**`Could not find a package configuration file provided by "milk"`**:

```bash
cmake .. -DCMAKE_PREFIX_PATH=/usr/local/milk-1.03.00
```

***

## API Reference

### fps_valkey.h

The C API is available for integration into other tools:

| Function | Description |
|----------|-------------|
| `fps_valkey_connect(vctx, host, port)` | Open dual connections to Valkey |
| `fps_valkey_disconnect(vctx)` | Stop subscriber + free connections |
| `fps_valkey_push_param(vctx, name, kw, val, type, cnt0)` | Push one parameter (HSET + PUBLISH) |
| `fps_valkey_push_metadata(vctx, name, md)` | Push FPS status/PID metadata |
| `fps_valkey_register_fps(vctx, name)` | Add to fps_list set |
| `fps_valkey_unregister_fps(vctx, name)` | Remove from set + delete hash |
| `fps_valkey_sub_start(vctx)` | Start PSUBSCRIBE subscriber thread |
| `fps_valkey_sub_stop(vctx)` | Stop subscriber thread |

### Supported Parameter Types

| FPTYPE | Valkey Representation | Example |
|--------|----------------------|---------|
| `INT32` | Decimal integer | `-42` |
| `UINT32` | Decimal integer | `200` |
| `INT64` | Decimal integer | `-100000` |
| `UINT64` | Decimal integer | `100000` |
| `FLOAT32` | Decimal float (10 sig. digits) | `3.141592741` |
| `FLOAT64` | Decimal float (17 sig. digits) | `3.1415926535897931` |
| `ONOFF` | `ON` / `OFF` | `ON` |
| `TIMESPEC` | `<sec>.<nsec>` | `1709654400.000000000` |
| `PID` | Decimal integer | `12345` |
| `STRING` | UTF-8 string | `hello` |
| `FILENAME` | Path string | `/tmp/data.fits` |
| `STREAMNAME` | Stream name | `dm01disp` |
| `FPSNAME` | FPS name | `dmcomb-00` |
| *other string types* | UTF-8 string | — |

***

## Files

| File | Purpose |
|------|---------|
| `CMakeLists.txt` | Standalone CMake build system |
| `fps_valkey.h` | API header |
| `fps_valkey.c` | Valkey client implementation (push + pull) |
| `milk-fps-valkey.c` | Main executable (FPS scan loop + Valkey) |
| `README.md` | This documentation |
