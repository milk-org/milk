# Valkey Integration

Real-time synchronization of FPS parameters across multiple hosts via a [Valkey](https://valkey.io/)
key-value store.

See also: [FPS](fps.md) · [Performance Tuning](performance.md) ·
[Programmer's Guide](programmers_guide.md)

---

## Overview

`milk-fps-valkey` is a standalone bridge that bidirectionally syncs FPS parameters between local
shared memory and a central Valkey (or Redis) server. This enables:

- **Multi-host parameter sharing:** FPS changes on one machine instantly appear on all others.
- **Remote monitoring and control:** inspect or modify parameters from any machine on the network.
- **External tool integration:** any Valkey client (Python, Node.js, CLI) can read/write FPS
  parameters.

```text
  Host A                 Valkey Server              Host B
  ┌──────────┐          ┌──────────────┐          ┌──────────┐
  │ FPS SHM  │◄────────►│  fps:hostA:* │◄────────►│ FPS SHM  │
  │          │  push/   │  fps:hostB:* │  push/   │          │
  │ milk-fps-│  pull    │              │  pull    │ milk-fps-│
  │ valkey   │          │  PubSub      │          │ valkey   │
  └──────────┘          └──────────────┘          └──────────┘
```

---

## Quick Start

### 1. Start Valkey

```bash
$ valkey-server --port 6379 &
```

### 2. Start FPS Processes

```bash
$ milk-fpsexec-mymodule fpsinit
$ milk-fpsexec-mymodule confstart
$ milk-fpsexec-mymodule runstart
```

### 3. Start the Sync Bridge

```bash
$ milk-fps-valkey
```

All local FPS parameters are now mirrored to Valkey.

### 4. Multi-Host

On each host, point to the same Valkey server:

```bash
## Host A
$ milk-fps-valkey -V 192.168.1.100

## Host B
$ milk-fps-valkey -V 192.168.1.100
```

---

## Key Features

| Feature            | Description                      |
| ------------------ | -------------------------------- |
| Bidirectional sync | Changes propagate both ways      |
| Low-latency pull   | PubSub notifications (ms)        |
| Echo prevention    | Local changes are not re-applied |
| Auto-reconnect     | Recovers from Valkey restarts    |
| Regex filtering    | Sync only matching FPS names     |

---

## Common Options

```bash
$ milk-fps-valkey [OPTIONS] [regex_pattern]
```

| Option              | Default     | Description             |
| ------------------- | ----------- | ----------------------- |
| `-V, --valkey-host` | `127.0.0.1` | Valkey server address   |
| `-P, --valkey-port` | `6379`      | Valkey server port      |
| `-i, --interval`    | `0.1`       | Poll interval (seconds) |
| `-h, --help`        | —           | Show help               |

---

## Detailed Documentation

For full architecture details, key schema, PubSub protocol, troubleshooting, and C API reference,
see the comprehensive README:

→ [`src/fpsvalkey/README.md`](../src/fpsvalkey/README.md)

---

## Building

`milk-fps-valkey` is **not** built by default. It requires `libvalkey` as an additional dependency:

```bash
$ cd src/fpsvalkey
$ mkdir build && cd build
$ cmake ..
$ make -j$(nproc)
$ sudo make install
```

See [`src/fpsvalkey/README.md`](../src/fpsvalkey/README.md) for detailed build instructions and
prerequisites.

---

← [Documentation Index](index.md)
