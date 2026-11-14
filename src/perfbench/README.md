# Performance Benchmark Harness (`src/perfbench/`)

Hardware-counter (`perf_event_open(2)`) and `processinfo` shared-memory benchmark harness for
measuring FPS compute-unit throughput, latency percentiles, cache/TLB miss rates, and system
resource usage.

---

## Purpose

`milk-perfbench` runs an `fpsexec` compute unit for a configured number of warmup and measured
iterations while collecting:

1. **Linux `perf_event_open(2)` hardware and software counters** (`perfbench_perf.c`) — cycles,
   instructions, L1D/L1I/LLC cache loads and misses, dTLB/iTLB misses, frontend/backend stalls,
   branch misses, page faults, context switches, and CPU migrations.
2. **Real-time `processinfo` shared-memory telemetry** (`perfbench_sysmon.c`,
   `milk-perfbench-readprocinfo.c`) — p50/p95/p99/p99.9/max iteration and execution latencies,
   jitter (`p99 - p50`), `/proc/<pid>/status` memory footprint (`VmPeak`, `VmHWM`, `VmRSS`),
   anonymous hugepages, CPU frequency range, and RAPL energy counters.
3. **Structured JSON and console summary reports** (`perfbench_report.c`) — per-iteration metrics
   tagged with the git commit hash and embedded binary build tags.

---

## Source Files

| File                              | Description                                                                        |
| :-------------------------------- | :--------------------------------------------------------------------------------- |
| `perfbench.h`                     | Hardware counter table, `pi_stats_t` / `bench_cfg_t` structs, and internal API     |
| `milk-perfbench.c`                | Main `milk-perfbench` CLI harness entry point and warmup/measured run orchestration|
| `perfbench_setup.c`               | FPS setup/teardown, stream preparation, build-tag extraction, and git hash lookup  |
| `perfbench_perf.c`                | `perf_event_open(2)` counter group setup, attach, read, and close                  |
| `perfbench_sysmon.c`              | `processinfo` SHM latency percentiles, `/proc` memory/hugepages, CPU freq & RAPL   |
| `perfbench_report.c`              | Per-iteration counter math, JSON result file writer, and terminal summary table    |
| `milk-perfbench-readprocinfo.c`   | Standalone utility to read a `processinfo` SHM file and emit timing/memory JSON    |
| `perfbench-mkstream.c`            | Standalone utility (`milk-perfbench-mkstream`) to create 2D float test streams     |
| `scripts/milk-perfbench`          | Reference Bash benchmark script (installed as `milk-perfbench.sh`)                 |
| `tests/test-perf-clitest.sh`      | End-to-end smoke test for the `milk-perfbench` CLI                                 |

---

## Standalone Executables

| Executable                    | Source                            | Description                                         |
| :---------------------------- | :-------------------------------- | :-------------------------------------------------- |
| `milk-perfbench`              | `milk-perfbench.c`                | C hardware-counter & `processinfo` benchmark runner |
| `milk-perfbench-readprocinfo` | `milk-perfbench-readprocinfo.c`   | Export `processinfo` SHM timing metrics as JSON     |
| `milk-perfbench-mkstream`     | `perfbench-mkstream.c`            | Create dummy 2D float shared-memory streams         |

---

## Dependencies & Build Tier

- **Build Tier**: Engine tier (no `CLIcore` dependency).
- **Internal Dependencies**: `milkprocessinfo`, `ImageStreamIO`.
