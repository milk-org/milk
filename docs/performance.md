---
tags:
  - performance
  - optimization
  - developer-guide
---

# Performance Tuning

Guidelines for optimizing `milk` pipeline throughput and
latency, covering OS-level configuration, process
scheduling, memory layout, and GPU acceleration.

See also: [Process Info](procinfo.md) ·
[Streams](streams.md) ·
[FPS](fps.md) ·
[Debugging](debugging.md) ·
[FAQ](faq.md) ·
[Code-Level Optimization Rules](https://github.com/milk-org/milk/blob/framework-dev/.agents/rules/performance-practices.md) ·
[Code Assist Tools](code_assist.md)

---

## 1. Measuring Performance

### 1.1. Loop Frequency

Use `milk-procinfo-list` to monitor the Hz column for
each compute unit. This is the most direct measure of
pipeline throughput.

### 1.2. Semaphore Latency

Benchmark the fundamental IPC latency:

```bash
$ milk-semloopspeed 10000
```

Typical values on modern hardware:

- **x86_64 (bare metal):** 200–500 kHz
- **ARM (embedded):** 50–150 kHz
- **VM / container:** 30–100 kHz (overhead from
  virtualization)

### 1.3. Stream Timing

Use `milk-streamCTRL` to observe per-stream frame rates
and detect bottlenecks in the pipeline.

---

## 2. CPU Pinning

Bind compute-critical processes to dedicated CPU cores
to avoid context-switch overhead and cache thrashing.

### 2.1. Using `taskset`

```bash
# Pin to core 4
$ taskset -c 4 milk-fpsexec-mymodule fpsinit

# Pin to cores 4-7
$ taskset -c 4-7 milk-fpsexec-mymodule fpsinit
```

### 2.2. Kernel Isolation (`isolcpus`)

For maximum determinism, isolate cores from the Linux
scheduler at boot time:

```bash
# Add to kernel command line (GRUB)
isolcpus=4,5,6,7
```

Then only explicitly pinned processes will run on those
cores.

!!! important
Isolated cores are invisible to normal scheduling.
All system services, interrupts, and other processes
will be confined to the remaining cores.

---

## 3. Real-Time Scheduling

### 3.1. `SCHED_FIFO`

For latency-critical loops, elevate the scheduling
policy:

```bash
$ sudo chrt -f 49 milk-fpsexec-mymodule fpsinit
```

Priority range: 1 (lowest) to 99 (highest). Avoid 99
which is reserved for kernel threads.

### 3.2. Permissions

To allow non-root real-time scheduling:

```bash
# /etc/security/limits.d/milk.conf
@realtime  -  rtprio  95
@realtime  -  memlock unlimited
```

Add users to the `realtime` group.

---

## 4. Shared Memory Configuration

### 4.1. tmpfs Size

By default, `/dev/shm` may be limited to 50% of RAM.
For large stream buffers, increase it:

```bash
# Temporary
$ sudo mount -o remount,size=16G /dev/shm

# Permanent: add to /etc/fstab
tmpfs /dev/shm tmpfs defaults,size=16G 0 0
```

### 4.2. Huge Pages

For very large streams, huge pages reduce TLB misses:

```bash
# Allocate 1024 huge pages (2 MB each = 2 GB)
$ echo 1024 | sudo tee \
    /proc/sys/vm/nr_hugepages
```

To enable huge pages for `ImageStreamIO` shared
memory streams ≥ 2 MB:

```bash
$ export MILK_SHM_HUGETLB=1
```

When set, `ImageStreamIO_createIm_gpu()` uses
`MAP_HUGETLB` for its `mmap()` call. If huge
pages are unavailable, it falls back to normal
pages automatically.

---

## 5. GPU Acceleration (CUDA)

### 5.1. Build with CUDA

```bash
$ cmake .. -DUSE_CUDA=ON \
    -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda
$ make -j$(nproc)
```

### 5.2. GPU Memory Pinning

Use `cudaHostRegister()` to pin shared memory buffers
for zero-copy GPU access. This is handled automatically
by GPU-enabled modules (e.g., `linalgebra` with cuBLAS).

### 5.3. Multi-GPU

Set the GPU device before launching:

```bash
$ CUDA_VISIBLE_DEVICES=1 milk-fpsexec-gpumodule fpsinit
```

---

## 6. Network Tuning (for Valkey sync)

When using `milk-fps-valkey` across hosts, minimize
network latency:

```bash
# Disable Nagle's algorithm
$ sysctl -w net.ipv4.tcp_nodelay=1

# Increase socket buffer sizes
$ sysctl -w net.core.rmem_max=16777216
$ sysctl -w net.core.wmem_max=16777216
```

See also: [Valkey Integration](valkey.md)

---

## 7. Quick Checklist

| Item                   | Command / Config                     |
| ---------------------- | ------------------------------------ |
| Check loop Hz          | `milk-procinfo-list`                 |
| Semaphore benchmark    | `milk-semloopspeed 10000`            |
| Pin to core 4          | `taskset -c 4 <cmd>`                 |
| RT priority 49         | `sudo chrt -f 49 <cmd>`              |
| Increase `/dev/shm`    | `mount -o remount,size=16G /dev/shm` |
| Huge pages for streams | `export MILK_SHM_HUGETLB=1`          |
| Enable CUDA            | `cmake .. -DUSE_CUDA=ON`             |
| Vec missed report      | `cmake .. -DVEC_REPORT=ON`           |
| PGO build              | `cmake .. -DUSE_PGO=GENERATE`        |
| Static LTO build       | `cmake .. -DUSE_STATIC_LTO=ON`       |
| PGO + Static LTO       | `-DUSE_PGO=USE -DUSE_STATIC_LTO=ON`  |
| Valkey low-latency     | `sysctl -w net.ipv4.tcp_nodelay=1`   |

---

← [Documentation Index](index.md)
