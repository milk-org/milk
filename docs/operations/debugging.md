# Debugging

Strategies for diagnosing issues in `milk` processes, from inspecting running pipelines to
post-mortem analysis of crashes.

See also: [Process Info](../guide/procinfo.md) · [FPS](../guide/fps.md) · [FAQ & Troubleshooting](../quickstart/faq.md) ·
[Performance Tuning](performance.md)

---

## 1. Inspecting Running Processes

### 1.1. `milk-procinfo-list`

The first tool to reach for. Displays all registered compute units with PID, state, and loop
frequency:

```bash
$ milk-procinfo-list
```

Look for:

- **Green PIDs:** healthy, actively looping.
- **Red / missing PIDs:** the process has died or its heartbeat has stopped.
- **Low Hz values:** the loop is running but may be starved of CPU or blocked on a semaphore.

### 1.2. `milk-fpsCTRL`

Open the FPS dashboard to inspect parameter values and process states interactively:

```bash
$ milk-fpsCTRL
```

Use arrow keys to navigate, `Enter` to inspect/edit parameters. Check the `confpid` and `runpid`
fields — if they show `0`, the corresponding loop is not running.

### 1.3. `milk-streamCTRL`

Inspect shared memory streams in real time:

```bash
$ milk-streamCTRL
```

Verify that streams are receiving frames (check frame counter `cnt0` is incrementing) and that
semaphore counts are not accumulating (which would indicate a stalled consumer).

---

## 2. Tmux Log Inspection

When processes are launched with `-tmux`, each runs in its own tmux session. To inspect output:

```bash
# List all milk-related tmux sessions
$ tmux ls | grep milk

# Attach to a specific session
$ tmux attach -t <session-name>

# Scroll up in tmux (Ctrl-b [, then PgUp)
```

<!-- prettier-ignore -->
!!! tip
    Each tmux session typically has three windows: `ctrl` (control commands), `run` (the main loop), and
    `conf` (the configuration loop). Switch windows with `Ctrl-b 0`, `Ctrl-b 1`, `Ctrl-b 2`.

---

## 3. GDB Debugging

### 3.1. Attaching to a Running Process

```bash
# Find the PID
$ milk-procinfo-list   # note the PID column

# Attach GDB
$ sudo gdb -p <PID>

# Inside GDB:
(gdb) bt           # backtrace
(gdb) info threads  # see all threads
(gdb) thread 2     # switch to thread 2
(gdb) bt           # backtrace for that thread
```

### 3.2. Running Under GDB from Start

```bash
$ gdb --args milk-fpsexec-mymodule fpsinit
(gdb) run
```

### 3.3. Core Dump Analysis

Enable core dumps:

```bash
$ ulimit -c unlimited
$ echo "/tmp/core.%e.%p" | \
    sudo tee /proc/sys/kernel/core_pattern
```

After a crash:

```bash
$ gdb /usr/local/milk/bin/milk-fpsexec-mymodule \
    /tmp/core.milk-fpsexec-mymodule.12345
(gdb) bt
```

---

## 4. Common Issues

### 4.1. Semaphore Deadlocks

**Symptom:** Process hangs, Hz drops to 0, but PID is still alive.

**Diagnosis:**

```bash
# Check semaphore state
$ milk-streamCTRL
# Look for sem values > 0 on the input stream
```

**Fix:** The producer may have died without posting. Restart the producer or manually post the
semaphore.

### 4.2. Shared Memory Leaks

**Symptom:** `/dev/shm/` fills up with stale `.im.shm` and `fps.*.shm` files.

**Cleanup:**

```bash
# List stale streams
$ ls /dev/shm/*.im.shm

# Remove orphaned streams
$ rm /dev/shm/mystale_stream.im.shm
```

### 4.3. Segfault in Stream Access

**Symptom:** `SIGSEGV` when accessing `img.im->array.F`.

**Likely cause:** The `IMGID` was not resolved or the stream was deleted. Always check the return
value of `mkIMGID_from_name()` and verify `img.im != NULL` before accessing pixel data.

---

## 5. Build-Time Debugging

Compile with debug symbols for meaningful backtraces:

```bash
$ cd _build
$ cmake .. -DCMAKE_BUILD_TYPE=Debug
$ make -j$(nproc)
$ sudo make install
```

<!-- prettier-ignore -->
!!! important
    Remember to switch back to `Release` or `RelWithDebInfo` for production use — `Debug` builds disable
    optimizations and are significantly slower.

---

← [Documentation Index](../index.md)
