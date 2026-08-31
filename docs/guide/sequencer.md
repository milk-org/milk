---
tags:
  - fps
  - sequencer
  - script
  - cli
---

# FPS Sequencer (`milk-seq`)

The `milk` framework provides a powerful standalone sequencer, `milk-seq`, which runs complex,
timing-sensitive command scripts across multiple FPS compute units. By decoupling the sequencer from
the `milk-fpsCTRL` TUI, `milk-seq` allows for robust, headless script execution and cross-process
orchestration via shared memory.

See also: [Function Processing System (FPS)](fps.md)

---

## 1. Architecture

`milk-seq` operates as an independent backend daemon. It creates a `MILKSEQ_STATE` structure in
`/dev/shm/milkseq.<name>.shm` and listens for commands on a designated FIFO pipe.

### The Component Stack

1. **`milk-seq` Daemon:** Headless orchestrator that executes `.seq` scripts and `milk-cli`
   commands.
2. **`milk-fpsCTRL` TUI:** Provides a real-time visual dashboard of the running sequencer. It now
   acts as a read-only viewer, reading the `MILKSEQ_STATE` from shared memory.
3. **`milk-cli` Commands (`seq.*`):** A suite of CLI commands for submitting inputs, checking
   status, and managing sequencer instances dynamically.

---

## 2. CLI Integration

You can manage sequencers directly from `milk-cli` or bash using the `seq.*` command suite:

| Command                            | Description                                                                     |
| ---------------------------------- | ------------------------------------------------------------------------------- |
| `seq.list`                         | List all running sequencer instances on the system.                             |
| `seq.start <name> [-f script.seq]` | Launch a new `milk-seq` daemon headless. Optionally start a script.             |
| `seq.stop <name>`                  | Safely terminate a sequencer daemon.                                            |
| `seq.status <name>`                | Display the current status, task counts, and errors for a specific sequencer.   |
| `seq.submit <name> <command>`      | Push an arbitrary command into a sequencer's FIFO pipe for immediate execution. |

**Example Workflow:**

```bash
milk-exec "seq.start calibloop -f calibration.seq"
milk-exec "seq.status calibloop"
milk-exec "seq.submit calibloop load next_step.seq"
```

---

## 3. Scripting Capabilities

The sequencer supports a subset of bash-like scripting constructs tailored for high-performance,
real-time AO scheduling. These commands are written in `.seq` files.

### 3.1 Loop Constructs

You can loop blocks of commands using `repeat N / endrepeat`:

```bash
repeat 5
    # Do something 5 times
    sleep 1.0
endrepeat
```

### 3.2 Conditional Blocks

Evaluate FPS configuration flags before executing commands:

```bash
if_fps_status dm_control ON
    echo "DM is running!"
endif
```

### 3.3 FPS Synchronization

The `wait_fps` and `wait_seq` commands pause execution until a condition is met in another process
or sequencer, ensuring lock-step synchronization.

```bash
# Block until FPS variable changes
wait_fps wfs_camera.frame_ready 1

# Wait for another sequencer to finish
wait_seq other_loop IDLE
```

### 3.4 Error Handling

You can define an error policy using `on_error`:

```bash
# Options: abort, skip, retry
on_error abort
```

### 3.5 Cross-Sequencer Injection

Use `seq_send` to inject a command into another running sequencer's FIFO securely:

```bash
seq_send other_loop "load failover.seq"
```

### 3.6 Modularity

You can compose scripts using the `include` directive:

```bash
# Insert contents of setup.seq inline
include setup.seq
```

---

## 4. The `milk-fpsCTRL` TUI

To visually monitor a running sequencer, launch `milk-fpsCTRL` and press `s` to toggle the Sequencer
panel.

The TUI will automatically locate the sequencer associated with the active module group. If multiple
sequencers exist, rely on the unified `MILKSEQ_STATE` tracking to view real-time process rates,
running tasks, and any reported errors.

The TUI handles the sequencer purely as a _viewer_. Pressing keys to start/stop the sequencer within
the TUI will inject commands into the sequencer's native FIFO path, ensuring thread-safe state
management.
