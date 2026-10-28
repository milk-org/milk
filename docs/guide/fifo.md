---
tags:
  - CLI
  - FIFO
  - Scripting
---

# FIFO Command Input

`milk-cli` supports **command FIFO** (named pipe) input, allowing external processes to inject
commands into a running interactive `milk-cli` process.

<!-- prettier-ignore -->
!!! tip
    To share images, variables, and commands across multiple shells or scripts without a named pipe,
    use named sessions (`@session new NAME` / `@session use NAME`; see
    [Sessions](sessions.md)).

## Quick Start

```text
milk-cli > milk.fifo create /tmp/myctl.fifo
[fifo] opened: /tmp/myctl.fifo (fd=4)
```

From another terminal:

```bash
echo "mem.listim" > /tmp/myctl.fifo
```

The `milk-cli` session shows ingestion feedback:

```text
[fifo] ← "mem.listim"
... (command output) ...
[fifo] ✓ (0.003s)
```

## The `milk.fifo` Command

| Sub-command               | Description                                                 |
| ------------------------- | ----------------------------------------------------------- |
| `milk.fifo create [path]` | Create and open a new FIFO. Auto-generates path if omitted. |
| `milk.fifo open <path>`   | Connect to an existing FIFO at the given path.              |
| `milk.fifo close`         | Close and remove the current FIFO.                          |
| `milk.fifo on`            | Re-enable FIFO input (after `milk.fifo off`).               |
| `milk.fifo off`           | Temporarily disable FIFO input without destroying the pipe. |
| `milk.fifo status`        | Print current FIFO state, path, and file descriptor.        |
| `milk.fifo`               | Same as `milk.fifo status`.                                 |

When `milk.fifo create` is called without a path (or `-f` is passed at startup), the default path is
`$MILK_SHM_DIR/.<processname>.fifo.<07d-PID>` (for example, `/milk/shm/.myctl.fifo.0012345` when
started with `milk-cli -n myctl -f`).

## Startup Flags

FIFOs can also be enabled at startup:

| Flag        | Description                                                             |
| ----------- | ----------------------------------------------------------------------- |
| `-f`        | Auto-create a FIFO (`$MILK_SHM_DIR/.<processname>.fifo.<07d-PID>`)      |
| `-F <path>` | Create a FIFO at the specified path                                     |

These are equivalent to running `milk.fifo create` or `milk.fifo create <path>` immediately after
startup.

## Examples

### Batch Command Injection

Feed a list of commands from a file into a running `milk-cli` session:

```bash
# Terminal 1: start milk-cli with a custom FIFO path
milk-cli -F /tmp/myctl.fifo

# Terminal 2: send commands
cat commands.milk > /tmp/myctl.fifo
```

### Dynamic Processing with awk

Process a list of images and save specific ones based on metadata:

```text
milk-cli > milk.fifo create /tmp/myctl.fifo
milk-cli > awk '{if ($4>200) print "iofits.saveFITS " $2 " " $2 "_save.fits"}' imlist.txt > /tmp/myctl.fifo
```

Each generated command appears with ingestion feedback:

```text
[fifo] ← "iofits.saveFITS im003 im003_save.fits"
[fifo] ✓ (0.012s)
[fifo] ← "iofits.saveFITS im007 im007_save.fits"
[fifo] ✓ (0.015s)
```

### Pausing and Resuming FIFO Input

Temporarily disable FIFO input while doing interactive work, then re-enable it:

```text
milk-cli > milk.fifo create /tmp/myctl.fifo
milk-cli > milk.fifo off
[fifo] input disabled

milk-cli > # ... do interactive work ...

milk-cli > milk.fifo on
[fifo] input enabled
```

### Connecting to an Existing FIFO

Connect to an existing named pipe:

```text
milk-cli > milk.fifo open /tmp/shared.fifo
[fifo] opened: /tmp/shared.fifo (fd=4)
```

### Using with xargs

Combine `find`, `xargs`, and a FIFO to batch-process files:

```text
milk-cli > milk.fifo create /tmp/myctl.fifo
milk-cli > find /data/frames -name "*.fits" | xargs -I {} echo "iofits.loadfits {} frame" > /tmp/myctl.fifo
```

### Scripted FIFO Workflow

Create a shell script that starts `milk-cli` and feeds commands through the FIFO:

```bash
#!/bin/bash
FIFO="/tmp/milk.fifo.$$"
milk-cli -F "$FIFO" &
MILKPID=$!
sleep 1

# Send commands
echo "mem.mk2Dim im1 128 128" > "$FIFO"
echo "mem.mk2Dim im2 256 256" > "$FIFO"
echo "mem.listim" > "$FIFO"
echo "exit" > "$FIFO"

wait $MILKPID
```
