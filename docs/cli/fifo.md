---
tags:
  - CLI
  - FIFO
  - Scripting
---

# FIFO Command Input

milk-cli supports **command FIFO** (named pipe) input, allowing external processes to inject
commands into a running CLI session. This enables powerful automation workflows where shell
pipelines feed commands into milk-cli in real time.

## Quick Start

```text
milk> fifo create
[fifo] opened: /milk/shm/.milk.fifo.12345 (fd=4)

milk> echo $MCLIFIFO
/milk/shm/.milk.fifo.12345
```

From another terminal:

```bash
echo "mem.listim" > /milk/shm/.milk.fifo.12345
```

The milk-cli session shows ingestion feedback:

```text
[fifo] ← "mem.listim"
... (command output) ...
[fifo] ✓ (0.003s)
```

## The `fifo` Command

| Sub-command          | Description                                                 |
| -------------------- | ----------------------------------------------------------- |
| `fifo create [path]` | Create and open a new FIFO. Auto-generates path if omitted. |
| `fifo open <path>`   | Connect to an existing FIFO at the given path.              |
| `fifo close`         | Close and remove the current FIFO.                          |
| `fifo on`            | Re-enable FIFO input (after `fifo off`).                    |
| `fifo off`           | Temporarily disable FIFO input without destroying the pipe. |
| `fifo status`        | Print current FIFO state, path, and file descriptor.        |
| `fifo`               | Same as `fifo status`.                                      |

## The `$MCLIFIFO` Variable

The special variable `$MCLIFIFO` expands to the current FIFO path when a FIFO is active, or an empty
string when no FIFO is open. This makes it easy to reference the FIFO in shell commands:

```text
milk> fifo create
milk> !echo "echo hello" > $MCLIFIFO
```

## Startup Flags

FIFOs can also be enabled at startup:

| Flag        | Description                                             |
| ----------- | ------------------------------------------------------- |
| `-f`        | Auto-create a FIFO (path based on process name and PID) |
| `-F <path>` | Create a FIFO at the specified path                     |

These are equivalent to running `fifo create` or `fifo create <path>` immediately after startup.

## Examples

### Batch Command Injection

Feed a list of commands from a file into a running milk-cli session:

```bash
# Terminal 1: start milk-cli with FIFO
milk-cli -f

# Terminal 2: send commands
cat commands.milk > $(cat /milk/shm/.milk.fifo.*)
```

### Dynamic Processing with awk

Process a list of images and save specific ones based on metadata:

```text
milk> fifo create
milk> !awk '{if ($4>200) print "iofits.saveFITS " $2 " " $2 "_save.fits"}' imlist.txt > $MCLIFIFO
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
milk> fifo create
milk> fifo off
[fifo] input disabled

milk> # ... do interactive work ...

milk> fifo on
[fifo] input enabled
```

### Connecting to Another Session's FIFO

If another milk-cli session has a FIFO open, you can connect to it:

```text
milk> fifo open /milk/shm/.othersession.fifo.54321
[fifo] opened: /milk/shm/.othersession.fifo.54321 (fd=4)
```

### Using with xargs

Combine `find`, `xargs`, and FIFO to batch-process files:

```text
milk> fifo create
milk> !find /data/frames -name "*.fits" | xargs -I {} echo "iofits.loadfits {} frame" > $MCLIFIFO
```

### Scripted FIFO Workflow

Create a shell script that starts milk-cli and feeds commands through the FIFO:

```bash
#!/bin/bash
# Start milk-cli with FIFO in the background
milk-cli -f &
MILKPID=$!
sleep 1

# Find the FIFO path
FIFO=$(ls /milk/shm/.milk.fifo.${MILKPID})

# Send commands
echo "mem.mk2Dim im1 128 128" > "$FIFO"
echo "mem.mk2Dim im2 256 256" > "$FIFO"
echo "mem.listim" > "$FIFO"
echo "exit" > "$FIFO"

wait $MILKPID
```
