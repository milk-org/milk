---
description: Guidance for running milk and cacao
  commands in the agent environment
---

# Running milk Commands

When running `milk-cli` or standalone executables
(`milk-fpsexec-*`, `cacao-fpsexec-*`), follow these
guidelines:

## Environment Setup

The milk environment must be sourced before running
any commands:

```bash
source ~/src/milk/local/bin/milk-setup.bash
```

## First-Time Build

After a fresh clone, initialize submodules before
building:

```bash
git submodule update --init --recursive
```

## Common Pitfalls

1. **Do not run `milk-cli` interactively** from the
   agent — it is a REPL and will block. Use
   standalone executables or pass commands via pipe:

   ```bash
   echo "command args" | milk-cli
   ```

2. **Shared memory permissions** — streams and FPS
   live in `/dev/shm/`. If a previous process
   crashed, stale SHM files may remain. Clean with:

   ```bash
   rm /dev/shm/fps.<name>.shm
   ```

3. **tmux sessions** — standalone executables with
   `-tmux` flag run inside tmux. Use `tmux ls` to
   see running sessions and `tmux kill-session -t`
   to clean up.

4. **Build directory** — always build from
   `~/src/milk/_build`, never from the source tree.
