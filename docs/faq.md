# FAQ & Troubleshooting

Common issues and solutions when building, installing, and
running `milk`.

---

## Installation

<details>
<summary><b>CMake cannot find cfitsio</b></summary>

```
Could NOT find CFITSIO
```

**Solution:** Install the development headers:
```bash
# Ubuntu/Debian
sudo apt-get install libcfitsio-dev

# CentOS/RHEL
sudo yum install cfitsio-devel
```

Or build without cfitsio:
```bash
cmake .. -DUSE_CFITSIO=OFF
```

See [Build Tiers](install/build_tiers.md) for details.

</details>

<details>
<summary><b>Build fails with missing readline/ncurses</b></summary>

```
fatal error: readline/readline.h: No such file or directory
```

**Solution:** Install the development headers, or build without CLI:
```bash
# Install headers
sudo apt-get install libreadline-dev libncurses5-dev

# Or build standalone-only (no interactive CLI)
cmake .. -DUSE_CLI=OFF
```

</details>

<details>
<summary><b>Library not found at runtime</b></summary>

```
error while loading shared libraries: libImageStreamIO.so
```

**Solution:** Add the install directory to the linker path:
```bash
echo "/usr/local/lib" > usrlocal.conf
sudo mv usrlocal.conf /etc/ld.so.conf.d/
sudo ldconfig
```

Or set `LD_LIBRARY_PATH` in your shell profile:
```bash
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
```

</details>

---

## Shared Memory

<details>
<summary><b>Permission denied when accessing /milk/shm</b></summary>

```
Cannot open /milk/shm/stream.im.shm: Permission denied
```

**Solution:** Ensure the SHM directory exists and is writable:
```bash
sudo mkdir -p /milk/shm
sudo chmod 1777 /milk/shm
```

For best performance, mount as tmpfs:
```bash
echo "tmpfs /milk/shm tmpfs rw,nosuid,nodev" | sudo tee -a /etc/fstab
sudo mount /milk/shm
```

</details>

<details>
<summary><b>Stale shared memory files</b></summary>

Old `.im.shm` files from crashed processes can interfere.

**Solution:**
```bash
# Remove all stale SHM files
milk-shmimpurge

# Or remove a specific stream
milk-shmim-rm <streamname>
```

</details>

<details>
<summary><b>SHM directory location</b></summary>

The default shared memory directory is `/milk/shm`. Override with:
```bash
export MILK_SHM_DIR=/path/to/custom/shm
```

</details>

---

## FPS / Process Control

<details>
<summary><b>FPS process won't start — "FPS already exists"</b></summary>

```
ERROR: FPS already exists
```

**Solution:** Remove the stale FPS, then retry:
```bash
milk-fps-set <fpsname> ..delete
```

</details>

<details>
<summary><b>milk-fpsCTRL shows no processes</b></summary>

Ensure the processinfo SHM directory exists and processes are
registered:
```bash
# Check for processinfo files
ls $MILK_SHM_DIR/proc.*.shm

# Scan for processes
milk-procinfo-list
```

</details>

<details>
<summary><b>tmux dispatch not working</b></summary>

If standalone executables launched with `-tmux` don't appear:

1. Ensure `tmux` is installed: `which tmux`
2. Check if the tmux session exists: `tmux ls`
3. Verify the FPS name has no spaces or special characters.

</details>

---

## CLI

<details>
<summary><b>milk-cli prompt jumps to bottom of terminal</b></summary>

This can happen when the startup banner clears the screen.

**Solution:** This is a known cosmetic issue. The prompt will
stabilize after the first command.

</details>

<details>
<summary><b>Command not found — "Unknown command"</b></summary>

```
Unknown command: mycommand
```

**Solution:** Check that the module is loaded:
```bash
# List all loaded modules
> m?

# Search for a command
> h? mycommand
```

If the module is a plugin, ensure it was compiled and the `.so`
file is in the library path.

</details>

---

## Performance

<details>
<summary><b>Real-time scheduling</b></summary>

For latency-critical applications (AO loops), configure real-time
scheduling:

```bash
# Create cpuset and enable RT scheduling
milk-makecsetandrt

# Set process priority (0-99, higher = higher priority)
milk -p 90
```

</details>

<details>
<summary><b>Semaphore loop speed</b></summary>

Benchmark semaphore performance:
```bash
milk-semloopspeed
```
Typical values: >100 kHz on modern hardware.

</details>

---

## Getting Help

- **CLI help:** Type `?` or `help` at the milk prompt
- **Command help:** `cmd? <command>` for detailed usage
- **Module list:** `m?` to list all loaded modules
- **Documentation:** See [docs/index.md](index.md)
- **Issues:** Report on [GitHub Issues](https://github.com/milk-org/milk/issues)
