# FAQ & Troubleshooting

Common issues and solutions when building, installing, and running `milk`.

---

## Installation

### CMake cannot find cfitsio

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

### Build fails with missing readline/ncurses

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

### Library not found at runtime

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

---

## Shared Memory

### Permission denied when accessing /milk/shm

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

### Stale shared memory files

Old `.im.shm` files from crashed processes can interfere.

**Solution:**
```bash
# Remove all stale SHM files
milk-shmimpurge

# Or remove a specific stream
milk-shmim-rm <streamname>
```

### SHM directory location

The default shared memory directory is `/milk/shm`. Override with:
```bash
export MILK_SHM_DIR=/path/to/custom/shm
```

---

## FPS / Process Control

### FPS process won't start

```
ERROR: FPS already exists
```

**Solution:** Remove the stale FPS, then retry:
```bash
milk-fps-set <fpsname> ..delete
```

### milk-fpsCTRL shows no processes

Ensure the processinfo SHM directory exists and processes are
registered:
```bash
# Check for processinfo files
ls $MILK_SHM_DIR/proc.*.shm

# Scan for processes
milk-procinfo-list
```

### tmux dispatch not working

If standalone executables launched with `-tmux` don't appear:

1. Ensure `tmux` is installed: `which tmux`
2. Check if the tmux session exists: `tmux ls`
3. Verify the FPS name has no spaces or special characters.

---

## CLI

### milk-cli prompt jumps to bottom of terminal

This can happen when the startup banner clears the screen.

**Solution:** This is a known cosmetic issue. The prompt will
stabilize after the first command.

### Command not found in CLI

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

---

## Performance

### Real-time scheduling

For latency-critical applications (AO loops), configure real-time
scheduling:

```bash
# Create cpuset and enable RT scheduling
milk-makecsetandrt

# Set process priority (0-99, higher = higher priority)
milk -p 90
```

### Semaphore loop speed

Benchmark semaphore performance:
```bash
milk-semloopspeed
```
Typical values: >100 kHz on modern hardware.

---

## Getting Help

- **CLI help:** Type `?` or `help` at the milk prompt
- **Command help:** `cmd? <command>` for detailed usage
- **Module list:** `m?` to list all loaded modules
- **Documentation:** See [docs/index.md](index.md)
- **Issues:** Report on [GitHub Issues](https://github.com/milk-org/milk/issues)
