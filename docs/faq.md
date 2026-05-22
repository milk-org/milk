# FAQ & Troubleshooting

Common issues and solutions when building, installing, and
running `milk`.

***

## 1. Installation

??? faq "CMake cannot find cfitsio"

    ```text
    Could NOT find CFITSIO
    ```

    **Solution:** Install the development headers:

    === "Ubuntu/Debian"
        ```bash
        sudo apt-get install libcfitsio-dev
        ```

    === "CentOS/RHEL"
        ```bash
        sudo yum install cfitsio-devel
        ```

    Or build without cfitsio:

    ```bash
    cmake .. -DUSE_CFITSIO=OFF
    ```

    See [Build Tiers](install/build_tiers.md) and
    [Compile Instructions](install/compile.md) for details.

??? faq "Build fails with missing readline"

    ```text
    fatal error: readline/readline.h: No such file or directory
    ```

    **Solution:** Install the development headers, or build without CLI:

    === "Install headers"
        ```bash
        sudo apt-get install libreadline-dev
        ```

    === "Build standalone-only (No CLI)"
        ```bash
        cmake .. -DUSE_CLI=OFF
        ```

??? faq "Library not found at runtime"

    ```text
    error while loading shared libraries: libImageStreamIO.so
    ```

    **Solution:** Add the install directory to the linker path:

    === "System-wide (ldconfig)"
        ```bash
        echo "/usr/local/lib" > usrlocal.conf
        sudo mv usrlocal.conf /etc/ld.so.conf.d/
        sudo ldconfig
        ```

    === "Per-user (LD_LIBRARY_PATH)"
        ```bash
        export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
        ```

***

## 2. Shared Memory

See also: [Streams](streams.md)

??? faq "Permission denied when accessing /milk/shm"

    ```text
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

??? faq "Stale shared memory files"

    Old `.im.shm` files from crashed processes can interfere.

    **Solution:**

    ```bash
    milk-shmimpurge              # (1)!
    milk-shmim-rm <streamname>   # (2)!
    ```

    1. Remove all stale SHM files.
    2. Remove a specific stream.

??? faq "SHM directory location"

    The default shared memory directory is `/milk/shm`. Override with:

    ```bash
    export MILK_SHM_DIR=/path/to/custom/shm
    ```

***

## 3. FPS / Process Control

See also: [FPS](fps.md) ·
[Process Info](procinfo.md) ·
[FPS Standalone Modes](FPS_Standalone_CMD_Modes.md)

??? faq "FPS process won't start — \"FPS already exists\""

    ```text
    ERROR: FPS already exists
    ```

    **Solution:** Remove the stale FPS, then retry:

    ```bash
    milk-fps-set <fpsname> ..delete
    ```

??? faq "milk-fpsCTRL shows no processes"

    Ensure the processinfo SHM directory exists and processes are
    registered:

    ```bash
    ls $MILK_SHM_DIR/proc.*.shm   # (1)!
    milk-procinfo-list             # (2)!
    ```

    1. Check for processinfo files
    2. Scan for processes

??? faq "tmux dispatch not working"

    If standalone executables launched with `-tmux` don't appear:

    1. Ensure `tmux` is installed: `which tmux`
    2. Check if the tmux session exists: `tmux ls`
    3. Verify the FPS name has no spaces or special characters.

***

## 4. CLI

See also: [CLI Reference](cli/CLIcore.md)

??? faq "milk-cli prompt jumps to bottom of terminal"

    This can happen when the startup banner clears the screen.

    **Solution:** This is a known cosmetic issue. The prompt will
    stabilize after the first command.

??? faq "Command not found — \"Unknown command\""

    ```text
    Unknown command: mycommand
    ```

    **Solution:** Check that the module is loaded:

    ```text
    milk-cli > m?                      # (1)!
    milk-cli > h? mycommand            # (2)!
    ```

    1. List all loaded modules
    2. Search for a command

    If the module is a plugin, ensure it was compiled and the `.so`
    file is in the library path.

***

## 5. Performance

??? faq "Real-time scheduling"

    For latency-critical applications (AO loops), configure
    real-time scheduling:

    ```bash
    milk-makecsetandrt           # (1)!
    milk-cli -p 90               # (2)!
    ```

    1. Create cpuset, enable RT
    2. Launch with high priority

??? faq "Semaphore loop speed"

    Benchmark semaphore performance:

    ```bash
    milk-semloopspeed
    ```

    Typical values: >100 kHz on modern hardware.

***

## 6. Getting Help

- **CLI help:** Type `?` or `help` at the `milk-cli >` prompt
- **Command help:** `cmd? <command>` for detailed usage
- **Module list:** `m?` to list all loaded modules
- **Documentation:** See [docs/index.md](index.md)
- **Issues:** Report on [GitHub Issues](https://github.com/milk-org/milk/issues)

***
← [Documentation Index](index.md)
