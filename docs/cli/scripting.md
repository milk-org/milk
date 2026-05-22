---
tags:
  - cli
  - scripting
---

# Scripting

The `milk-cli` interpreter is designed to provide a seamless scripting experience by transparently integrating with your system's `bash` shell. Any command that is not a native `milk-cli` command is automatically evaluated by the underlying OS shell using `wordexp()`.

This means **all standard Bash scripting features are fully supported** inside `milk-cli`, including:

- Variables (`$var`), Arithmetic (`$(( ))`), array indexing, and bash-style parameter expansions
- Flow Control (`if`, `for`, `while`, `case`), including C-style `for ((i=0; i<N; i++))` loops
- Built-ins (`sleep`, `read`, `printf`, `trap`, `shift`)
- I/O Redirection (`>`. `<`. `|`), Heredocs, and Background Jobs (`&`)

This page documents the **native `milk-cli` extensions** and how to use them alongside standard bash in your scripts.

See also: [CLI Syntax](CLIcore.md) · [FPS](../fps.md) · [Streams](../streams.md)

## Script Files

### Running Scripts

You can execute a milk script file in several ways:

- **Shebang**: Use `#!/usr/bin/env milk-script` as the first line.
  The standalone `milk-script` binary runs the file
  non-interactively with no readline dependency.
- **Command line**: Run `milk-cli -s script.milk` from a shell.
- **Interactive**: Use the `source` (or `.`) command from within
  an active `milk-cli` session.

```bash
#!/usr/bin/env milk-script

# setup.milk — initialize processing pipeline
mem.mk2Dim wfs 128 128
echo "Environment initialized."
```

```bash
# Inside milk-cli interactive mode:
source setup.milk
```

### Include Guard

`include_once` sources a file only once per session, even if called multiple times.
This is useful for loading helper function libraries without redundant evaluation:

```bash
#!/usr/bin/env milk-script
include_once helpers.milk
include_once helpers.milk   # no-op
```

### Startup Profile

If the `~/.milkrc` file exists, it is automatically sourced at interactive startup.
Use this file to define persistent aliases, load standard variables, or register helper functions.

### Saving State

Export all current variables, aliases, and function definitions to a file so they can be reloaded later:

```bash
#!/usr/bin/env milk-script
savescript state.milk
```

Similarly, write your interactive Readline command history to a replayable script text file:

```bash
#!/usr/bin/env milk-script
savehistory replay.milk
```

## `milk-cli` Native Features

The following commands and syntactic expansions are implemented natively by the `milk-cli` engine logic and take precedence over standard bash built-ins.

### Stream Event Triggers (`on_update`)

The `on_update` command blocks execution until a shared-memory stream posts its semaphore (i.e., a new frame arrives), then executes a provided command block.

This is extremely useful for event-driven processing and synchronous scripts:

```bash
#!/usr/bin/env milk-script
on_update wfs_cam { echo "New WFS frame received!" }
```

### Enhanced Conditional Tests (`[ ]`)

The native interpreter supports a robust set of file and logic tests inside `[ ]` that run instantly without invoking `test` subprocesses:

- **Filesystem**: `-f` (file), `-d` (directory), `-e` (exists), `-s` (non-empty), `-r` (readable), `-w` (writable), `-x` (executable), `-L` (symlink).
- **Strings**: `-n` (not empty), `-z` (empty), `==`, `!=`, `=~` (POSIX regex match).
- **Numbers**: `-eq`, `-ne`, `-lt`, `-le`, `-gt`, `-ge`.
- **Logic**: `-a` (AND), `-o` (OR), `!` (NOT).

### Wait for Resources (`waitfor_*`)

You can tell your script to pause and wait for shared memory streams or FPS parameter blocks to be initialized by other compute units before proceeding:

```bash
#!/usr/bin/env milk-script
waitfor_stream wfs_cam 30    # wait up to 30s for the stream
waitfor_fps dmcomb 10        # wait up to 10s for the FPS
```

This returns `0` on success and `1` on timeout. The default timeout if omitted is 10 seconds.
This allows for robust orchestration in startup scripts:

```bash
#!/usr/bin/env milk-script
waitfor_stream wfs_cam 60
if [ $? -eq 0 ]; then
    echo "Stream is ready!"
fi
```

### Unified Event Multiplexing

The `wait_any` command blocks until **any one** of multiple heterogeneous events fires, eliminating polling loops in orchestration scripts:

```bash
wait_any [-t timeout] event1 [event2] [event3] ...
```

Each event is a single token with a prefix:

| Prefix | Fires when | Example |
|--------|------------|---------|
| `S:<stream>` | Stream `cnt0` changes | `S:wfs_cam` |
| `F:<fps>.<param><op><val>` | FPS param matches | `F:dmcomb.abort=1` |
| `P:<proc>:<state>` | Process reaches state | `P:wfsloop:STOP` |

FPS comparison operators: `=`, `!=`, `>=`, `<=`.

The return value `$?` is the 0-based index of the event that fired, `254` on timeout, or `255` on parse error.

```bash
#!/usr/bin/env milk-script
# Wait for new frame OR operator abort
wait_any -t 60 S:wfs_cam F:dmcomb.abort=1
if [ $? -eq 0 ]; then
    echo "New frame arrived"
elif [ $? -eq 1 ]; then
    echo "Operator requested abort"
fi
```

### Engine Event Traps

The `trap` command supports non-blocking engine event handlers using the same prefix syntax as `wait_any`. Traps fire automatically between CLI commands, without blocking the main thread.

```bash
# Register traps
trap 'echo "new frame"' STREAM:wfs_cam
trap 'echo "gain changed"' FPS:dmcomb.loopgain>=0.5
trap 'echo "loop stopped"' PROC:wfsloop:STOP

# Optional flags (before the quoted command)
trap -i 200 'echo "throttled"' STREAM:fast_cam   # min 200ms between fires
trap -n 5 'echo "five times"' STREAM:slow_cam    # fire at most 5 times

# Clear a trap
trap '' STREAM:wfs_cam

# List all active traps (POSIX + engine)
trap -l
```

**Auto-re-arm**: Engine traps automatically re-arm after firing. The baseline state (e.g., stream `cnt0`, FPS parameter value) is updated after each fire, so the same event won't trigger twice.

**Throttle**: A minimum interval (default 100ms) prevents high-frequency streams from flooding the handler. Override with `-i ms` (set to 0 for no throttle). Use `-n N` to limit the total number of fires.

**Supported events**: Same event types as `wait_any`, using the engine-trap forms `STREAM:name`, `FPS:fps.param<op>val`, and `PROC:name:STATE`.

### System Snapshot (`milkquery`)

The `milkquery` command emits a unified JSON object containing FPS instances, shared-memory streams, and active processes in one call:

```bash
milkquery                     # Full snapshot
milkquery --fps [pattern]     # FPS instances only
milkquery --streams [pattern] # Streams only
milkquery --procs             # Active processes only
```

Output is a JSON object with top-level keys `"fps"`, `"streams"`, and `"processes"`, each containing an array of entries. When a filter flag is used, only the requested section(s) appear.

Individual list commands also support `--json`:

```bash
fpslist --json [pattern]      # Same as milkquery --fps
streamlist --json [pattern]   # Same as milkquery --streams
proclist --json               # Same as milkquery --procs
```

### FPS Parameter Expansion

You can effortlessly read FPS (Function Processing System) parameters directly into bash variables using the native `@fpsname.param` syntax:

```bash
#!/usr/bin/env milk-script
echo "The current gain is: @myloop.loopgain"

# Store in a variable
g=@myloop.loopgain
```

To write FPS parameters programmatically, natively use the `fpsset` command:

```bash
#!/usr/bin/env milk-script
# Note: Do not use the '@' prefix when writing
fpsset myloop loopgain 0.5
```

### Advanced Variable Expansion and Arrays

`milk-cli`'s native interpreter includes powerful bash-like variable expansions:

- **Default values**: `${var:-default}` returns `default` if `var` is empty/unset.
- **Assign default**: `${var:=default}` sets `var` to `default` if it was empty/unset.
- **Error if empty**: `${var:?message}` prints an error if `var` is empty/unset.
- **Substring**: `${var:offset:length}` returns a slice of the string. Negative offsets count from the end.
- **String length**: `${#var}` returns the number of characters in the variable.
- **Array indexing**: `${myarray[idx]}` or `${myassoc[key]}` returns the value at the given element of the respective array.
- **Array splat**: `${myarray[@]}` expands to all elements in the array joined by a space.
- **Array size**: `${#myarray[@]}` returns the number of elements in the array.
For mathematical expressions, the `$(( ... ))` expansion natively supports standard arithmetic and bitwise logic:

- Basic operators: `+  -  *  /  %`
- Bitwise operators: `&  |  ^  <<  >>  ~`
- Comparisons: `==  !=  <  >  <=  >=`

### Stream & FPS Metadata Expansion

Similar to FPS parameters, you can access properties of shared memory streams via the `@s.` namespace expansion:

```bash
#!/usr/bin/env milk-script
echo "Geometry: ${@s.mystream.xsize}x${@s.mystream.ysize}x${@s.mystream.zsize}"
echo "Datatype Code: ${@s.mystream.type}"
echo "Frame Counter: ${@s.mystream.cnt0}"
echo "Number of axes: ${@s.mystream.naxis}"
```

For FPS compute units, you can check their allocation status:

```bash
#!/usr/bin/env milk-script
echo "Status: ${myfps.status}"     # 1 if exists, 0 if unconnected
```

## Scripting Showcase

Here are several examples demonstrating how `milk-cli` native features combine with standard bash utilities to form powerful automation scripts.

??? example "Example 1: Parameter Defaults and String Manipulation"

    Because `milk-cli` correctly delegates back to `bash`, you can safely use standard recursive shell expansions (e.g. `##` suffix stripping) to quickly parse paths:

    ```bash
    #!/usr/bin/env milk-script
    function process_image {
        local img_path=${1:-/data/default.fits}
        local filename=${img_path##*/}   # strip path prefix
        local basename=${filename%.*}    # strip extension

        echo "Processing ${basename}..."
    }

    process_image
    process_image /tmp/test_image.fits
    ```

    **Output:**
    ```text
    Processing default...
    Processing test_image...
    ```

??? example "Example 2: Transparent OS Fallback"

    Combine Linux shell utilities directly with native `milk-cli` commands. In scripts, there is no need to prefix standard bash commands with `!` like inside interactive mode.

    ```bash
    #!/usr/bin/env milk-script
    prefix="output_"
    ext=".fits"

    # Run the native milk command to process the file
    milk-FITS2shm image.fits wfs_cam

    # Use standard shell binaries to print system info
    count=$(ls -1 | grep -c "${ext}")
    echo "Found $count FITS files."

    tag=$(date +%Y%m%d_%H%M%S)
    echo "Saving to: ${prefix}${tag}${ext}"
    ```

??? example "Example 3: Waiting for Streams and reading Metadata"

    Block until multiple shared-memory streams become available during startup, then dynamically read their geometry properties via native dot-expansion:

    ```bash
    #!/usr/bin/env milk-script
    function wait_and_monitor {
        local stream=$1
        echo "Waiting for stream ${stream}..."

        waitfor_stream $stream 60
        if [ $? -ne 0 ]; then
            echo "Error: Stream ${stream} timed out."
            return 1
        fi

        # Read stream metadata via @ namespace
        echo "Ready! Shape: ${@s.${stream}.xsize} x ${@s.${stream}.ysize}"
    }

    wait_and_monitor wfs_cam
    wait_and_monitor dm_disp
    ```

??? example "Example 4: Batch FPS Parameter Manipulation"

    Read and write FPS parameters from a script to automate configuration changes across multiple compute units, leveraging standard bash `for` loop integer evaluation to calculate vector indices:

    ```bash
    #!/usr/bin/env milk-script

    # Retrieve current value and log it
    gain=$(milk-fps-set dmcomb.loopgain)
    echo "DM combiner gain was: $gain"

    # Set the loop gain on the DM combiner FPS native memory
    fpsset dmcomb loopgain 1.0

    # Apply identical gain to all modal channels using a bash loop
    nmodes=50
    for m in $(seq 0 $(( nmodes - 1 ))); do
        fpsset dmcomb modesgain[$m] 0.1
    done

    echo "Set $nmodes modal gains to 0.1"
    ```

??? example "Example 5: Stream Diagnostic Report"

    Collect live metadata from multiple streams and natively format a compact diagnostic table using string expansion and alignment flags.

    ```bash
    #!/usr/bin/env milk-script
    streams=(wfs_cam dm_disp wfs_ref)

    echo "--------------------------------------------"
    echo "  Stream           XSize  YSize  Frame"
    echo "--------------------------------------------"

    for s in ${streams[@]}; do
        waitfor_stream $s 5
        if [ $? -ne 0 ]; then
            printf "  %-18s OFFLINE\n" $s
            continue
        fi

        # The @s.${s}.prop syntax queries SHM image
        # metadata directly; $VAR substitution inside
        # @ tokens is supported so stream names can
        # be dynamic.
        xs=@s.${s}.xsize
        ys=@s.${s}.ysize
        cnt=@s.${s}.cnt0
        printf "  %-18s %-6s %-6s %s\n" $s $xs $ys $cnt
    done

    echo "--------------------------------------------"
    ```

??? example "Example 6: AO Loop Startup Orchestration"

    A complete startup script that initialises an AO loop step by step, verifies each stage using `milk-cli` conditionals, and aborts cleanly on hardware failure:

    ```bash
    #!/usr/bin/env milk-script

    # ---------- helpers ----------
    function die {
        echo "FATAL: $1"
        exit 1
    }

    function wait_stream {
        local s=$1 timeout=${2:-30}
        waitfor_stream $s $timeout
        [ $? -ne 0 ] && die "Stream '$s' not available after ${timeout}s"
        echo "  [OK] $s"
    }

    # ---------- 1. verify hardware streams ----------
    echo "=== 1. Checking hardware streams ==="
    wait_stream wfs_cam 60
    wait_stream dm_volt

    # ---------- 2. load reference PSF ----------
    echo "=== 2. Loading WFS reference ==="
    milk-FITS2shm wfs_ref.fits wfs_ref
    wait_stream wfs_ref

    # ---------- 3. start modal decomposition FPS ----------
    echo "=== 3. Starting modal decomposition ==="
    milk-fpsexec-cacaoloop-WFS -n wfs01 -tmux
    sleep 2
    waitfor_stream wfs_modes 20
    [ $? -ne 0 ] && die "Modal decomposition failed to produce wfs_modes"

    # ---------- 4. configure loop gains ----------
    echo "=== 4. Configuring loop gains ==="
    nmodes=100
    for m in $(seq 0 $(( nmodes - 1 ))); do
        fpsset dmcomb modesgain[$m] 0.05
    done
    fpsset dmcomb loopgain 1.0
    fpsset dmcomb loopON 1

    # ---------- 5. confirm loop is running ----------
    echo "=== 5. Loop status ==="
    sleep 1
    fpsgain=$(milk-fps-set dmcomb.loopgain)
    echo "  loopgain = $fpsgain"
    echo "AO loop started successfully."
    ```

??? example "Example 7: Real-Time Process Triggering & Monitoring"

    Use FPS parameters to configure `procinfo` settings, binding a compute unit to trigger automatically on a stream and monitoring its health from the script.

    ```bash
    #!/usr/bin/env milk-script

    # 1. Start the compute unit process
    milk-fpsexec-examplefunc2_FPS -tmux &
    sleep 1

    # 2. Configure it to be triggered by a stream (Mode 3 = SEMAPHORE)
    fpsset examplefunc2_FPS procinfo.triggermode 3
    fpsset examplefunc2_FPS procinfo.triggersname wfs_cam

    # 3. Enable the background loop
    fpsset examplefunc2_FPS procinfo.enabled 1

    # 4. Monitor its execution natively
    for i in {1..5}; do
        # Dot-expansion accesses the live telemetry via procinfo
        status=${examplefunc2_FPS.procinfo.status}
        loopcnt=${examplefunc2_FPS.procinfo.loopcnt}

        echo "Check $i: Status=$status, Iterations=$loopcnt"
        sleep 2
    done
    ```

??? example "Example 8: Advanced Native Math and Image Calculus"

    You can run mathematical expressions natively, manipulate image values directly with constants, and calculate norms and conditions efficiently without needing external parsing like `bc` or `awk`.

    ```bash
    #!/usr/bin/env milk-script

    # 1. Provide an initial constant to initialize variables
    a = 2 + 3 * 4
    echo "Variable 'a' evaluates natively to: $a"

    # 2. Utilize mathematical functions like abs, max, min natively
    e = min(abs(-5), max(3, fmod(17, 5)))
    echo "Compound math 'e' limits safely to: $e"

    # 3. Quickly generate and fill a virtual flat dummy space.
    mem.mk2Dim imtest 10 10
    imtest = imtest * 0.0 + 5.0

    # 4. Use vector constraints natively over memory blocks
    d_im_im = dot(imtest, imtest)
    n_im = norm(imtest)
    echo "Dot product: $d_im_im | Norm vector length: $n_im"

    # 5. Native Ternary Mask mapping limits logic cleanly:
    imtest_plus = imtest + 3
    imtest_mask = (imtest_plus == 8)
    im_where = where(imtest_mask, imtest, -imtest)
    h = imean(im_where)
    echo "Filtered conditionally merged mean of masked image: $h"
    ```

??? example "Example 9: Bi-Directional Environment Variable Sync"

    `milk-cli` natively exports all active workspace variables to the underlying POSIX environment when dispatching shell commands (like `!cmd` or pipes), and can concurrently read standard Linux environment variables without requiring manual exports.

    ```bash
    #!/usr/bin/env milk-script

    # 1. Read standard OS variables natively (Inward Sync)
    echo "Operating as user: $USER in home directory: $HOME"

    # 2. Define a milk-cli local variable
    PREFIX="WFS_DATA"

    # 3. Use it transparently inside a dispatched shell command (Outward Sync)
    !echo "Saving to prefix: $PREFIX" > /tmp/${PREFIX}_log.txt
    !cat /tmp/${PREFIX}_log.txt
    ```

??? example "Example 10: Advanced Hybrid Scripting (Bash + milk-cli Native)"

    Combine the robust argument parsing capabilities of `bash` with the zero-overhead execution of the `milk-cli` interpreter. By parsing arguments in bash and then dropping into a `milk-cli` heredoc, the script gains native access to FPS, streams, and image calculus without sacrificing traditional command-line interfaces.

    See `scripts/milk-script-advanced` for the full template.

    ```bash
    #!/usr/bin/env bash

    # 1. BASH HEADER: Argument Parsing
    MSdescr="Advanced Native milk-cli Script Template"
    MSarg+=( "streamname:string:Name of the input stream to wait for" )

    # Parse arguments (milk-argparse automatically exports $STREAMNAME)
    source milk-argparse

    # 2. NATIVE EXECUTION: Zero-Overhead milk-cli Block
    milk-cli -s << 'EOF'

    echo "Waiting natively for stream: $STREAMNAME..."
    # Event-Driven Execution without polling loops
    waitfor_stream $STREAMNAME 10

    # Fast Native Image Math & FPS
    mem.mk2Dim mask_img ${@s.in.xsize} ${@s.in.ysize}
    mask_img = mask_img * 0.0 + 1.0

    on_update $STREAMNAME {
        echo "Frame arrived!"
    }
    EOF
    ```

## Syntax Highlighting (Neovim)

The `tree-sitter-milkcli` package provides a
[tree-sitter](https://tree-sitter.github.io/) grammar
for `.milk` script files, enabling rich syntax
highlighting in **Neovim** (≥ 0.9) and other
tree-sitter-capable editors.

### Quick Setup

From the milk source tree:

```bash
cd tree-sitter-milkcli

# 1. Build the parser from the grammar
./scripts/build.sh

# 2. Install into Neovim (queries + config)
./scripts/nvim-install.sh

# 3. Open a .milk file — highlighting is active
nvim examples/demo.milk
```

The install script copies highlight queries to
`~/.config/nvim/after/queries/milkcli/` and creates
a Lua config snippet at
`~/.config/nvim/plugin/milkcli-treesitter.lua`.

### What Gets Highlighted

| Syntax Element | Highlight Group | Example |
|----------------|-----------------|---------|
| Flow control | `@keyword` | `if`, `for`, `while`, `fi` |
| Shell builtins | `@function.builtin` | `echo`, `export`, `source` |
| milk commands | `@function.macro` | `assigncheck`, `procctl` |
| FPS variables | `@property` | `@fps.loop.gain` |
| Stream metadata | `@type` | `${s.wfs.cnt0}` |
| Variables | `@variable.builtin` | `$VAR`, `${VAR}` |
| Strings | `@string` | `"hello"`, `'literal'` |
| Numbers | `@number` | `42`, `3.14` |
| Comments | `@comment` | `# comment` |

### Dynamic Module Commands

Module commands (`listim`, `saveFITS`, `imcrop`, etc.)
are registered at runtime, not hardcoded in the grammar.
To highlight them, run the generator script after
installing new milk modules:

```bash
# Generate and install module highlight queries
./scripts/gen-module-highlights.sh --install

# Or generate to stdout for inspection
./scripts/gen-module-highlights.sh --stdout
```

### Management

```bash
# Check installation status (dry-run)
./scripts/nvim-install.sh --check

# Remove from Neovim
./scripts/nvim-install.sh --uninstall

# Rebuild parser after grammar changes
./scripts/build.sh
./scripts/build.sh --test    # build + parse example
./scripts/build.sh --clean   # remove generated files
```

!!! info "Requirements"
    - **Node.js** — for the `tree-sitter-cli` build tool
    - **Neovim ≥ 0.9** — with built-in tree-sitter support
    - **nvim-treesitter** plugin — installed automatically
      by `nvim-install.sh`

---
← [CLI Syntax](CLIcore.md) · [Documentation Index](../index.md)
