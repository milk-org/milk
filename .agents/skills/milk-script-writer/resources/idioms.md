# milk-cli Scripting Idioms

Common patterns and recipes for milk-cli scripts.
Use these as building blocks when generating scripts.

---

## 1. Script Boilerplate

```bash
#!/usr/bin/env milk-script
#
# myscript.milk — Brief description
#

set -e

# Parse arguments with defaults
_input=${1:-default_stream}
_size=${2:-128}

# Cleanup handler (remove temp images)
trap '_cleanup' EXIT
function _cleanup {
    mem.rm _tmp_a 0 2>/dev/null
    mem.rm _tmp_b 0 2>/dev/null
}
```

---

## 2. Stream Creation and Cleanup

```bash
# Create a temporary image, use it, clean up
mem.mk2Dim _work ${_xsize} ${_ysize}
_work = _work * 0.0    # Zero-fill

# ... computation ...

# Cleanup (or let trap handle it)
mem.rm _work 0
```

---

## 3. Wait for a Resource

```bash
# Wait for a stream to appear (10s timeout)
if ! waitfor_stream mystream 10; then
    echo "ERROR: stream not found"
    exit 1
fi

# Wait for an FPS to appear (5s timeout)
if ! waitfor_fps myfps 5; then
    echo "ERROR: FPS not available"
    exit 1
fi

# Wait for a process to be active
procwait myproc ACTIVE 10
if [ $? -ne 0 ]; then
    echo "ERROR: process didn't start"
    exit 1
fi
```

---

## 4. FPS Parameter Polling Loop

```bash
# Poll an FPS parameter until it reaches value
_timeout=30
_start=$(date +%s)
while true; do
    _val=@myfps.status
    if [ $_val = READY ]; then
        echo "FPS is ready"
        break
    fi
    _now=$(date +%s)
    _elapsed=$(( _now - _start ))
    if [ $_elapsed -gt $_timeout ]; then
        echo "ERROR: timeout"
        exit 1
    fi
    usleep 100000
done
```

Or use the built-in wait:

```bash
# Wait for FPS param to reach a value
wait -F myfps status=READY 30
if [ $? -ne 0 ]; then
    echo "ERROR: FPS not ready"
    exit 1
fi
```

---

## 5. Read/Write FPS Parameters

```bash
# Read a parameter
_gain=@loopctrl.gain
echo "Current gain: $_gain"

# Write a parameter
fpsset loopctrl gain 0.5

# Or using @ syntax
@loopctrl.gain=0.5

# Conditionally update
if [ @loopctrl.loopON -eq 0 ]; then
    fpsset loopctrl loopON 1
fi
```

---

## 6. Image Arithmetic Pipeline

```bash
# Create working images
mem.mk2Dim _raw 128 128
mem.mk2Dim _dark 128 128

# Fill dark with a constant
_dark = _dark * 0.0 + 100.0

# Subtract dark, normalize
_corrected = _raw - _dark
_maxval = imax(_corrected)
if [ $_maxval -gt 0 ]; then
    _norm = _corrected / $_maxval
fi
```

---

## 7. Image Statistics

```bash
# Query image statistics
_total = itot(myimage)
_avg = imean(myimage)
_lo = imin(myimage)
_hi = imax(myimage)

echo "sum=$_total mean=$_avg"
echo "min=$_lo max=$_hi"
```

---

## 8. Stream Metadata Query

```bash
# Check if stream exists
if [ -S mystream ]; then
    _w=@s.mystream.xsize
    _h=@s.mystream.ysize
    _t=@s.mystream.typename
    echo "Stream: ${_w}x${_h} ${_t}"
else
    echo "Stream not found"
fi
```

---

## 9. Process Lifecycle

```bash
# Start a process, wait, then stop
procctl myprocess run
procwait myprocess ACTIVE 10

# ... do work ...

procctl myprocess stop
procwait myprocess IDLE 10
```

---

## 10. Loop Over Stream Updates

```bash
# React to each stream update (10 iterations)
_count=0
while [ $_count -lt 10 ]; do
    wait -S mystream 5
    if [ $? -eq 0 ]; then
        _val = imean(mystream)
        echo "Frame $_count: mean=$_val"
        _count=$(( _count + 1 ))
    fi
done
```

---

## 11. Configuration via getopts

```bash
#!/usr/bin/env milk-script
# Parse -s <stream> -g <gain> -n <niter>

_stream=default
_gain=0.5
_niter=100

while getopts "s:g:n:" _opt; do
    case $_opt in
        s) _stream=$OPTARG ;;
        g) _gain=$OPTARG ;;
        n) _niter=$OPTARG ;;
        *) echo "Usage: -s stream"
           echo "       -g gain -n niter"
           exit 1 ;;
    esac
done
```

---

## 12. Error Handling with Traps

```bash
set -e

function _on_exit {
    # Clean up temp images
    for _img in _tmp1 _tmp2 _tmp3; do
        mem.rm $_img 0 2>/dev/null
    done
    echo "Cleanup complete"
}
trap '_on_exit' EXIT

function _on_error {
    echo "ERROR at line, aborting"
}
trap '_on_error' ERR
```

---

## 13. Iterating with Brace Expansion

```bash
# Process modes 00 through 09
for _m in {0..9}; do
    _name=mode_${_m}
    echo "Processing $_name"
done

# Step through gain values
for _g in 0.1 0.2 0.5 1.0; do
    fpsset loopctrl gain $_g
    sleep 2
    _avg = imean(residual)
    echo "gain=$_g mean=$_avg"
done
```

---

## 14. Conditional Stream/FPS Check

```bash
# Guard entire script on resource availability
for _req in stream1 stream2 stream3; do
    if [ ! -S $_req ]; then
        echo "ERROR: missing stream $_req"
        exit 1
    fi
done

# Check FPS exists before accessing
if [ -F loopctrl ]; then
    _gain=@loopctrl.gain
else
    _gain=0.0
fi
```

---

## 15. Background and Wait

```bash
# Run two commands in parallel
long_command_1 &
long_command_2 &
wait
echo "Both commands finished"
```

---

## 16. Saving and Loading FITS

```bash
# Load a FITS file into shared memory
iofits.loadfits darkframe.fits _dark

# Save shared memory to FITS
iofits.savefits _result output.fits
```

---

## Anti-Patterns to Avoid

1. **Don't use `VAR = value` for string assignment.**
   The spaced `=` triggers the calculator. Use
   `VAR=value` (no spaces) for strings.

2. **Don't assume bash process substitution.**
   `<(cmd)` is not supported. Use temp files or
   pipes instead.

3. **Don't nest `$()` command substitutions.**
   `$(cmd1 $(cmd2))` will not parse correctly.
   Use intermediate variables.

4. **Don't forget `mem.rm ... 0` for temp images.**
   SHM images persist after script exit. Clean up
   explicitly at end of script. `mem.rm` accepts an
   optional second argument `errmode` (e.g. `0` to ignore errors);
   it's good practice to pass it explicitly when cleaning up.
5. **Don't use `printf` in tight loops.**
   I/O in hot paths hurts performance. Guard
   with a verbosity check.
