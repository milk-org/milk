---
name: milk-script-writer
description: Generate correct milk-cli scripts from
  natural language prompts, with full knowledge of
  the scripting language features
---

# milk-script-writer

This skill enables AI agents to write correct,
idiomatic `milk-cli` scripts from natural language
prompts. It contains a complete reference of the
scripting language, common idioms, and example
scripts.

## When to Use

- User asks to write a `.milk` script
- User wants to automate a milk workflow
- User needs to control FPS/streams/processes
  from a script
- User wants to create image processing pipelines
- User asks how to do something in the milk CLI
  scripting language

## Output Rules

Generated scripts MUST:

1. Start with `#!/usr/bin/env milk-script`
2. Use `set -e` for error safety
3. Clean up streams/images at end of script
4. Use functions for reusable logic
5. Document purpose with comments
6. Keep lines ≤ 100 characters
7. Use descriptive variable names (≥ 3 chars)
8. Prefer milk-native commands over shell calls

## Script Template

```bash
#!/usr/bin/env milk-script
#
# script_name.milk
# Brief description of what this script does.
#
# Usage: milk-cli -s script_name.milk [args]
#

set -e

# --- Configuration ---
_param1=${1:-default_value}
_param2=${2:-64}

# --- Main logic ---
# ...

echo "Done."
```

## Resource Files

For detailed references, read these files:

- `resources/cheatsheet.md` — One-page quick ref
- `resources/idioms.md` — Common patterns & recipes

Example scripts in `examples/`:

- `basic_image_ops.milk` — Calculator & image math
- `fps_control.milk` — FPS parameter access
- `stream_monitor.milk` — Stream metadata & wait
- `process_orchestration.milk` — Process lifecycle

---

## Critical: Differences from Bash

> These are the top sources of bugs when generating
> milk-cli scripts. Know them by heart.

### 1. Spaced `=` vs unspaced `=` in assignments

```bash
# Spaced '=': full calculator / image-expression evaluation
a = 2 + 3 * 4       # a is 14
# (can also use images/FPS/etc. in the expression)

# Unspaced '=': math-only evaluation via cli_var_set()
a=2+3               # a is "5" (pure math expression)
a=hello             # a is "hello" (no math operators → plain string)
```

### 2. `@` prefix reads FPS/process properties

```bash
val=@myfps.gain      # Read FPS param
@myfps.gain=0.5      # Write FPS param
fpsset myfps gain 0.5  # Alternate write
```

### 3. `@s.name.prop` for stream metadata

Dynamic stream names are supported — use `$VAR` inside
`@` tokens and it expands before namespace dispatch:

```bash
w=@s.myimg.xsize     # Stream width
h=@s.myimg.ysize     # Stream height
n=@s.myimg.naxis     # Number of axes

# Dynamic: variable holds the stream name
stream=myimg
w=@s.${stream}.xsize
```

### 4. Image variables are SHM streams

When the calculator detects a name that matches an
existing SHM image, operations become element-wise:

```bash
mem.mk2Dim img 64 64
img = img * 0.0 + 1.0   # Fill with 1.0
out = img + 5.0          # Element-wise add
mask = (img > 0.5)       # Boolean mask
s = itot(mask)           # Sum all pixels
```

### 5. Milk-specific test operators

```bash
[ -S streamname ]    # SHM stream exists
[ -F fpsname ]       # FPS instance exists
[ -P procname ]      # Process is active
[ -v varname ]       # Variable is defined
```

> **Caveat:** `-S` and `-F` check `/dev/shm/`
> hardcoded. On systems using a non-default
> `MILK_SHM_DIR` (e.g., `/milk/shm`), these
> tests will return false even for existing
> streams/FPS. Use direct access or `image_ID()`
> as alternatives.

### 6. `on_update` is blocking

```bash
on_update mystream { echo "stream updated" }
# Blocks until mystream is written to
```

### 7. Limited pipe and nesting support

- Pipes work via `popen()`, not full bash pipes
- `$()` command substitution does not nest
- No process substitution `<(cmd)`

---

## Language Reference

### Variables

| Feature | Syntax |
|---------|--------|
| Set | `VAR=val` (no spaces) |
| Math set | `VAR = expr` (spaces) |
| Read | `$VAR` or `${VAR}` |
| Unset | `unset VAR` |
| List | `vars` |
| Export | `export VAR=val` |
| Readonly | `readonly VAR=val` |
| Local | `local VAR=val` |
| Last status | `$?` |

**String operations:**
`${#v}` length, `${v:n:m}` substr,
`${v%%pat}` strip suffix, `${v##pat}` strip prefix,
`${v/p/r}` replace first, `${v//p/r}` replace all,
`${v^^}` upper, `${v,,}` lower,
`${v:-def}` default, `${v:=def}` assign default,
`${v:+alt}` alt if set, `${v:?msg}` error if unset,
`${!v}` indirect.

### Arrays

```bash
arr=(a b c)          # Create
${arr[0]}            # Element access
${arr[@]}            # All elements
${#arr[@]}           # Length
read -a arr          # Read into array
mapfile -t arr < f   # Lines to array
declare -A map       # Associative array
map[key]=value       # Assoc set
${map[key]}          # Assoc get
```

### Arithmetic

```bash
y=$(( x + 5 ))       # Integer arithmetic
(( x > 5 ))          # Arith conditional
let "x = 1 + 2"      # Let assignment
```

Operators: `+`, `-`, `*`, `/`, `%`,
`&`, `|`, `^`, `~`, `<<`, `>>`,
`<`, `<=`, `>`, `>=`, `==`, `!=`

### Native Calculator

Triggered by `VAR = expr` (spaced `=`).
Works on both scalars and images.

**Math functions** (scalar and image):
`sin`, `cos`, `tan`, `asin`, `acos`, `atan`,
`atan2`, `sinh`, `cosh`, `tanh`,
`exp`, `log`, `log2`, `log10`,
`sqrt`, `cbrt`, `pow`,
`floor`, `ceil`, `round`, `trunc`,
`abs`, `fabs`, `min`, `max`, `fmod`,
`where(cond, true_val, false_val)`

**Image reduction functions:**
`itot(img)` sum, `imean(img)` mean,
`imin(img)` min, `imax(img)` max,
`dot(a,b)` dot product,
`norm(img)` L2 norm,
`imcrop(img, x, y, w, h)` crop

### Flow Control

```bash
# If/elif/else
if [ $x -gt 5 ]; then
    echo big
elif [ $x -gt 2 ]; then
    echo medium
else
    echo small
fi

# While
while [ $n -lt 10 ]; do
    n=$(( n + 1 ))
done

# Until
until [ $done -eq 1 ]; do
    body
done

# For (word list)
for item in a b c; do
    echo $item
done

# For (brace expansion)
for i in {1..10}; do
    echo $i
done

# For (C-style)
for ((i=0; i<10; i++)); do
    body
done

# Case
case $mode in
    fast) echo speed ;;
    safe) echo caution ;;
    *)    echo default ;;
esac
```

`break [N]`, `continue [N]`, `return [val]`

### Test Operators

**Numeric:** `-eq` `-ne` `-gt` `-ge` `-lt` `-le`

**String:** `=` `!=` `-n` (non-empty)
`-z` (empty)

**File:** `-f` `-d` `-e` `-s` `-r` `-w` `-x` `-L`

**Milk:** `-S` (stream) `-F` (FPS)
`-P` (process) `-v` (variable)

**Extended:** `[[ $s =~ ^regex$ ]]`

**Negation:** `[ ! expr ]`

### Functions

```bash
function myfunc {
    local x=$1
    echo "arg: $x"
    return 0
}
myfunc hello
```

Params: `$1`..`$9`, `shift [N]`,
`getopts "ab:" opt`

### FPS Access

```bash
@fpsname.param           # Read value
@fpsname.param=value     # Write value
fpsset fpsname param val # Write (command)
fpsdump fpsname          # Dump all params
fpsdump --json fpsname   # Dump as JSON object
fpslist                  # List FPS instances
fpslist --json           # List as JSON array
waitfor_fps name 10      # Wait up to 10s
wait -F name p=v 5       # Wait for param=val
```

### Stream Access

```bash
mem.mk2Dim name 64 64   # Create 2D image
mem.rm name 0            # Delete image
mem.listim               # List all images
streamlist               # List SHM streams
streamlist --json        # List as JSON array
@s.<name>.xsize          # Width
@s.<name>.ysize          # Height
@s.<name>.naxis          # Number of axes
@s.<name>.type           # Datatype code
@s.<name>.typename       # Datatype name
@s.<name>.cnt0           # Frame counter
@s.<name>.nelement       # Total elements
@s.${var}.xsize          # Dynamic name via $VAR
waitfor_stream name 10   # Wait up to 10s
wait -S name 5           # Wait for update
on_update name { cmd }   # Trigger on write
```

> Note: `mem.rm` accepts an optional second argument
> `errmode` (0=ignore, 1=warn, 2=err, 3=exit). If omitted,
> the default is `0` (ignore).
### Process Control

```bash
procctl name run         # Start
procctl name pause       # Pause
procctl name stop        # Stop
procwait name ACTIVE 10  # Wait for state
procstat name            # Show status
```

Process properties via `@name.prop`:
`pid`, `loopstat`, `loopcnt`, `loopfreq`,
`exectime`, `rtprio`, `ctrlval`, `trigmode`,
`statusmsg`, `tmux`, `description`

```bash
proclist                 # List active procs
proclist --json          # List as JSON array
```

### Unified Event Wait

```bash
# Block until any event fires; $? = event index
wait_any [-t T] S:stream F:fps.p=v P:proc:STATE
# Operators: = != >= <=
wait_any -t 10 S:cam F:dmcomb.gain>=0.5
wait_any -t -1 P:wfsloop:STOP P:wfsloop:CRASHED
# $? = 0..N-1 (event), 254 (timeout), 255 (error)
```

### System Snapshot

```bash
milkquery                    # Full JSON snapshot
milkquery --fps [pattern]    # FPS only
milkquery --streams [pat]    # Streams only
milkquery --procs            # Processes only
```

### Engine Event Traps

```bash
trap 'cmd' STREAM:name       # Non-blocking
trap 'cmd' FPS:f.p>=v        # fire-on-match
trap 'cmd' PROC:n:STATE      # proc state
trap -i 200 'cmd' STREAM:s   # 200ms throttle
trap -n 5 'cmd' STREAM:s     # fire 5x max
trap '' STREAM:name           # clear
trap -l                       # list all
```

### I/O

```bash
echo [-n] args           # Print
printf "fmt" args        # Formatted print
read [-p prompt] var     # Read input
read -t 5 var            # Timed read
read -a arr              # Read to array
cmd > file               # Redirect out
cmd >> file              # Append
cmd < file               # Redirect in
read v <<< "string"      # Here-string
$(cmd)                   # Command subst
cmd &                    # Background
```

### Utility Commands

```bash
sleep N                  # Pause (float OK)
usleep N                 # Microsecond pause
time cmd                 # Measure time
timeout N cmd            # Kill after N sec
watch -n N cmd           # Repeat every N sec
source file              # Execute script
basename path            # Filename part
dirname path             # Directory part
seq START [STEP] END     # Number sequence
pushd / popd / dirs      # Dir stack
alias n='cmd'            # Create alias
true / false             # Set $? to 0 / 1
```

### Error Handling

```bash
set -e                   # Exit on error
set -x                   # Trace commands
set +e / +x              # Disable above
trap 'cmd' EXIT INT TERM # Signal handler
```

### Command Chaining

```bash
cmd1 ; cmd2              # Sequential
cmd1 && cmd2             # AND
cmd1 || cmd2             # OR
```

---

## Known Gotchas

1. **Variable names**: Use ≥ 3 character names.
   Very short names like `_h` or `_w` may fail
   to expand due to parser edge cases. Use
   `_hh`, `_ww`, `_width`, `_height` instead.

2. **`mem.rm` errmode and defaults**: The second
   argument (`errmode`, e.g. `0` = ignore errors) is
   optional. `mem.rm imgname` and even `mem.rm` (no
   args) are accepted; when arguments are omitted,
   the command uses its default image selection and
   error-handling behavior (see `cmd? mem.rm`).

3. **Calc temp images leak**: The calculator
   creates `_tmpcalcN` images in SHM for
   intermediate results. The CLI normally
   deletes `_tmpcalc*` after each line and at
   the end of calculator evaluation, but if
   milk-cli is killed or crashes mid-eval,
   stale temps may remain and a subsequent
   startup may print `CALC_PARSER_ERROR`
   when it tries to resolve them.

4. **`-S` / `-F` test operators**: These check
   `/dev/shm/` hardcoded. On systems using
   `MILK_SHM_DIR=/milk/shm` (or similar),
   they will always return false.

5. **`printf` format**: Printf supports `%d`,
   `%s`, `%f` but format specifiers and
   variable expansion happen separately.
   Use `echo` for simple output.

---

## Discovering Commands

When the user needs a command you don't know:

```bash
m?                       # List all commands
m? modulename            # Commands in module
cmd? commandname         # Help for command
h? searchterm            # Search descriptions
fhelp                    # Fuzzy search
```

## Testing Generated Scripts

Run a script:
```bash
milk-cli -s script.milk
```

Validate syntax interactively:
```bash
milk-cli
source script.milk
```
