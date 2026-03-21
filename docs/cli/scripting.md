---
tags:
  - cli
  - scripting
---

# Scripting

The `milk-cli` shell supports bash-like scripting
constructs: variables, arithmetic, flow control,
user-defined functions, and script files. This page
documents the full scripting system.

See also: [CLI Syntax](CLIcore.md) ·
[FPS](../fps.md) · [Streams](../streams.md)

## Variables

<details markdown="1"><summary><b>Click to expand</b></summary>

### Setting and Reading

```bash
#!/usr/bin/env milk-cli -s
x=42
name=hello
echo $x          # prints: 42
echo ${name}     # prints: hello
```

Variable names are case-sensitive and may contain
letters, digits, and underscores. No spaces around `=`.

### Listing and Removing

```bash
#!/usr/bin/env milk-cli -s
vars              # list all variables
unset x           # remove variable x
```

### Special Variables

| Variable | Description |
|----------|-------------|
| `$?` | Exit status of last command (0=success) |
| `$HOME`, `$PATH`, … | Environment variables |

### String Operations

The `${...}` expansion supports bash-like string
manipulation:

| Syntax | Description |
|--------|-------------|
| `${#var}` | String length |
| `${var:offset:length}` | Substring extraction |
| `${var%%pattern}` | Strip longest suffix match |
| `${var##pattern}` | Strip longest prefix match |
| `${var^^}` | Uppercase all characters |
| `${var,,}` | Lowercase all characters |
| `${var^}` | Capitalize first character |
| `${var,}` | Lowercase first character |

```bash
#!/usr/bin/env milk-cli -s
path=/home/user/file.txt
echo ${#path}           # prints: 19
echo ${path:6:4}        # prints: user
echo ${path%%/*}        # prints: (empty)
echo ${path##*/}        # prints: file.txt
name=hello
echo ${name^^}          # prints: HELLO
name=WORLD
echo ${name,,}          # prints: world
```

### Array Variables

Arrays store multiple values indexed by integer:

```bash
#!/usr/bin/env milk-cli -s
colors=(red green blue)
echo ${colors[0]}       # prints: red
echo ${colors[2]}       # prints: blue
echo ${colors[@]}       # prints: red green blue
echo ${#colors[@]}      # prints: 3
```

### FPS Parameter Access

Read FPS parameters using `@fpsname.param` syntax:

```bash
#!/usr/bin/env milk-cli -s
echo @myloop.loopgain      # read FPS param
g=@myloop.loopgain         # store in variable
```

Write FPS parameters with `fpsset`:

```bash
#!/usr/bin/env milk-cli -s
fpsset myloop loopgain 0.5
```


</details>

## Arithmetic

<details markdown="1"><summary><b>Click to expand</b></summary>

`$(( expr ))` evaluates integer arithmetic with
`+`, `-`, `*`, `/`, `%` operators. Variables and FPS
parameters are expanded inside the expression:

```bash
#!/usr/bin/env milk-cli -s
x=10
y=$(( x + 5 ))        # y = 15
z=$(( y * 2 - x ))    # z = 20
echo $y $z            # prints: 15 20
```


</details>

## Built-in Commands

<details markdown="1"><summary><b>Click to expand</b></summary>

### sleep

Pause execution for a given number of seconds
(float-capable):

```bash
#!/usr/bin/env milk-cli -s
sleep 1.5             # pause 1.5 seconds
sleep 0.01            # 10 ms pause
```

### printf

Formatted output supporting `%s`, `%d`, `%f`, `%%`,
and `\n`, `\t` escape sequences:

```bash
#!/usr/bin/env milk-cli -s
printf "value = %d\n" 42
printf "%s has %d items\n" list 5
```

### read

Read a line from standard input into a variable:

```bash
#!/usr/bin/env milk-cli -s
read name
hello world
echo $name         # prints: hello world
```

Flags:

| Flag | Description |
|------|-------------|
| `-p "prompt"` | Display a prompt before reading |
| `-t N` | Timeout after N seconds |
| `-a arrayname` | Split words into an array |

```bash
#!/usr/bin/env milk-cli -s
read -p "Enter value: " val
Enter value: 42
echo $val           # prints: 42
read -t 5 response  # wait up to 5 sec
read -a words       # split into array
hello world
echo ${words[0]}    # prints: hello
```

### Logical Operators

Chain commands with `&&` (AND) and `||` (OR):

```bash
#!/usr/bin/env milk-cli -s
cmd1 && cmd2        # cmd2 runs if cmd1 succeeds
cmd1 || cmd2        # cmd2 runs if cmd1 fails
```

### Brace Expansion

Expand `{N..M}` and `{N..M..S}` into integer
sequences:

```bash
#!/usr/bin/env milk-cli -s
echo {1..5}         # prints: 1 2 3 4 5
echo {0..10..2}     # prints: 0 2 4 6 8 10
echo {5..1}         # prints: 5 4 3 2 1
```

Useful with `for` loops:

```bash
#!/usr/bin/env milk-cli -s
for i in {1..5}; do
    echo item $i
done
```

### Pipes

Pipe the output of one command into another:

```bash
#!/usr/bin/env milk-cli -s
echo hello | read greeting
listim | grep wfs
```

### Output Redirection

Redirect stdout to a file:

```bash
#!/usr/bin/env milk-cli -s
echo "log entry" > output.txt
echo "more" >> output.txt  # append
```

### Stderr Redirection

Redirect stderr independently:

```bash
#!/usr/bin/env milk-cli -s
cmd 2>&1          # stderr to stdout
cmd 2>/dev/null    # discard stderr
cmd 2>errors.txt   # stderr to file
```

### Here-Strings

Provide a string as stdin to a command:

```bash
#!/usr/bin/env milk-cli -s
read name <<< "world"
echo $name            # prints: world
```

### Glob Expansion

Tokens containing `*` or `?` are expanded to
matching filenames:

```bash
#!/usr/bin/env milk-cli -s
echo *.fits           # all FITS files
echo data_??.bin      # data_01.bin etc.
```

Quoted globs are not expanded:
`echo "*.fits"` prints the literal `*.fits`.

### exit

Exit the CLI session with an optional status code:

```bash
#!/usr/bin/env milk-cli -s
exit          # exit with status 0
exit 1        # exit with status 1
```

### shift

Shift positional parameters (`$1`→`$2`, etc.) inside
functions:

```bash
#!/usr/bin/env milk-cli -s
function process_all {
    while [ -n "$1" ]; do
        echo processing $1
        shift
    done
}
process_all a b c
```

### trap

Register signal handlers that execute a
command when a signal is received:

```bash
#!/usr/bin/env milk-cli -s
trap 'echo cleanup' EXIT
trap 'rm /tmp/lockfile' INT TERM
```

Supported signals: `EXIT`, `INT`, `TERM`,
`HUP`, `USR1`, `USR2`, or numeric.

### set

Control shell behavior flags:

```bash
#!/usr/bin/env milk-cli -s
set -e      # exit on error
set -x      # trace commands (+ prefix)
set +e      # disable -e
set +x      # disable -x
set -ex     # enable both
```

### export

Set environment variables visible to child
processes:

```bash
#!/usr/bin/env milk-cli -s
export DMSHMDIR="/tmp/shm"
export VERBOSE=1
export PATH              # push current $PATH
```

### Extended Test `[[ ]]`

Extended conditional test with regex support:

```bash
#!/usr/bin/env milk-cli -s
[[ $filename =~ ^data_[0-9]+$ ]]
[[ -n $myvar ]]
[[ -f $path ]]
```

### Tilde Expansion

`~` expands to `$HOME` at the start of tokens:

```bash
#!/usr/bin/env milk-cli -s
echo ~/data          # /home/user/data
ls ~/scripts/*.sh
```

### Input Redirection

Read a file as standard input:

```bash
#!/usr/bin/env milk-cli -s
read line < config.txt
cmd < input.dat
```

### select

Interactive numbered menu loop:

```bash
#!/usr/bin/env milk-cli -s
select mode in fast normal slow; do
    echo "Selected: $mode"
    break
done
```

### Arithmetic For

C-style counted loop using `for ((;;))`:

```bash
#!/usr/bin/env milk-cli -s
for ((i=0; i<5; i++)); do
    echo "iteration $i"
done
```

### Parameter Defaults

Use `${var:-default}` to substitute a default
value when a variable is unset or empty:

```bash
#!/usr/bin/env milk-cli -s
echo ${name:-anonymous}
echo ${dir:=/tmp}
echo ${v:+yes}
echo ${v:?error message}
```

| Syntax | Meaning |
|--------|---------|
| `${v:-def}` | Expand to `def` if unset |
| `${v:=def}` | Assign `def` if unset |
| `${v:+alt}` | Expand to `alt` if set |
| `${v:?err}` | Error message if unset |

### String Operations

```bash
#!/usr/bin/env milk-cli -s
path="/home/user/file.txt"
echo ${path/user/admin}
echo ${path//\//|}
echo ${path#/home/}
echo ${path%.txt}
```

| Syntax | Meaning |
|--------|---------|
| `${v/pat/rep}` | Replace first match |
| `${v//pat/rep}` | Replace all matches |
| `${v#pat}` | Strip shortest prefix |
| `${v%pat}` | Strip shortest suffix |

### Source (Include Files)

Execute commands from a file in the current
environment:

```bash
#!/usr/bin/env milk-cli -s
source config.sh
. config.sh
```

### Read-Only Variables

```bash
#!/usr/bin/env milk-cli -s
readonly PI=3.14159
```

### Break / Continue with Count

Exit or skip multiple loop levels:

```bash
#!/usr/bin/env milk-cli -s
for i in 1 2 3; do
    for j in a b c; do
        break 2
    done
done
```

### Printf

Format output with `%s`, `%d`, `%f` specifiers:

```bash
#!/usr/bin/env milk-cli -s
printf "Name: %s Age: %d\n" Alice 30
printf "PI = %f\n" 3.14159
```

### Getopts

Parse command-line options:

```bash
#!/usr/bin/env milk-cli -s
OPTIND=1
while getopts "vf:" opt; do
    echo "opt=$opt OPTARG=$OPTARG"
done
```

### Mapfile / Readarray

Read lines from a file into an array:

```bash
#!/usr/bin/env milk-cli -s
mapfile -t lines < input.txt
echo ${lines[0]}
echo ${#lines[@]}
```

### Background Jobs

Run commands in the background and wait:

```bash
#!/usr/bin/env milk-cli -s
long_command &
echo "PID is $!"
wait
```

### Subshell Grouping

Execute commands in an isolated sub-environment:

```bash
#!/usr/bin/env milk-cli -s
(x=42; echo $x)
echo $x    # empty — subshell is isolated
```

### declare / typeset

Declare variable attributes:

```bash
#!/usr/bin/env milk-cli -s
declare -i count=0    # integer
declare -r PI=3.14    # read-only
declare -a arr        # array
declare -x MYVAR=val  # export
typeset -i x=5        # alias
```

| Flag | Meaning |
|------|---------|
| `-i` | Integer variable |
| `-a` | Array variable |
| `-r` | Read-only |
| `-x` | Export to environment |

### let

Evaluate arithmetic expressions:

```bash
#!/usr/bin/env milk-cli -s
let "x=5+3"       # x = 8
let "y=x*2"       # y = 16
let "x++"         # x = 9
```

### eval

Construct and execute a command string:

```bash
#!/usr/bin/env milk-cli -s
cmd="echo hello"
eval $cmd          # prints: hello
vname=foo
eval "$vname=42"
echo $foo           # prints: 42
```

### type / command -v

Check whether a command exists:

```bash
#!/usr/bin/env milk-cli -s
type echo           # echo is a builtin
command -v ls        # /usr/bin/ls
type nonexistent     # not found
```

### timeout

Run a command with a deadline:

```bash
#!/usr/bin/env milk-cli -s
timeout 5 long_running_cmd
```

If the command does not finish within N seconds,
it is terminated with `$?` set to 124.


</details>

## Flow Control

<details markdown="1"><summary><b>Click to expand</b></summary>

### if / elif / else / fi

```bash
#!/usr/bin/env milk-cli -s
x=10
if [ $x -gt 100 ]; then
    echo big
elif [ $x -gt 5 ]; then
    echo medium
else
    echo small
fi
# prints: medium
```

`elif` branches are evaluated in order. The first
matching condition executes its body.

The `[ ... ]` syntax supports these test operators:

| Operator | Meaning |
|----------|---------|
| `-eq` | equal |
| `-ne` | not equal |
| `-gt` | greater than |
| `-ge` | greater or equal |
| `-lt` | less than |
| `-le` | less or equal |
| `=` | string equal |
| `!=` | string not equal |
| `-n str` | string is non-empty |
| `-z str` | string is empty |
| `-f path` | regular file exists |
| `-d path` | directory exists |
| `-e path` | path exists (any type) |
| `-s path` | file exists and is non-empty |
| `! expr` | logical NOT (negate result) |

### while / do / done

```bash
#!/usr/bin/env milk-cli -s
n=1
while [ $n -le 5 ]; do
    echo iteration $n
    n=$(( n + 1 ))
done
```

Use `break` to exit early, `continue` to skip to the
next iteration:

```bash
#!/usr/bin/env milk-cli -s
n=0
while [ $n -lt 10 ]; do
    n=$(( n + 1 ))
    if [ $n -eq 3 ]; then
        continue
    fi
    if [ $n -eq 7 ]; then
        break
    fi
    echo $n
done
```

### for / do / done

```bash
#!/usr/bin/env milk-cli -s
for item in alpha beta gamma; do
    echo processing $item
done
```

### Nesting

All flow control constructs can be nested:

```bash
#!/usr/bin/env milk-cli -s
n=0
while [ $n -lt 3 ]; do
    n=$(( n + 1 ))
    if [ $n -eq 2 ]; then
        echo found two
    fi
    echo n=$n
done
```

### case / esac

Multi-way branching by pattern matching:

```bash
#!/usr/bin/env milk-cli -s
mode=fast
case $mode in
  fast) echo speed ;;
  safe|careful) echo caution ;;
  *) echo unknown ;;
esac
# prints: speed
```

Patterns support `|` alternation and `*` wildcard.
Only the first matching pattern's body is executed.


</details>

## Functions

<details markdown="1"><summary><b>Click to expand</b></summary>

Define reusable functions with `function name { ... }`.
Inside the body, `$1` through `$9` are the call
arguments:

```bash
#!/usr/bin/env milk-cli -s
function greet {
    echo Hello $1
}
greet world           # prints: Hello world
```

### Return Values

Use `return` to exit a function early. An optional
integer argument sets `$?`:

```bash
#!/usr/bin/env milk-cli -s
function check {
    if [ $1 -gt 10 ]; then
        return 0
    fi
    return 1
}
check 20
echo $?               # prints: 0
```

### Local Variable Scoping

By default, variables assigned within a function are global.
Use the `local` keyword to explicitly bound a variable
to the function's scope. The `local` keyword safely shadows
any existing global variables, restoring them when returning.

```bash
#!/usr/bin/env milk-cli -s
x=global
y=global
function test {
    x=modified
    local y=localized
}
test
echo $x               # prints: modified
echo $y               # prints: global
```


</details>

## Heredocs

<details markdown="1"><summary><b>Click to expand</b></summary>

Assign multi-line text to a variable using heredoc
syntax:

```bash
#!/usr/bin/env milk-cli -s
config=<<EOF
gain=0.5
nframes=100
mode=closed
EOF
echo $config
```

Lines between `=<<DELIM` and `DELIM` are concatenated
with newlines and stored in the variable.


</details>

## Stream Event Triggers

<details markdown="1"><summary><b>Click to expand</b></summary>

The `on_update` command waits for a shared-memory
stream to be updated, then runs a command:

```bash
#!/usr/bin/env milk-cli -s
on_update wfs { echo frame received }
```

This blocks until the stream's semaphore is posted
(i.e., a new frame arrives), then executes the
command body. Useful for event-driven processing
in scripts.


</details>

## Script Files

<details markdown="1"><summary><b>Click to expand</b></summary>

### Running Scripts

Use `source` (or the shorthand `.`) to execute a
script file:

```bash
#!/usr/bin/env milk-cli -s
source myscript.milk
. myscript.milk        # same thing
```

Blank lines and `#` comments are skipped. On error,
the filename and line number are printed.

### Include Guard

`include_once` sources a file only once per session,
even if called multiple times. Uses resolved paths:

```bash
#!/usr/bin/env milk-cli -s
include_once helpers.milk
include_once helpers.milk   # no-op
```

### Startup Profile

`~/.milkrc` is automatically sourced at startup if
it exists. Use this for persistent aliases, variables,
and function definitions.

### Saving State

Export all current variables and function definitions
to a file:

```bash
#!/usr/bin/env milk-cli -s
savescript state.milk
```

The saved file can be loaded later with `source`.

### Saving History

Write the readline command history to a file:

```bash
#!/usr/bin/env milk-cli -s
savehistory replay.milk
```

This creates a script that replays your interactive
session.

### Example Script

```bash
#!/usr/bin/env milk-cli -s
# setup.milk — initialize processing pipeline

# Configuration
nframes=100
gain=0.5

# Create shared memory images
mem.mk2Dim wfs 128 128
mem.mk2Dim dm 32 32

# Define a helper function
function run_loop {
    n=0
    while [ $n -lt $nframes ]; do
        n=$(( n + 1 ))
        echo frame $n
    done
}

# Execute
run_loop
echo "Processing complete"
```

### Script File Convention

| Feature | Detail |
|---------|--------|
| Extension | `.milk` (by convention) |
| Comments | Lines starting with `#` |
| Shebang | `#!/usr/bin/env milk-cli -s` |
| Startup | `-s FILE` flag on command line |
| Auto-load | `~/.milkrc` at startup |


</details>

## Command Reference

<details markdown="1"><summary><b>Click to expand</b></summary>

| Command | Description |
|---------|-------------|
| `source <file>` | Execute a script file |
| `. <file>` | Same as `source` |
| `include_once <file>` | Source only once |
| `savescript <file>` | Save variables and functions |
| `savehistory <file>` | Save command history |
| `echo <args>` | Print arguments |
| `printf "fmt" args` | Formatted output |
| `sleep <seconds>` | Pause (float-capable) |
| `vars` | List all variables |
| `unset <var>` | Remove a variable |
| `readonly VAR=val` | Mark variable read-only |
| `declare [-i\|-a\|-r\|-x\|-A]` | Typed declaration |
| `typeset` | Alias for `declare` |
| `local VAR=val` | Set variable in current scope |
| `let "expr"` | Arithmetic evaluation |
| `eval "cmd"` | Execute string as command |
| `type <cmd>` | Check command existence |
| `command -v <cmd>` | Print command path |
| `timeout N cmd` | Run cmd with deadline |
| `fpsset <fps> <param> <val>` | Write FPS parameter |
| `return [val]` | Exit function, set `$?` |
| `break [N]` | Exit N loop levels |
| `continue [N]` | Skip N loop levels |
| `read [-p\|-t\|-a] VAR` | Read input |
| `getopts spec opt` | Parse options |
| `mapfile -t arr < file` | Read lines to array |
| `on_update <stream> { cmd }` | Run cmd on stream update |
| `true` / `false` | Set `$?` to 0 / 1 |
| `alias name='cmd'` | Create alias |
| `unalias name` | Remove alias |
| `basename path` | Extract filename |
| `dirname path` | Extract directory |
| `pushd dir` | Push dir and cd |
| `popd` | Pop dir and cd back |
| `dirs` | Show directory stack |
| `seq [start] [step] end` | Print number sequence |
| `waitfor_stream s [T]` | Wait for SHM stream |
| `waitfor_fps name [T]` | Wait for FPS SHM |


</details>

## Associative Arrays

<details markdown="1"><summary><b>Click to expand</b></summary>

Associative arrays store key-value pairs:

```bash
#!/usr/bin/env milk-cli -s
declare -A config
config[host]=localhost
config[port]=8080
echo ${config[host]}   # localhost
```

Assignment uses `map[key]=value` syntax. Lookup
uses `${map[key]}` expansion.


</details>

## Indirect Expansion

<details markdown="1"><summary><b>Click to expand</b></summary>

`${!var}` expands to the value of the variable
whose *name* is stored in `var`:

```bash
#!/usr/bin/env milk-cli -s
target=myval
myval=42
echo ${!target}   # 42
```


</details>

## Aliases

<details markdown="1"><summary><b>Click to expand</b></summary>

```bash
#!/usr/bin/env milk-cli -s
alias ll='listim'
alias s='readshmim'
ll                    # runs: listim
alias                 # list all aliases
unalias ll            # remove alias
```

Aliases are expanded before command dispatch.


</details>

## Path Utilities

<details markdown="1"><summary><b>Click to expand</b></summary>

```bash
#!/usr/bin/env milk-cli -s
basename /data/img.fits   # img.fits
dirname /data/img.fits    # /data
```


</details>

## Directory Stack

<details markdown="1"><summary><b>Click to expand</b></summary>

```bash
#!/usr/bin/env milk-cli -s
pushd /data           # cd to /data
pushd /tmp            # cd to /tmp
dirs                  # show stack
popd                  # back to /data
popd                  # back to original
```


</details>

## Number Sequences

<details markdown="1"><summary><b>Click to expand</b></summary>

```bash
#!/usr/bin/env milk-cli -s
seq 5                 # 1 2 3 4 5
seq 2 5               # 2 3 4 5
seq 0 0.1 1.0         # 0 0.1 ... 1.0
seq 10 -2 0           # 10 8 6 4 2 0
```

Supports floating-point steps.


</details>

## Builtins

<details markdown="1"><summary><b>Click to expand</b></summary>

```bash
#!/usr/bin/env milk-cli -s
true                  # $? = 0
false                 # $? = 1
if (( x > 5 )); then echo big; fi
```


</details>

## Variable Testing

<details markdown="1"><summary><b>Click to expand</b></summary>

`test -v VAR` checks if a variable is set:

```bash
#!/usr/bin/env milk-cli -s
x=1
if [ -v x ]; then echo set; fi
```


</details>

## Stream & FPS Metadata

<details markdown="1"><summary><b>Click to expand</b></summary>

Access shared memory stream properties via
dot-syntax variable expansion:

```bash
#!/usr/bin/env milk-cli -s
echo ${mystream.xsize}   # X dimension
echo ${mystream.ysize}   # Y dimension
echo ${mystream.zsize}   # Z dimension
echo ${mystream.type}    # datatype code
echo ${mystream.cnt0}    # frame counter
echo ${mystream.naxis}   # number of axes
```

FPS status can be checked:

```bash
#!/usr/bin/env milk-cli -s
echo ${myfps.status}     # 1 if exists
```


</details>

## Wait for Resources

<details markdown="1"><summary><b>Click to expand</b></summary>

Wait for shared memory streams or FPS to appear:

```bash
#!/usr/bin/env milk-cli -s
waitfor_stream wfs 30    # wait up to 30s
waitfor_fps dmcomb 10    # wait up to 10s
```

Returns 0 on success, 1 on timeout. Default timeout:
10 seconds.

```bash
#!/usr/bin/env milk-cli -s
# Pattern: wait for stream, then process
waitfor_stream wfs 60
if [ $? -eq 0 ]; then
    echo "Stream ready: ${wfs.xsize}x${wfs.ysize}"
fi
```



</details>

## Scripting Showcase

<details markdown="1"><summary><b>Click to expand</b></summary>

Here are examples demonstrating the combined capabilities of `milk-cli`'s
built-in scripting engine, ordered from simple to complex.

---

### Simple

<details markdown="1">
<summary><b>Example 1: Hello World and Variables</b></summary>

The simplest possible script — setting variables and printing them:

```bash
#!/usr/bin/env milk-cli -s
name=world
echo "Hello, ${name}!"
x=42
echo "The answer is $x"
```

Expected output:

```text
Hello, world!
The answer is 42
```

</details>

<details markdown="1">
<summary><b>Example 2: Arithmetic and String Length</b></summary>

Use `$(( ))` for integer arithmetic and `${#var}` for string length:

```bash
#!/usr/bin/env milk-cli -s
a=8
b=3
echo "Sum:     $(( a + b ))"
echo "Product: $(( a * b ))"
echo "Division: $(( a / b ))"
echo "Modulo:  $(( a % b ))"

word=spectral
echo "Length of '${word}': ${#word}"
```

Expected output:

```text
Sum:     11
Product: 24
Division: 2
Modulo:  2
Length of 'spectral': 8
```

</details>

<details markdown="1">
<summary><b>Example 3: Conditional Logic</b></summary>

Branch on exit status or integer comparisons:

```bash
#!/usr/bin/env milk-cli -s
temp=72

if [ $temp -lt 60 ]; then
    echo "Cold"
elif [ $temp -lt 80 ]; then
    echo "Comfortable"
else
    echo "Hot"
fi
```

Expected output:

```text
Comfortable
```

</details>

<details markdown="1">
<summary><b>Example 4: Looping Over a Range</b></summary>

`for` loops over sequences with `seq`, or literal lists:

```bash
#!/usr/bin/env milk-cli -s
# Loop over a numeric sequence
for i in $(seq 1 5); do
    echo "Frame $i"
done

# Loop over a literal list
for color in red green blue; do
    echo "Channel: $color"
done
```

Expected output:

```text
Frame 1
Frame 2
Frame 3
Frame 4
Frame 5
Channel: red
Channel: green
Channel: blue
```

</details>

---

### Intermediate

<details markdown="1">
<summary><b>Example 5: Parameter Defaults and String Manipulation</b></summary>

Safely provide defaults for arguments and cleanly parse paths:

```bash
#!/usr/bin/env milk-cli -s
function process_image {
    local img_path=${1:-/data/default.fits}
    local filename=${img_path##*/}   # strip path prefix
    local basename=${filename%.*}    # strip extension

    echo "Processing ${basename}..."
}

process_image
process_image /tmp/test_image.fits
```

Expected output:

```text
Processing default...
Processing test_image...
```

</details>

<details markdown="1">
<summary><b>Example 6: Functions and Return Values</b></summary>

Functions use `return` to pass integer exit status back to `$?`.
Use command substitution `$(...)` to capture printed output:

```bash
#!/usr/bin/env milk-cli -s
function clamp {
    local val=$1 lo=$2 hi=$3
    if [ $val -lt $lo ]; then
        echo $lo
    elif [ $val -gt $hi ]; then
        echo $hi
    else
        echo $val
    fi
}

result=$(clamp 150 0 100)
echo "Clamped: $result"

result=$(clamp -5 0 100)
echo "Clamped: $result"
```

Expected output:

```text
Clamped: 100
Clamped: 0
```

</details>

<details markdown="1">
<summary><b>Example 7: Arrays and Iteration</b></summary>

Build an array and iterate over its elements:

```bash
#!/usr/bin/env milk-cli -s
streams=(wfs_cam dm_disp tt_disp)

echo "Registered streams: ${#streams[@]}"

for i in $(seq 0 $(( ${#streams[@]} - 1 ))); do
    echo "  [$i] ${streams[$i]}"
done
```

Expected output:

```text
Registered streams: 3
  [0] wfs_cam
  [1] dm_disp
  [2] tt_disp
```

</details>

<details markdown="1">
<summary><b>Example 8: While Loop and Counter</b></summary>

Accumulate a running sum with a `while` loop:

```bash
#!/usr/bin/env milk-cli -s
total=0
count=0
max=10

while [ $count -lt $max ]; do
    total=$(( total + count ))
    count=$(( count + 1 ))
done

echo "Sum 0..$(( max - 1 )) = $total"
```

Expected output:

```text
Sum 0..9 = 45
```

</details>

<details markdown="1">
<summary><b>Example 9: Local Variables and Recursion</b></summary>

Use `local` variables and `return` to safely write recursive functions:

```bash
#!/usr/bin/env milk-cli -s
function factorial {
    local n=$1
    if [ $n -le 1 ]; then
        return 1
    else
        local prev=$(( n - 1 ))
        factorial $prev
        return $(( n * $? ))
    fi
}

factorial 5
echo "5! = $?"
```

Expected output:

```text
5! = 120
```

</details>

<details markdown="1">
<summary><b>Example 10: Transparent OS Fallback and Command Substitution</b></summary>

Combine Linux shell utilities with native `milk-cli` variables
without the `!` prefix required for interactive mode:

```bash
#!/usr/bin/env milk-cli -s
prefix="output_"
ext=".fits"

# Count FITS files in the current directory
count=$(ls -1 | grep -c "${ext}")
echo "Found $count FITS files."

# Timestamp tag for unique filenames
tag=$(date +%Y%m%d_%H%M%S)
echo "Saving to: ${prefix}${tag}${ext}"
```

</details>

---

### Advanced

<details markdown="1">
<summary><b>Example 11: Waiting for Streams</b></summary>

Block until shared-memory streams become available, then
read their geometry via dot-expansion:

```bash
#!/usr/bin/env milk-cli -s
function wait_and_monitor {
    local stream=$1
    echo "Waiting for stream ${stream}..."

    waitfor_stream $stream 60
    if [ $? -ne 0 ]; then
        echo "Error: Stream ${stream} timed out."
        return 1
    fi

    # Read stream metadata via dot-expansion
    echo "Ready! Shape: ${${stream}.xsize}x${${stream}.ysize}"
}

wait_and_monitor wfs_cam
wait_and_monitor dm_disp
```

</details>

<details markdown="1">
<summary><b>Example 12: FPS Parameter Manipulation</b></summary>

Read and write FPS parameters from a script to automate
configuration changes across multiple compute units:

```bash
#!/usr/bin/env milk-cli -s
# Set loop gain on the DM combiner FPS
milk-fps-set dmcomb.loopgain 0.05

# Retrieve current value and log it
gain=$(milk-fps-set dmcomb.loopgain)
echo "DM combiner gain set to: $gain"

# Apply identical gain to all modal channels
nmodes=50
for m in $(seq 0 $(( nmodes - 1 ))); do
    milk-fps-set dmcomb.modesgain[$m] 0.1
done

echo "Set $nmodes modal gains to 0.1"
```

</details>

<details markdown="1">
<summary><b>Example 13: Stream Diagnostic Report</b></summary>

Collect live metadata from multiple streams and format a
compact diagnostic table:

```bash
#!/usr/bin/env milk-cli -s
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

    xs=${${s}.xsize}
    ys=${${s}.ysize}
    cnt=${${s}.cnt0}
    printf "  %-18s %-6s %-6s %s\n" $s $xs $ys $cnt
done

echo "--------------------------------------------"
```

</details>

<details markdown="1">
<summary><b>Example 14: AO Loop Startup Orchestration</b></summary>

A complete startup script that initialises an AO loop step by
step, verifies each stage, and aborts cleanly on failure:

```bash
#!/usr/bin/env milk-cli -s

# ---------- helpers ----------
function die {
    echo "FATAL: $1"
    exit 1
}

function wait_stream {
    local s=$1 timeout=${2:-30}
    waitfor_stream $s $timeout
    [ $? -ne 0 ] && die "Stream '$s' not available after ${timeout}s"
    echo "  [OK] $s  ${${s}.xsize}x${${s}.ysize}"
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
    milk-fps-set dmcomb.modesgain[$m] 0.05
done
milk-fps-set dmcomb.loopgain 1.0
milk-fps-set dmcomb.loopON 1

# ---------- 5. confirm loop is running ----------
echo "=== 5. Loop status ==="
sleep 1
fpsgain=$(milk-fps-set dmcomb.loopgain)
echo "  loopgain = $fpsgain"
echo "AO loop started successfully."
```

</details>


</details>

---
← [CLI Syntax](CLIcore.md) · [Documentation Index](../index.md)
