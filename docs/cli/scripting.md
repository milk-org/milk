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

### Setting and Reading

```bash
milk-cli > x=42
milk-cli > name=hello
milk-cli > echo $x          # prints: 42
milk-cli > echo ${name}     # prints: hello
```

Variable names are case-sensitive and may contain
letters, digits, and underscores. No spaces around `=`.

### Listing and Removing

```bash
milk-cli > vars              # list all variables
milk-cli > unset x           # remove variable x
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
milk-cli > path=/home/user/file.txt
milk-cli > echo ${#path}           # prints: 19
milk-cli > echo ${path:6:4}        # prints: user
milk-cli > echo ${path%%/*}        # prints: (empty)
milk-cli > echo ${path##*/}        # prints: file.txt
milk-cli > name=hello
milk-cli > echo ${name^^}          # prints: HELLO
milk-cli > name=WORLD
milk-cli > echo ${name,,}          # prints: world
```

### Array Variables

Arrays store multiple values indexed by integer:

```bash
milk-cli > colors=(red green blue)
milk-cli > echo ${colors[0]}       # prints: red
milk-cli > echo ${colors[2]}       # prints: blue
milk-cli > echo ${colors[@]}       # prints: red green blue
milk-cli > echo ${#colors[@]}      # prints: 3
```

### FPS Parameter Access

Read FPS parameters using `@fpsname.param` syntax:

```bash
milk-cli > echo @myloop.loopgain      # read FPS param
milk-cli > g=@myloop.loopgain         # store in variable
```

Write FPS parameters with `fpsset`:

```bash
milk-cli > fpsset myloop loopgain 0.5
```

## Arithmetic

`$(( expr ))` evaluates integer arithmetic with
`+`, `-`, `*`, `/`, `%` operators. Variables and FPS
parameters are expanded inside the expression:

```bash
milk-cli > x=10
milk-cli > y=$(( x + 5 ))        # y = 15
milk-cli > z=$(( y * 2 - x ))    # z = 20
milk-cli > echo $y $z            # prints: 15 20
```

## Built-in Commands

### sleep

Pause execution for a given number of seconds
(float-capable):

```bash
milk-cli > sleep 1.5             # pause 1.5 seconds
milk-cli > sleep 0.01            # 10 ms pause
```

### printf

Formatted output supporting `%s`, `%d`, `%f`, `%%`,
and `\n`, `\t` escape sequences:

```bash
milk-cli > printf "value = %d\n" 42
milk-cli > printf "%s has %d items\n" list 5
```

### read

Read a line from standard input into a variable:

```bash
milk-cli > read name
hello world
milk-cli > echo $name         # prints: hello world
```

Flags:

| Flag | Description |
|------|-------------|
| `-p "prompt"` | Display a prompt before reading |
| `-t N` | Timeout after N seconds |
| `-a arrayname` | Split words into an array |

```bash
milk-cli > read -p "Enter value: " val
Enter value: 42
milk-cli > echo $val           # prints: 42
milk-cli > read -t 5 response  # wait up to 5 sec
milk-cli > read -a words       # split into array
hello world
milk-cli > echo ${words[0]}    # prints: hello
```

### Logical Operators

Chain commands with `&&` (AND) and `||` (OR):

```bash
milk-cli > cmd1 && cmd2        # cmd2 runs if cmd1 succeeds
milk-cli > cmd1 || cmd2        # cmd2 runs if cmd1 fails
```

### Brace Expansion

Expand `{N..M}` and `{N..M..S}` into integer
sequences:

```bash
milk-cli > echo {1..5}         # prints: 1 2 3 4 5
milk-cli > echo {0..10..2}     # prints: 0 2 4 6 8 10
milk-cli > echo {5..1}         # prints: 5 4 3 2 1
```

Useful with `for` loops:

```bash
milk-cli > for i in {1..5}; do
milk-cli >     echo item $i
milk-cli > done
```

### Pipes

Pipe the output of one command into another:

```bash
milk-cli > echo hello | read greeting
milk-cli > listim | grep wfs
```

### Output Redirection

Redirect stdout to a file:

```bash
milk-cli > echo "log entry" > output.txt
milk-cli > echo "more" >> output.txt  # append
```

### Stderr Redirection

Redirect stderr independently:

```bash
milk-cli > cmd 2>&1          # stderr to stdout
milk-cli > cmd 2>/dev/null    # discard stderr
milk-cli > cmd 2>errors.txt   # stderr to file
```

### Here-Strings

Provide a string as stdin to a command:

```bash
milk-cli > read name <<< "world"
milk-cli > echo $name            # prints: world
```

### Glob Expansion

Tokens containing `*` or `?` are expanded to
matching filenames:

```bash
milk-cli > echo *.fits           # all FITS files
milk-cli > echo data_??.bin      # data_01.bin etc.
```

Quoted globs are not expanded:
`echo "*.fits"` prints the literal `*.fits`.

### exit

Exit the CLI session with an optional status code:

```bash
milk-cli > exit          # exit with status 0
milk-cli > exit 1        # exit with status 1
```

### shift

Shift positional parameters (`$1`→`$2`, etc.) inside
functions:

```bash
milk-cli > function process_all {
milk-cli >     while [ -n "$1" ]; do
milk-cli >         echo processing $1
milk-cli >         shift
milk-cli >     done
milk-cli > }
milk-cli > process_all a b c
```

### trap

Register signal handlers that execute a
command when a signal is received:

```bash
milk-cli > trap 'echo cleanup' EXIT
milk-cli > trap 'rm /tmp/lockfile' INT TERM
```

Supported signals: `EXIT`, `INT`, `TERM`,
`HUP`, `USR1`, `USR2`, or numeric.

### set

Control shell behavior flags:

```bash
milk-cli > set -e      # exit on error
milk-cli > set -x      # trace commands (+ prefix)
milk-cli > set +e      # disable -e
milk-cli > set +x      # disable -x
milk-cli > set -ex     # enable both
```

### export

Set environment variables visible to child
processes:

```bash
milk-cli > export DMSHMDIR="/tmp/shm"
milk-cli > export VERBOSE=1
milk-cli > export PATH              # push current $PATH
```

### Extended Test `[[ ]]`

Extended conditional test with regex support:

```bash
milk-cli > [[ $filename =~ ^data_[0-9]+$ ]]
milk-cli > [[ -n $myvar ]]
milk-cli > [[ -f $path ]]
```

### Tilde Expansion

`~` expands to `$HOME` at the start of tokens:

```bash
milk-cli > echo ~/data          # /home/user/data
milk-cli > ls ~/scripts/*.sh
```

### Input Redirection

Read a file as standard input:

```bash
milk-cli > read line < config.txt
milk-cli > cmd < input.dat
```

### select

Interactive numbered menu loop:

```bash
milk-cli > select mode in fast normal slow; do
milk-cli >     echo "Selected: $mode"
milk-cli >     break
milk-cli > done
```

### Arithmetic For

C-style counted loop using `for ((;;))`:

```bash
milk-cli > for ((i=0; i<5; i++)); do
milk-cli >     echo "iteration $i"
milk-cli > done
```

### Parameter Defaults

Use `${var:-default}` to substitute a default
value when a variable is unset or empty:

```bash
milk-cli > echo ${name:-anonymous}
milk-cli > echo ${dir:=/tmp}
milk-cli > echo ${v:+yes}
milk-cli > echo ${v:?error message}
```

| Syntax | Meaning |
|--------|---------|
| `${v:-def}` | Expand to `def` if unset |
| `${v:=def}` | Assign `def` if unset |
| `${v:+alt}` | Expand to `alt` if set |
| `${v:?err}` | Error message if unset |

### String Operations

```bash
milk-cli > path="/home/user/file.txt"
milk-cli > echo ${path/user/admin}
milk-cli > echo ${path//\//|}
milk-cli > echo ${path#/home/}
milk-cli > echo ${path%.txt}
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
milk-cli > source config.sh
milk-cli > . config.sh
```

### Read-Only Variables

```bash
milk-cli > readonly PI=3.14159
```

### Break / Continue with Count

Exit or skip multiple loop levels:

```bash
milk-cli > for i in 1 2 3; do
milk-cli >     for j in a b c; do
milk-cli >         break 2
milk-cli >     done
milk-cli > done
```

### Printf

Format output with `%s`, `%d`, `%f` specifiers:

```bash
milk-cli > printf "Name: %s Age: %d\n" Alice 30
milk-cli > printf "PI = %f\n" 3.14159
```

### Getopts

Parse command-line options:

```bash
milk-cli > OPTIND=1
milk-cli > while getopts "vf:" opt; do
milk-cli >     echo "opt=$opt OPTARG=$OPTARG"
milk-cli > done
```

### Mapfile / Readarray

Read lines from a file into an array:

```bash
milk-cli > mapfile -t lines < input.txt
milk-cli > echo ${lines[0]}
milk-cli > echo ${#lines[@]}
```

### Background Jobs

Run commands in the background and wait:

```bash
milk-cli > long_command &
milk-cli > echo "PID is $!"
milk-cli > wait
```

### Subshell Grouping

Execute commands in an isolated sub-environment:

```bash
milk-cli > (x=42; echo $x)
milk-cli > echo $x    # empty — subshell is isolated
```

### declare / typeset

Declare variable attributes:

```bash
milk-cli > declare -i count=0    # integer
milk-cli > declare -r PI=3.14    # read-only
milk-cli > declare -a arr        # array
milk-cli > declare -x MYVAR=val  # export
milk-cli > typeset -i x=5        # alias
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
milk-cli > let "x=5+3"       # x = 8
milk-cli > let "y=x*2"       # y = 16
milk-cli > let "x++"         # x = 9
```

### eval

Construct and execute a command string:

```bash
milk-cli > cmd="echo hello"
milk-cli > eval $cmd          # prints: hello
milk-cli > vname=foo
milk-cli > eval "$vname=42"
milk-cli > echo $foo           # prints: 42
```

### type / command -v

Check whether a command exists:

```bash
milk-cli > type echo           # echo is a builtin
milk-cli > command -v ls        # /usr/bin/ls
milk-cli > type nonexistent     # not found
```

### timeout

Run a command with a deadline:

```bash
milk-cli > timeout 5 long_running_cmd
```

If the command does not finish within N seconds,
it is terminated with `$?` set to 124.

## Flow Control

### if / elif / else / fi

```bash
milk-cli > x=10
milk-cli > if [ $x -gt 100 ]; then
milk-cli >     echo big
milk-cli > elif [ $x -gt 5 ]; then
milk-cli >     echo medium
milk-cli > else
milk-cli >     echo small
milk-cli > fi
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
milk-cli > n=1
milk-cli > while [ $n -le 5 ]; do
milk-cli >     echo iteration $n
milk-cli >     n=$(( n + 1 ))
milk-cli > done
```

Use `break` to exit early, `continue` to skip to the
next iteration:

```bash
milk-cli > n=0
milk-cli > while [ $n -lt 10 ]; do
milk-cli >     n=$(( n + 1 ))
milk-cli >     if [ $n -eq 3 ]; then
milk-cli >         continue
milk-cli >     fi
milk-cli >     if [ $n -eq 7 ]; then
milk-cli >         break
milk-cli >     fi
milk-cli >     echo $n
milk-cli > done
```

### for / do / done

```bash
milk-cli > for item in alpha beta gamma; do
milk-cli >     echo processing $item
milk-cli > done
```

### Nesting

All flow control constructs can be nested:

```bash
milk-cli > n=0
milk-cli > while [ $n -lt 3 ]; do
milk-cli >     n=$(( n + 1 ))
milk-cli >     if [ $n -eq 2 ]; then
milk-cli >         echo found two
milk-cli >     fi
milk-cli >     echo n=$n
milk-cli > done
```

### case / esac

Multi-way branching by pattern matching:

```bash
milk-cli > mode=fast
milk-cli > case $mode in
milk-cli >   fast) echo speed ;;
milk-cli >   safe|careful) echo caution ;;
milk-cli >   *) echo unknown ;;
milk-cli > esac
# prints: speed
```

Patterns support `|` alternation and `*` wildcard.
Only the first matching pattern's body is executed.

## Functions

Define reusable functions with `function name { ... }`.
Inside the body, `$1` through `$9` are the call
arguments:

```bash
milk-cli > function greet {
milk-cli >     echo Hello $1
milk-cli > }
milk-cli > greet world           # prints: Hello world
```

### Return Values

Use `return` to exit a function early. An optional
integer argument sets `$?`:

```bash
milk-cli > function check {
milk-cli >     if [ $1 -gt 10 ]; then
milk-cli >         return 0
milk-cli >     fi
milk-cli >     return 1
milk-cli > }
milk-cli > check 20
milk-cli > echo $?               # prints: 0
```

### Local Variable Scoping

By default, variables assigned within a function are global.
Use the `local` keyword to explicitly bound a variable
to the function's scope. The `local` keyword safely shadows
any existing global variables, restoring them when returning.

```bash
milk-cli > x=global
milk-cli > y=global
milk-cli > function test {
milk-cli >     x=modified
milk-cli >     local y=localized
milk-cli > }
milk-cli > test
milk-cli > echo $x               # prints: modified
milk-cli > echo $y               # prints: global
```

## Heredocs

Assign multi-line text to a variable using heredoc
syntax:

```bash
milk-cli > config=<<EOF
milk-cli > gain=0.5
milk-cli > nframes=100
milk-cli > mode=closed
milk-cli > EOF
milk-cli > echo $config
```

Lines between `=<<DELIM` and `DELIM` are concatenated
with newlines and stored in the variable.

## Stream Event Triggers

The `on_update` command waits for a shared-memory
stream to be updated, then runs a command:

```bash
milk-cli > on_update wfs { echo frame received }
```

This blocks until the stream's semaphore is posted
(i.e., a new frame arrives), then executes the
command body. Useful for event-driven processing
in scripts.

## Script Files

### Running Scripts

Use `source` (or the shorthand `.`) to execute a
script file:

```bash
milk-cli > source myscript.milk
milk-cli > . myscript.milk        # same thing
```

Blank lines and `#` comments are skipped. On error,
the filename and line number are printed.

### Include Guard

`include_once` sources a file only once per session,
even if called multiple times. Uses resolved paths:

```bash
milk-cli > include_once helpers.milk
milk-cli > include_once helpers.milk   # no-op
```

### Startup Profile

`~/.milkrc` is automatically sourced at startup if
it exists. Use this for persistent aliases, variables,
and function definitions.

### Saving State

Export all current variables and function definitions
to a file:

```bash
milk-cli > savescript state.milk
```

The saved file can be loaded later with `source`.

### Saving History

Write the readline command history to a file:

```bash
milk-cli > savehistory replay.milk
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

## Command Reference

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

## Associative Arrays

Associative arrays store key-value pairs:

```bash
milk-cli > declare -A config
milk-cli > config[host]=localhost
milk-cli > config[port]=8080
milk-cli > echo ${config[host]}   # localhost
```

Assignment uses `map[key]=value` syntax. Lookup
uses `${map[key]}` expansion.

## Indirect Expansion

`${!var}` expands to the value of the variable
whose *name* is stored in `var`:

```bash
milk-cli > target=myval
milk-cli > myval=42
milk-cli > echo ${!target}   # 42
```

## Aliases

```bash
milk-cli > alias ll='listim'
milk-cli > alias s='readshmim'
milk-cli > ll                    # runs: listim
milk-cli > alias                 # list all aliases
milk-cli > unalias ll            # remove alias
```

Aliases are expanded before command dispatch.

## Path Utilities

```bash
milk-cli > basename /data/img.fits   # img.fits
milk-cli > dirname /data/img.fits    # /data
```

## Directory Stack

```bash
milk-cli > pushd /data           # cd to /data
milk-cli > pushd /tmp            # cd to /tmp
milk-cli > dirs                  # show stack
milk-cli > popd                  # back to /data
milk-cli > popd                  # back to original
```

## Number Sequences

```bash
milk-cli > seq 5                 # 1 2 3 4 5
milk-cli > seq 2 5               # 2 3 4 5
milk-cli > seq 0 0.1 1.0         # 0 0.1 ... 1.0
milk-cli > seq 10 -2 0           # 10 8 6 4 2 0
```

Supports floating-point steps.

## Builtins

```bash
milk-cli > true                  # $? = 0
milk-cli > false                 # $? = 1
milk-cli > if (( x > 5 )); then echo big; fi
```

## Variable Testing

`test -v VAR` checks if a variable is set:

```bash
milk-cli > x=1
milk-cli > if [ -v x ]; then echo set; fi
```

## Stream & FPS Metadata

Access shared memory stream properties via
dot-syntax variable expansion:

```bash
milk-cli > echo ${mystream.xsize}   # X dimension
milk-cli > echo ${mystream.ysize}   # Y dimension
milk-cli > echo ${mystream.zsize}   # Z dimension
milk-cli > echo ${mystream.type}    # datatype code
milk-cli > echo ${mystream.cnt0}    # frame counter
milk-cli > echo ${mystream.naxis}   # number of axes
```

FPS status can be checked:

```bash
milk-cli > echo ${myfps.status}     # 1 if exists
```

## Wait for Resources

Wait for shared memory streams or FPS to appear:

```bash
milk-cli > waitfor_stream wfs 30    # wait up to 30s
milk-cli > waitfor_fps dmcomb 10    # wait up to 10s
```

Returns 0 on success, 1 on timeout. Default timeout:
10 seconds.

```bash
# Pattern: wait for stream, then process
waitfor_stream wfs 60
if [ $? -eq 0 ]; then
    echo "Stream ready: ${wfs.xsize}x${wfs.ysize}"
fi
```

---
← [CLI Syntax](CLIcore.md) · [Documentation Index](../index.md)
