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

```bash
milk-cli > path=/home/user/file.txt
milk-cli > echo ${#path}           # prints: 19
milk-cli > echo ${path:6:4}        # prints: user
milk-cli > echo ${path%%/*}        # prints: (empty)
milk-cli > echo ${path##*/}        # prints: file.txt
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

Use `-p "prompt"` to display a prompt first:

```bash
milk-cli > read -p "Enter value: " val
Enter value: 42
milk-cli > echo $val           # prints: 42
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

Variables created inside a function are automatically
local — they are removed when the function returns.
Variables that existed before the call are restored to
their original values:

```bash
milk-cli > x=outer
milk-cli > function test {
milk-cli >     x=inner
milk-cli >     y=local_only
milk-cli > }
milk-cli > test
milk-cli > echo $x               # prints: outer
milk-cli > echo $y               # prints: (empty)
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
| `fpsset <fps> <param> <val>` | Write FPS parameter |
| `return [val]` | Exit function, set `$?` |
| `break` | Exit loop |
| `continue` | Skip to next iteration |
| `on_update <stream> { cmd }` | Run cmd on stream update |

---
← [CLI Syntax](CLIcore.md) · [Documentation Index](../index.md)
