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

## Flow Control

### if / then / else / fi

```bash
milk-cli > x=10
milk-cli > if [ $x -gt 5 ]; then
milk-cli >     echo x is big
milk-cli > else
milk-cli >     echo x is small
milk-cli > fi
```

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

### while / do / done

```bash
milk-cli > n=1
milk-cli > while [ $n -le 5 ]; do
milk-cli >     echo iteration $n
milk-cli >     n=$(( n + 1 ))
milk-cli > done
```

Use `break` to exit early:

```bash
milk-cli > n=0
milk-cli > while [ $n -lt 100 ]; do
milk-cli >     n=$(( n + 1 ))
milk-cli >     if [ $n -eq 5 ]; then
milk-cli >         break
milk-cli >     fi
milk-cli > done
milk-cli > echo stopped at $n    # prints: stopped at 5
```

### for / do / done

```bash
milk-cli > for item in alpha beta gamma; do
milk-cli >     echo processing $item
milk-cli > done
```

For loops with arithmetic:

```bash
milk-cli > sum=0
milk-cli > for val in 10 20 30; do
milk-cli >     sum=$(( sum + val ))
milk-cli > done
milk-cli > echo total=$sum       # prints: total=60
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

## Functions

Define reusable functions with `function name { ... }`.
Inside the body, `$1` through `$9` are the call
arguments:

```bash
milk-cli > function greet {
milk-cli >     echo Hello $1
milk-cli > }
milk-cli > greet world           # prints: Hello world
milk-cli > greet milk            # prints: Hello milk
```

Functions can contain flow control:

```bash
milk-cli > function classify {
milk-cli >     if [ $1 -gt 100 ]; then
milk-cli >         echo big
milk-cli >     else
milk-cli >         echo small
milk-cli >     fi
milk-cli > }
milk-cli > classify 200          # prints: big
milk-cli > classify 5            # prints: small
```

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

## Command Reference

| Command | Description |
|---------|-------------|
| `source <file>` | Execute a script file |
| `. <file>` | Same as `source` |
| `savescript <file>` | Save variables and functions |
| `savehistory <file>` | Save command history |
| `echo <args>` | Print arguments |
| `vars` | List all variables |
| `unset <var>` | Remove a variable |
| `fpsset <fps> <param> <val>` | Write FPS parameter |

---
← [CLI Syntax](CLIcore.md) · [Documentation Index](../index.md)
