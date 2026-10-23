---
tags:
  - cli
  - interactive
---

# Command Line Interface Syntax

The interactive CLI is provided by the `milk-cli` executable (typically aliased as `milk`). Source
code is in `src/cli/CLIcore/`.

<!-- prettier-ignore -->
!!! note
    Standalone executables (`milk-fpsexec-*`) have their own command-line interfaces. See
    [FPS](fps.md#6-standalone-executables-fpsexec).

See also: [FPS](fps.md) · [Streams](streams.md) · [FAQ](../quickstart/faq.md) ·
[Build Tiers](../quickstart/build_tiers.md)

---

## Starting & Configuring

This section covers launching `milk-cli`, configuring its behavior, and loading additional modules.

### Command-Line Options

When launching `milk-cli`, the following arguments are available:

| Option                      | Description                                         |
| --------------------------- | --------------------------------------------------- |
| `-h`, `--help`              | Print help and exit                                 |
| `-h1`, `--help-oneline`     | Print one-line description and exit                 |
| `-h2`, `--help-description` | Print verbose description and exit                  |
| `-hm`, `--help-mono`        | Print help message without ANSI colors and exit     |
| `-v`, `--version`           | Print version and exit                              |
| `-i`, `--info`              | Print version, settings, info and exit              |
| `--verbose`                 | Be verbose                                          |
| `-d`, `--debug=LEVEL`       | Set debug level at startup                          |
| `-o`, `--overwrite`         | Auto-overwrite FITS files (**use with caution**)    |
| `-e`, `--errorexit`         | Exit on command error                               |
| `-Z`, `--idle`              | Only run process when X is idle                     |
| `--listimf`                 | Write image list to `imlist.txt`                    |
| `-n`, `--pname=NAME`        | Rename process                                      |
| `-p`, `--priority=PR`       | Set RT priority (0–99, higher = higher)             |
| `-f`, `--fifoflag`          | Enable fifo input; auto-generate fifo path          |
| `-F`, `--fifoname=PATH`     | Enable fifo input; use custom fifo path             |
| `-c`, `--command=CMD`       | Execute single command and exit                     |
| `-s`, `--startup=FILE`      | Run a script file and exit (with `-f`/`-F`: feed it through the fifo) |
| `-A`, `--autocomplete`      | Enable autocomplete preview                         |
| `--no-autocomplete`         | Disable autocomplete                                |
| `--no-history-suggest`      | Disable ghost history suggestions                   |
| `--no-arg-hints`            | Disable argument hint bar                           |
| `--no-fuzzy`                | Disable fuzzy matching in completions               |

**Examples:**

```bash
$ milk-cli -m /dev/tty2            # memory monitor on tty2
$ milk-cli -p 90                   # high RT priority
$ milk-cli -f                      # enable fifo (auto-named)
$ milk-cli -F /tmp/myfifo          # enable fifo, custom path
$ milk-cli -F /tmp/myfifo -s init.milk  # fifo + startup script
```

### Startup Script

`~/.milkrc` is sourced at interactive startup, like a bash script.

```bash
# Example ~/.milkrc
alias li=mem.listim
@config synhl on
```

### Configurable Prompt

Customize the prompt format using `@config prompt`:

```text
milk-cli > @config prompt "%u@%h %d > "
user@host src >
```

| Token | Expands to                 |
| ----- | -------------------------- |
| `%h`  | hostname                   |
| `%u`  | username                   |
| `%d`  | current directory basename |
| `%t`  | HH:MM:SS                   |
| `%n`  | process name               |

### Module Loading

Modules are shared objects (`lib<ModuleName>.so`) installed in `$MILK_INSTALLDIR/lib`. Commands of a
loaded module are called as `<shortname>.<command>`:

```text
milk-cli > @module load mylib.so        # load shared object
milk-cli > @module load milkimagegen     # load module by library name (libmilkimagegen.so)
milk-cli > @module load milkfft --as ft  # load module with a custom short name
milk-cli > @m?                           # list loaded modules
milk-cli > @m? <modulename>              # list the commands of a module
```

Modules can also be loaded at startup with the `MILKCLI_ADD_LIBS` environment variable (shared
object paths separated by space, comma or semicolon):

```bash
MILKCLI_ADD_LIBS="/path/libMyFirstModule.so /path/libMySecondModule.so" milk-cli
```

---

## Input & Editing

The CLI provides a rich interactive editing experience with tab completion, inline suggestions, and
syntax highlighting — all built on GNU readline.

### Syntax Rules

- Spaces separate arguments (count doesn't matter)
- Comments follow `#`
- Unrecognized input is interpreted as arithmetic

```text
milk-cli > <command> <arg1> <arg2>   # comment
```

### Tab Completion

- **First argument:** matches command → image → filename
- **Additional arguments:** context-aware based on command's parameter types:
  - `FILENAME` / `FITSFILENAME` args → filesystem paths
  - `FPSNAME` args → scan `/dev/shm/fps.*.shm` entries
  - Other args → match image stream names
- Fuzzy (substring) matching automatically kicks in when prefix matching finds nothing

```text
milk-cli > mem.<TAB>           # list commands starting with mem.
milk-cli > iofits.loadfits my<TAB>    # complete filename
```

### Ghost Suggestions

As you type, a dim "ghost" suggestion appears inline based on your command history. Press the
**right arrow** key to accept it.

### Argument Hints

When a known command (or `@<cmd>`) is typed, the CLI displays its `CLICMDARGDEF` argument signature
both as an inline ghost suffix after the cursor and in the bottom hint bar. The active argument
position is highlighted in bold cyan (`<name:type>`), advancing automatically as you type each
argument. You can also query a command's compact argument signature on demand via
`@hc --sig <cmd> [arg_pos]`.

### Readline Editing

GNU readline is used for line editing. Type `@helprl` at the prompt for a quick reference. See
[GNU readline documentation](http://tiswww.case.edu/php/chet/readline/rltop.html).

Key bindings include `Ctrl+A` (start of line), `Ctrl+E` (end of line), `Ctrl+R` (reverse history
search), and arrow keys for navigation.

### Backslash Line Continuation

End a line with `\` to continue on the next line. The prompt changes to `>` for continuation lines:

```text
milk-cli > arith.imfunc \
>   im1 im2 outim
```

### Syntax Highlighting

The first word is colored **green** (valid command) or **red** (unknown). Toggle with:

```text
milk-cli > @config synhl off            # disable
milk-cli > @config synhl on             # enable (default)
```

<!-- prettier-ignore -->
!!! tip
    If you encounter rendering issues with syntax highlighting, disable it with `@config synhl off`.

### Auto-Correction

When a command is not found, the CLI suggests the closest match using Levenshtein distance:

```text
milk-cli > mem.lisim
Command 'mem.lisim' not found. Did you mean 'mem.listim'?
```

---

## Help & Discovery

Several commands help you explore the available commands and modules.

### Help Commands

```text
milk-cli > @?                      # print help
milk-cli > @help                   # same as @?
milk-cli > @hb                     # bash-enhanced @ syntax & primitives guide
milk-cli > @help --json            # print help index in JSON format
milk-cli > @help --porcelain       # print help index in TSV format
milk-cli > @helprl                 # readline quick reference
milk-cli > @lm?                    # list all loaded modules
milk-cli > @m? <module>            # list commands for a module
milk-cli > @m?                     # list commands for all modules
milk-cli > @cmd? <command>         # detailed command description (or @hc <command>)
milk-cli > @hc --sig <cmd> [pos]   # compact CLICMDARGDEF signature hint
milk-cli > @cmd?                   # describe all commands
milk-cli > cmdinfo? <pattern>      # search command info by regex
```

Command names and descriptions are color-coded: cyan for command names, green for descriptions,
yellow for examples, dim for source file paths.

### Important Commands

```text
milk-cli > @ci                     # compilation info & memory usage
milk-cli > mem.listim              # list all images in memory
milk-cli > mem.listim --json       # list all images in JSON format
milk-cli > mem.listim --porcelain  # list all images in TSV format
milk-cli > mem.listim > list.txt    # list images into a file
milk-cli > exit                    # leave the shell
milk-cli > mem.mk2Dim <im> <xs> <ys>   # create 2D image
milk-cli > @usleep <usec>          # sleep for <usec> microseconds
milk-cli > @config precision single               # set default precision to float
milk-cli > @config precision double               # set default precision to double
milk-cli > cd <dir>                # change current working directory
milk-cli > pwd                     # print current working directory
```

---

## History

The CLI maintains both readline history (for arrow-key recall) and a persistent log of every command
with timestamps and session IDs.

### Persistent History

Command history is saved to `~/.milk_history` and loaded at startup. Up to 1000 entries are retained
between sessions.

```text
milk-cli > @history --global             # show last 20 entries (all sessions)
milk-cli > @history --global 50          # show last 50 entries
milk-cli > @history --local             # show entries from current session
```

Use `Ctrl+R` for interactive reverse search through history.

### Fuzzy History Search

```text
milk-cli > @history search mem.listim
```

Finds all history entries containing "mem.listim" (case- insensitive) and highlights the matching
substring in yellow. Shows match count at the end.

### Command Statistics

```text
milk-cli > @history --stats             # top 20 most-used commands
```

Shows command name and call count for the current session.

### Session Logging

Log all executed commands with timestamps:

```text
milk-cli > @config sessionlog on         # log to ~/.milk_session.log
milk-cli > @config sessionlog mylog.txt  # log to custom file
milk-cli > @config sessionlog off        # stop logging
```

Each entry includes a timestamp and elapsed time since the session started.

---

## Shell Features

The CLI supports many shell-like features: scripting, chaining, piping, redirection, environment
variable expansion, and arithmetic.

### Script Execution

```text
milk-cli > source myscript.milk  # run commands from file
milk-cli > . myscript.milk       # same thing (dot-source)
```

Blank lines and `#` comments are skipped. On error, the filename and line number are printed in red.

The CLI supports bash-like scripting constructs including variables, arithmetic, flow control
(`if`/`while`/`for`), and user-defined functions. See the [Scripting](scripting.md) page for full
documentation.

| Command               | Description                          |
| --------------------- | ------------------------------------ |
| `source <file>`       | Execute a script file                |
| `. <file>`            | Same as `source` (dot-source)        |
| `@history save <file>` | Save command history to file         |

### Command Timing

```text
milk-cli > time mem.listim      # measure execution time
```

Prints the elapsed wall-clock time after the command completes.

### Command Chaining

Sequential, conditional, and unconditional chaining:

```text
milk-cli > cmd1 ; cmd2         # always run cmd2 after cmd1
milk-cli > cmd1 && cmd2        # run cmd2 only if cmd1 succeeds
milk-cli > cmd1 || cmd2        # run cmd2 only if cmd1 fails
```

Operators inside double-quoted strings are not treated as chain separators.

### Pipe to Shell

Pipe a CLI command's output to a standard shell command:

```text
milk-cli > mem.listim | grep wfs     # filter image list
milk-cli > @cmd? | head -5           # show first 5 help lines
```

### Output Redirect

Redirect command output to a file:

```text
milk-cli > mem.listim > imlist.txt  # write image list to file
```

### Environment Variable Expansion

`$VAR` and `${VAR}` are expanded before execution:

```text
milk-cli > iofits.loadfits ${HOME}/data/im.fits im1
milk-cli > iofits.saveFITS im1 $OUTDIR/result.fits
```

### Command Substitution

Replace `$(cmd)` or `` `cmd` `` in the command line with the standard output of the command
execution.

```text
milk-cli > cd $(echo /tmp)
milk-cli > iofits.loadfits `findim.sh` result
```

### Wildcard Expansion (Globbing)

File path wildcards like `*`, `?`, and `[]` are automatically expanded into matching files if placed
unquoted in arguments.

```text
milk-cli > iofits.imgs2cube *.fits cube.fits
```

### Arithmetic & Logic Operations

The calculator is reached with `@calc 'expr'`, the statement form `@ expr`, or inline as `@{expr}`
(see [Scripting](scripting.md#calculator)). It supports images, scalars, and logical operations.

**Assignments & Math:**

```text
milk-cli > @ a = 5.0               # scalar assignment
milk-cli > @ b = a * 10            # scalar arithmetic
milk-cli > @ im1 = sqrt(im + 2.0)  # image arithmetic
```

**Functions:** Standard math functions are supported on both scalars and images (per-pixel): `abs`,
`fabs`, `round`, `fmod`, `min`, `max`, `sqrt`, `exp`, `cos`, `sin`, `tan`, `acos`, `asin`, `atan`.

```text
milk-cli > @ m = max(im1, 5)       # clip image below 5
```

**Relational & Logical Operators:** Supported operators: `<`, `<=`, `>`, `>=`, `==`, `!=`, `&&`,
`||`, `!`. When applied to images, these return a binary mask (1.0 or 0.0) for each pixel.

```text
milk-cli > @ mask = (im1 > 100) && (im2 != 0)
```

**Ternary Conditionals:** The `where(cond, true_val, false_val)` function works like a ternary
operator. If `cond` is an image mask, it blends the `true_val` and `false_val` images/scalars based
on the mask.

```text
milk-cli > @ im_clean = where(im > 100, 100, im)  # cap at 100
```

**Vector Reductions:**

```text
milk-cli > @ d = dot(im1, im2)     # sum(im1 * im2)
milk-cli > @ n = norm(im1)         # sqrt(dot(im1, im1))
```

### Repeating a Command

Use a bash loop; Ctrl-C stops it:

```text
milk-cli > while :; do clear; mem.listim; milksleep 1; done
```

---

## Customization

The CLI supports aliases and bookmarks for frequently-used commands.

### Command Aliases

These are bash aliases; put them in `~/.milkrc` to keep them:

```text
milk-cli > alias li=mem.listim   # create alias
milk-cli > unalias li            # remove alias
milk-cli > alias                 # list all aliases
```

### Command Bookmarks

Save and recall multi-command sequences:

```text
milk-cli > bookmark save setup "cmd1 ; cmd2 ; cmd3"
milk-cli > bookmark run setup
milk-cli > bookmark list
milk-cli > bookmark rm setup
```

Bookmarks persist in `~/.milk_bookmarks`.

---

## External Integration

The CLI integrates with external processes and standard Linux tools via FIFO input, FITS file I/O,
and pipe-based workflows.

### FIFO Input Mode

The CLI can receive commands from a named pipe (FIFO) in addition to interactive keyboard input.
This allows external processes to send commands to a running `milk-cli` session.

**Enabling FIFO mode:**

| Flag                          | Effect                               |
| ----------------------------- | ------------------------------------ |
| `-f` / `--fifoflag`           | Enable fifo; path auto-generated     |
| `-F PATH` / `--fifoname=PATH` | Enable fifo; use `PATH` as fifo file |

The auto-generated path follows the template:

```text
$SHMDIR/.<processname>.fifo.<PID>
```

For example, with default settings and PID 12345:

```text
/dev/shm/.1740801234p12345.fifo.0012345
```

Use `-n NAME` together with `-f` to produce a friendlier path:

```bash
$ milk-cli -n myctl -f   # /dev/shm/.myctl.fifo.<PID>
```

**Sending commands to a running session:**

From a separate shell, write newline-terminated commands to the fifo:

```bash
$ echo "mem.listim" > /dev/shm/.myctl.fifo.0012345
```

The CLI processes one line per select iteration and returns to interactive input once the pipe is
drained.

**Startup script via fifo:**

Use `-s FILE` together with `-f` or `-F` to pre-load a startup script. The file content is piped
into the fifo before accepting interactive input:

```bash
$ milk-cli -n myctl -F /tmp/myfifo -s setup.milk
```

### FITS File I/O

FITSIO is used for FITS file I/O. Requires `USE_CFITSIO=ON`. See
[Build Tiers](../quickstart/build_tiers.md).

```text
milk-cli > iofits.loadfits im1.fits imf1       # load as "imf1"
milk-cli > iofits.loadfits im1.fits            # auto-name from filename
milk-cli > iofits.loadfits im1.fits.gz im1     # load compressed FITS
milk-cli > iofits.save_fl im1 imf1.fits        # save as float
milk-cli > iofits.save_fl im1 "!im1.fits"      # overwrite existing file
```

### Driving milk-cli via FIFO

Start `milk-cli` with the `-f` flag to enable FIFO input (see [FIFO Input Mode](#fifo-input-mode)).
External processes can then send commands through the named pipe.

**From inside `milk-cli`** (assuming FIFO is at `/dev/shm/.myctl.fifo.<PID>`):

```text
milk-cli > !ls im*.fits | xargs -I {} echo iofits.loadfits {} > /dev/shm/.myctl.fifo.0012345
```

**From a separate shell** (while `milk-cli -n myctl -f` is running):

```bash
$ ls im*.fits | xargs -I {} echo iofits.loadfits {} > /dev/shm/.myctl.fifo.0012345
```

### Using `imlist.txt`

Start `milk-cli` with `--listimf` to maintain `imlist.txt` — an ASCII table of all images
currently in memory. Standard Unix tools can filter this list and generate commands.

The `-l` flag works independently of FIFO mode. You can feed the generated commands back into the
CLI via a FIFO:

**Via FIFO** (requires `-f` or `-F`; start with `milk-cli -l -n myctl -f`):

```text
milk-cli > !awk '{if ($4>200) print $2}' imlist.txt \
        | xargs -I {} echo iofits.save_fl {} {}_tmp.fits > /dev/shm/.myctl.fifo.0012345
```

---

← [Documentation Index](../index.md)
