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
    [FPS](fps.md#5-standalone-executables-fpsexec).

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
| `-v`, `--version`           | Print version and exit                              |
| `-i`, `--info`              | Print version, settings, info and exit              |
| `-q`, `--quiet`             | Start without the banner and startup messages       |
| `--verbose`                 | Be verbose                                          |
| `-d`, `--debug=LEVEL`       | Set debug level at startup                          |
| `-o`, `--overwrite`         | Auto-overwrite FITS files (**use with caution**)    |
| `-e`, `--errorexit`         | Exit on command error                               |
| `--listimf`                 | Write image list to `imlist.txt`                    |
| `-n`, `--pname=NAME`        | Rename process                                      |
| `--session NAME`            | Commands run in session NAME, started if it does not run (`@session`) |
| `--server SOCKET`           | Run as a session's server on abstract socket SOCKET (how milk.bash and [systemd](../operations/systemd.md#7-a-named-session-as-a-service) start one) |
| `--idle-timeout S`          | With `--server`: end after S seconds without a command while no shell uses it |
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
| `--idle`                    | Idle mode: only run process when X is idle          |
| `--catalog`                 | Dump module/command catalog in TSV format and exit  |
| `--format FILE`             | Format a `.milk` script to stdout and exit          |
| `--format-in-place FILE`    | Format a `.milk` script in place and exit           |
| `--outline FILE`            | Print structural outline of a `.milk` script        |

**Examples:**

```bash
$ milk-cli --session bench1        # the REPL's bash uses session bench1
$ milk-cli -p 90                   # high RT priority
$ milk-cli -f                      # enable fifo (auto-named)
$ milk-cli -F /tmp/myfifo          # enable fifo, custom path
$ milk-cli -F /tmp/myfifo -s init.milk  # fifo + startup script
```

### Startup Script

`~/.milkrc` is sourced when milk-cli starts an interactive prompt, like a bash script: not with
`-c`, `-s` or piped input, and not in milk-script, a session's server or plain bash.

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
milk-cli > @module list                  # list loaded modules (--all: also installed ones)
milk-cli > @help <modulename>            # list the commands of a module
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
- Module commands are written `<module>.<command>`, language keywords `@<keyword>`, and everything
  else is executed by the embedded bash shell (unless interactive `@calc` mode is toggled on)

```text
milk-cli > <module>.<command> <arg1> <arg2>   # comment
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

When a known `<module>.<cmd>` (or `@<keyword>`) is typed, the CLI displays its `CLICMDARGDEF`
argument signature both as an inline ghost suffix after the cursor and in the bottom hint bar. The
active argument position is highlighted in bold cyan (`<name:type>`), advancing automatically as you
type each argument. You can also inspect a command's argument list with `<module>.<cmd> -h` or
`@help <module>.<cmd>`.

### Readline Editing

GNU readline is used for line editing. Type `@help readline` at the prompt for a quick reference.
See [GNU readline documentation](http://tiswww.case.edu/php/chet/readline/rltop.html).

Key bindings include `Ctrl+A` (start of line), `Ctrl+E` (end of line), `Ctrl+R` (reverse history
search), and arrow keys for navigation.

### Backslash Line Continuation

End a line with `\` to continue on the next line. The prompt changes to `>` for continuation lines:

```text
milk-cli > arith.imtrunc \
>   im1 0.0 1.0 outim
```

### Syntax Highlighting

Syntax highlighting defaults to level `2` on 256-color terminals (full tree-sitter bash + milk
highlighting with live syntax diagnostics), or level `1` (8-color command/argument highlighting).
Configure it with:

```text
milk-cli > @config synhl off            # disable (same as 0)
milk-cli > @config synhl 1              # basic 8-color mode
milk-cli > @config synhl on             # enable 256-color tree-sitter mode (same as 2, default)
```

<!-- prettier-ignore -->
!!! tip
    If you encounter rendering issues with syntax highlighting, disable it with `@config synhl off`.

### Auto-Correction

When a command dispatched to the server is not found, the CLI suggests the closest match using
Levenshtein distance:

```text
milk-cli > @mem.lisim
milk: unknown command '@mem.lisim'
Did you mean?
  mem.listim (COREMOD_memory) -- list images in memory
```

---

## Help & Discovery

Several commands help you explore the available commands, keywords, and modules.

### Help Commands

```text
milk-cli > @help                   # list language keywords and help overview
milk-cli > @help <keyword>         # detailed help for a keyword (or <keyword> --help)
milk-cli > @help bash              # bash-enhanced @ syntax & primitives guide
milk-cli > @help readline          # readline quick reference
milk-cli > @help --tui             # browse help in an interactive fuzzy finder
milk-cli > @help --json            # print help index in JSON format
milk-cli > @help --porcelain       # print help index in TSV format
milk-cli > @module list            # list loaded modules (--all: also installed ones)
milk-cli > @help <module>          # list commands for a module
milk-cli > @help <module>.<cmd>    # detailed command description (or <module>.<cmd> -h)
milk-cli > @help -k <regex>        # search keywords, modules, and commands by regex
```

Command names and descriptions are color-coded: cyan for command names, green for descriptions,
yellow for examples, dim for source file paths.

### Important Commands

```text
milk-cli > milk.ci                 # compilation info & memory usage
milk-cli > mem.listim              # list all images in memory
milk-cli > mem.listim --json       # list all images in JSON format
milk-cli > mem.listim --porcelain  # list all images in TSV format
milk-cli > mem.listim > list.txt   # list images into a file
milk-cli > exit                    # leave the shell
milk-cli > mem.mk2Dim <im> <xs> <ys>   # create 2D image
milk-cli > @sleep <sec>            # sleep for <sec> seconds (supports fractional seconds)
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

Each line typed at a milk-cli prompt is appended to `~/.milk_history` as it runs, and the file is
loaded at startup, so arrow-key recall sees the lines of every milk-cli; the file keeps the last 1000.
The log (`~/.milk_history_log`) records the session each line ran in: with `--session NAME` (or
after `@session use NAME`), that session, else milk-cli's own.

```text
milk-cli > @history                      # the last 20 lines of this session (--local)
milk-cli > @history --global 50          # the last 50, all sessions
```

`@history`, `--local` and `--global` work from any front-end: in a bash shell, they show the lines
of the session it uses.

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

Shows each command's name and how often this milk-cli ran it; the commands its bash runs in another
session (`--session`) are not counted.

### Session Logging

Log the lines read at milk-cli's prompt, or piped into it:

```text
milk-cli > @config sessionlog on         # log to ~/.milk_session.log
milk-cli > @config sessionlog mylog.txt  # log to custom file
milk-cli > @config sessionlog off        # stop logging
```

Each entry is `[<time>] [<ms since logging started>] <line>`. Lines that arrive together are logged
as they are read, before they run, so the ones that follow `@config sessionlog on` in the same
batch are not.

---

## Shell Features

The CLI supports many shell-like features: scripting, chaining, piping, redirection, environment
variable expansion, and arithmetic.

### Script Execution

```text
milk-cli > @source myscript.milk         # translate and run a milk script in the current shell
milk-cli > @source --once myscript.milk  # skip if already sourced in this shell
milk-cli > source myscript.sh            # source a plain bash script (no @{...} translation)
```

Blank lines and `#` comments are skipped. `@source` translates milk constructs (`@{...}`) before
running the file in the current shell.

The CLI supports full bash scripting constructs including variables, arithmetic, flow control
(`if`/`while`/`for`), and user-defined functions. See the [Scripting](scripting.md) page for full
documentation.

| Command                  | Description                                            |
| ------------------------ | ------------------------------------------------------ |
| `@source [--once] <file>`| Translate and execute a `.milk` script file            |
| `source <file>` / `. <file>` | Source a plain bash file (without `@{...}` expansion) |
| `@history save <file>`   | Save command history to file                           |

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
milk-cli > @help | head -5           # show first 5 help lines
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

The calculator is reached with `@calc "expr"`, inline as `@{expr}`, or by entering interactive
calculator mode with a bare `@calc` (see [Scripting](scripting.md#calculator)). It supports images,
scalars, complex arrays, linear algebra, and logical operations.

**Assignments & Math:**

```text
milk-cli > @calc "a = 5.0"            # scalar assignment
milk-cli > @calc "b = a * 10"         # scalar arithmetic
milk-cli > @calc "im1 = sqrt(im + 2)" # one-pass fused image arithmetic
```

**Functions:** Standard math functions are supported on both scalars and images (per-pixel): `abs`,
`fabs`, `sign`, `round`, `floor`, `ceil`, `fmod`, `min`, `max`, `clamp`, `sqrt`, `cbrt`, `hypot`,
`exp`, `exp2`, `ln` (natural logarithm), `log` (base 10), `log2`, `cos`, `sin`, `tan`, `acos`,
`asin`, `atan`, `atan2`, `cosh`, `sinh`, `tanh`, `erf`, `erfc`
([all functions](scripting.md#calculator)).

```text
milk-cli > @calc "m = max(im1, 5)"    # clip image below 5
```

**Relational & Logical Operators:** Supported operators: `<`, `<=`, `>`, `>=`, `==`, `!=`, `&&`,
`||`, `!`. When applied to images, these return a binary mask (1.0 or 0.0) for each pixel.

```text
milk-cli > @calc "mask = (im1 > 100) && (im2 != 0)"
```

**Ternary Conditionals:** The `where(cond, true_val, false_val)` function and `cond ? a : b`
select per pixel when `cond` is an image mask.

```text
milk-cli > @calc "im_clean = where(im > 100, 100, im)"  # cap at 100
```

**Vector Reductions & Linear Algebra:**

```text
milk-cli > @calc "d = dot(im1, im2)"  # fused map-reduce sum(im1 * im2)
milk-cli > @calc "n = norm(im1)"      # Euclidean norm sqrt(dot(im1, im1))
milk-cli > @calc "C = matmul(A, B)"   # matrix product (BLAS when enabled)
```

### Repeating a Command

Use a bash loop; Ctrl-C stops it:

```text
milk-cli > while :; do clear; mem.listim; @sleep 1; done
```

---

## Customization

The CLI supports bash aliases and shell functions for frequently-used command sequences; put them in
`~/.milkrc` to load them automatically in interactive sessions.

### Command Aliases & Functions

```text
milk-cli > alias li=mem.listim              # create alias
milk-cli > unalias li                       # remove alias
milk-cli > alias                            # list all aliases
milk-cli > setup() { cmd1; cmd2; cmd3; }    # define a reusable multi-command function
```

---

## External Integration

The CLI integrates with external processes and standard Linux tools via named sessions (`@session`),
FIFO input (`milk.fifo`), FITS file I/O, and pipe-based workflows.

### FIFO Input Mode

The CLI can receive commands from a named pipe (FIFO) in addition to interactive keyboard input.
This allows external processes to send commands to a running `milk-cli` process.

**Enabling FIFO mode:**

| Flag                          | Effect                               |
| ----------------------------- | ------------------------------------ |
| `-f` / `--fifoflag`           | Enable fifo; path auto-generated     |
| `-F PATH` / `--fifoname=PATH` | Enable fifo; use `PATH` as fifo file |

The auto-generated path follows the template (with a 7-digit zero-padded PID):

```text
$MILK_SHM_DIR/.<processname>.fifo.<07d-PID>
```

For example, with default settings and PID 12345:

```text
/milk/shm/.1740801234p12345.fifo.0012345
```

Use `-n NAME` together with `-f` (or `-F PATH`) to produce a predictable path:

```bash
$ milk-cli -n myctl -f   # /milk/shm/.myctl.fifo.0012345
```

**Sending commands to a running session:**

From a separate shell, write newline-terminated commands to the fifo:

```bash
$ echo "mem.listim" > /milk/shm/.myctl.fifo.0012345
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

CFITSIO is used for FITS file I/O. Requires `USE_CFITSIO=ON`. See
[Build Tiers](../quickstart/build_tiers.md).

```text
milk-cli > iofits.loadfits im1.fits imf1       # load as "imf1"
milk-cli > iofits.loadfits im1.fits            # auto-name from filename
milk-cli > iofits.loadfits im1.fits.gz im1     # load compressed FITS
milk-cli > iofits.saveFITS im1 imf1.fits       # save to FITS file
milk-cli > iofits.saveFITS im1 "!im1.fits"     # overwrite existing file
```

### Driving milk-cli via FIFO

Start `milk-cli` with `-F /tmp/myctl.fifo` (or `-f`) to enable FIFO input (see
[FIFO Input Mode](#fifo-input-mode) and [FIFO Command Input](fifo.md)). External processes can then
send commands through the named pipe.

**From inside `milk-cli`**:

```text
milk-cli > ls im*.fits | xargs -I {} echo "iofits.loadfits {}" > /tmp/myctl.fifo
```

**From a separate shell**:

```bash
$ ls im*.fits | xargs -I {} echo "iofits.loadfits {}" > /tmp/myctl.fifo
```

### Using `imlist.txt`

Start `milk-cli` with `--listimf` to maintain `imlist.txt` — an ASCII table of all images
currently in memory. Standard Unix tools can filter this list and generate commands.

The `--listimf` flag works independently of FIFO mode. You can feed the generated commands back into
the CLI via a FIFO (start with `milk-cli --listimf -F /tmp/myctl.fifo`):

```text
milk-cli > awk '{if ($4>200) print $2}' imlist.txt \
        | xargs -I {} echo "iofits.saveFITS {} {}_tmp.fits" > /tmp/myctl.fifo
```

---

← [Documentation Index](../index.md)
