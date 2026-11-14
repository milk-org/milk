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

Module commands are called as `<module>.<command>`; an installed module loads at the first use of
one of its commands. `@module`, `MILKCLI_ADD_LIBS` and `milk-cli-all` are described in
[Scripting](scripting.md#module-commands).

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

The prompt and bash (after `source milk.bash`) complete with the same completer:

- **First word:** module commands, `@` keywords and bash words
- **Arguments of a module command:** by the command's parameter types:
  - `FILENAME` / `FITSFILENAME` args → filesystem paths
  - `FPSNAME` args → FPS instances
  - Other args → image and stream names, and entity paths (`s.cam.`, `f.loop.`, `p.loop.`)
- **Arguments of a keyword:** its options after `-`, and its words (`@config` settings, `@session`
  subcommands); the arguments of `@calc`, `@test`, `@assert`, calc mode lines and `@{...}`
  complete calculator functions, variables and images
- **Any other command** completes file names, as in bash
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

Syntax highlighting is on by default: tree-sitter bash + milk highlighting with live syntax
diagnostics, in 256 colors, or 16 on terminals with fewer. Calc mode lines are not highlighted.
Configure it with:

```text
milk-cli > @config synhl off            # disable
milk-cli > @config synhl on             # enable (default)
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

`@help` lists the keywords and topics, `@help <keyword|module|module.cmd>` describes one, and
`@help -k <regex>` searches them: see [Scripting](scripting.md#keywords). At the prompt,
`@help --tui` browses the help in a fuzzy finder and `@help readline` is a quick reference of the
editing keys. Command names are shown in cyan, descriptions in green, examples in yellow and source
file paths dim.

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

Each line that is not a module command or a keyword is run by the embedded bash, in which the
commands and keywords are functions and builtins. Chaining (`;`, `&&`, `||`), pipes, redirection,
`$VAR`, `$(cmd)`, globs, loops, `time`, aliases and functions therefore work as in bash:

```text
milk-cli > mem.listim | grep wfs
milk-cli > iofits.loadfits ${HOME}/data/im.fits im1 && iofits.saveFITS im1 $OUTDIR/result.fits
milk-cli > while :; do clear; mem.listim; @sleep 1; done      # Ctrl-C stops it
```

Scripts, `@source` and the keywords are described in [Scripting](scripting.md); `@calc "expr"`,
`@{expr}` and the calculator mode that a bare `@calc` enters, in [Calculator](calculator.md).
Aliases and functions put in `~/.milkrc` are there at every interactive start.

---

## External Integration

The CLI integrates with external processes and standard Linux tools via named sessions (`@session`),
FIFO input (`milk.fifo`), FITS file I/O, and pipe-based workflows.

### FIFO Input

With `-f` or `-F PATH`, `milk-cli` also reads commands from a named pipe, one per line, so that
another process can drive it: see [FIFO Command Input](fifo.md).

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
