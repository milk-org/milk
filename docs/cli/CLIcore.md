---
tags:
  - cli
  - interactive
---

# Command Line Interface Syntax

The interactive CLI is provided by the `milk-cli` executable
(typically aliased as `milk`). Source code is in
`src/cli/CLIcore/`.

> [!NOTE]
> Standalone executables (`milk-fpsexec-*`) have their own
> command-line interfaces. See
> [FPS Standalone Modes](../FPS_Standalone_CMD_Modes.md).

See also: [FPS](../fps.md) ·
[Streams](../streams.md) ·
[FAQ](../faq.md) ·
[Build Tiers](../install/build_tiers.md)

## 1. Command Line Options

When launching `milk-cli`, the following arguments are
available:

| Option | Description |
|--------|-------------|
| `-h`, `--help` | Print help and exit |
| `-i`, `--info` | Print version, settings, info and exit |
| `-j`, `--journal` | Write all commands to `milk_cmdlog.txt` |
| `--verbose` | Be verbose |
| `-d`, `--debug=LEVEL` | Set debug level at startup |
| `-o`, `--overwrite` | Auto-overwrite FITS files (**use with caution**) |
| `-l` | Write image list to `imlist.txt` |
| `-m`, `--mmon=TTY` | Open memory monitor on tty device |
| `-n`, `--pname=NAME` | Rename process |
| `-p`, `--priority=PR` | Set RT priority (0–99, higher = higher) |
| `-f`, `--fifo=FIFO` | Specify fifo name |
| `-s`, `--startup=FILE` | Execute script on startup (requires `-f`) |

**Examples:**

```bash
$ milk-cli -m /dev/tty2       # memory monitor
$ milk-cli -p 90              # high priority
$ milk-cli -f /tmp/fifo24     # custom fifo
```

## 2. Syntax Rules and Parser

- Spaces separate arguments (count doesn't matter)
- Comments follow `#`
- Unrecognized input is interpreted as arithmetic

```text
milk-cli > <command> <arg1> <arg2>   # comment
```

## 3. Tab Completion

- **First argument:** matches command → image → filename
- **Additional arguments:** context-aware based on command's
  parameter types:
  - `FILENAME` / `FITSFILENAME` args → filesystem paths
  - `FPSNAME` args → scan `/dev/shm/fps.*.shm` entries
  - Other args → match image stream names
- Fuzzy (substring) matching automatically kicks in when
  prefix matching finds nothing

```text
milk-cli > mem.<TAB>           # list commands starting with mem.
milk-cli > loadfits my<TAB>    # complete filename starting with my
```

## 4. Ghost Suggestions

As you type, a dim "ghost" suggestion appears inline based
on your command history. Press the **right arrow** key to
accept it.

## 5. Argument Hints

When a known command is typed, the bottom line shows the
command's syntax with `<angle bracket>` parameter tokens.
The active argument position is highlighted in bold cyan,
advancing as you type each argument.

## 6. Readline Editing

GNU readline is used for line editing. Type `helprl` at the
prompt for a quick reference. See
[GNU readline documentation](http://tiswww.case.edu/php/chet/readline/rltop.html).

Key bindings include `Ctrl+A` (start of line), `Ctrl+E`
(end of line), `Ctrl+R` (reverse history search), and
arrow keys for navigation.

The CLI also reads commands from `cmdfile.txt` if it exists,
executing them top-to-bottom and removing each line as it
is read.

## 7. Help Commands

```text
milk-cli > ?                       # print help
milk-cli > help                    # same as ?
milk-cli > helprl                  # readline quick reference
milk-cli > lm?                     # list all loaded modules
milk-cli > m? <module>             # list commands for a module
milk-cli > m?                      # list commands for all modules
milk-cli > cmd? <command>          # detailed command description
milk-cli > cmd?                    # describe all commands
milk-cli > cmdinfo? <pattern>      # search command info by regex
```

Command names and descriptions are color-coded: cyan for
command names, green for descriptions, yellow for examples,
dim for source file paths.

## 8. Important Commands

```text
milk-cli > ci                      # compilation info & memory usage
milk-cli > listim                  # list all images in memory
milk-cli > listimf <file>          # list images, write to file
milk-cli > !<syscmd>               # execute system command
milk-cli > quit                    # exit (or: exit, exitCLI)
milk-cli > creaim <im> <xs> <ys>   # create 2D image
milk-cli > usleep <usec>           # sleep for <usec> microseconds
milk-cli > dpsingle                # set default precision to float
milk-cli > dpdouble                # set default precision to double
```

## 9. FITS File I/O

FITSIO is used for FITS file I/O. Requires `USE_CFITSIO=ON`.
See [Build Tiers](../install/build_tiers.md).

```text
milk-cli > loadfits im1.fits imf1       # load as "imf1"
milk-cli > loadfits im1.fits            # auto-name from filename
milk-cli > loadfits im1.fits.gz im1     # load compressed FITS
milk-cli > save_fl im1 imf1.fits        # save as float
milk-cli > save_fl im1 "!im1.fits"      # overwrite existing file
```

## 10. Persistent History

Command history is saved to `~/.milk_history` and loaded
at startup. Up to 1000 entries are retained between
sessions.

```text
milk-cli > history              # show last 20 commands
milk-cli > history 50           # show last 50 commands
```

Use `Ctrl+R` for interactive reverse search through
history.

## 11. History Expansion

| Shortcut | Description |
|----------|-------------|
| `!!` | Re-run the last command |
| `!! args` | Append `args` to the last command |
| `!$` | Insert the last argument of the previous command |
| `!prefix` | Re-run the last command starting with `prefix` |

The expanded command is printed with a `>>` prefix before
execution.

```text
milk-cli > loadfits im1.fits im1
milk-cli > !!                   # re-runs: loadfits im1.fits im1
milk-cli > save_fl !$           # expands to: save_fl im1
```

## 12. Fuzzy History Search

```text
milk-cli > searchhist listim
```

Finds all history entries containing "listim" (case-
insensitive) and highlights the matching substring in
yellow. Shows match count at the end.

## 13. Startup Script

Commands in `~/.milkrc` are executed line-by-line on
startup. Blank lines and `#` comments are skipped.

```bash
# Example ~/.milkrc
alias li mem.listim
synhl on
```

## 14. Script Execution

```text
milk-cli > source myscript.milk  # run commands from file
```

Blank lines and `#` comments are skipped. On error, the
filename and line number are printed in red.

**Example `myscript.milk`:**

```bash
# This is a sample milk script
echo "Starting image processing..."

# Create a 100x100 image named 'myimg'
mem.mk2Dim myimg 100 100

# Perform some basic arithmetic
im1=myimg+5.0

# Save to FITS using an environment variable
iofits.saveFITS im1 ${HOME}/im1_test.fits

# Print current images in memory
mem.listim
```

## 15. Command Timing

```text
milk-cli > time mem.listim      # measure execution time
```

Prints the elapsed wall-clock time after the command
completes.

## 16. Command Chaining

Sequential, conditional, and unconditional chaining:

```text
milk-cli > cmd1 ; cmd2         # always run cmd2 after cmd1
milk-cli > cmd1 && cmd2        # run cmd2 only if cmd1 succeeds
milk-cli > cmd1 || cmd2        # run cmd2 only if cmd1 fails
```

Operators inside double-quoted strings are not treated as
chain separators.

## 17. Pipe to Shell

Pipe a CLI command's output to a standard shell command:

```text
milk-cli > mem.listim | grep wfs     # filter image list
milk-cli > cmd? | head -5            # show first 5 help lines
```

## 18. Output Redirect

Redirect command output to a file:

```text
milk-cli > mem.listim > imlist.txt  # write image list to file
```

## 19. Environment Variable Expansion

`$VAR` and `${VAR}` are expanded before execution:

```text
milk-cli > iofits.loadfits ${HOME}/data/im.fits im1
milk-cli > iofits.saveFITS im1 $OUTDIR/result.fits
```

## 20. Backslash Line Continuation

End a line with `\` to continue on the next line. The
prompt changes to `>` for continuation lines:

```text
milk-cli > arith.imfunc \
>   im1 im2 outim
```

## 21. Syntax Highlighting

The first word is colored **green** (valid command) or
**red** (unknown). Toggle with:

```text
milk-cli > synhl off             # disable
milk-cli > synhl on              # enable (default)
```

> [!TIP]
> If you encounter rendering issues with syntax
> highlighting, disable it with `synhl off`.

## 22. Auto-Correction

When a command is not found, the CLI suggests the closest
match using Levenshtein distance:

```text
milk-cli > mem.lisim
Command 'mem.lisim' not found. Did you mean 'mem.listim'?
```

## 23. Command Statistics

```text
milk-cli > cmdstats              # top 20 most-used commands
```

Shows command name and call count for the current session.

## 24. Configurable Prompt

Customize the prompt format using `setprompt`:

```text
milk-cli > setprompt "%u@%h %d > "
user@host src >
```

| Token | Expands to |
|-------|------------|
| `%h` | hostname |
| `%u` | username |
| `%d` | current directory basename |
| `%t` | HH:MM:SS |
| `%n` | process name |

## 25. Command Aliases

```text
milk-cli > alias li mem.listim   # create alias
milk-cli > unalias li            # remove alias
milk-cli > aliases               # list all aliases
```

Aliases persist in `~/.milk_aliases`.

## 26. Command Bookmarks

Save and recall multi-command sequences:

```text
milk-cli > bookmark save setup "cmd1 ; cmd2 ; cmd3"
milk-cli > bookmark run setup
milk-cli > bookmark list
milk-cli > bookmark rm setup
```

Bookmarks persist in `~/.milk_bookmarks`.

## 27. Session Logging

Log all executed commands with timestamps:

```text
milk-cli > sessionlog on          # log to ~/.milk_session.log
milk-cli > sessionlog mylog.txt   # log to custom file
milk-cli > sessionlog off         # stop logging
```

Each entry includes a timestamp and elapsed time since
the session started.

## 28. Watch Command

```text
milk-cli > watch 1000 mem.listim   # repeat every 1000ms
```

Press any key to stop the repeating execution.

## 29. Arithmetic Operations

Expressions not matching any command are evaluated as
image arithmetic:

```text
milk-cli > im1=sqrt(im+2.0)       # arithmetic on images
```

## 30. Module Loading

Load shared library modules at runtime:

```text
milk-cli > soload mylib.so         # load shared object
milk-cli > mload COREMOD_arith     # load module by name
```

## 31. Integration with Standard Linux Tools

### Using `cmdfile.txt` to drive milk-cli

`milk-cli` executes commands from `cmdfile.txt` if the file
exists. This enables scripting from both inside and outside
the CLI.

**From inside `milk-cli`:**

```text
milk-cli > !ls im*.fits | xargs -I {} echo loadfits {} > cmdfile.txt
```

**From a separate shell (while `milk-cli` is running):**

```bash
$ ls im*.fits | xargs -I {} echo loadfits {} > cmdfile.txt
```

### Using `imlist.txt` and `cmdfile.txt`

Start `milk-cli` with `-l` to maintain `imlist.txt`. Then
filter and act on the list:

```text
milk-cli > !awk '{if ($4>200) print $2}' imlist.txt \
        | xargs -I {} echo save_fl {} {}_tmp.fits > cmdfile.txt
```

---
← [Documentation Index](../index.md)
