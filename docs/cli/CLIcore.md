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

## 4. Argument Hints

When a known command is typed, the bottom line shows the
command's syntax with `<angle bracket>` parameter tokens.
The active argument position is highlighted in bold cyan.

## 5. Input

GNU readline is used for line editing. Type `helprl` at the
prompt for a quick reference. See
[GNU readline documentation](http://tiswww.case.edu/php/chet/readline/rltop.html).

The CLI reads commands from `cmdfile.txt` if it exists,
executing them top-to-bottom and removing each line as it
is read.

<details markdown="1">
<summary><b>Help Commands</b></summary>

```text
milk-cli > ?                       # print help
milk-cli > help                    # same as ?
milk-cli > helprl                  # readline quick reference
milk-cli > lm?                     # list all loaded modules
milk-cli > m? <module>             # list commands for a module
milk-cli > m?                      # list commands for all modules
milk-cli > cmd? <command>          # detailed command description
milk-cli > cmd?                    # describe all commands
milk-cli > h? <str>                # search commands by string
```

</details>

<details markdown="1">
<summary><b>Important Commands</b></summary>

```text
milk-cli > ci                      # compilation time & memory usage
milk-cli > listim                  # list all images in memory
milk-cli > listimf <file>          # list images, write to file
milk-cli > !<syscmd>               # execute system command
milk-cli > showhist                # print command history
milk-cli > quit                    # exit (or: exit)
milk-cli > setdp <val>             # precision: 0=float, 1=double
milk-cli > creaim <im> <xs> <ys>   # create 2D image
```

</details>

<details markdown="1">
<summary><b>FITS Files I/O</b></summary>

FITSIO is used for FITS file I/O. See also `COREMOD_memory`
and `COREMOD_iofits`.

**ℹ️ Note:** FITS I/O requires `USE_CFITSIO=ON` at build
time. See [Build Tiers](../install/build_tiers.md).

**Loading:**

```text
milk-cli > loadfits im1.fits imf1       # load as "imf1"
milk-cli > loadfits im1.fits            # load as "im1" (auto-name)
milk-cli > loadfits im1.fits.gz im1     # load compressed
```

**Saving:**

```text
milk-cli > save_fl im1 imf1.fits        # save as float
milk-cli > save_fl im1                  # save as im1.fits (auto)
milk-cli > save_fl im1 "!im1.fits"      # overwrite existing
milk-cli > save_fl im1 ../dir2/im1.fits # specify path
milk-cli > save_fl im1 im1.fits.gz      # save compressed
```

</details>

<details markdown="1">
<summary><b>Integration with Standard Linux Tools</b></summary>

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

</details>

## 5. Persistent History

Command history is saved to `~/.milk_history` and loaded
at startup. Up to 1000 entries are retained between sessions.

## 6. History Expansion

| Shortcut | Description |
|----------|-------------|
| `!!` | Re-run the last command |
| `!!args` | Append `args` to the last command |
| `!$` | Insert the last argument of the previous command |
| `!prefix` | Re-run the last command starting with `prefix` |

The expanded command is printed with a `>>` prefix before
execution.

## 7. Startup Script

Commands in `~/.milkrc` are executed line-by-line on startup.
Blank lines and `#` comments are skipped.

## 8. Command Timing

```text
milk-cli > time mem.listim      # measure execution time
```

## 9. Command Chaining, Pipes, and Redirects

```text
milk-cli > cmd1 ; cmd2           # sequential execution
milk-cli > mem.listim | grep im  # pipe to shell
milk-cli > mem.listim > out.txt  # redirect to file
```

## 10. Command Statistics

```text
milk-cli > cmdstats              # top 20 most-used commands
```

## 11. Script Execution

```text
milk-cli > source myscript.milk  # run commands from file
```

Blank lines and `#` comments are skipped. Errors show the
file name and line number.

## 12. Syntax Highlighting

The first word is colored **green** (valid command) or
**red** (unknown). Toggle with:

```text
milk-cli > synhl off             # disable
milk-cli > synhl on              # enable (default)
```

> [!TIP]
> If you encounter rendering issues with syntax
> highlighting, disable it with `synhl off`.

## 13. Auto-Correction

When a command is not found, the CLI suggests the closest
match using Levenshtein distance:

```text
milk-cli > mem.lisim
Command 'mem.lisim' not found. Did you mean 'mem.listim'?
```

## 14. Configurable Prompt

Customize the prompt format using `setprompt`:

```text
milk-cli > setprompt "%u@%h %d > "
```

| Token | Expands to |
|-------|------------|
| `%h` | hostname |
| `%u` | username |
| `%d` | current directory basename |
| `%t` | HH:MM:SS |
| `%n` | process name |

## 15. Command Bookmarks

Save and recall multi-command sequences:

```text
milk-cli > bookmark save setup "cmd1 ; cmd2 ; cmd3"
milk-cli > bookmark run setup
milk-cli > bookmark list
milk-cli > bookmark rm setup
```

Bookmarks persist in `~/.milk_bookmarks`.

## 16. Session Logging

Log all commands with timestamps:

```text
milk-cli > sessionlog on          # log to ~/.milk_session.log
milk-cli > sessionlog mylog.txt   # log to custom file
milk-cli > sessionlog off         # stop logging
```

## 17. Command Aliases

```text
milk-cli > alias li mem.listim   # create alias
milk-cli > unalias li            # remove alias
milk-cli > aliaslist             # list all aliases
```

Aliases persist in `~/.milk_aliases`.

## 18. Watch Command

```text
milk-cli > watch 1000 mem.listim   # repeat every 1000ms
```

Press any key to stop.

## 19. Arithmetic Operations

```text
milk-cli > im1=sqrt(im+2.0)       # arithmetic on images
```

---
← [Documentation Index](../index.md)
