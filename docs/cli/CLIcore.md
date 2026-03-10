# Command Line Interface Syntax

The interactive CLI is provided by the `milk-cli` executable
(typically aliased as `milk`). Source code is in
`src/cli/CLIcore/`.

> [!NOTE]
> Standalone executables (`milk-fpsexec-*`) have their own
> command-line interfaces. See
> [FPS Standalone Modes](../FPS_Standalone_CMD_Modes.md).

## Command Line Options

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

## Syntax Rules and Parser

- Spaces separate arguments (count doesn't matter)
- Comments follow `#`
- Unrecognized input is interpreted as arithmetic

```text
milk> <command> <arg1> <arg2>   # comment
```

## Tab Completion

- **First argument:** matches command → image → filename
- **Additional arguments:** matches image → filename

## Input

GNU readline is used for line editing. Type `helprl` at the
prompt for a quick reference. See
[GNU readline documentation](http://tiswww.case.edu/php/chet/readline/rltop.html).

The CLI reads commands from `cmdfile.txt` if it exists,
executing them top-to-bottom and removing each line as it
is read.

<details>
<summary><b>Help Commands</b></summary>

```text
milk> ?                       # print help
milk> help                    # same as ?
milk> helprl                  # readline quick reference
milk> lm?                     # list all loaded modules
milk> m? <module>             # list commands for a module
milk> m?                      # list commands for all modules
milk> cmd? <command>          # detailed command description
milk> cmd?                    # describe all commands
milk> h? <str>                # search commands by string
```

</details>

<details>
<summary><b>Important Commands</b></summary>

```text
milk> ci                      # compilation time & memory usage
milk> listim                  # list all images in memory
milk> listimf <file>          # list images, write to file
milk> !<syscmd>               # execute system command
milk> showhist                # print command history
milk> quit                    # exit (or: exit)
milk> setdp <val>             # precision: 0=float, 1=double
milk> creaim <im> <xs> <ys>   # create 2D image
```

</details>

<details>
<summary><b>FITS Files I/O</b></summary>

FITSIO is used for FITS file I/O. See also `COREMOD_memory`
and `COREMOD_iofits`.

> [!NOTE]
> FITS I/O requires `USE_CFITSIO=ON` at build time.
> See [Build Tiers](../install/build_tiers.md).

**Loading:**

```text
milk> loadfits im1.fits imf1       # load as "imf1"
milk> loadfits im1.fits            # load as "im1" (auto-name)
milk> loadfits im1.fits.gz im1     # load compressed
```

**Saving:**

```text
milk> save_fl im1 imf1.fits        # save as float
milk> save_fl im1                  # save as im1.fits (auto)
milk> save_fl im1 "!im1.fits"      # overwrite existing
milk> save_fl im1 ../dir2/im1.fits # specify path
milk> save_fl im1 im1.fits.gz      # save compressed
```

</details>

<details>
<summary><b>Integration with Standard Linux Tools</b></summary>

### Using `cmdfile.txt` to drive milk-cli

`milk-cli` executes commands from `cmdfile.txt` if the file
exists. This enables scripting from both inside and outside
the CLI.

**From inside `milk-cli`:**

```text
milk> !ls im*.fits | xargs -I {} echo loadfits {} > cmdfile.txt
```

**From a separate shell (while `milk-cli` is running):**

```bash
$ ls im*.fits | xargs -I {} echo loadfits {} > cmdfile.txt
```

### Using `imlist.txt` and `cmdfile.txt`

Start `milk-cli` with `-l` to maintain `imlist.txt`. Then
filter and act on the list:

```text
milk> !awk '{if ($4>200) print $2}' imlist.txt \
        | xargs -I {} echo save_fl {} {}_tmp.fits > cmdfile.txt
```

</details>

## Arithmetic Operations

```text
milk> im1=sqrt(im+2.0)       # arithmetic on images
```
