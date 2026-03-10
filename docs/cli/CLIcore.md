# Command Line Interface Syntax

The interactive CLI is provided by the `milk-cli` executable (typically aliased as `milk`). Source code is in `src/cli/CLIcore/`.

> [!NOTE]
> Standalone executables (`milk-fpsexec-*`) have their own command-line interfaces. See [FPS Standalone Modes](../FPS_Standalone_CMD_Modes.md).

## Command Line Options

When launching `milk` (or `milk-cli`), the following arguments are available:

| Option | Description |
|---|---|
| `-h`, `--help` | Print this message and exit |
| `-i`, `--info` | Print version, settings, info and exit |
| `-j`, `--journal` | Keeps journal of commands (Write all commands to file `milk_cmdlog.txt` as they are entered) |
| `--verbose` | Be verbose |
| `-d`, `--debug=DEBUGLEVEL` | Set debug level at startup |
| `-o`, `--overwrite` | Automatically overwrite files if necessary (**USE WITH CAUTION - WILL OVERWRITE EXISTING FITS FILES**) |
| `-l` | Keeps a list of images in file `imlist.txt` |
| `-m`, `--mmon=TTYDEVICE` | Open memory monitor on tty device. <br> Example: `milk -m /dev/tty2` |
| `-n`, `--pname=<processname>` | Rename process to `<processname>` |
| `-p`, `--priority=<PR>` | Change process priority (0-99). Higher number = higher priority. <br> Example: `milk -p 90` |
| `-f`, `--fifo=<FIFONAME>` | Specify fifo name. <br> Example: `milk -f /tmp/fifo24` |
| `-s`, `--startup=STARTUPFILE` | Execute specified script on startup. Requires the `-f` option, as the script is loaded into fifo. |

## Syntax Rules and Parser

- Spaces are used to separate arguments. The number of spaces is irrelevant.
- Comments are written after the special character `#`.
- If a command is not found, the input string will be interpreted as an arithmetic operation (See **Arithmetic Operations** below).

```bash
<command> <arg1> <arg2>   # comment
```

## Tab Completion

Tab completion is provided and behaves as follows:
- **First argument:** Try to match command, then image, then filename.
- **Additional arguments:** Try to match image, then filename.

## Input

GNU readline is used to read input. See [GNU readline documentation](http://tiswww.case.edu/php/chet/readline/rltop.html). For a quick help on readline input, type:
```text
> helprl
```

The CLI will take input from file `cmdfile.txt` if it exists. If file `cmdfile.txt` exists, commands will be read one by one from top to bottom, and will be removed from the file as they are read, until the file is empty.

<details>
<summary><b>Help Commands</b></summary>

```text
> ?
> help
	# print this help file
> helprl
	# print readline quick help
> lm?
	# list all modules loaded
> m? <module>
	# list all commands for a module
> m?
	# perform m? on all modules loaded
> cmd? <command>
	# command description for <command>
> cmd?
	# command description for all commands
> h? str
	# search for string <str> in all commands and their descriptions
```

</details>

<details>
<summary><b>Important Commands</b></summary>

```text
> ci
	# compilation time and memory usage
> listim
	# list all images in memory
> listimf <filename>
	# list all images in memory and write output to file <filename>
> !<syscommand>
	# execute system command
> showhist
	# prints history of all commands
> quit
	# exit the shell (exit also works)

> setdp <val>
	# set default precision to float (<val> = 0) or double (<val> = 1)
> creaim <im> <xs> <ys>
	# creates a 2D image named <im>, size = <xs> x <ys> pixels
```

</details>

<details>
<summary><b>FITS Files I/O</b></summary>

FITSIO is used for FITS files I/O. See FITSIO documentation for more detailed instructions. (See also modules `COREMOD_memory` and `COREMOD_iofits`).

> [!NOTE]
> FITS I/O requires `USE_CFITSIO=ON` at build time.
> See [Build Tiers](../install/build_tiers.md).

### Loading Files

```text
> loadfits <fname> <imname>
	# load FITS file <fname> into image <imname>
> loadfits im1.fits imf1
	# load file im1.fits in memory with name imf1
> loadfits im1.fits
	# load file im1.fits in memory with name im1 (default name is composed of all chars before first ".")
> loadfits im1.fits.gz im1
	# load compressed file
```

### Saving Files

```text
> save_fl  <imname> <fname>
	# save image <imname> into FITS file <fname> (float)
> save_fl im1 imf1.fits
	# write image im1 to disk file imf1.fits
> save_fl im1
	# write image im1 to disk file im1.fits (default file name = image name + ".fits")
> save_fl im1 "!im1.fits"
	# overwrite file im1.fits if it exists
> save_fl im1 "../dir2/im1.fits"
	# specify full path
> save_fl im1 im1.fits.gz
	# save compressed image
```

</details>


<details>
<summary><b>Integration with Standard Linux Tools</b></summary>

### Using `cmdfile.txt` to Drive milk from UNIX Prompt

`milk` can use standard Linux tools and commands thanks to the `cmdfile.txt` file, which, if it exists, is executed as `milk` commands.

For example, to load all `im*.fits` files in memory, you can type within `milk`:
```bash
> !ls im*.fits | xargs -I {} echo loadfits {} > cmdfile.txt
```

You can also drive `milk` from the unix command line if you are not in the `milk` interactive shell, but `milk` is running in the same directory. For example, the following command will load all `im*.fits` into `milk` from the unix command line:
```bash
$ ls im*.fits | xargs -I {} echo loadfits {} > cmdfile.txt
```

### Using `imlist.txt` and `cmdfile.txt`

If you start `milk` with the `-l` option, the file `imlist.txt` contains the list of images currently in memory in an ASCII table. You can use standard unix tools to process this list and issue commands. For example, if you want to save all images with an x-size > 200 onto disk as single precision FITS files:

```bash
> !awk '{if ($4>200) print $2}' imlist.txt | xargs -I {} echo save_fl {} {}_tmp.fits > cmdfile.txt
```

</details>

## Arithmetic Operations

```text
> im1=sqrt(im+2.0)
	# will perform an arithmetic operation on image im and store the result in image im1
```
