---
tags:
  - scripts
  - cli
  - automation
---

# Scripts Reference

`milk` includes shell scripts for common operations. Scripts are installed to `bin/` alongside the
compiled executables.

See also: [Streams](streams.md) · [FPS](fps.md) · [FAQ](../quickstart/faq.md)

## 1. `@` Session Commands vs `milk-*` Executables

After sourcing `milk.bash` (`source milk.bash`), Milk provides two complementary command families:

| Family | Where to Use | Primary Examples |
| ------ | ------------ | ---------------- |
| **`@...` Commands** | Inside an active `milk.bash` or `milk-cli` session (zero-fork RPC + direct `/dev/shm` access) | `@` (`@calc`), `@test`, `@assert`, `@get`, `@set`, `@snapshot`, `@diff`, `@restore`, `@exists`, `@running`, `@waitfor`, `@waitany`, `@grab`, `@replay`, `@batch`, `@pipe`, `@defer`, `@on`, `@prompt`, `@server`, `@log`, `@h`, `@hb`, `@hc`, `@hm`, `@hf`, and all C module functions (`@imgen.mk2Dim`, `@info.imstats`, `@fits.loadfits`, ...) |
| **`milk-*` Executables** | Standalone OS programs (TUIs, background compute units, external shell tools) | `milk-CTRL`, `milk-fpsCTRL`, `milk-streamCTRL`, `milk-termview`, `milk-fpsexec-*`, `milk-commands`, `milk-help`, `milk-*-help` |

### Unified SHM & FPS Lifecycle Control via `@get` / `@set`

Within a `milk.bash` session, `@get` and `@set` replace external `milk-fps-*` and `milk-stream-*`
subprocess calls, supporting zero-subshell variable/array binding (`-v`, `-a`, `-A`), globbed
discovery and updates (`s.*`, `f.*`, `p.*`, `q.*`), multi-assignment, snapshots (`@snapshot`,
`@diff`, `@restore`), assertions (`@assert`), and JSON export (`--json`):

```bash
$ source milk.bash
$ @get -v nx s.cam0.xsize
$ @get -a streams 's.*'
$ @get -A widths 's.dm*.xsize'
$ @get --json s.cam0
$ @ -v rms 'irms(cam0)'
$ @set f.loopctrl.gain = 0.25 f.loopctrl.mult = 0.99
$ @set 'f.loop*.xsize' = 128 's.cam*.kw.OBSERVER' = "Subaru"
$ @set f.loopctrl.conf = start
$ @set f.loopctrl.run  = start:tmux
$ @snapshot -o baseline.snap 'f.loop*' 's.cam*'
$ @diff baseline.snap
$ @restore --dry-run baseline.snap
$ @assert --running --fps-min 500 --dim 128x128 s.cam0 "Camera stream degraded"
$ @assert "irms(cam0) < 0.5" "Residual diverged"
$ @prompt watch f.loopctrl.run gain=f.loopctrl.gain s.cam0.kw.OBSERVER
$ @batch "mem.mk2Dim a 64 64; a = 5; b = a * 2"
$ @pipe "mem.mk2Dim p1 32 32 |> p1 = 4 |> p2 = _ * 3"
$ @grab --fits telem.fits --timestamps telem.txt cam0 100
$ @replay --fps 200 telem.fits cam0_sim
$ @hc --sig mem.mk2Dim
```

## 2. Core Scripts (`scripts/`)

Installed from the root `CMakeLists.txt` into `bin/`:

| Script                       | Description                                        |
| ---------------------------- | -------------------------------------------------- |
| `milk.bash`                  | Sourceable GNU Bash integration layer (`@` commands, `@server`, `@log`, SHM streams, FPS) |
| `milk-check`                 | Verify installation (libraries, paths, SHM)        |
| `milk-argparse`              | Argument parsing helper for milk scripts           |
| `milk-script-std-config`     | Standard configuration for milk scripts            |
| `milk-scriptexample`         | Example/template for writing new scripts           |
| `milk-script-advanced`       | Advanced template combining bash & native milk-cli |
| `milk-exec`                  | Execute a milk CLI command non-interactively       |
| `milk-cli-all`               | Launch milk-cli on all accessible instances        |
| `milk-commands`              | List all available milk CLI commands               |
| `milk-completion.sh`         | Bash tab-completion for milk commands              |
| `milk-fpsinit`               | Initialize FPS instances from configuration        |
| `milk-fpslist-addentry`      | Add an entry to `fpslist.txt`                      |
| `milk-fpsmkcmd`              | Generate FPS command scripts                       |
| `merge3DfitsTelemetry`       | Merge 3D FITS telemetry cubes                      |
| `tmuxkillall`                | Kill all tmux sessions                             |
| `tmuxsessionname`            | Print the current tmux session name                |
| `waitforfile`                | Block until a specified file appears               |

## 2. Stream Utilities (`scripts/`)

| Script               | Description                                |
| -------------------- | ------------------------------------------ |
| `milk-FITS2shm`      | Load a FITS file into shared memory        |
| `milk-cubeslice2shm` | Extract a slice from a FITS cube into SHM  |
| `milk-streamlink`    | Create symbolic links to stream SHM files  |
| `milk-shm2FITSloop`  | Continuously save SHM stream to FITS files |

## 3. Logging Scripts (`scripts/`)

| Script             | Description                    |
| ------------------ | ------------------------------ |
| `milk-logshim`     | Start logging a stream to disk |
| `milk-logshimkill` | Kill a running log process     |
| `milk-logshimon`   | Enable logging for a stream    |
| `milk-logshimoff`  | Disable logging for a stream   |
| `milk-logshimstat` | Show logging status            |

## 4. Image Utilities (`scripts/`)

| Script              | Description                           |
| ------------------- | ------------------------------------- |
| `milk-cr2tofits`    | Convert Canon RAW (CR2) files to FITS |
| `milk-fitsheader`   | Display FITS file header              |
| `milk-images-merge` | Merge multiple FITS images            |

## 5. FPS Scripts (`src/engine/libfps/scripts/`)

| Script               | Description                                        |
| -------------------- | -------------------------------------------------- |
| `milk-demopipeline`  | Start a multi-stage FPS pipeline for demonstration |
| `milk-latency-audit` | Profile latency, cache misses, and IRQ affinities  |

## 6. milk-cli Script Payloads (`share/milk/scripts/`)

These are `.milk` files evaluated by the milk-cli interpreter. They are installed as **data files**
(no execute bit) under `share/milk/scripts/` and must be invoked via the CLI orchestrator:

```bash
milk-cli -s $(milk --install-prefix)/share/milk/scripts/makecircleofdisks.milk \
         [ndisks] [disk_radius] [circle_radius] [image_size] [output_name]
```

| Script                   | Description                                  |
| ------------------------ | -------------------------------------------- |
| `makecircleofdisks.milk` | Generate a procedural ring of circular disks |

## 7. COREMOD Scripts

### 7.1. COREMOD_memory (`src/coremods/COREMOD_memory/scripts/`)

| Script                   | Description                              |
| ------------------------ | ---------------------------------------- |
| `milk-semloopspeed`      | Benchmark semaphore loop speed           |
| `milk-shmimave`          | Compute running average of a SHM stream  |
| `milk-shmimcopy-semtrig` | Copy stream on semaphore trigger         |
| `milk-shmimpoke`         | Write test pattern to a SHM image        |
| `milk-shmimpoke-semtrig` | Poke SHM image on semaphore trigger      |
| `milk-streamFITSlog`     | Log stream to FITS files with timestamps |
| `milk-nettransmit`       | Transmit SHM stream over network         |

### 7.2. COREMOD_arith (`src/coremods/COREMOD_arith/scripts/`)

| Script                        | Description                       |
| ----------------------------- | --------------------------------- |
| `milk-stream-crop`            | Crop a sub-region from a stream   |
| `milk-fpslistadd-SCICROPMASK` | Add science crop mask to FPS list |
| `milk-fpslistadd-WFSCROPMASK` | Add WFS crop mask to FPS list     |

---

## 7. Usage Examples

### Loading a FITS file into shared memory

```bash
$ milk-FITS2shm myimage.fits mystream
```

This creates the shared memory stream `mystream` from the FITS file `myimage.fits`. Other processes
can then connect to `mystream` via `ImageStreamIO`.

### Benchmarking semaphore speed

```bash
$ milk-semloopspeed
```

Measures the round-trip semaphore post/wait latency. Typical values on modern hardware exceed 100
kHz.

### Logging a stream to disk

```bash
# Start logging stream "wfs" to FITS files
$ milk-logshim wfs

# Check logging status
$ milk-logshimstat

# Stop logging
$ milk-logshimkill wfs
```

### Streaming a FITS cube slice into shared memory

```bash
$ milk-cubeslice2shm mycube.fits 0 myslice
```

Extracts slice 0 from the FITS cube and writes it to the shared memory stream `myslice`.

---

← [Documentation Index](../index.md)
