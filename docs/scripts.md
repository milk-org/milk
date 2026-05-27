---
tags:
  - scripts
  - cli
  - automation
---

# Scripts Reference

`milk` includes shell scripts for common operations. Scripts
are installed to `bin/` alongside the compiled executables.

See also: [Streams](streams.md) ·
[FPS](fps.md) ·
[FAQ](faq.md)

## 1. Core Scripts (`scripts/`)

| Script                         | Description                                               |
| ------------------------------ | --------------------------------------------------------- |
| `milk-check`                   | Verify installation (libraries, paths, SHM)               |
| `milk-check-standalone-deps`   | Check standalone build dependencies                       |
| `milk-argparse`                | Argument parsing helper for milk scripts                  |
| `milk-script-std-config`       | Standard configuration for milk scripts                   |
| `milk-scriptexample`           | Example/template for writing new scripts                  |
| `milk-script-advanced`         | Advanced script template combining bash & native milk-cli |
| `milk-setup-worktrees`         | Set up parallel development Git worktrees                 |
| `milk-exec`                    | Execute a milk CLI command non-interactively              |
| `milk-cli-all`                 | Launch milk-cli on all accessible instances               |
| `milk-commands`                | List all available milk CLI commands                      |
| `milk-completion.sh`           | Bash tab-completion for milk commands                     |
| `milk-debug`                   | Launch milk with GDB debugging                            |
| `milk-demopipeline`            | Start a multi-stage FPS pipeline for demonstration        |
| `milk-fpsinit`                 | Initialize FPS instances from configuration               |
| `milk-fpslist-addentry`        | Add an entry to `fpslist.txt`                             |
| `milk-fpsmkcmd`                | Generate FPS command scripts                              |
| `milk-fps-set-completion.bash` | Bash completion for `milk-fps-set`                        |
| `milk-latency-audit`           | Profile latency, cache misses, and IRQ affinities         |

## 2. Stream Utilities (`scripts/`)

| Script               | Description                                |
| -------------------- | ------------------------------------------ |
| `milk-FITS2shm`      | Load a FITS file into shared memory        |
| `milk-cubeslice2shm` | Extract a slice from a FITS cube into SHM  |
| `milk-stream-scan`   | Scan and list active shared memory streams |
| `milk-streamlink`    | Create symbolic links to stream SHM files  |
| `milk-shmim-rm`      | Remove a shared memory image               |
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

| Script               | Description                                |
| -------------------- | ------------------------------------------ |
| `milk-cr2tofits`     | Convert Canon RAW (CR2) files to FITS      |
| `milk-fitsheader`    | Display FITS file header                   |
| `milk-images-merge`  | Merge multiple FITS images                 |
| `milk-makecsetandrt` | Create cpuset and set real-time scheduling |

## 5. milk-cli Script Payloads (`share/milk/scripts/`)

These are `.milk` files evaluated by the milk-cli interpreter. They are
installed as **data files** (no execute bit) under `share/milk/scripts/` and
must be invoked via the CLI orchestrator:

```bash
milk-cli -s $(milk --install-prefix)/share/milk/scripts/makecircleofdisks.milk \
         [ndisks] [disk_radius] [circle_radius] [image_size] [output_name]
```

| Script                   | Description                                  |
| ------------------------ | -------------------------------------------- |
| `makecircleofdisks.milk` | Generate a procedural ring of circular disks |

## 6. COREMOD Scripts

### 6.1. COREMOD_memory (`src/coremods/COREMOD_memory/scripts/`)

| Script                   | Description                              |
| ------------------------ | ---------------------------------------- |
| `milk-semloopspeed`      | Benchmark semaphore loop speed           |
| `milk-shmimave`          | Compute running average of a SHM stream  |
| `milk-shmimcopy-semtrig` | Copy stream on semaphore trigger         |
| `milk-shmimlog`          | Log shared memory stream to disk         |
| `milk-shmimpoke`         | Write test pattern to a SHM image        |
| `milk-shmimpoke-semtrig` | Poke SHM image on semaphore trigger      |
| `milk-shmimpurge`        | Remove stale SHM files                   |
| `milk-streamFITSlog`     | Log stream to FITS files with timestamps |
| `milk-nettransmit`       | Transmit SHM stream over network         |
| `milk-rmshmim`           | Remove a shared memory image             |

### 6.2. COREMOD_arith (`src/coremods/COREMOD_arith/scripts/`)

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

This creates the shared memory stream `mystream` from the
FITS file `myimage.fits`. Other processes can then connect
to `mystream` via `ImageStreamIO`.

### Benchmarking semaphore speed

```bash
$ milk-semloopspeed
```

Measures the round-trip semaphore post/wait latency.
Typical values on modern hardware exceed 100 kHz.

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

Extracts slice 0 from the FITS cube and writes it to the
shared memory stream `myslice`.

---

← [Documentation Index](index.md)
