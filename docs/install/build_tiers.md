# milk Build Tiers

The milk build system supports three build tiers, controlled by
the CMake options `USE_COREMODS`, `USE_CFITSIO`, and `USE_CLI`.
Lower tiers have fewer external dependencies and produce a
smaller footprint.

---

## Tier overview

| Tier | Components | External deps |
|------|------------|---------------|
| **Engine** | ImageStreamIO, libprocessinfo, libfps, libmilkdata | POSIX only (pthread, rt, m, dl) |
| **Core** | Engine + COREMOD_arith, COREMOD_memory, COREMOD_tools | Same as Engine |
| **Core + FITS** | Core + COREMOD_iofits | cfitsio |
| **Full** | Core + FITS + CLI + plugins | cfitsio, readline, ncurses, GSL (all optional) |

---

## CMake options

```
option(USE_COREMODS  "Build core modules (COREMOD_*)"    ON)
option(USE_CFITSIO   "Build FITS I/O (needs cfitsio)"    ON)
option(USE_CLI       "Build interactive CLI (milk-cli)"   ON)
option(USE_NCURSES   "Enable ncurses and TUI support"     ON)
option(USE_READLINE  "Enable readline support"            ON)
option(USE_GSL       "Enable GSL for plugins"             ON)
```

### Dependency chain

* `USE_CLI=ON` requires `USE_COREMODS=ON` (auto-enabled).
* Plugins are only built when `USE_COREMODS=ON`.
* `USE_CFITSIO=OFF` disables `COREMOD_iofits` entirely and
  compiles the remaining coremods without cfitsio linkage.
  Functions that would normally write FITS files print a
  warning instead.

---

## Build examples

### Engine-only (POSIX deps only)

```bash
mkdir build && cd build
cmake .. -DUSE_COREMODS=OFF -DUSE_CLI=OFF
make -j$(nproc)
```

Produces: `libImageStreamIO`, `libmilkprocessinfo`, `libmilkfps`,
`libmilkdata`, and engine-level standalone tools.

### Core (engine + coremods, no cfitsio)

```bash
cmake .. -DUSE_CLI=OFF -DUSE_CFITSIO=OFF
make -j$(nproc)
```

Produces everything above plus `COREMOD_arith`,
`COREMOD_memory`, and `COREMOD_tools` — all without
requiring cfitsio.

### Core + FITS

```bash
cmake .. -DUSE_CLI=OFF
make -j$(nproc)
```

Same as Core, and also builds `COREMOD_iofits` (requires
cfitsio).

### Full build (default)

```bash
cmake ..
make -j$(nproc)
```

Builds everything: engine, all coremods, CLI (`milk-cli`),
all plugins, and example module.

### Full build with CUDA

```bash
cmake .. -DUSE_CUDA=ON
make -j$(nproc)
```

---

## What is disabled when cfitsio is off

When `USE_CFITSIO=OFF`:

* `COREMOD_iofits` is not built; `loadfits` / `saveFITS`
  commands are unavailable.
* `COREMOD_memory` builds normally, but:
  * `logshmim` (streamFITSlog) prints a warning instead of
    writing FITS cubes.
  * `saveall` (imsaveallsnap, imsaveallseq) prints a warning
    instead of writing FITS files.
  * `stream_pixmapdecode` prints a warning instead of
    saving the pixel-slice index.
* `COREMOD_arith` builds normally (no runtime impact; it
  never used cfitsio directly).

Source files use `#ifdef USE_CFITSIO` guards around all FITS
I/O code paths.
