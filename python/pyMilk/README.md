# pyMilk

__Welcome to pyMilk.__

pyMilk means to provide a user-side interaction channel between your python scripts and the interprocess communication (IPC) shared memory structures used in [MILK](https://github.com/milk-org/milk) and it's AO-centric superset [CACAO](https://git@github.com/cacao-org/cacao):
- __SHM__ Shared memory image, a databuffer with IPC capabilities, as defined in ImageStreamIO (ISIO)
- __ProcessInfo__ a telemetry structure for monitoring RTC processes, also hosted in IPC-accessible shared memory.
- __FPS__ Function Parameter Structure, an IPC key-value structure that is usually used to channel information between a real-time runtime program, and a user-facing control environment or automated orchestrator.

For interfacing with the underlying C API of MILK we use a [pybind](https://github.com/pybind/pybind11) linking deployed with the C library [ImageStreamIO](https://github.com/milk-org/ImageStreamIO) (ISIO).

- Credit to Arnaud Sevin for the pybind11 modules and many other contributions.
- Credit to Frantz Martinache for defining the API for the SHM python binding.

## Installing

Installing the software is straightforward and done entirely with `pip`, with some caveats to avoid collisions with the underlying MILK installation.

### There is no existing MILK installation:

Clone the repo with ImageStreamIO as submodule:
```
git clone https://github.com/milk-org/pyMilk --recurse-submodules
```

Install pyMilk **in "developer mode"** (`-e` flag; CMake is misbehaved with `pip install .`). Developer mode does not copy source files in your python package manager.
The source code actually used is the one in the present pyMilk folder. If you intend to change python sources, this approach is more convenient.

```
cd pyMilk
pip install -e .
```

### You have otherwise installed MILK/CACAO

In case you need MILK/CACAO as well, beyond merely interacting with SHMs and FPSs with pyMilk in standalone mode, you may encounter some weirdness with name and library resolution.
Please let me know in the Github Issues.

At this stage, having a full installation of MILK/CACAO is the only way to use the FPS features of pyMilk. You will need to compile the CacaoProcessTools python module that is packaged with MILK. See installation instructions there.

## Usage

### SHM — Shared Memory Streams

The `SHM` class provides read/write access to MILK/CACAO shared memory image streams.

```python
from pyMilk.interfacing.shm import SHM
import numpy as np

# Connect to an existing stream
shm = SHM('mystream')
data = shm.get_data()                        # read immediately
data = shm.get_data(check=True, timeout=5.0) # wait for a new frame

# Write to it
shm.set_data(np.random.rand(*shm.shape).astype(shm.nptype))

# Create a new stream (512x512, float32, on CPU)
shm = SHM('newstream', np.zeros((512, 512), dtype=np.float32))

# Or specify shape and type without providing data
shm = SHM('newstream', ((512, 512), np.float32))

# Grab 1000 successive frames synchronously
cube = shm.multi_recv_data(1000, output_as_cube=True)

# Keyword access
shm.set_keywords({'EXPTIME': 0.001, 'FRATE': 1000.0})
kw = shm.get_keywords()

# Context manager
with SHM('mystream') as s:
    data = s.get_data()

# Export to FITS
shm.save_as_fits('output.fits')

shm.close()
```

A CLI helper is also available to create streams from the shell:

```bash
# Create a 256x256 float32 stream with 50 keywords
creashmim mystream 256 256 --type=f32 --kw=50
```

### GPU Capability



### FPS — Function Parameter Structures

The `FPS` class interfaces with CACAO parameter structures used to configure and control real-time processes.

```python
from pyMilk.interfacing.fps import FPS

# Connect to an existing FPS
fps = FPS('myprocess')

# Read / write parameters (dict-style)
fps['gain'] = 0.3
print(fps['gain'])

# Process lifecycle
fps.conf_start()
fps.run_start()
print(fps.run_isrunning())  # True
fps.run_stop()
fps.conf_stop()
```

### Viewers

pyMilk ships three real-time viewers that can be launched from the command line:

- **`shmImshow`** — 2D image display (PyQtGraph). Includes histogram, dark subtraction, log scale.
- **`shmPlot`** — 1D multi-curve plotter (PyQtGraph). Plots one curve per row of a 2D stream.
- **`shmTermshow`** — ASCII art display in the terminal. No GUI dependency required.

```bash
# 2D viewer at 20 fps
shmImshow.py camstream --fr=20

# 1D plotter
shmPlot.py wfs_slopes --fr=10

# Terminal viewer with zoom factor 2
shmTermshow.py camstream --fr=10 -z=2
```

All three viewers support interactive keyboard shortcuts (press `H` for help) and live dark-frame subtraction when a `<name>_dark` stream exists.

<!-- TODO: add screenshot -->
![Viewer screenshot]()

## Technical notes

### Python package manager

We heavily recommend __NOT__ using anaconda or miniconda, which lock you in their package repositories, but rather we prefer the[miniforge](https://github.com/conda-forge/miniforge) python package manager distribution.

- It provides `mamba`, a faster and smarter rework of `conda`
- It uses conda-forge upstream, which are better furnished and updated than the anaconda counterparts.
- We have on average experienced less package dependency hell, in particular regarding the graphical components of pyMilk, which depend upon Qt.

### NotImplemented.s

pyMilk's SHM API does not support all the features provided by `xaosim`, and therefore a lot of functions are *here* but are *empty*.
This includes a number of things related to legacy keyword writing.

### Contributions

Contributors are welcome to add python contributions, or if they feel up to it, to dig into underlying C libraries and bindings.
Useful python contributions to the interface will be ported deeper into the C for speed and reliability.

Please:
- Notify maintainers of any significant workplan through Github issues.
- Work through a pull-request workflow only, and rebase you branch into a commit line with sufficient separation of intention.
- Install and use pre-commit locally for all contributions.


### Data ordering

Data ordering is **column-major** by default upon read-write.
A **symcode** parameter enables to switch through all transposed-flipped representations of rectangular images.
The default is 4 as to provide retro-compatibility. We lack, at this point, the adequate compatibility with the fits files.

We hope to be properly dealing with data-ordering at the scale of MILK at some point in the future.
