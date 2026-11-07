---
tags:
  - python
  - pymilk
  - api
render_macros: true
---

# pyMilk

[pyMilk](https://github.com/milk-org/pyMilk) is the Python side of milk's shared memory: the
`SHM` class for streams and the `FPS` class for function parameter structures, with numpy
arrays in and out. It is what scripts, viewers and offline analysis use to talk to a running
system.

See also: [Streams](streams.md) · [FPS](fps.md) · [Programmer's Guide](../arch/programmers_guide.md)

To deploy and operate pipelines of milk compute units from Python (the `milk` package,
`milk-pipeinit`, `milk-pipes`), see [Pipelines](pipelines.md).

---

## 1. Installation

```bash
$ git clone https://github.com/milk-org/pyMilk --recurse-submodules
$ cd pyMilk
$ pip install -e .
```

The `-e` (developer mode) install is the supported one: the bindings are compiled in place
from the ImageStreamIO submodule. Streams need nothing else. The `FPS` class needs a full
milk installation, since it binds milk's own FPS library.

`MILK_SHM_DIR` must be set: a stream named `mystream` is the file `$MILK_SHM_DIR/mystream.im.shm`.

---

## 2. Streams: the `SHM` class

### 2.1. Reading

```python
from pyMilk.interfacing.shm import SHM

shm = SHM("mystream")          # FileNotFoundError if the stream does not exist
frame = shm.get_data()         # the current frame, as a numpy array (a copy)
print(shm.shape, shm.nptype)   # the user-side shape and dtype
```

`get_data()` returns a copy, so the array stays valid whatever the writer does next.
`get_data(copy=False)` returns a view on the shared memory instead: faster, but it changes
under you.

### 2.2. Waiting for a new frame

```python
frame = shm.get_data(check=True)                 # block until the next write (5 s timeout)
frame = shm.get_data(check=True, timeout=0.1)    # warns and returns the stale frame on timeout
frame = shm.get_data(check=True, timeout=0.1, return_none_on_timeout=True)   # None on timeout
frame = shm.get_data(check=True, timeout=None)   # wait forever
```

`check=True` waits on one of the stream's semaphores, the same mechanism as the C API, with
microsecond wake-up latency. The semaphore is flushed before waiting, so the frame returned was
written *after* the call started: this is the way to wait for a fresh frame, not just a
recent one.

### 2.3. Grabbing many frames: `multi_recv_data()`

```python
frames = shm.multi_recv_data(100)                        # a list of 100 arrays, in order
cube   = shm.multi_recv_data(1000, output_as_cube=True)  # one (1000, *shm.shape) array
```

`multi_recv_data(n)` waits for `n` successive writes and returns every frame, with no frame
read twice and none skipped as long as the reader keeps up (each frame is waited for with a
`timeout`, 5 s by default). `output_as_cube=True` fills one preallocated array, which avoids
a copy per frame and gives you something to average or save directly:

```python
dark = shm.multi_recv_data(500, output_as_cube=True).mean(axis=0)
```

`monitorCount=True` prints a summary of the stream counter deltas around each read, to check
that no frame was missed.

### 2.4. Writing and creating

```python
import numpy as np
from pyMilk.interfacing.shm import SHM

# Create a 128x128 float32 stream (overwrites a stream of that name)
shm = SHM("pystream", np.zeros((128, 128), dtype=np.float32))
shm = SHM("pystream", ((128, 128), np.float32))     # same, from a shape and a dtype

shm.set_data(np.random.randn(128, 128).astype(np.float32))   # write and post the semaphores
shm.set_data(some_float64_array, check_dt=True)                # cast to the stream's dtype
```

`set_data` writes the array, updates the counters and timestamps and posts the semaphores,
which wakes every reader blocked in `get_data(check=True)`. `repost()` posts again without
new data.

To reuse an existing stream when its shape and dtype already match, rather than recreate it:

```python
from pyMilk.interfacing.shm_functions import creashmim

shm = creashmim("pystream", (128, 128), np.float32)
```

### 2.5. Keywords and metadata

```python
shm.set_keywords({"EXPTIME": 0.001, "FRATE": (1000.0, "frame rate, Hz")})   # add or update
shm["GAIN"] = 2                              # same, one keyword, no comment
print(shm["EXPTIME"], shm.get_keywords())    # value; {name: value}
print(shm.get_keywords(comments=True))       # {name: (value, comment)}

print(shm.get_counter())                     # cnt0: number of writes so far
print(shm.md.cnt0, shm.md.size)              # the ImageStreamIO metadata struct
shm.print_meta_data()
```

### 2.6. Closing and context manager

A `SHM` object holds a semaphore slot on the stream. Close it when you are done, or use it as a
context manager, in particular in code that opens streams repeatedly:

```python
with SHM("mystream") as s:
    frame = s.get_data(check=True)
```

`destroy()` removes the stream itself from shared memory (not if the file was recreated by
someone else since this object opened it).

### 2.7. Export to FITS

```python
shm.save_as_fits("frame.fits")              # the current frame
```

---

## 3. Data ordering

Data is column-major in shared memory, and a `symcode` constructor argument (0–7, default 4)
selects the transposed / flipped representation you get on the Python side; the inverse is
applied on write. `triDim` chooses how a 3D stream is ordered, and `autoSqueeze` removes
singleton dimensions. Use the same values on both sides of a Python-to-Python exchange, and
`symcode=0` when a stream must match a FITS file axis for axis (which is what the pipeline
`[[dataloads]]` do).

---

## 4. Real-time loop example

```python
import numpy as np
from pyMilk.interfacing.shm import SHM

shm_in = SHM("wfs_raw")
shm_out = SHM("wfs_mean", np.zeros(shm_in.shape, dtype=np.float32))   # same shape, float32
running_mean = shm_in.get_data().astype(np.float32)

while True:
    frame = shm_in.get_data(check=True, timeout=None)   # one iteration per new frame
    running_mean += 0.01 * (frame - running_mean)
    shm_out.set_data(running_mean)                      # wakes the readers of wfs_mean
```

<!-- prettier-ignore -->
!!! warning
    Python's GIL limits true parallel performance. For latency-critical loops (>1 kHz), use C modules.
    Python is best suited for monitoring, scripting, and offline analysis.

---

## 5. Function parameter structures: the `FPS` class

```python
from pyMilk.interfacing.fps import FPS

fps = FPS("basic_delay0")            # FPSDoesntExistError if $MILK_SHM_DIR/basic_delay0.fps.shm is absent
fps["delaysec"] = 0.02               # set a parameter (ValueError on an unknown key)
print(fps["delaysec"], "delaysec" in fps)
print(fps.conf_isrunning(), fps.run_isrunning())

fps.conf_start(); fps.run_start()    # signal the conf / run processes
fps.run_stop();   fps.conf_stop()
```

`fps.procinfo` (present when the FPS has `procinfo.*` keys) exposes the process-info
parameters as attributes. `FPSManager("aol0_*")` finds and caches every FPS matching a glob.
In a pipeline, FPS names are `<name>_<cu>` and the start/stop sequencing is done by
[tasks](pipelines_api.md#plans-and-tasks); reach for `FPS` directly to read or tune parameters
of a running compute unit.

---

## 6. Viewers

pyMilk ships three real-time viewers:

```bash
$ shmImshow.py camstream --fr=20       # 2D image (PyQtGraph): histogram, dark subtraction, log scale
$ shmPlot.py wfs_slopes --fr=10        # 1D multi-curve plot, one curve per row
$ shmTermshow.py camstream --fr=10 -z=2    # ASCII art in the terminal, no GUI dependency
```

Press `H` in a viewer for the keyboard shortcuts. A `<name>_dark` stream, when it exists, is
subtracted live.

---

← [Documentation Index](../index.md)
