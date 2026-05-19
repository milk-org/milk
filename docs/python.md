---
tags:
  - python
  - pymilk
  - api
---

# Python API

Accessing `milk` shared memory streams from Python using
the `ImageStreamIOWrap` bindings and `pyMilk`.

See also: [Streams](streams.md) ·
[FPS](fps.md) ·
[Programmer's Guide](programmers_guide.md)

---

## 1. Installation

### 1.1. `ImageStreamIOWrap` (low-level bindings)

The C-extension bindings are built automatically when
`milk` is compiled with Python support:

```bash
$ cd _build
$ cmake .. -DUSE_PYTHON=ON
$ make -j$(nproc)
$ sudo make install
```

Verify:

```python
>>> import ImageStreamIOWrap
>>> print(ImageStreamIOWrap.__file__)
```

### 1.2. `pyMilk` (high-level API)

`pyMilk` provides a Pythonic wrapper around
`ImageStreamIOWrap` with numpy integration:

```bash
$ pip install pyMilk
```

Or install from source:

```bash
$ pip install git+https://github.com/milk-org/pyMilk.git
```

---

## 2. Connecting to Streams

### 2.1. Reading a Stream

```python
from pyMilk.interfacing.isio_shmlib import SHM

# Connect to existing shared memory stream
shm = SHM("mystream")

# Read the current frame as a numpy array
frame = shm.get_data()
print(frame.shape, frame.dtype)
```

### 2.2. Writing to a Stream

```python
import numpy as np
from pyMilk.interfacing.isio_shmlib import SHM

# Create a new 128x128 float32 stream
shm = SHM(
    "pystream",
    data=np.zeros((128, 128), dtype=np.float32),
)

# Write new data
new_frame = np.random.randn(128, 128).astype(np.float32)
shm.set_data(new_frame)
```

### 2.3. Waiting for New Frames

```python
# Block until a new frame arrives (semaphore-based)
frame = shm.get_data(check=True)
```

This uses the same semaphore mechanism as the C API,
providing microsecond-level wake-up latency.

---

## 3. Stream Metadata

```python
# Get stream dimensions
print(shm.mtdata["size"])

# Get frame counter
print(shm.mtdata["cnt0"])

# Get keywords
keywords = shm.get_keywords()
for kw in keywords:
    print(kw["name"], kw["value"])
```

---

## 4. Integration with NumPy

All data returned by `pyMilk` is a standard numpy
array. This means you can use the full numpy/scipy
ecosystem directly:

```python
import numpy as np
from pyMilk.interfacing.isio_shmlib import SHM

shm = SHM("wfs_image")
frame = shm.get_data()

# Compute statistics
print(f"Mean: {np.mean(frame):.2f}")
print(f"RMS:  {np.std(frame):.4f}")
print(f"Max:  {np.max(frame):.2f}")

# Apply processing
from scipy.ndimage import median_filter
filtered = median_filter(frame, size=3)
```

---

## 5. Real-Time Loop Example

A minimal Python loop that reads frames and computes
running statistics:

```python
import numpy as np
from pyMilk.interfacing.isio_shmlib import SHM

shm_in = SHM("wfs_raw")
cnt = 0
running_mean = None

while True:
    frame = shm_in.get_data(check=True)
    if running_mean is None:
        running_mean = frame.astype(np.float64)
    else:
        running_mean = 0.99 * running_mean + \
            0.01 * frame
    cnt += 1
    if cnt % 100 == 0:
        print(f"Frame {cnt}, "
              f"mean={np.mean(running_mean):.2f}")
```

!!! warning
    Python's GIL limits true parallel performance.
    For latency-critical loops (>1 kHz), use C modules.
    Python is best suited for monitoring, scripting,
    and offline analysis.

---

## 6. CacaoProcessTools

The `python_module/` directory contains
`CacaoProcessTools`, a set of Python utilities for
controlling `cacao` processes:

```bash
$ cd python_module
$ pip install .
```

See
[`python_module/README.md`](../python_module/README.md)
for usage details.

---
← [Documentation Index](index.md)
