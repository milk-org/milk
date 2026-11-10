# Example 04: an AO loop with milk only

An adaptive optics loop built from milk compute units, without cacao and without C code. Each
cacao process becomes a calculator program of one to four lines, run by
[`milk-fpsexec-milk-calc`](../../docs/guide/calculator.md#compute-units-without-c) and compiled
to native code when its loop starts.

The wavefront sensor is a numerical shortcut: each pixel of its frame is the wavefront phase at that
point. There is no optical simulation.

## The loop

| Unit        | Program                                    | Trigger  | Role (cacao equivalent)                        |
| ----------- | ------------------------------------------ | -------- | ---------------------------------------------- |
| `wfssim`    | [`wfssim.calc`](units/wfssim.calc)         | 2 kHz    | test hardware: phase = turbulence + DM, noise  |
| `dmtrig`    | `milk-fpsexec-mem-trigmerge`               | channels | fires when any DM channel is written           |
| `dmcomb`    | [`dmcomb.calc`](units/dmcomb.calc)         | `dmtrig` | `dm` = sum of `dmch0..3`, clipped (DMcomb)     |
| `acquwfs`   | [`acquwfs.calc`](units/acquwfs.calc)       | `wfsim`  | dark and reference subtraction (acquWFS)       |
| `wfs2modes` | [`wfs2modes.calc`](units/wfs2modes.calc)   | `wfss`   | `mval = CM.T @ wfss` (MVM to modes)            |
| `mfilt`     | [`mfilt.calc`](units/mfilt.calc)           | `mval`   | leaky integrator, modal gains, clip (mfilt)    |
| `modes2dm`  | [`modes2dm.calc`](units/modes2dm.calc)     | `mint`   | `dmch1 = M2A @ mint` (MVM to DM)               |

```text
wfssim -> wfsim -> acquwfs -> wfss -> wfs2modes -> mval -> mfilt -> mint -> modes2dm -> dmch1
dmch0..3 -> dmtrig -> dmcomb -> dm -> (read by wfssim at its next frame)
```

A unit has a single trigger stream, so the DM combiner needs `trigmerge` to wake on any of its four
channels ([Merging triggers](../../docs/guide/procinfo.md#merging-triggers)).

Sizes are set at the top of [`aoloop.sh`](aoloop.sh): a 12x12 DM, a 24x24 WFS (2 pixels per
actuator, circular pupil), 100 controlled modes, 2 kHz camera.

## Calibration

The response of a phase WFS is known: the frame is `IF @ dm`, where `IF` holds one Gaussian
influence function per actuator. `start` therefore computes the matrices instead of measuring them.
It takes the SVD of `IF` and keeps 100 modes. `M2A` holds their DM shapes (the right singular
vectors) and `CM` holds `U / s`, so `CM.T @ wfss` gives the mode coefficients.

## Running it

Needs a milk install on `PATH` with the calculator (`milk-fpsexec-milk-calc`, `milk.bash`).

```bash
./aoloop.sh start      # streams, model, matrices, units; loop open
```

```bash
./aoloop.sh close      # close the loop
```

```bash
./aoloop.sh measure    # residual open and closed, latency, time per unit (leaves the loop closed)
```

```bash
./aoloop.sh stop       # stop the units, remove the streams and the work directory
```

`start` prints the `MILK_SHM_DIR` that holds the streams, and the other commands use the same one.
By default it is a directory of the example's own, `${TMPDIR:-/tmp}/milk-aoloop-<uid>/shm`, so
nothing goes to `/milk/shm`. To watch the loop from another shell, export the printed value there:

```bash
export MILK_SHM_DIR=${TMPDIR:-/tmp}/milk-aoloop-$(id -u)/shm
```

```bash
milk-fpsCTRL
```

With `--default-shm`, `start` uses the usual directory instead (`$MILK_SHM_DIR`, else
`/milk/shm`), where `milk-CTRL` and the other tools see the streams as they are:

```bash
./aoloop.sh --default-shm start
```

The example's stream names (`dm`, `CM`, `IF`, ...) then replace any stream of the same name in that
directory. `stop` removes them by name, and runs `@session prune` there for the session files that
the script's shells left.

```bash
source milk.bash; @set f.mfilt.gain=0.3 f.wfssim.noise=0.05
```

Live parameters: `f.mfilt.gain`, `.mult`, `.limit` and `.loopON`; `f.wfssim.ar` (turbulence
correlation per frame, 1 freezes it), `.tgain` and `.noise`; `f.dmcomb.maxstroke`. Modal gains are
the stream `mgain`.

## Performance

`./aoloop.sh measure` on an i9-12950HX laptop (powersave governor, no real-time priority):

```text
residual RMS over 4000 frames (phase units): loop open 0.297716652210843, closed 0.0404447476357945
camera frame to DM write: median 91 us, 90% below 200 us
unit       rate (Hz)  busy (us)   max (us)  missed
wfssim          2000       38.7       93.3       0
dmtrig          1983      502.6      641.7       0
dmcomb          1983        4.1       16.7      15
acquwfs         2007        6.2       36.6      63
wfs2modes       2010       14.6       45.7     211
mfilt           2002        5.4       24.2     183
modes2dm        1988        4.6       18.3     172
```

- **busy** is the median time per frame from the unit's timing ring (`p.<unit>.exectime`). It is
  about 4 µs for a unit with almost nothing to compute, the per-frame floor of a calc unit.
  `dmtrig` waits for its inputs inside its loop, so its busy time is that wait.
- **latency** runs from the camera frame's write time to the `dm` write it caused. Two probe units
  take each write time from `_ts`, the write time of their trigger frame.
- **missed** counts frames posted before the unit was back to waiting during the measurement. The
  unit then processes the newest frame at once. Without real-time scheduling, about 1% of the
  wake-ups come late; see [Real-Time Scheduling](../../docs/operations/performance.md#3-real-time-scheduling).
- `@calc --explain` shows how a program compiles once its parameters exist as variables, e.g.
  `@calc "maxstroke = 1"; @calc --explain "$(cat units/dmcomb.calc)"`. The element-wise units
  (`acquwfs`, `mfilt`, `dmcomb`) are fused native passes. The two MVMs are BLAS calls.
