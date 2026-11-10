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

## One unit for the whole iteration

```bash
./aoloop.sh --fused start
```

With `--fused`, a single unit `aoloop`, triggered by `wfsim`, runs the programs of `acquwfs`,
`wfs2modes`, `mfilt`, `modes2dm` and `dmcomb` one after the other: `aoloop.sh` concatenates them
into `units/aoloop.calc` in the work directory. It writes the same streams, so the other commands
work as before. The `mfilt` and `dmcomb` parameters become `f.aoloop.gain`, `.mult`, `.limit`,
`.loopON` and `.maxstroke`. There is no `dmtrig`: a write to `dmch0`, `dmch2` or `dmch3` reaches
`dm` at the next camera frame, not at once.

## Performance

`./aoloop.sh measure` on an i9-12950HX laptop (powersave governor, no real-time priority), one unit
per step:

```text
residual RMS over 4000 frames (phase units): loop open 0.291094822970909, closed 0.0409870936380916
camera frame to DM write: median 74 us, 90% below 153 us
unit       rate (Hz)  busy (us)   max (us)  missed
wfssim          2000       16.9       64.1       0
dmtrig          1998      499.0      670.2       0
dmcomb          1998        3.1       12.7      20
acquwfs         2000        3.6       21.1      87
wfs2modes       2000        8.8       35.7     123
mfilt           2000        4.8       19.7     118
modes2dm        1998        3.9       18.7     131
```

and with `--fused`, right after:

```text
residual RMS over 4000 frames (phase units): loop open 0.273583985405153, closed 0.0412681103061038
camera frame to DM write: median 19 us, 90% below 51 us
unit       rate (Hz)  busy (us)   max (us)  missed
wfssim          2000       33.7       63.2       0
aoloop          2000       26.0       62.3      57
```

Over three alternating runs of each, the median camera-to-DM latency was 74 to 84 µs with one unit
per step, and 19 to 28 µs fused (90% below 146 to 185 µs, and 51 to 129 µs). The fused unit is busy
26 µs per frame, about the sum of the five steps (17 to 43 µs over the same runs), so fusing saves
no computation. It saves hand-offs: the split loop wakes six more processes or threads per frame
(four units, `trigmerge`'s input thread and loop, `dmcomb`), about 9 µs each.

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
