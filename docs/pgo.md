# Profile-Guided Optimization (PGO)

PGO trains GCC with real runtime profiles to
optimize branch prediction, function layout, and
inlining. Typical speedups: **10–30%** on
branch-heavy real-time loops.

## Quick Start

```bash
$ cd _build

# Step 1 — Instrument
$ cmake .. -DUSE_PGO=GENERATE
$ make -j$(nproc) && sudo make install

# Step 2 — Run representative workloads
$ milk-fpsexec-streamcopy -n scopy01
$ # ... exercise your typical AO loop patterns

# Step 3 — Rebuild with profiles
$ cmake .. -DUSE_PGO=USE
$ make -j$(nproc) && sudo make install
```

## How It Works

| Step | CMake Flag | GCC Flags | Effect |
|------|-----------|-----------|--------|
| 1 | `-DUSE_PGO=GENERATE` | `-fprofile-generate` | Emits `.gcda` profile data at runtime |
| 2 | *(run workload)* | — | Collects branch/call counts |
| 3 | `-DUSE_PGO=USE` | `-fprofile-use -fprofile-correction` | Optimizes using collected data |

## Per-Executable Optimization

PGO profiles are inherently **per-executable**.
Each standalone `milk-fpsexec-*` binary produces
its own `.gcda` profile files (stored next to the
corresponding `.o` files in the build tree).

This means:

- Running `milk-fpsexec-streamcopy` profiles
  only `stream_copy.c` code paths.
- Running `milk-fpsexec-linalg-SGEMM` profiles
  only `SGEMM.c` code paths.
- Each executable gets its own optimized branch
  layout and inlining decisions.

### Workflow for specific executables

```bash
$ cd _build

# Step 1 — Build everything with instrumentation
$ cmake .. -DUSE_PGO=GENERATE
$ make -j$(nproc) && sudo make install

# Step 2 — Run ONLY the executables you want to
#          optimize, with realistic workloads
$ milk-fpsexec-streamcopy -n scopy01
$ milk-fpsexec-linalg-SGEMM -n sgemm01
$ # let them process enough frames to build
$ # a representative profile, then stop

# Step 3 — Rebuild with profiles
$ cmake .. -DUSE_PGO=USE
$ make -j$(nproc) && sudo make install
```

Only the executables that were actually exercised
in Step 2 receive meaningful PGO optimization.
Others are compiled normally — GCC silently
ignores missing profiles when
`-fprofile-correction` is set.

### What gets profiled

| Component | Profile source |
|-----------|---------------|
| Standalone `.c` | Profiled independently per executable |
| Shared libraries (`libImageStreamIO`, `libmilkdata`, etc.) | Aggregated across all runs — optimized for the common case |

> [!TIP]
> For the best results, run each fpsexec with a
> workload that closely matches production use:
> same stream sizes, same number of modes, same
> loop rate. The more representative the
> training run, the better the optimization.

## Notes

- Profile data (`.gcda` files) is written to the
  build directory alongside each `.o` file.
- `-fprofile-correction` handles minor mismatches
  from multi-threaded execution and missing
  profiles.
- Re-run the full cycle whenever you make
  significant code changes.
- To disable PGO, omit the `-DUSE_PGO` flag
  (or set it to empty).

---
← [Documentation Index](index.md)
