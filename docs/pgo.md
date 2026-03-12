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

## Per-Executable Profile Isolation

Each standalone `milk-fpsexec-*` and
`cacao-fpsexec-*` binary gets its own profile
subdirectory under `_build/pgo/`:

```
_build/pgo/
├── shared/                          ← shared libs
├── milk-fpsexec-streamcopy/         ← streamcopy
├── milk-fpsexec-linalg-SGEMM/       ← SGEMM
├── cacao-fpsexec-cacaoloop-WFS/     ← WFS
└── ...
```

This isolation is automatic — the
`milk_pgo_target()` CMake helper (called by
`add_milk_standalone()` / `add_cacao_standalone()`)
sets per-target `-fprofile-dir`.

### Optimizing specific executables

```bash
$ cd _build

# Step 1 — Build everything with instrumentation
$ cmake .. -DUSE_PGO=GENERATE
$ make -j$(nproc) && sudo make install

# Step 2 — Run ONLY the executables you want to
#          optimize, with realistic workloads
$ milk-fpsexec-streamcopy -n scopy01
$ # let it process several thousand frames, then ^C
$ cacao-fpsexec-cacaoloop-WFS -n wfs01
$ # exercise another workload

# Step 3 — Rebuild with profiles
$ cmake .. -DUSE_PGO=USE
$ make -j$(nproc) && sudo make install
```

Only the executables exercised in Step 2 receive
PGO optimization. Others compile normally — GCC
silently ignores missing profiles when
`-fprofile-correction` is set.

### What gets profiled

| Component | Profile directory | Scope |
|-----------|------------------|-------|
| Standalone `.c` | `pgo/<exe-name>/` | Independent per executable |
| Shared libraries | `pgo/shared/` | Aggregated across all runs |

> [!TIP]
> For the best results, run each fpsexec with a
> workload that closely matches production use:
> same stream sizes, same number of modes, same
> loop rate. The more representative the
> training run, the better the optimization.

## Notes

- Profile data (`.gcda` files) is written to
  `PGO_DIR` (default: `_build/pgo/`).
  Override with `-DPGO_DIR=/path/to/profiles`.
- `-fprofile-correction` handles minor mismatches
  from multi-threaded execution and missing
  profiles.
- Re-run the full cycle whenever you make
  significant code changes.
- To disable PGO, omit the `-DUSE_PGO` flag
  (or set it to empty).

---
← [Documentation Index](index.md)
