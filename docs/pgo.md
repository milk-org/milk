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

## Notes

- Profile data (`.gcda` files) is written to the
  build directory alongside each `.o` file.
- `-fprofile-correction` handles minor mismatches
  from multi-threaded execution.
- Re-run the full cycle whenever you make
  significant code changes.
- To disable PGO, omit the `-DUSE_PGO` flag
  (or set it to empty).

---
← [Documentation Index](index.md)
