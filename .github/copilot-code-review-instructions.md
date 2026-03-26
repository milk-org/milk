# Copilot Code Review Instructions for milk

milk is a real-time image-processing framework for
Adaptive Optics. Code runs in latency-critical shared
memory loops. Review with performance and correctness
as top priorities.

## C Code Style

- Lines must not exceed **80 characters**.
- Use **Linux kernel C coding style**.
- Function prototypes: **one argument per line**.
- Document functions with **Kernel-Doc** (`/** ... */`)
  in `.c` files; brief descriptions in `.h` files.
- Minimize variable scope using **code blocks** `{ }`.
- Every `.c` file must **explicitly include** every
  header it uses — never rely on transitive includes
  (e.g., do not assume `CLIcore.h` provides `math.h`).

## Architecture

- Flag any **new cross-module `#include`** that may
  violate the layered dependency graph. Engine libraries
  must never depend on CLI or plugin code.
- Standalone executables must **never link `CLIcore`**.
  They link `_compute` variants of libraries instead.
- Files compiled in dual mode must use the conditional
  include pattern:
  ```c
  #ifdef MILK_NO_CLI
  #include "CLIcore_standalone.h"
  #else
  #include "CLIcore.h"
  #endif
  ```

## Performance (Critical)

- Flag **`printf`/`fprintf`/`fflush`** in compute
  functions or hot loops. These must be guarded with
  `if (VERBOSE > 0)`.
- Flag **`malloc`/`free` inside per-frame loops**.
  Allocations belong in initialization, not compute.
- Flag **`sqrt()`, `pow()`, `fabs()`, `floor()`,
  `ceil()`** on float data — use `sqrtf()`, `powf()`,
  `fabsf()`, `floorf()`, `ceilf()` instead.
- Flag bare **double literals** like `0.5` in float
  arithmetic — should be `0.5f`.
- Flag **standalone `if`** (not `else if`) chains for
  datatype dispatch — wastes ~9 redundant comparisons.
- Flag **hand-written matrix multiply loops** — should
  use `cblas_sgemv()` / `cblas_sgemm()`.
- Flag **`pow(2, n)`** for integer `n` — should be
  `1 << n`.
- Suggest **`restrict`** on array/pixel pointer
  parameters in compute-heavy functions.

## FPS Compute Units (V2 Template)

- New standalone executables must follow the **8-section
  layout** from `src/milk_module_example/examplefunc_fps_cli_poc.c`.
- Every fpsexec must have a valid `.description` in
  `FPS_APP_INFO` for `-h1` one-line help.
- CMake for standalone targets should use
  `add_milk_standalone()` or `add_cacao_standalone()`,
  not the manual 4-line pattern.

## Commit & PR Standards

- Commit messages should use **conventional prefixes**:
  `feat:`, `fix:`, `perf:`, `refactor:`, `docs:`,
  `chore:`, `test:`.
- Subject line: max **72 characters**, imperative mood.
- PRs must target **`framework-dev`** — never `dev` or
  `main`.
- AI-authored PRs must include "Prompt Summary" and
  "AI Authorship" sections in the body.

## Common Mistakes to Flag

1. Lines exceeding 80 characters.
2. Missing Kernel-Doc on new public functions.
3. Implicit header dependencies.
4. `printf` in hot paths without `VERBOSE` guard.
5. Float/double precision mixing in inner loops.
6. New cross-module dependencies without justification.
7. `malloc`/`free` inside compute loops.
8. Standalone executables linking `CLIcore`.
