# Naming Conventions

## 1. Files and Directories

### 1.1 Source Files (`.c` / `.h`)

| Category        | Pattern                  | Examples                   |
| --------------- | ------------------------ | -------------------------- |
| Module init     | `<module_name>.c/h`      | `COREMOD_arith.c`, `fft.c` |
| Compute unit    | `<action>_<object>.c/h`  | `image_crop2D.c`           |
| V2 fpsexec      | `<funcname>.c`           | `MVM_CPU.c`, `findspots.c` |
| Framework       | `<subsys>_<topic>.c/h`   | `fps_cli_sync.c`           |
| Standalone tool | `milk-<subsys>-<verb>.c` | `milk-fps-set.c`           |

**Rules:**

- Use `snake_case` for multi-word file names.
- Name after the primary function or object the file implements.
- Module prefixes (`AOloopControl_DM_`) are acceptable for cacao
  modules to preserve namespace grouping.
- Keep names concise: `image_crop2D.c` over
  `image_crop_two_dimensional.c`.

### 1.2 Directories (Modules)

| Tier           | Pattern                  | Examples                |
| -------------- | ------------------------ | ----------------------- |
| Engine library | `lib<name>`              | `libfps`, `libmilkdata` |
| Core module    | `COREMOD_<domain>`       | `COREMOD_arith`         |
| Plugin module  | `<domain>`               | `fft`, `linalgebra`     |
| cacao module   | `AOloopControl_<subsys>` | `AOloopControl_DM`      |

- New milk modules: lowercase `snake_case`.
- Established `CamelCase` prefixes (`AOloopControl`) are preserved.

### 1.3 Scripts

Use the following patterns for script names.

| Type          | Pattern                | Examples                 |
| ------------- | ---------------------- | ------------------------ |
| OS executable | `milk-<subsys>-<verb>` | `milk-fps-set`           |
| milk fpsexec  | `milk-fpsexec-<name>`  | `milk-fpsexec-MVM`       |
| cacao fpsexec | `cacao-fpsexec-<name>` | `cacao-fpsexec-dmcomb`   |
| CLI script    | `<descriptive>.milk`   | `makecircleofdisks.milk` |

---

## 2. Functions

### 2.1 Scope-Based Naming

| Scope                      | Convention                         |
| -------------------------- | ---------------------------------- |
| **Public API** (in `.h`)   | `<subsystem>_<verb>_<object>()`    |
| **Module-internal static** | `<verb>_<object>()` or descriptive |
| **CLI registration**       | `CLIADDCMD_<module>__<func>()`     |
| **CLI entry point**        | `CLIfunction()`                    |
| **Module init**            | `init_module_CLI()`                |

**Rules:**

- Public functions must carry a subsystem or module prefix to
  avoid symbol collisions. **New subsystems must use lowercase
  `snake_case`** for the entire name (e.g., `fps_save2disk()`).
  Legacy mixed-case prefixes (e.g., `ImageStreamIO_createIm()`)
  are preserved but not used for new APIs.
- Static functions need no prefix. Use descriptive `snake_case`:
  `crop_region()`, `validate_input_stream()`.
- The V2 template reserves these standard names:
  - `fpsexec()` — core computation
  - `compute_function()` — processinfo wrapper
  - `CLIfunction()` — CLI entry
  - `customCONFcheck()` — FPS config validator
  - `customCONFsetup()` — FPS config one-time setup

### 2.2 Verb Conventions

| Intent  | Verbs                        |
| ------- | ---------------------------- |
| Create  | `create`, `make`, `init`     |
| Destroy | `destroy`, `free`, `remove`  |
| Read    | `get`, `read`, `check`, `is` |
| Write   | `set`, `write`, `update`     |
| Connect | `connect`, `open`            |
| Close   | `close`, `disconnect`        |
| Compute | `compute`, `process`, `exec` |
| Print   | `print`, `display`, `show`   |
| Search  | `scan`, `search`, `find`     |
| Persist | `load`, `save`               |

### 2.3 Function Name Length

- Aim for **15–40 characters** for public API names (with prefix).
- Prefer searchable over cryptic: `fps_save2disk` > `fps_s2d`.
- Avoid excessive verbosity: `fps_save2disk` >
  `fps_save_parameters_to_disk_file`.
- Use approved abbreviations (see §5) when names get long.

---

## 3. Variables

### 3.1 General Principles

**Balance length against clarity.** A variable name should be:

1. Long enough to convey **intent** without reading context.
2. Short enough to fit in expressions within 100 characters.
3. **Grep-searchable** — no single-letter names outside trivial loops.

### 3.2 Naming Rules by Scope

| Scope                    | Length | Convention                                    |
| ------------------------ | ------ | --------------------------------------------- |
| Loop index (< 10 lines)  | 1–2    | `ii`, `jj`, `kk`, `nn`                        |
| Block-local (< 20 lines) | 3–12   | `xsize`, `total`, `val`                       |
| Function-local (> 20 ln) | 5–20   | `byte_copy_size`                              |
| Static file-scope        | 6–25   | `param_streamname`                            |
| Global / extern          | 8–30   | `milk_data`, `dcimg`                          |
| FPS parameter local      | 8–25   | `param_loopgain` (recommended, not mandatory) |

### 3.3 Loop Index Conventions

- **Hard rule for agent-generated code**: use **doubled letters**
  `ii`, `jj`, `kk` (not `i`, `j`, `k`) **exclusively** for loops
  iterating over pixels in images or data vectors.
- Reserve `ii` for x-axis / innermost, `jj` for y, `kk` for z.
- For pixel count loops, `nn` is acceptable.
- **For all other non-pixel loops**: Use descriptive index names.
  Appending `_idx` is encouraged (e.g., `fps_idx`, `proc_idx`, `arg_idx`, `term_idx`).
- **Forbidden**: Bare single-letter variables (`i`, `j`, `k`) are
  forbidden everywhere as they are ungrepable.
- Examples of descriptive named indices for non-trivial loops:
  ```c
  for (int fps_idx = 0; fps_idx < nb_fps; fps_idx++)
  for (uint32_t col = 0; col < ncol; col++)
  ```

### 3.4 Dimension Variables

Use these standard names. Types must match `IMAGE_METADATA` fields
for vectorization (see §3.8).

| Variable   | Type       | Source / Meaning                   |
| ---------- | ---------- | ---------------------------------- |
| `xsize`    | `uint32_t` | `md->size[0]` — width (axis 0)     |
| `ysize`    | `uint32_t` | `md->size[1]` — height (axis 1)    |
| `zsize`    | `uint32_t` | `md->size[2]` — depth / slices     |
| `xysize`   | `uint64_t` | `(uint64_t)xsize * ysize`          |
| `xyzsize`  | `uint64_t` | `(uint64_t)xsize * ysize * zsize`  |
| `naxis`    | `uint8_t`  | `md->naxis` — number of axes       |
| `nelement` | `uint64_t` | `md->nelement` — total pixel count |
| `datatype` | `uint8_t`  | `md->datatype` — pixel data type   |

### 3.5 Pointer Naming

- Pixel data pointers (`float *restrict`): descriptive name, add
  type suffix when disambiguation is needed:
  ```c
  float *restrict pix_in  = imgin.im->array.F;
  float *restrict pix_out = imgout.im->array.F;
  ```
- **GPU device pointers**: prefix with `d_` to distinguish
  device memory from host memory:
  ```c
  float *d_modes  = NULL;  /* GPU device memory */
  float *d_in     = NULL;  /* GPU input buffer  */
  float *d_wfsVec = NULL;  /* GPU WFS vector    */
  ```
  This convention is already established in `linalgebra`,
  `ImageStreamIO`, and `cudacomp` modules.
- Raw pointers (`void *`): `raw`, `buf`, `ptr`
- IMGID variables (`IMGID`): `img`, `imgptr`, `img_input`
- Stream metadata (`IMAGE_METADATA *`): `md`
- FPS pointer (`FUNCTION_PARAMETER_STRUCT *`): `fps`

### 3.6 Boolean / Flag Variables

- Type: `int` for boolean-intent variables (project preference:
  `int` matches kernel style and avoids a `<stdbool.h>` dependency).
- Prefix with `is_`, `has_`, `do_`, or `flag_`:
  ```c
  int is_shared  = img.mdt->shared;
  int do_verbose = (VERBOSE > 0);
  int has_gpu    = (gpu_count > 0);
  ```
- Bitmask flags: `uint32_t` or `uint64_t` matching the field they
  test (e.g., `uint64_t fpflag` for `FPFLAG_*` tests).

### 3.7 Counter / Accumulator Variables

- Stream counters (`uint64_t`): `cnt`, `cnt0`, `cnt1` — matches
  `IMAGE_METADATA` fields.
- Accumulators: `total`, `sum`, `acc` — use `double` for float
  sums (precision), `uint64_t` for integer sums.
- Item counts: `nb_<thing>` or `n<thing>` — use `long`,
  `uint32_t`, or `int` depending on range.
  Examples: `long nb_frames`, `uint32_t nbimg`, `int naxis`.
- Return status: always `errno_t` (= `int`).

### 3.8 Loop Index Types

> [!CAUTION]
> **A type mismatch between a loop index and its bound destroys
> GCC auto-vectorization.** Verified empirically: `uint32_t`
> index vs `uint64_t` bound produces scalar code (`vmulss`,
> 1 float/cycle) instead of vectorized (`vmulps`, 8 floats/cycle)
> — an **8× slowdown**.

**Rule: match the index type to the bound type.**

| Bound source               | Bound type | Index type |
| -------------------------- | ---------- | ---------- |
| `md->size[]`, `xsize`      | `uint32_t` | `uint32_t` |
| `md->nelement`, `xysize`   | `uint64_t` | `uint64_t` |
| `strlen()`, `we_wordc`     | `size_t`   | `size_t`   |
| Fixed small count (< ~100) | `int`      | `int`      |

Do **not** use `size_t` for pixel loops — it mismatches
`IMAGE_METADATA` fields and obscures intent (`size_t` means
"memory size", not "pixel count").

```c
/* GOOD — types match, vectorizes */
uint64_t xysize = (uint64_t)xsize * ysize;
for (uint64_t ii = 0; ii < xysize; ii++)

/* GOOD — types match, vectorizes */
for (uint32_t ii = 0; ii < xsize; ii++)

/* BAD — type mismatch, scalar fallback */
for (uint32_t ii = 0; ii < xysize; ii++)

/* BAD — semantic mismatch */
for (size_t ii = 0; ii < xsize; ii++)
```

Cast `uint32_t` axis sizes to `uint64_t` **before** multiplying
to avoid overflow and produce a `uint64_t` bound:

```c
uint64_t xysize = (uint64_t)xsize * ysize;
```

---

## 4. Types

### 4.1 Struct / Typedef Names

All `typedef struct` names use `UPPER_CASE`. This is the
established codebase convention and must not be changed.

| Category      | Examples                                   |
| ------------- | ------------------------------------------ |
| Core data     | `IMAGE`, `IMAGE_METADATA`, `VARIABLE`      |
| API structs   | `FUNCTION_PARAMETER_STRUCT`, `MILK_DATA`   |
| Helpers       | `CBFRAMEMD`, `FRAMEWRITEMD`, `SEMFILEDATA` |
| Handles / IDs | `IMGID`, `CLICMDDATA`, `CLICMDARGDEF`      |

### 4.2 Enum Values

Use `UPPER_CASE` with a category prefix:

```c
_DATATYPE_FLOAT        /* legacy — see §6.1 */
FPTYPE_INT32
FPFLAG_DEFAULT_INPUT
CLIARG_IMG
```

---

## 5. Approved Abbreviations

Pre-approved abbreviations — prefer these over long forms:

| Abbr          | Meaning                    | Context        |
| ------------- | -------------------------- | -------------- |
| `img`         | image                      | everywhere     |
| `im`          | image (struct member)      | `IMAGE` fields |
| `shm`         | shared memory              | everywhere     |
| `sem`         | semaphore                  | everywhere     |
| `fps`         | function parameter struct  | everywhere     |
| `proc`        | process                    | processinfo    |
| `cmd`         | command                    | CLI            |
| `cli`         | command-line interface     | CLI            |
| `conf`        | configuration              | FPS            |
| `param`       | parameter                  | FPS            |
| `buf`         | buffer                     | data           |
| `cb`          | circular buffer            | streams        |
| `cnt`         | counter                    | everywhere     |
| `nb`          | number of                  | counts         |
| `idx`         | index                      | arrays         |
| `ptr`         | pointer                    | everywhere     |
| `val`         | value                      | FPS params     |
| `str`         | string (**var only** §6.1) | FPS params     |
| `fn` / `func` | function                   | callbacks      |
| `init`        | initialize                 | lifecycle      |
| `alloc`       | allocate                   | memory         |
| `sz`          | size                       | dimensions     |
| `len`         | length                     | strings        |
| `max` / `min` | maximum / minimum          | limits         |
| `avg`         | average                    | statistics     |
| `rms`         | root mean square           | statistics     |
| `dm`          | deformable mirror          | AO domain      |
| `wfs`         | wavefront sensor           | AO domain      |
| `ao`          | adaptive optics            | AO domain      |
| `rm`          | response matrix            | AO domain      |
| `cm`          | control matrix             | AO domain      |
| `pf`          | predictive filter          | AO domain      |
| `mvm`         | matrix-vector multiply     | compute        |

This list is **guidance, not enforcement**. Agents must use these
abbreviations in all new code they generate. If a new abbreviation
is needed, use it consistently and add it to this list.

---

## 6. Macros and Constants

| Category      | Convention               | Examples                   |
| ------------- | ------------------------ | -------------------------- |
| Numeric const | `UPPER_CASE`             | `IMAGE_NB_SEMAPHORE`       |
| String limits | `STRINGMAXLEN_<WHAT>`    | `STRINGMAXLEN_IMAGE_NAME`  |
| Data types    | `_DATATYPE_<TYPE>`       | `_DATATYPE_FLOAT` (legacy) |
| FPS types     | `FPTYPE_<TYPE>`          | `FPTYPE_INT32`             |
| FPS flags     | `FPFLAG_<NAME>`          | `FPFLAG_DEFAULT_INPUT`     |
| CLI args      | `CLIARG_<TYPE>`          | `CLIARG_IMG`               |
| Size-of       | `SIZEOF_DATATYPE_<TYPE>` | `SIZEOF_DATATYPE_FLOAT`    |
| Feature-test  | `UPPER_CASE`             | `MILK_NO_CLI`, `USE_CLI`   |

All macros use `UPPER_CASE_WITH_UNDERSCORES`.

Header guards: prefer `FILENAME_H` (no leading underscore). The
`_FILENAME_H` form in legacy headers is technically reserved by
C11 (see §6.1).

### 6.1 C and POSIX Reserved Namespaces

> [!WARNING]
> The C standard (C11 §7.1.3) and POSIX reserve several
> identifier patterns. **New code must not create identifiers
> in these reserved namespaces.**

**Reserved patterns — avoid in new code:**

| Pattern                  | Reserved by  | Risk                   |
| ------------------------ | ------------ | ---------------------- |
| `_Uppercase...`          | C11 §7.1.3   | UB; compiler internals |
| `__anything`             | C11 §7.1.3   | UB; compiler built-ins |
| `str` + lowercase (func) | `<string.h>` | libc name collision    |
| `mem` + lowercase (func) | `<string.h>` | libc name collision    |
| `is` + lowercase (func)  | `<ctype.h>`  | libc name collision    |
| `to` + lowercase (func)  | `<ctype.h>`  | libc name collision    |
| `SIG` + uppercase        | `<signal.h>` | signal name collision  |
| `E` + digit/uppercase    | `<errno.h>`  | errno code collision   |
| names ending in `_t`     | POSIX        | type name collision    |

**Practical rules:**

- Header guards: `FILENAME_H`, not `_FILENAME_H`.
- Boolean prefix `is_` (with underscore) is safe — it does not
  match the `is`+lowercase reservation. Never use `isXxx`
  without an underscore as a function name.
- The `str` abbreviation is safe for **variable names**
  (`char *str`, `param_str`), but never name a function
  `str` + lowercase (e.g., avoid `strname()`; use
  `stream_name()` instead).
- Do not define new file-scope `typedef` names ending in `_t`.
  Use `UPPER_CASE` for all new type names.

**Known legacy conflicts** (preserved, not renamed):

| Legacy identifier      | Conflict        | Status                |
| ---------------------- | --------------- | --------------------- |
| `_DATATYPE_FLOAT` etc. | `_` + uppercase | Entrenched; do not    |
|                        |                 | rename without shim.  |
| `_IMAGESTRUCT_H` etc.  | `_` + uppercase | Legacy header guard.  |
|                        |                 | Use `IMAGESTRUCT_H`   |
|                        |                 | for new headers.      |
| `errno_t`              | POSIX `_t`      | Conditionally defined |
|                        |                 | in `milkdata.h`;      |
|                        |                 | matches C11 Annex K.  |

---

## 7. FPS Parameter Keywords

FPS keywords use dot-separated hierarchical names:

```
.in_name        — input stream name
.out_name       — output stream name
.loopgain       — loop gain
.maxNBiter      — max iterations
.NBmodes        — number of modes
```

**Rules:**

- Always start with a leading dot: `.keyword`.
- **Hard rule for descriptive parameter names**: Do not use generic, non-descriptive names like
  `.param1`, `.param2`, `.param3`, or `.arg1`. Every parameter must have a descriptive name
  reflecting its physical meaning or algorithmic function (e.g. `.fringe_period` instead of
  `.param1`, `.fringe_angle` instead of `.param2`).
- Use `snake_case` after the dot (preferred) or `camelCase` for
  established parameters.
- Group related params with common prefix: `.loop_gain`,
  `.loop_maxiter`, `.loop_multcoeff`.
- Keep keywords < 20 chars after the dot for `fpsCTRL` readability.

---

## 8. Legacy Exceptions

> [!IMPORTANT]
> The existing codebase contains naming patterns that predate
> these conventions. **Do not rename existing public API symbols.**
> These conventions apply to **new code** and to code undergoing
> active refactoring.

Known legacy patterns to preserve:

- `ImageStreamIO_*` prefix (mixed case, well established)
- `functionparameter_*` prefix (verbose but consistent)
- `FUNCTION_PARAMETER_STRUCT` typedef (long but entrenched)
- `COREMOD_*` module directory names
- `AOloopControl_*` module directory names

---

## 9. Summary Cheat Sheet

```
Files:        snake_case.c/h
Dirs:         snake_case (new), COREMOD_X (legacy)
Public funcs: subsystem_verb_object()
Static funcs: verb_object()
Variables:    snake_case, length ∝ scope
Loop indices: ii, jj, kk (pixels); _idx or descriptive (other); not i, j, k
Structs:      UPPER_CASE
Macros:       UPPER_CASE_WITH_UNDERSCORES
FPS keywords: .snake_case
Scripts:      milk-subsys-verb
```
