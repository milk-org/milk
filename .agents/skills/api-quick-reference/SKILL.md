---
name: api-quick-reference
description: Deep reference for milk APIs including IMGID, datatype dispatch, processinfo macros, magic context variables, FPS parameter types, and stream write protocols.
---

# milk API Quick Reference

This skill provides a condensed "cheat sheet" of critical APIs used in milk compute units.

## 1. IMGID API Reference

Creating, connecting to and format-checking a stream through `IMGID` is in
[Shared Memory Streams §5](../../../docs/guide/streams.md#5-c-api-imgid). Configure a new stream
through `img.mdt->` (the metadata template) before it is created or connected, never through the
struct fields; `resolveIMGID(&img, ERRMODE_WARN, dcimg, dcnimg)` resolves a name in the process's
image table (§4).

## 2. Datatype Dispatch Table

When a function supports multiple datatypes, use an `else if` chain matching `_DATATYPE_*` to the correct union member of `im->array`.

| Type Constant              | Union Member | C Type           | Size Macro                       |
| -------------------------- | ------------ | ---------------- | -------------------------------- |
| `_DATATYPE_UINT8`          | `.UI8`       | `uint8_t`        | `SIZEOF_DATATYPE_UINT8`          |
| `_DATATYPE_INT8`           | `.I8`        | `int8_t`         | `SIZEOF_DATATYPE_INT8`           |
| `_DATATYPE_UINT16`         | `.UI16`      | `uint16_t`       | `SIZEOF_DATATYPE_UINT16`         |
| `_DATATYPE_INT16`          | `.I16`       | `int16_t`        | `SIZEOF_DATATYPE_INT16`          |
| `_DATATYPE_UINT32`         | `.UI32`      | `uint32_t`       | `SIZEOF_DATATYPE_UINT32`         |
| `_DATATYPE_INT32`          | `.I32`       | `int32_t`        | `SIZEOF_DATATYPE_INT32`          |
| `_DATATYPE_UINT64`         | `.UI64`      | `uint64_t`       | `SIZEOF_DATATYPE_UINT64`         |
| `_DATATYPE_INT64`          | `.I64`       | `int64_t`        | `SIZEOF_DATATYPE_INT64`          |
| `_DATATYPE_FLOAT`          | `.F`         | `float`          | `SIZEOF_DATATYPE_FLOAT`          |
| `_DATATYPE_DOUBLE`         | `.D`         | `double`         | `SIZEOF_DATATYPE_DOUBLE`         |
| `_DATATYPE_COMPLEX_FLOAT`  | `.CF`        | `complex float`  | `SIZEOF_DATATYPE_COMPLEX_FLOAT`  |
| `_DATATYPE_COMPLEX_DOUBLE` | `.CD`        | `complex double` | `SIZEOF_DATATYPE_COMPLEX_DOUBLE` |

For untyped generic copies, use `.raw` with `__builtin_memcpy`.

## 3. processinfo Macro Reference

`MILK_PROCINFO_INIT`, `MILK_PROCINFO_LOOPSTART` and `MILK_PROCINFO_END`, which wrap a unit's
compute loop, are in
[Process Info §7](../../../docs/guide/procinfo.md#7-compute-function-macros).

## 4. Magic Context Variables

These global variables are defined by the framework and are always available in compute functions.

| Variable        | Type           | Defined By              | Description                                                                                                    |
| --------------- | -------------- | ----------------------- | -------------------------------------------------------------------------------------------------------------- |
| `dcfpsptr`      | `FPS*`         | `milkdata_macros.h`     | Pointer to the current FPS instance.                                                                           |
| `dcfpsname`     | `char[]`       | `milkdata_macros.h`     | Name of the current FPS instance.                                                                              |
| `dcimg`         | `IMAGE*`       | `milkdata_macros.h`     | The global image array (used in `resolveIMGID`).                                                               |
| `dcnimg`        | `long`         | `milkdata_macros.h`     | Size of the `dcimg` array.                                                                                     |
| `processinfo`   | `PROCESSINFO*` | `fps_procinfo_macros.h` | Struct tracking process timing, loop limits, and triggers. Injected by `MILK_PROCINFO_INIT`. |
| `processloopOK` | `int`          | `fps_procinfo_macros.h` | Loop condition variable. Injected by `INIT`, checked in `LOOPSTART`.                                           |

## 5. Stream Write Protocol

Writing an output stream is in
[Process Info — Output Stream Updates](../../../docs/guide/procinfo.md#output-stream-updates), and
the ancestry it carries over in
[Shared Memory Streams §6](../../../docs/guide/streams.md#6-stream-process-trace-ancestry) (more
internals: the `imagestream-internals` skill). In a compute unit, never call
`ImageStreamIO_UpdateIm()`, `SHMIM_CNT0_INCREMENT`, `SHMIM_WRITE_RELEASE` or
`ImageStreamIO_sempost()` yourself: `processinfo_update_output_stream()` does it.

## 6. FPS Parameter Quick-Reference

The `FPTYPE_*` types with their C types, the `FPFLAG_*` flags and the X-macro column order are in
[FPS §4](../../../docs/guide/fps.md#4-parameter-data-types-and-flags).

## 7. Required Header Includes

Every `.c` file must explicitly include the headers
it uses. Here are the common APIs and their headers:

| API / Function                                                                                           | Required `#include`                                          |
| -------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------ |
| `IMGID`, `imgid_make_from_name`, `imgid_copy`, `imgid_free`, `imcreateIMGID`, `resolveIMGID`             | `"COREMOD_memory/COREMOD_memory.h"`                          |
| `MILK_RUNNABLE`, `MILK_PARAMS_DEFINE`, `EXPORT_FPSEXEC`, `MILK_PROCINFO_*`, `FPS_APP_INFO`, `FPS_CLI_BINDING`, `CLICMDDATA`, `CLICMDARGDEF` | `"milk_runnable.h"` (first include of a unit; never `CLIcore.h`) |
| `processinfo_update_output_stream`                                                                       | `"processtools.h"` (via `milk_runnable.h`)                  |
| `cblas_sgemv`, `cblas_sgemm`                                                                             | `<cblas.h>` (link `${BLAS_LIBRARIES}`)                       |
| `sqrtf`, `powf`, `sinf`, `cosf`                                                                          | `<math.h>` (link `-lm`)                                      |
| `MILK_HOT`, `MILK_COLD`, `UNLIKELY`, `MILK_ASSUME_ALIGNED`                                               | `"milkDebugTools.h"` (via `milk_runnable.h`)                 |
| `DEBUG_TRACE_FSTART`, `DEBUG_TRACE_FEXIT`                                                                | `"milkDebugTools.h"` (via `milk_runnable.h`)                 |
