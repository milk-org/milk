---
name: pseudocode-to-compute-unit
description: Step-by-step methodology for translating algorithm pseudocode into a working milk V2 compute unit.
---

# Translating Pseudocode to a milk Compute Unit

Use this methodology to convert an abstract algorithm into a fully functional milk compute unit using the V2 FPS architecture.

## Prerequisite: Quick Reference

Before you begin, consult the `api-quick-reference` skill to review the `IMGID` API, datatype dispatch tables, and stream write protocols.

## Step 1: Algorithm Analysis

Analyze the pseudocode to identify:

1. **Inputs:** Are they continuous image streams? Static files? Scalar parameters?
2. **Outputs:** Does it produce a new stream? Does it modify a stream in place? Does it return a scalar value?
3. **Tunables:** What numerical thresholds, gains, or switches does the algorithm need?
4. **Execution Mode:**
   - _Stream Processor:_ Runs continuously, triggered by a new frame in an input stream.
   - _One-shot:_ Runs once when called, then exits.
   - _Generator:_ Runs continuously, self-timed (e.g. at a set delay), outputting data.

## Step 2: Parameter Mapping

Map each identified variable to an `FPTYPE_*` parameter.

1. **Streams:** Map input/output stream names to `FPTYPE_STREAMNAME`. Give the input a flag like `FPFLAG_DEFAULT_TRIGGER_STREAM` if it drives the loop.
2. **Tunables:** Map floats to `FPTYPE_FLOAT32` or `FLOAT64`, integers to `FPTYPE_INT32` or `INT64`.
3. **Switches:** Map booleans to `FPTYPE_ONOFF`.

Set up the `FPS_PARAMS` X-Macro block. Follow the strict column order:
`X(keyword, &var, type, is_primary, flags, description)`

## Step 3: Stream I/O Design

Determine how streams will be connected in your `compute_function`:

- **Input Stream:** Use `imgid_make_from_name()` + `resolveIMGID()`.
- **Output Stream:**
  - If it matches the input exactly: `imgid_copy(&in, &out)` + `imcreateIMGID(&out)`.
  - If it differs: manually set `out.mdt->naxis`, `out.mdt->size[]`, `out.mdt->datatype`, `out.mdt->shared`, then `imcreateIMGID(&out)`.

## Step 4: Code Structure Selection

Based on the execution mode, choose the template:

- **Stream Processor:** Look at `src/milk_module_example/examplefunc4_streamprocess.c`. You will need `INSERT_STD_PROCINFO_COMPUTEFUNC_INIT`, `LOOPSTART`, and `END`.
- **One-shot / Custom:** Look at `src/milk_module_example/examplefunc_fps_cli_poc.c`.

## Step 5: Datatype Handling

If the algorithm must handle multiple datatypes (e.g., FLOAT and UINT16), use the standard milk dispatch pattern inside your inner compute block:

```c
uint32_t dt = inimg->mdt->datatype;
if (dt == _DATATYPE_FLOAT)
{
    float * restrict in_ptr = inimg->im->array.F;
    float * restrict out_ptr = outimg->im->array.F;
    // ... compute ...
}
else if (dt == _DATATYPE_UINT16)
{
    uint16_t * restrict in_ptr = inimg->im->array.UI16;
    // ... compute ...
}
```

## Step 6: Memory and Performance

- **No malloc in the loop:** Allocate any scratch buffers _before_ `INSERT_STD_PROCINFO_COMPUTEFUNC_LOOPSTART`.
- **Pointer Alignment:** Use `MILK_RESTRICT` and `MILK_ASSUME_ALIGNED(ptr)` in compute-heavy inner loops to help GCC vectorize.
- **Write Protocol:** Use `SHMIM_WRITE_ACQUIRE(out.im->md)`, modify pixels, `SHMIM_WRITE_RELEASE(...)`, and `processinfo_update_output_stream()`.

## Step 7: Scaffold and Finalize

Copy the chosen template, replace `FPS_APP_INFO`, inject your `FPS_PARAMS`, and place your core logic inside the computation function. Make sure to define the CMake target with `add_milk_standalone()` or equivalent in the module's `CMakeLists.txt`.
