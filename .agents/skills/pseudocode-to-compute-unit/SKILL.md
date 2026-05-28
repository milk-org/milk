---
name: pseudocode-to-compute-unit
description: Step-by-step methodology for translating algorithm pseudocode into a working milk V2 compute unit.
---

# Translating Pseudocode to a milk Compute Unit

Use this methodology to convert an abstract algorithm
into a fully functional milk compute unit using the
V2 FPS architecture.

## Prerequisite: Quick Reference

Before you begin, consult the `api-quick-reference`
skill to review the `IMGID` API, datatype dispatch
tables, stream write protocols, and required headers.

## Step 1: Algorithm Analysis

Analyze the pseudocode to identify:

1. **Inputs:** Are they continuous image streams?
   Static files? Scalar parameters?
2. **Outputs:** Does it produce a new stream? Does
   it modify a stream in place?
3. **Tunables:** What numerical thresholds, gains,
   or switches does the algorithm need?
4. **Execution Mode:**
   - _Stream Processor:_ Runs continuously,
     triggered by a new frame in an input stream.
   - _One-shot:_ Runs once when called, then exits.
   - _Generator:_ Runs continuously, self-timed.

## Step 2: Parameter Mapping

Map each identified variable to an `FPTYPE_*`
parameter.

1. **Streams:** Map input/output stream names to
   `FPTYPE_STREAMNAME`. Give the input a flag like
   `FPFLAG_DEFAULT_TRIGGER_STREAM` if it drives
   the loop.
2. **Tunables:** Map floats to `FPTYPE_FLOAT32` or
   `FLOAT64`, integers to `FPTYPE_INT32` or `INT64`.
3. **Switches:** Map booleans to `FPTYPE_ONOFF`.

Set up the `FPS_PARAMS` X-Macro block. Follow the
strict column order:
`X(keyword, &var, type, is_primary, flags, descr)`

## Step 3: Stream I/O Design

Determine how streams will be connected in your
`compute_function`:

- **Input Stream:** Use `imgid_make_from_name()`
  - `resolveIMGID()`.
- **Output Stream:**
  - If it matches the input exactly:
    `imgid_copy(&in, &out)` +
    `imcreateIMGID(&out)`.
  - If it differs: manually set
    `out.mdt->naxis`, `out.mdt->size[]`,
    `out.mdt->datatype`, `out.mdt->shared`,
    then `imcreateIMGID(&out)`.

## Step 4: Code Structure Selection

Based on the execution mode, choose the template:

- **Stream Processor:** Copy
  `src/milk_module_example/examplefunc4_streamprocess.c`.
  You will need `INSERT_STD_PROCINFO_COMPUTEFUNC_INIT`,
  `LOOPSTART`, and `END`.
- **One-shot / Custom:** Copy
  `src/milk_module_example/examplefunc_fps_cli_poc.c`.

## Step 5: Datatype Handling

If the algorithm must handle multiple datatypes
(e.g., FLOAT and UINT16), use the standard milk
dispatch pattern inside your inner compute block:

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

- **No malloc in the loop:** Allocate any scratch
  buffers _before_
  `INSERT_STD_PROCINFO_COMPUTEFUNC_LOOPSTART`.
- **Pointer Alignment:** Use `MILK_RESTRICT` and
  `MILK_ASSUME_ALIGNED(ptr)` in compute-heavy
  inner loops to help GCC vectorize.
- **Write Protocol:** Set `outimg->md->write = 1`
  before modifying pixels, then call
  `processinfo_update_output_stream(processinfo,
outimg.im, inimg.im)` to finalize. Do NOT call
  `SHMIM_WRITE_RELEASE` or `ImageStreamIO_UpdateIm`
  yourself — `processinfo_update_output_stream`
  handles everything.

## Step 7: Scaffold and Finalize

Copy the chosen template, replace `FPS_APP_INFO`,
inject your `FPS_PARAMS`, and place your core logic
inside the computation function. Define the CMake
target with `add_milk_standalone()` or equivalent.

---

## Worked Example: Gain-Multiply Stream Processor

**Pseudocode:**

```
for each frame in input_stream:
    for each pixel ii:
        output[ii] = input[ii] * gain
```

**Resulting V2 compute unit:**

```c
#ifdef MILK_NO_CLI
#    include "CLIcore_standalone.h"
#else
#    include "CLIcore.h"
#endif
#include "fps.h"
#include "COREMOD_memory/COREMOD_memory.h"

/* Section 1: Identity */
static FPS_APP_INFO FPS_app_info = {
    .fps_name    = "gainmul",
    .cmdkey      = "gainmul",
    .description = "multiply stream by gain"
};

/* Section 2: Local variables */
static char  inimname[FUNCTION_PARAMETER_STRMAXLEN]
    = "stream_in";
static char  outimname[FUNCTION_PARAMETER_STRMAXLEN]
    = "stream_out";
static float gain = 1.0f;

/* Section 3: FPS_PARAMS X-macro */
#define FPS_PARAMS(X)                             \
    X(".in_name", inimname, FPTYPE_STREAMNAME,    \
      1, FPFLAG_DEFAULT_TRIGGER_STREAM,           \
      "input stream")                             \
    X(".out_name", outimname, FPTYPE_STREAMNAME,  \
      1, FPFLAG_DEFAULT_INPUT, "output stream")   \
    X(".gain", &gain, FPTYPE_FLOAT32,             \
      0,                                          \
      FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN,     \
      "gain factor")

/* Section 4: Core computation */
static errno_t streamprocess(
    IMGID *inimg,
    IMGID *outimg,
    float  g)
{
    resolveIMGID(inimg, ERRMODE_WARN,
                 dcimg, dcnimg);
    if (inimg->ID == -1)
    {
        return RETURN_FAILURE;
    }
    uint64_t xysize =
        inimg->mdt->size[0] * inimg->mdt->size[1];

    float * restrict in_ptr = inimg->im->array.F;
    float * restrict out_ptr = outimg->im->array.F;
    for (uint64_t ii = 0; ii < xysize; ii++)
    {
        out_ptr[ii] = in_ptr[ii] * g;
    }

    return RETURN_SUCCESS;
}

/* Section 5: Bindings & CLIcmddata */
static FPS_CLI_BINDING my_bindings[] = {
    FPS_PARAMS(FPS_X_BINDING) };
static const int nb_bindings =
    sizeof(my_bindings) / sizeof(FPS_CLI_BINDING);
static CLICMDARGDEF farg[] = {
    FPS_PARAMS(FPS_X_FARG) };

#ifdef FPS_STANDALONE
CLICMDDATA CLIcmddata = {
#else
static CLICMDDATA CLIcmddata = {
#endif
    "", "", CLICMD_FIELDS_DEFAULTS
};
FPS_CMDSETTINGS_INIT(dft, CLIcmddata, FPS_app_info)

/* Section 6: Compute wrapper */
static MILK_HOT errno_t __attribute__((unused))
compute_function()
{
    IMGID inimg = imgid_make_from_name(inimname);
    resolveIMGID(&inimg, ERRMODE_WARN,
                 dcimg, dcnimg);
    if (inimg.ID == -1)
    {
        return RETURN_FAILURE;
    }

    IMGID outimg = imgid_make_from_name(outimname);
    imgid_copy(&inimg, &outimg);
    imcreateIMGID(&outimg);

    INSERT_STD_PROCINFO_COMPUTEFUNC_INIT
    INSERT_STD_PROCINFO_COMPUTEFUNC_LOOPSTART
    {
        streamprocess(&inimg, &outimg, gain);
        processinfo_update_output_stream(
            processinfo, outimg.im, inimg.im);
    }
    INSERT_STD_PROCINFO_COMPUTEFUNC_END

    imgid_free(&inimg);
    imgid_free(&outimg);
    return RETURN_SUCCESS;
}

/* Section 7: CLI registration */
#if !defined(FPS_STANDALONE) && !defined(MILK_NO_CLI)
static errno_t CLIfunction(void)
{
    return safe_fps_generic_CLIfunction(
        &FPS_app_info, farg, &CLIcmddata,
        my_bindings, nb_bindings,
        compute_function);
}

errno_t CLIADDCMD_mymodule__gainmul()
{
    safe_fps_fill_farg_examples(
        farg, my_bindings, nb_bindings);
    INSERT_STD_CLIREGISTERFUNC
    return RETURN_SUCCESS;
}
#endif

/* Section 8: Standalone main */
#ifdef FPS_STANDALONE
FPS_MAIN_STANDALONE_V2(
    FPS_app_info, FPS_PARAMS, compute_function)
#endif
```
