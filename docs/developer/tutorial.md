# Developer Tutorial: Writing Your First Milk Module

Welcome to `milk`! This tutorial will guide you through creating your first module and standalone
`fpsexec` compute block. By the end of this guide, you will have a working module linked to the
`milk` core frameworks.

See also: [Programmer's Guide](../arch/programmers_guide.md) · [Code Assist Tools](code_assist.md) ·
[Adding Plugins](plugins.md) · [FPS](../guide/fps.md)

## 1. Setting Up the Directory Structure

The easiest way to start is by copying the provided example module into `plugins/` (outside
`milk-extra-src/`, which is reserved for built-in plugins). We will create a new module named
`my_first_module`:

```bash
cd milk/plugins/
cp -r ../src/milk_module_example my_first_module
cd my_first_module
```

## 2. Configuring CMake

Open `CMakeLists.txt` in your new directory (see the [Programmer's Guide](../arch/programmers_guide.md#6-cmakeliststxt-conventions) for its layout).

Set `LIBNAME` and `SRCNAME` to your module's name:

```cmake title="CMakeLists.txt"
set(LIBNAME "my_first_module")
set(SRCNAME "my_first_module")
```

Source files are picked up automatically; the module is built by `milk_add_runnable_module()` and one
executable is generated per `EXPORT_FPSEXEC` line of the sources (see below). Update the matching
`add_test` lines to the executable ids you export.

## 3. Writing a Compute Unit

The Function Parameter Structure (FPS) is the standard way milk exposes configuration variables to
the outside world (like the `milk-fpsCTRL` TUI). A compute unit is a `MILK_RUNNABLE` descriptor that
gathers an identity, a parameter table and a compute function. We write one in
`my_first_module_fps.c`.

### A. Parameters

Use an X-macro table to define the configuration parameters you want to expose; `MILK_PARAMS_DEFINE`
expands it into the FPS bindings and the CLI argument definitions:

```c title="my_first_module_fps.c"
#include "milk_runnable.h"

// Local variables
static int32_t param_iterations = 100;
static float   param_gain       = 0.5f;

// Parameter table (X-macro)
#define MYFIRST_PARAMS(X)                                                    \
    X(".iterations", &param_iterations, FPTYPE_INT32, 1,                     \
      FPFLAG_DEFAULT_INPUT, "Number of iterations")                          \
    X(".gain", &param_gain, FPTYPE_FLOAT32, 1,                               \
      FPFLAG_DEFAULT_INPUT, "Loop gain")

MILK_PARAMS_DEFINE(myfirst, MYFIRST_PARAMS);
```

### B. Implement the Compute Function

This is the core compute function. It is called after parameters are synced from FPS shared memory;
the processinfo macros provide the heartbeat, loop control and timing:

```c title="my_first_module_fps.c"
static errno_t myfirst_compute(const MILK_RUNNABLE *self)
{
    MILK_PROCINFO_INIT(self)
    MILK_PROCINFO_LOOPSTART(self)
    {
        printf("Running with gain %f, %d iterations\n", param_gain, param_iterations);

        for (int i = 0; i < param_iterations; i++)
        {
            // Do math here...
        }
    }
    MILK_PROCINFO_END(self)

    return RETURN_SUCCESS;
}
```

### C. The Descriptor

The descriptor gives the unit its identity. `EXPORT_FPSEXEC` makes CMake generate the
`milk-fpsexec-my-first-exec` executable, with a `main()` that handles the FPS lifecycle (create, exec,
confstart, runstart, etc.):

```c title="my_first_module_fps.c"
const MILK_RUNNABLE runnable_my_first_module_myfirst = {
    .kind    = MILK_RUNNABLE_FPS,
    .app     = { .fps_name    = "myfirstexec",  // SHM name on disk
                 .cmdkey      = "myexec",       // CLI keyword
                 .description = "My very first milk compute block!" },
    .srcfile = __FILE__,
    MILK_PARAMS_REF(myfirst),
    .compute = myfirst_compute,
};
EXPORT_FPSEXEC(milk, "my-first-exec", runnable_my_first_module_myfirst);
```

### D. Register the Unit in the Module

List the unit in the module registry (`my_first_module.c`), which is what makes it a `milk-cli` command
once the module is loaded:

```c title="my_first_module.c"
#define MODULE_SHORTNAME_DEFAULT "myfirst"
#define MODULE_DESCRIPTION       "My first milk module"

#include "milk_runnable.h"

extern const MILK_RUNNABLE runnable_my_first_module_myfirst;

static const MILK_RUNNABLE *const runnables[] = {
    &runnable_my_first_module_myfirst,
};

MILK_MODULE_RUNNABLES(my_first_module, NULL, runnables);
```

## 4. Compile and Run!

Go back to the root `milk` directory, re-run CMake so it discovers your new plugin, and install:

```bash
cd ../../
bash compile.sh $PWD/local
make -C _build install
export MILK_INSTALLDIR="$PWD/local/milk"
export PATH="$MILK_INSTALLDIR/bin:$PATH"
```

Now, create your FPS compute block and start it in the background (`runstart` returns once its loop
runs):

```bash
milk-fpsexec-my-first-exec fpsinit
milk-fpsexec-my-first-exec runstart
```

Inspect your running parameter tree, and what the unit prints:

```bash
milk-fpsCTRL myfirstexec
milk-fps-log -f myfirstexec
```

Congratulations! You've successfully written a milk module and an FPS compute block.

## 5. Next Steps

Now that you've written your first module, you can automate this process and learn more advanced
patterns using our built-in tools:

- Run the `/add-module` workflow to scaffold new plugins automatically.
- Run the `/add-runnable` workflow to scaffold new compute units.
- Consult the `api-quick-reference` skill for a cheat sheet on streams and FPS parameters.

---

← [Documentation Index](../index.md)
