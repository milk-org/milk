# Developer Tutorial: Writing Your First Milk Module

Welcome to `milk`! This tutorial will guide you through
creating your first module and standalone `fpsexec` compute
block. By the end of this guide, you will have a working
module linked to the `milk` core frameworks.

See also: [Programmer's Guide](../programmers_guide.md) ·
[Code Assist Tools](../code_assist.md) ·
[Adding Plugins](plugins.md) ·
[FPS](../fps.md) ·
[FPS Standalone Modes](../FPS_Standalone_CMD_Modes.md)

## 1. Setting Up the Directory Structure

The easiest way to start is by copying the provided example module. We will create a new module named `my_first_module`.

```bash
cd milk/plugins/milk-extra-src/
cp -r ../../src/milk_module_example my_first_module
cd my_first_module
```

## 2. Configuring CMake

Open `CMakeLists.txt` in your new directory. You need to identify your module and tell the build system which files to compile.

Change the `LIBNAME` to your module's name:

```cmake title="CMakeLists.txt"
# Change this:
# set(LIBNAME milk_module_example)
set(LIBNAME my_first_module)
```

Define your source files and the standalone executables you want to build:

```cmake title="CMakeLists.txt"
set(SOURCEFILES
    examplefunc.c
    examplefunc2_FPS.c
    # Add your own source files here
)

# Register a standalone executable
# add_milk_standalone(short_name source_file.c)
add_milk_standalone(my-first-exec examplefunc2_FPS.c)
```

## 3. Writing an FPS Compute Block

The Function Parameter Structure (FPS) is the standard way milk exposes configuration variables to the outside world (like the `milk-fpsCTRL` TUI).

Let's look at a basic FPS setup in `my_first_module_fps.c`. You need an info struct, a parameter mapping, and an execution loop.

### A. Define the Application Info

Every `fpsexec` program needs an ID string, a command key, and a short description:

```c title="my_first_module_fps.c"
#include "fps.h"

FPS_APP_INFO FPS_app_info = {
    .fps_name = "myfirstexec", // SHM name on disk
    .cmdkey = "myexec",        // CLI keyword
    .description = "My very first milk compute block!"
};
```

### B. Map Parameters

Use the `FPS_PARAMS` X-Macro to define the configuration parameters you want to expose:

```c title="my_first_module_fps.c"
// Local Variables
static int32_t param_iterations = 100;
static float   param_gain = 0.5f;

// Parameter mapping (X-macro)
#define FPS_PARAMS(X) \
    X(".iterations", &param_iterations, \
      FPTYPE_INT32, 1, \
      FPFLAG_DEFAULT_INPUT, "Number of iterations") \
    X(".gain", &param_gain, \
      FPTYPE_FLOAT32, 1, \
      FPFLAG_DEFAULT_INPUT, "Loop gain")
```

### C. Implement the Compute Function

This is the core compute function. It is called after parameters are synced from FPS shared memory.

```c title="my_first_module_fps.c"
static errno_t fpsexec(void) {
    printf("Running with gain %f, %d iterations\n",
           param_gain, param_iterations);

    for (int i = 0; i < param_iterations; i++) {
        // Do math here...
    }

    return RETURN_SUCCESS;
}
```

### D. The Main Entry Point

The V2 macro generates the standalone `main()` function that handles the FPS lifecycle (create, exec, confstart, runstart, etc.):

```c title="my_first_module_fps.c"
// Processinfo-wrapped entry point
static errno_t compute_function() {
    INSERT_STD_PROCINFO_COMPUTEFUNC_START
    fpsexec();
    INSERT_STD_PROCINFO_COMPUTEFUNC_END
    return RETURN_SUCCESS;
}

// Standalone main (compiled with -DFPS_STANDALONE)
#ifdef FPS_STANDALONE
FPS_MAIN_STANDALONE_V2(FPS_app_info, FPS_PARAMS, compute_function)
#endif
```

## 4. Compile and Run!

Go back to the root `milk` directory and re-run standard compilation. `CMake` will automatically discover your newly linked plugin.

```bash
cd ../../../
bash compile.sh $PWD/local
```

Now, launch your standalone FPS compute block!

```bash
milk-fpsexec-my-first-exec -tmux
```

Open a new terminal and inspect your running parameter tree:

```bash
milk-fpsCTRL myfirstexec
```

Congratulations! You've successfully written a milk module and an FPS compute block.

## 5. Next Steps

Now that you've written your first module, you can automate this process and learn more advanced patterns using our built-in tools:

- Run the `/create-plugin` workflow to scaffold new plugins automatically.
- Run the `/create-fpsexec` workflow to scaffold new compute units.
- Consult the `api-quick-reference` skill for a cheat sheet on streams and FPS parameters.
- Review the `pseudocode-to-compute-unit` skill to learn how to translate abstract algorithms into V2 templates.

---

← [Documentation Index](../index.md)
