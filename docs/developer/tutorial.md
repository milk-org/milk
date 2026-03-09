# Developer Tutorial: Writing Your First Milk Module

Welcome to `milk`! This tutorial will guide you through creating your first module and standalone `fpsexec` compute block. By the end of this guide, you will have a working module linked to the `milk` core frameworks.

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

```cmake
# Change this:
# set(LIBNAME milk_module_example)
set(LIBNAME my_first_module)
```

Define your source files and the standalone executables you want to build:

```cmake
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

```c
#include "fps.h"

FPS_APP_INFO FPS_app_info = {
    .fps_name = "myfirstexec", // SHM name on disk
    .cmdkey = "myexec",        // CLI keyword
    .description = "My very first milk compute block!"
};
```

### B. Map Parameters
Use the `FPS_PARAMS` X-Macro to define the configuration parameters you want to expose:

```c
// Local Variables
static int param_iterations = 100;
static float param_gain = 0.5f;

// Parameter mapping
#define FPS_PARAMS \
    FPSEXEC_PARAM_INT32("Iterations", &param_iterations) \
    FPSEXEC_PARAM_FLOAT32("Gain", &param_gain)
```

### C. Implement the Execution Loop
This is the core compute function. It is automatically called after parameters are synced.

```c
int run_my_compute(void) {
    FPS_PRINTINFO("Starting compute with gain %f and %d iterations.", param_gain, param_iterations);
    
    // Core loop controlled by processinfo
    while(data.processinfo->loopstat == 0) {
        // Do math here...
        
        // Let the CPU rest
        usleep(10000); 
    }
    
    return 0;
}
```

### D. The Main Entry Point
Use the standardized multi-mode macro to build both the CLI command and the standalone `main()` function automatically:

```c
#ifdef FPS_STANDALONE
CLICMDDATA CLIcmddata = {"myexec", "Run my compute block", CLICMD_FIELDS_DEFAULTS};
#else
static CLICMDDATA CLIcmddata = {"myexec", "Run my compute block", CLICMD_FIELDS_DEFAULTS};
#endif

// This macro creates the main() entry point for the standalone executable.
FPS_MAIN_STANDALONE_V2(FPS_app_info, FPS_PARAMS, run_my_compute)
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
