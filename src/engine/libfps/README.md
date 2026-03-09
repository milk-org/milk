# libfps: Function Parameter Structure Library

## Overview

`libfps` is a core component of the MILK framework, providing a standardized way to manage function parameters in a distributed environment. It allows processes to share and dynamically update configuration parameters through shared memory (SHM).

## Key Features

- **Shared Memory Storage:** Parameters are stored in SHM, allowing multiple processes to read and write them with near-zero latency.
- **Dynamic Reconfiguration:** Processes can monitor for parameter changes and update their internal state accordingly.
- **Hierarchical Keywords:** Parameters use dot-notated keywords (e.g., `ao.gain.loop`) for organized management.
- **Persistent Storage:** Parameters can be automatically saved to disk on change or on closure, ensuring settings persist across restarts.
- **Tmux Integration:** Helper functions to manage tmux sessions for running processes (`ctrl`, `conf`, `run` windows).

## Standalone Command Convention

Standalone applications using `libfps` typically support the following commands:

- **fpsinit**: One-time setup. Creates the FPS shared memory structure and sets default values.
- **confstart**: Starts the configuration loop process (usually runs in `conf` window).
- **confstep**: Runs a single iteration of the configuration logic.
- **confstop**: Stops the configuration process.
- **runstart**: Starts the main processing run loop (usually runs in `run` window).
- **runstop**: Stops the main processing loop.

## Tmux Management

`libfps` provides functions to automate the setup of a tmux session with three standard windows:
- **ctrl**: For user interaction and control.
- **conf**: For the configuration process (`confstart`).
- **run**: For the main run process (`runstart`).

### Example Usage (Standalone)

```c
#include "fps_tmux.h"

// ... inside main(argc, argv) ...

if (use_tmux) {
    // 1. Ensure session exists (creates ctrl, conf, run windows)
    functionparameter_FPS_tmux_standalone_setup(fps_name);

    // 2. Dispatch command to appropriate window
    if (strcmp(command, "confstart") == 0) {
        functionparameter_FPS_tmux_send(fps_name, "conf", "my_app confstart");
    } else if (strcmp(command, "runstart") == 0) {
        functionparameter_FPS_tmux_send(fps_name, "run", "my_app runstart");
    }
}
```

## Basic Usage (Code)

### 1. Initialization (CONF process)
```c
#include "fps.h"

// Setup
FPS_SETUP_INIT("my_fps", FPSCMDCODE_FPSINIT);
function_parameter_add_entry(&fps, ".gain", "Loop gain", FPTYPE_FLOAT32, FPFLAG_DEFAULT_INPUT, &default_gain, NULL);
function_parameter_FPCONFexit(&fps);
```

### 2. Configuration Loop (CONF process)
```c
FPS_CONNECT("my_fps", FPSCMDCODE_CONFSTART);
FPS_CONFLOOP_START
{
    // Validate parameters
    float gain = functionparameter_GetParamValue_FLOAT32(&fps, ".gain");
    if (gain < 0) { /* clamp or warn */ }
}
FPS_CONFLOOP_END
```

### 3. Run Loop (RUN process)
```c
FPS_CONNECT("my_fps", FPSCONNECT_RUN);
// ... setup loop ...
while(loop) {
    // Read parameters
    float gain = functionparameter_GetParamValue_FLOAT32(&fps, ".gain");
    // Compute
}
```
