# libprocessinfo: MILK Process Management Library

## Overview

`libprocessinfo` provides a lightweight framework for monitoring, controlling, and synchronizing real-time loops within the MILK ecosystem. It uses shared memory to expose the internal state of a process, such as its iteration count, execution timing, and trigger status.

## Key Features

- **Status Monitoring:** Exposes process state (ACTIVE, PAUSED, STOPPED, ERROR, CRASHED) to external tools.
- **Loop Control:** Allows external tools to signal a loop to pause, resume, single-step, or exit gracefully.
- **Timing Statistics:** Measures and records high-resolution timing for each loop iteration (start time, end time, iteration frequency).
- **Trigger Management:** Standardizes how loops are triggered (e.g., waiting on an image stream semaphore or a time delay).
- **Provenance & Telemetry:** Updates output stream metadata with PID, timestamps, and upstream data counters.
- **Signal Handling:** Integrates standard UNIX signals for clean shutdown and inter-process communication.

## Core Concepts

### PROCESSINFO
The main structure stored in shared memory (`$MILK_SHM_DIR/proc.<name>.<PID>.shm`, where `MILK_SHM_DIR` defaults to `/milk/shm`). It contains:
- Process identification (PID, name, source code location).
- Loop counters and control flags.
- Timing buffers for performance analysis.
- Input trigger configuration.

### Global Process List
A shared memory registry of all active MILK processes, enabling discovery by monitoring tools like `milk-procCTRL`.

## Basic Usage

### 1. Loop Initialization
```c
#include "processtools.h"

// Setup process info
PROCESSINFO *pinfo = processinfo_setup("my_loop", "Process description", "starting", __FUNCTION__, __FILE__, __LINE__);

// Configure a trigger (e.g., wait for a semaphore on an input image)
processinfo_waitoninputstream_init(pinfo, input_image, PROCESSINFO_TRIGGERMODE_SEMAPHORE, -1);
```

### 2. Main Loop
```c
int processloopOK = 1;
processinfo_loopstart(pinfo);

while (processloopOK) {
    // 1. Handle signals and check if loop should exit/pause
    processloopOK = processinfo_loopstep(pinfo);
    
    // 2. Wait for the trigger
    processinfo_waitoninputstream(pinfo);
    
    // 3. Mark start of computation
    processinfo_exec_start(pinfo);
    
    // --- Perform Work Here ---
    
    // 4. Mark end of computation
    processinfo_exec_end(pinfo);
    
    // 5. Update output stream metadata
    processinfo_update_output_stream(pinfo, output_image, input_image);
}

processinfo_cleanExit(pinfo);
```

## Monitoring Tools
- `milk-procCTRL`: A TUI tool for monitoring all active MILK processes, displaying CPU loads, timing stats, and providing control hooks.
- `milk-procinfo-list`: A CLI tool to list basic info for all active processes.