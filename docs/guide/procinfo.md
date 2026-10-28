---
tags:
  - processinfo
  - telemetry
  - profiling
---

# Process Info (`procinfo`)

The `procinfo` (Process Information) system is a critical telemetry and heartbeat monitoring layer
within `milk`. It works symbiotically with the Function Processing System (FPS) to ensure health
checks, profiling, and state visibility across the entire execution environment.

See also: [FPS](fps.md) · [Streams](streams.md) · [Programmer's Guide](../arch/programmers_guide.md)

## 1. Design and Purpose

Instead of blindly assuming processes are running simply because their process IDs exist, `procinfo`
maintains active heartbeats and state markers in shared memory. This is particularly crucial for
real-time control applications (like Adaptive Optics) where a process might "hang" mathematically
without crashing the executable.

## 2. The Output: `milk-procinfo-list`

The primary user-facing tool for this system is the `milk-procinfo-list` command. This tool scans
the system and provides a visual overview of all processes that have registered themselves with
`procinfo`.

### 2.1. State Tracking

A correctly implemented compute unit will constantly update its state so tools like
`milk-procinfo-list` can display:

- **IDLE / WAITING:** The loop is paused or blocking on a stream semaphore waiting for new data.
- **ACTIVE / RUNNING:** The compute loop is actively processing data. PIDs are highlighted green to
  indicate healthy processing.
- **FAILURE / ERROR:** If processing errors occur or the heartbeat stops, monitoring tools can flag
  the process for restart.

## 3. Loop Profiling

Along with boolean states, `procinfo` actively measures loop execution frequencies. It can display
the **Hz** (loops per second) for active pipelines. This means performance drops or bottlenecks in
stream processing are immediately visible from the top-level dashboard.

## 4. Standalone Executable Integration (`fpsexec`)

For `MILK_RUNNABLE` units run as an fpsexec (see [Programmer's Guide](../arch/programmers_guide.md)),
`procinfo` is automatically enabled and registered for you. Your executable simply needs to
correctly execute its while loop, and the framework pulses the heartbeat for you.

## 5. C Code Template

For custom usage without a `MILK_RUNNABLE` unit, here is how you initialize and use `PROCESSINFO`:

```c
#include "CLIcore.h"

int functiontemplate_usingprocessinfo() {

    PROCESSINFO *processinfo;
    char pinfoname[STRINGMAXLEN_PROCESSINFO_NAME];
    sprintf(pinfoname, "aol%ld-acqRM", loopindex);

    char pinfodescr[STRINGMAXLEN_PROCESSINFO_DESCRIPTION];
    sprintf(pinfodescr, "NBcycle=%ld", NBcycle);

    char pinfomsg[STRINGMAXLEN_PROCESSINFO_DESCRIPTION];
    sprintf(pinfomsg, "starting setup");

    processinfo = processinfo_setup(
        pinfoname,	         // short name for the processinfo instance
        pinfodescr,    // description
        pinfomsg,  // message on startup
        __FUNCTION__, __FILE__, __LINE__
        );

	// OPTIONAL SETTINGS
    processinfo->MeasureTiming = 1; // Measure timing
    processinfo->RT_priority = 20;  // RT_priority, 0-99. Larger number = higher priority. If <0, ignore
    processinfo->loopcntMax = 100;  // -1 if infinite loop

    // =============================================
    // OPTIONAL: TESTING CONDITION FOR LOOP ENTRY
    // =============================================
    int loopOK = 1;
    if(0 /* error condition */) {
        processinfo_error(processinfo, "ERROR: no WFS reference");
        return RETURN_FAILURE;
    }

    // ==================================
    // STARTING LOOP
    // ==================================
    processinfo_loopstart(processinfo); // Notify processinfo that we are entering loop

    while(loopOK==1) {
	    loopOK = processinfo_loopstep(processinfo);

        // Semaphore wait goes here

        processinfo_exec_start(processinfo);
        if(processinfo_compute_status(processinfo)==1) {
            // computation ....
        }

        // Post semaphore(s) and counter(s)
        // process signals, increment loop counter
        processinfo_exec_end(processinfo);

        // OPTIONAL: MESSAGE WHILE LOOP RUNNING
        processinfo_WriteMessage(processinfo, "loop running fine");
    }

    // ==================================
    // ENDING LOOP
    // ==================================
    processinfo_cleanExit(processinfo);

    return RETURN_SUCCESS;
}
```

## 6. Important PROCESSINFO Struct Fields

Beyond the fields set automatically by the framework, these commonly-used fields can be configured
by the developer:

| Field               | Type        | Description                              |
| ------------------- | ----------- | ---------------------------------------- |
| `name`              | `char[80]`  | Short process name (no spaces, no dots)  |
| `description`       | `char[200]` | Human-readable description string        |
| `unitname`          | `char[100]` | Unit the process runs (`MILK_UNIT_NAME`) |
| `MeasureTiming`     | `int`       | Enable loop timing measurement           |
| `RT_priority`       | `int`       | RT priority 0–99 (higher = more urgent)  |
| `loopcntMax`        | `long`      | Max loop iterations (−1 = infinite)      |
| `CPUmask`           | `cpu_set_t` | CPU affinity mask for core pinning       |
| `triggermode`       | `int`       | Trigger type (semaphore, cnt, delay)     |
| `triggerstreamname` | `char[80]`  | Name of input trigger stream             |
| `triggersem`        | `int`       | Semaphore index for trigger              |

## 7. Compute Function Macros

For `MILK_RUNNABLE` compute units, `milk_runnable.h` provides macros that wrap the `processinfo` API
for a standard compute loop; `self` is the `const MILK_RUNNABLE *` argument of the compute function:

| Macro                          | Purpose                                                  |
| ------------------------------ | -------------------------------------------------------- |
| `MILK_PROCINFO_INIT(self)`      | Allocate and register PROCESSINFO, set up signal handler |
| `MILK_PROCINFO_LOOPSTART(self)` | Start the compute loop (opens the loop body), set loopOK |
| `MILK_PROCINFO_END(self)`       | Per-iteration end (timing, counters), closes the loop    |

```c
static errno_t my_compute(const MILK_RUNNABLE *self)
{
    MILK_PROCINFO_INIT(self)
    MILK_PROCINFO_LOOPSTART(self)

    // ... per-iteration computation ...
    processinfo_update_output_stream(processinfo, imgout.im, imgin.im);

    MILK_PROCINFO_END(self)
    return RETURN_SUCCESS;
}
```

### Output Stream Updates

After writing to an output stream, call:

```c
processinfo_update_output_stream(processinfo, img.im, NULL);
```

Pass the triggering input `IMAGE *` as the third argument (or `NULL` if there is no upstream
trigger stream) to propagate stream process trace ancestry. This posts semaphores, increments
`cnt0`/`cnt1`, clears the `write` flag, and updates timing metadata. Always set
`img.im->md->write = 1` before modifying pixels.

### Input Stream Triggers

To wait on an input stream:

```c
processinfo_waitoninputstream_init(
    processinfo, imgin.im, PROCESSINFO_TRIGGERMODE_SEMAPHORE, -1);
// ... inside loop:
processinfo_waitoninputstream(processinfo);
```

---

← [Documentation Index](../index.md)
