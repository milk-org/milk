---
tags:
  - processinfo
  - telemetry
  - profiling
---

# Process Info (`procinfo`)

The `procinfo` (Process Information) system is a critical
telemetry and heartbeat monitoring layer within `milk`. It
works symbiotically with the Function Processing System
(FPS) to ensure health checks, profiling, and state
visibility across the entire execution environment.

See also: [FPS](fps.md) ·
[Streams](streams.md) ·
[Programmer's Guide](programmers_guide.md)

## 1. Design and Purpose

Instead of blindly assuming processes are running simply
because their process IDs exist, `procinfo` maintains
active heartbeats and state markers in shared memory.
This is particularly crucial for real-time control
applications (like Adaptive Optics) where a process might
"hang" mathematically without crashing the executable.

## 2. The Output: `milk-procinfo-list`

The primary user-facing tool for this system is the
`milk-procinfo-list` command. This tool scans the system
and provides a visual overview of all processes that have
registered themselves with `procinfo`.

### 2.1. State Tracking

A correctly implemented compute unit will constantly
update its state so tools like `milk-procinfo-list` can
display:

- **IDLE / WAITING:** The loop is paused or blocking on
  a stream semaphore waiting for new data.
- **ACTIVE / RUNNING:** The compute loop is actively
  processing data. PIDs are highlighted green to indicate
  healthy processing.
- **FAILURE / ERROR:** If processing errors occur or the
  heartbeat stops, monitoring tools can flag the process
  for restart.

## 3. Loop Profiling

Along with boolean states, `procinfo` actively measures
loop execution frequencies. It can display the **Hz**
(loops per second) for active pipelines. This means
performance drops or bottlenecks in stream processing are
immediately visible from the top-level dashboard.

## 4. Standalone Executable Integration (`fpsexec`)

When utilizing the standard V2 templates for standalone
modules (`FPS_MAIN_STANDALONE_V2`), `procinfo` is
automatically enabled and registered for you. Your
executable simply needs to correctly execute its while
loop, and the framework pulses the heartbeat for you.

## 5. C Code Template

For custom usage without the standard V2 template, here is how you initialize and use `PROCESSINFO`:

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

Beyond the fields set automatically by the framework,
these commonly-used fields can be configured by the
developer:

| Field                 | Type           | Description                              |
| --------------------- | -------------- | ---------------------------------------- |
| `name`                | `char[80]`     | Short process name (no spaces, no dots)  |
| `description`         | `char[200]`    | Human-readable description string        |
| `tmuxname`            | `char[100]`    | Name of the tmux session hosting process |
| `MeasureTiming`       | `int`          | Enable loop timing measurement           |
| `RT_priority`         | `int`          | RT priority 0–99 (higher = more urgent)  |
| `loopcntMax`          | `int64_t`      | Max loop iterations (−1 = infinite)      |
| `CPUmask`             | `cpu_set_t`    | CPU affinity mask for core pinning       |
| `triggermode`         | `int`          | Trigger type (semaphore, cnt, delay)     |
| `triggerstreamname`   | `char[80]`     | Name of input trigger stream             |
| `triggersem`          | `int`          | Semaphore index for trigger              |

## 7. Compute Function Macros

For V2 fpsexec compute units, the FPS framework
provides macros (defined in `fps_procinfo_macros.h`)
that wrap the `processinfo` API for a standard
compute loop:

| Macro                                      | Purpose                                                  |
| ------------------------------------------ | -------------------------------------------------------- |
| `INSERT_STD_PROCINFO_COMPUTEFUNC_INIT`     | Allocate and register PROCESSINFO, set up signal handler |
| `INSERT_STD_PROCINFO_COMPUTEFUNC_LOOPSTART`| Start the compute loop, set loopOK flag                  |
| `INSERT_STD_PROCINFO_COMPUTEFUNC_START`    | Per-iteration start (check signals, processinfo step)    |
| `INSERT_STD_PROCINFO_COMPUTEFUNC_END`      | Per-iteration end (timing, counters, signal handling)    |

These macros are used inside `compute_function()`
(Section 6 of the V2 template):

```c
static errno_t compute_function()
{
    INSERT_STD_PROCINFO_COMPUTEFUNC_INIT
    INSERT_STD_PROCINFO_COMPUTEFUNC_LOOPSTART
    {
        INSERT_STD_PROCINFO_COMPUTEFUNC_START
        // ... per-iteration computation ...
        processinfo_update_output_stream(
            processinfo, imgout.im);
        INSERT_STD_PROCINFO_COMPUTEFUNC_END
    }
    INSERT_STD_PROCINFO_COMPUTEFUNC_LOOPEND
}
```

### Output Stream Updates

After writing to an output stream, call:

```c
processinfo_update_output_stream(processinfo, img.im);
```

This posts semaphores, increments `cnt0`/`cnt1`,
clears the `write` flag, and updates timing metadata.
Always set `img.im->md->write = 1` before modifying
pixels.

### Input Stream Triggers

To wait on an input stream:

```c
processinfo_waitoninputstream_init(
    processinfo, imgin.im, TRIGGERMODE_SEMAPHORE, -1);
// ... inside loop:
processinfo_waitoninputstream(processinfo);
```

---

← [Documentation Index](index.md)
