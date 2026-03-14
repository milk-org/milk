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
    char pinfoname[200];   // short name for the processinfo instance, no spaces, no dot
    sprintf(pinfoname, "aol%ld-acqRM", loop);

    char pinfodescr[200];
    sprintf(pinfodescr, "NBcycle=%ld", NBcycle);

    char pinfomsg[200];
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

---
← [Documentation Index](index.md)
