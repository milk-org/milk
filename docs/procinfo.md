# Process Info (`procinfo`)

The `procinfo` (Process Information) system is a critical telemetry and heartbeat monitoring layer within `milk`. It works symbiotically with the Function Processing System (FPS) to ensure health checks, profiling, and state visibility across the entire execution environment.

## Design and Purpose

Instead of blindly assuming processes are running simply because their process IDs exist, `procinfo` maintains active heartbeats and state markers in shared memory. This is particularly crucial for real-time control applications (like Adaptive Optics) where a process might "hang" mathematically without crashing the executable.

## The Output: `milk-procinfo-list`

The primary user-facing tool for this system is the `milk-procinfo-list` command. This tool scans the system and provides a visual overview of all processes that have registered themselves with `procinfo`. 

### State Tracking
A correctly implemented compute unit will constantly update its state so tools like `milk-procinfo-list` can display:
- **IDLE / WAITING:** The loop is paused or blocking on a stream semaphore waiting for new data.
- **ACTIVE / RUNNING:** The compute loop is dynamically churning data. PIDs are often highlighted green to indicate healthy processing.
- **FAILURE / ERROR:** If processing errors occur or the heartbeat stops arbitrarily, monitoring tools can flag the process for restart.

## Loop Profiling
Along with boolean states, `procinfo` actively measures loop execution frequencies. It can display the **Hz** (loops per second) for active pipelines. This means performance drops or bottlenecks in stream processing are immediately visible from the top-level dashboard.

## Standalone Executable Integration (`fpsexec`)
When utilizing the standard V2 templates for standalone modules (`FPS_MAIN_STANDALONE_V2`), `procinfo` is automatically enabled and registered for you. Your executable simply needs to correctly execute its while loop, and the framework under-the-hood pulses the heartbeat for you.
