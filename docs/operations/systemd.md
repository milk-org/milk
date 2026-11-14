# Units as systemd Services

A unit can run as a systemd service. Nothing in milk needs systemd: without it, units start and
stop as described in [FPS](../guide/fps.md#3-running-units). Where systemd runs, a service adds an
exact exit status, restarts, a watchdog, ordering between units, and CPU placement before the
process starts.

See also: [FPS](../guide/fps.md) · [Unit Lifecycle](../arch/unit_lifecycle.md) ·
[Performance Tuning](performance.md)

---

## 1. Quick Start (User Units)

`share/milk/systemd/milk-unit@.service` is a template for any unit:

```bash
milk-fpsexec-milk-calc -n dsub -t cam -e "out = cam - dark" fpsinit   # make the unit
mkdir -p ~/.config/systemd/user
cp $MILK_INSTALLDIR/share/milk/systemd/milk-unit@.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user start milk-unit@dsub     # returns once the loop runs
systemctl --user status milk-unit@dsub    # Status: "running, 99.8 Hz"
milk-fps-runstop dsub                     # stops the service: milk goes through systemctl now
milk-fps-runstart dsub                    # starts it again
```

The template names the shm directory of the build (`MILK_UNIT_SHM_DIR`, by default the
`MILK_SHM_DIR` of the shell that configured it, else `/milk/shm`). For another directory, replace
both paths in a drop-in (`systemctl --user edit milk-unit@.service`):

```ini
[Service]
EnvironmentFile=
EnvironmentFile=/other/shm/%i.fps.d/env
StandardOutput=append:/other/shm/%i.fps.d/run.out
```

## 2. What the Template Does

| Setting | Why |
| --- | --- |
| `Type=notify` | `systemctl start` returns once the unit's loop runs (`READY=1`), or fails with the worker's reason |
| `ExecStart=milk-fps-run %i` | checks the unit (an fpsexec's FPS, owned by the service's user, not running), then runs its fpsexec as `NAME:runstart --fg` in its work directory |
| `EnvironmentFile=<shm>/%i.fps.d/env` | the environment of the shell that ran `fpsinit`, as for any start (`milk-fps-info --env NAME`) |
| `StandardOutput=append:<shm>/%i.fps.d/run.out` | the output goes where `milk-fps-log`, fpsCTRL's `Ctrl+a` and milk-CTRL's OUTPUT read it; it is rotated as at a milk start. systemd's own lines stay in the journal |
| `Restart=on-failure` | a crash restarts the unit; a stop does not |
| `TimeoutStopSec=5` | systemd's stop is SIGTERM, which ends the loop cleanly, then SIGKILL |
| `#WatchdogSec=5` | see [Watchdog](#5-watchdog) |

A unit run by `milk-fps-run` tells the service manager what it tells a milk starter: `STATUS=`
while it connects, `READY=1` once its loop runs, `ERRNO=` and `STATUS=` when it cannot start, the
loop rate in `STATUS=` about once a second, and `STOPPING=1` when it exits. `milk-fps-run NAME conf`
runs the conf process instead; it needs a template of its own.

The template uses no `PrivateTmp=`, `PrivateDevices=` or `ProtectSystem=strict`: they hide the
shm directory from the unit.

## 3. One Controller per Unit

The first time systemd runs a unit, the unit records the service in `<shm>/NAME.fps.d/systemd`.
From then on:

- `milk-fps-runstart` and `milk-fps-runstop`, `@set f.NAME.run=start|stop`, fpsCTRL and
  milk-CTRL run `systemctl [--user] start|stop` on that service instead of starting or signalling
  the process themselves; a refusal (a system unit and no polkit rule) is printed as is.
- `milk-fps-info NAME` shows `run by <service> (user|system)`.
- `milk-fps-run` refuses to start a unit that milk already runs, and milk refuses to start a unit
  the service runs.

`milk-fps-run --release NAME` deletes the record: milk starts and stops the unit itself again.
Do this before removing the service, or milk keeps calling `systemctl` for it.

## 4. User and System Units

A user manager (`systemctl --user`) runs units as you, needs no root, and is enough for most
units. It cannot give real-time scheduling (its `RLIMIT_RTPRIO` is 0) and is not delegated every
cgroup controller.

For real-time loops, install the template in `/etc/systemd/system` and give each unit the drop-in
`share/milk/systemd/milk-unit-rt.conf` (in `/etc/systemd/system/milk-unit@NAME.service.d/`):
`User=` (the owner of the FPS: a unit always runs as its owner), `AllowedCPUs=`, `NUMAPolicy=`,
`CPUSchedulingPolicy=fifo`, `CPUSchedulingPriority=`, `LimitRTPRIO=` and `LimitMEMLOCK=infinity`.

Placement under systemd:

- `.procinfo.cset` is skipped, with a warning: on cgroup v2 moving the process to a cpuset takes it
  out of its service, and systemd loses track of it. Use `AllowedCPUs=` instead.
- `.procinfo.taskset` and `.procinfo.RTprio` still apply (RT priority needs `LimitRTPRIO=`).

## 5. Watchdog

With `WatchdogSec=`, a unit sends `WATCHDOG=1` from its loop every half interval; a loop that stops
iterating (a computation that hangs) is killed and, with `Restart=on-failure`, restarted. An idle
loop waiting for its trigger stream counts too: each timed-out wait is an iteration. Set
`WatchdogSec=` above the trigger timeout (`.procinfo.triggertimeout`, 2 s by default) plus 1 s.
Units whose loop is not a `MILK_PROCINFO` loop send no `WATCHDOG=1`: leave the watchdog off for
them.

## 6. Groups of Units

`share/milk/systemd/milk-loop.target` starts and stops units together: list them in its `Wants=`,
and give each unit a drop-in with `PartOf=milk-loop.target` and `After=` the unit that feeds it.

## 7. A Named Session as a Service

A named session's milk-cli server can run as a user service too: it tells `$NOTIFY_SOCKET`
`READY=1` once it listens, stops cleanly on the manager's SIGTERM, and sends `STOPPING=1` as it
exits. A service gets the user manager's environment, not your shell's: set the milk variables in
the unit. For example, in `~/.config/systemd/user/milk-session@.service`:

```ini
[Service]
Type=notify
ExecStart=/usr/local/milk/bin/milk-cli -q --server @milk-cli-%U-%i
Environment=MILK_SHM_DIR=/milk/shm
```

`systemctl --user start milk-session@NAME` then returns once `@session use NAME` can connect;
`@session end NAME` and `systemctl --user stop milk-session@NAME` both stop it cleanly. The servers
milk starts itself (`@session new`, the session of each shell) never answer a service manager, even
from a shell that has `$NOTIFY_SOCKET`.

## 8. When a Unit Does Not Start

- `systemctl --user status milk-unit@NAME` shows the status line and the last log lines;
  `journalctl --user -u milk-unit@NAME` the rest of systemd's messages.
- `milk-fps-log NAME` shows what the unit printed, `NAME.fps.d/run.info` how it started and ended.
- "Failed to load environment files": the unit has no FPS in the template's shm directory (or
  `fpsinit` was never run).
- "already running": milk started the unit; stop it with `milk-fps-runstop NAME` first.
