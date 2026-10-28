# Loose ends

Deliberately unfinished items: one line each, with the slice that should pick it up. Sweep this
list at each slice boundary.

- [R4] B6: detect a busy tmux session in `StartConfProcesses.can()`. R1c only hints at it in the
  timeout error. Teardown owns tmux in R4.
- [R4?] `Pipeline.compute_units` rebuilds the views (and remaps every FPS) on each access. Cache
  them only if profiling shows a cost.
- [later] `tests/cli/cli_hardening_test.py` paces pty/FIFO input with short sleeps (about 0.5 s
  in total). Left alone by R7.
- [R4] A shared `milk` wait primitive (predicate, upper bound) for the RunStart, RunStop and
  ConfStop tasks and StartConfProcesses. `tests/conftestaux/polling.py` should then delegate
  to it.
