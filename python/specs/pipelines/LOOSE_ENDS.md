# Loose ends

Deliberately unfinished items: one line each, with the slice that should pick it up. Sweep this
list at each slice boundary.

- [R4] B6: detect a busy tmux session in `StartConfProcesses.can()`. R1c only hints at it in the
  timeout error. Teardown owns tmux in R4.
- [R4?] `Pipeline.compute_units` rebuilds the views (and remaps every FPS) on each access. Cache
  them only if profiling shows a cost.
