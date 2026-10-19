# milk-termview

Standalone terminal viewer for shared-memory streams (links ImageStreamIO and libm only). User
guide: [docs/guide/termview.md](../../../../docs/guide/termview.md).

| File | Description |
|------|-------------|
| milk-termview.c | Entry point: options, `-h` help, then `termview_run()` |
| termview.h | Public options and `termview_run()` |
| termview_internal.h | Shared types (cells, layout, view, state), inline helpers, module interfaces |
| termview.c | Viewer state, actions (`tv_do()`), input dispatch, scheduler loop, flush budget, flow control |
| tv_ui.c | Interface: menubar and dropdown menus, colorbar, status bar, range minibuffer, help box |
| tv_image.c | Layout, view transform, sampling tables, typed samplers, percentiles, image layer |
| tv_flush.c | Diff emitter: minimal ANSI per changed cell, 24-bit/256/16 colours, byte budget |
| tv_input.c | Input decoder: keys, SGR mouse and cursor position reports from raw bytes and arrival times |
| tv_stream.c | Read-only stream mapping, validation, truncation guard, reopen |
| tv_term.c | Raw mode, enter/leave sequences, signals, stderr while the screen is in use |
| tv_cmap.c | Colormaps, scale curves, the LUT, option name tables |
| tests/test_termview_unit.c | Unit tests (decoder, emitter, pipeline, interface, stream lifecycle) |
| tests/termview_pty_test.py | pty tests and benchmarks (`--perf`, `--bench`) |
| tests/tv-test-mkstream.c | Test stream generator (patterns, rates; MILK_SHM_DIR only) |
