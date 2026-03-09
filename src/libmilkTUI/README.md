# libmilkTUI

Terminal User Interface (TUI) library for `milk`.

## Purpose

Provides ncurses-based terminal rendering utilities shared by
`milk-fpsCTRL`, `milk-procCTRL`, and `milk-streamCTRL`.

## Dependencies

- `ImageStreamIO` — Stream data display
- `ncurses` — Terminal rendering

## Key Features

- Color management and screen printing helpers
- Keyboard input handling (non-blocking)
- Screen layout and formatting utilities

## Notes

Only built when `USE_CLI=ON` (default). Standalone executables
do not use this library.
