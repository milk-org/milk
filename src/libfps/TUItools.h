#ifndef TUI_TOOLS_H
#define TUI_TOOLS_H

#include <stdio.h>

// Stubs for TUI tools to allow compilation without TUI library
#define screenprint_setcolor(c)
#define TUI_printfw(...) printf(__VA_ARGS__)
#define TUI_newline() printf("\n")

// Simple ANSI color codes for terminal output
#define AECBOLDHIGREEN "\033[1;32m"
#define AECNORMAL      "\033[0m"

#endif
