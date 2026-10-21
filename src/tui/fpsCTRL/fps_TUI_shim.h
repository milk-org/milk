// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file fps_TUI_shim.h
 * @brief Color pair definitions (adjust to match your theme or standard defaults)
 */

#ifndef _FPS_TUI_SHIM_H
#define _FPS_TUI_SHIM_H

#include <ncurses.h>

// Color pair definitions (adjust to match your theme or standard defaults)
#define COLOR_DIRECTORY 4
#define COLOR_OK 2
#define COLOR_ERROR 1
#define COLOR_WARNING 3
#define COLOR_BLACK_ON_WHITE 7

// Shim macros for screenprint functions mapping to ncurses
#define screenprint_setcolor(c) attron(COLOR_PAIR(c))
#define screenprint_unsetcolor(c) attroff(COLOR_PAIR(c))

#define screenprint_setbold() attron(A_BOLD)
#define screenprint_unsetbold() attroff(A_BOLD)

#define screenprint_setreverse() attron(A_REVERSE)
#define screenprint_unsetreverse() attroff(A_REVERSE)

#define screenprint_setblink() attron(A_BLINK)
#define screenprint_unsetblink() attroff(A_BLINK)

#define screenprint_setdim() attron(A_DIM)
#define screenprint_unsetdim() attroff(A_DIM)

#define screenprint_setnormal() attrset(A_NORMAL)

#define TUI_printfw printw

int get_singlechar_nonblock();
int get_singlechar_block();

#endif
