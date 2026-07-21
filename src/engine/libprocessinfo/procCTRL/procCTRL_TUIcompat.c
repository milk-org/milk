// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file streamCTRL_TUIcompat.c
 * @brief Global state storage for the TUIcompat frame buffer
 *
 * Defines storage for all externally-declared globals in
 * streamCTRL_TUIcompat.h and streamCTRL_standalone_defs.h.
 */

#include "procCTRL_TUIcompat.h"


/* Frame buffer and cursor state */
char sc_framebuf[SC_FRAMEBUF_SIZE];
int  sc_framebuf_pos = 0;
int  sc_cursor_row   = 1;
int  sc_cursor_col   = 0;
int  sc_term_rows    = 24;
int  sc_term_cols    = 80;

/* Terminal state */
struct termios ansi__orig_termios;
int            ansi__raw_active = 0;

/* Signal flags */
