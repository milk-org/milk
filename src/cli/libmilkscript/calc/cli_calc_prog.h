// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file cli_calc_prog.h
 * @brief Calculator programs: statements compiled once and run on each frame (milk.calc units).
 */

#ifndef CLI_CALC_PROG_H
#define CLI_CALC_PROG_H

#include <stddef.h>
#include <stdint.h>

/* A calculator variable bound to a value outside the calculator (a unit's FPS parameter) */
typedef struct
{
    const char *name; /* kept by the caller while the program exists */
    uint64_t    type; /* FPTYPE_INT64 UINT64 FLOAT64 ONOFF (int32_t) or STRING (char[64]) */
    void       *ptr;
} cli_calc_binding;

typedef struct cli_calc_prog cli_calc_prog;

/** cli_calc_prog_attach() - in an fpsexec process, work on the runtime's image and variable tables. */
void cli_calc_prog_attach(void);

/**
 * cli_calc_prog_new() - compile @text: statements ended by `;` or a newline, `#` to the end of the line a
 * comment. A statement that makes a new image runs once now, and the image becomes a shared stream.
 * NULL, with the reason in @err, on error.
 */
cli_calc_prog *cli_calc_prog_new(const char             *text,
                                 const cli_calc_binding *vars,
                                 int                     nvars,
                                 char                   *err,
                                 size_t                  errsz);

/** cli_calc_prog_assigns() - 1 if a statement assigns bound variable @i. */
int cli_calc_prog_assigns(const cli_calc_prog *p, int i);

/** cli_calc_prog_run() - set the bound variables, run the statements, copy back the assigned ones; 0 or -1. */
int cli_calc_prog_run(cli_calc_prog *p, char *err, size_t errsz);

void cli_calc_prog_free(cli_calc_prog *p);

#endif
