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
#include <stdio.h>

#include "milk_calc_ext.h"

typedef void (*cli_calc_ext_param_cb_t)(const char                   *label,
                                        const milk_calc_param_spec_t *spec,
                                        int                           is_op,
                                        void                         *ud);

typedef void (*cli_calc_ext_scalar_out_cb_t)(const char *var_name, const char *descr, void *ud);

int                          cli_calc_ext_load(const char *path, char *err, size_t errsz);
int                          cli_calc_ext_load_list(const char *libs_str, char *err, size_t errsz);
int                          cli_calc_ext_count(void);
const milk_calc_ext_entry_t *cli_calc_ext_get(int idx);
void                         cli_calc_ext_scan_source(const char                  *src,
                                                      cli_calc_ext_param_cb_t      param_cb,
                                                      cli_calc_ext_scalar_out_cb_t scalar_out_cb,
                                                      void                        *ud);

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

/** cli_calc_prog_run() - set the bound variables, run the statements, copy back the assigned ones; 0 or -1.
 * Each output image is posted once, after the last statement that wrote it. */
int cli_calc_prog_run(cli_calc_prog *p, char *err, size_t errsz);

/** cli_calc_torn_count() - kept passes re-run, or run on a frame still being written, so far (`_torn`). */
uint64_t cli_calc_torn_count(void);

/** cli_calc_prog_set_processinfo() - the unit's PROCESSINFO (or NULL): posted frames then carry its trigger trace. */
void cli_calc_prog_set_processinfo(cli_calc_prog *p, void *processinfo);

/**
 * cli_calc_prog_exec_once() - execute @text once (for .init / .initfile): statements ended by `;` or
 * a newline, `#` to the end of the line a comment. Any new image target becomes a shared stream, and
 * assigned bound variables are copied back and marked in @assigned_out (if non-NULL); 0 or -1.
 */
int cli_calc_prog_exec_once(const char             *text,
                            const cli_calc_binding *vars,
                            int                     nvars,
                            unsigned char          *assigned_out,
                            char                   *err,
                            size_t                  errsz);

void cli_calc_prog_free(cli_calc_prog *p);

/** cli_calc_prog_explain() - compile @text as a unit would and print to @out how each statement runs
 * (kept fused pass, merged into which statement, scalar tape, or text with the reason); 0 if it compiled. */
int cli_calc_prog_explain(const char *text, FILE *out);

#endif
