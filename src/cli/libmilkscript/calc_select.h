// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef MILK_CALC_SELECT_H
#define MILK_CALC_SELECT_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "mcalc.h"

/** calc_select_use_new() - return 1 when MILK_CALC_ENGINE=new, 0 otherwise (default old). */
int calc_select_use_new(void);

/** calc_select_local() - dispatch local scalar fast path to mcalc_local or cli_calc_local. */
int calc_select_local(const char *expr, int digits, char *out, size_t outsz, int *truth);

/** calc_select_print_catalog() - dispatch function/extension catalog listing. */
int calc_select_print_catalog(const char *pattern, int json_mode);

/** calc_select_eval_bash() - dispatch @calc / @test evaluation. */
int calc_select_eval_bash(const char *input, int is_test, int *out_rc);

/** calc_select_eval_str() - dispatch string-captured calculator evaluation. */
int calc_select_eval_str(const char *input, int is_test, int *out_rc, char *out, size_t outsz);

/** calc_select_eval_math_to_val() - dispatch silent scalar math evaluation. */
int calc_select_eval_math_to_val(const char *input,
                                 int        *out_type,
                                 long       *out_lval,
                                 double     *out_dval);

/** calc_select_tmp_cleanup() - dispatch temporary _tmpcalcN image cleanup. */
void calc_select_tmp_cleanup(void);

/** calc_select_prog_explain() - dispatch program execution plan inspection. */
int calc_select_prog_explain(const char *text, FILE *out);

/** calc_select_ext_load_list() - dispatch C extension list loading. */
int calc_select_ext_load_list(const char *libs_str, char *err, size_t errsz);

/** calc_select_ext_count() - dispatch registered C extension count. */
int calc_select_ext_count(void);

/** calc_select_ext_get() - dispatch registered C extension lookup by index. */
const milk_calc_ext_entry_t *calc_select_ext_get(int idx);

/** calc_select_ext_scan_source() - dispatch C extension unbound parameter scan. */
void calc_select_ext_scan_source(const char               *src,
                                 mcalc_ext_param_cb_t      param_cb,
                                 mcalc_ext_scalar_out_cb_t scalar_out_cb,
                                 void                     *ud);

/** calc_select_prog_attach() - dispatch unit table attachment. */
void calc_select_prog_attach(void);

/** calc_select_prog_new() - dispatch unit calculator program compilation. */
void *calc_select_prog_new(const char          *text,
                           const mcalc_binding *vars,
                           int                  nvars,
                           char                *err,
                           size_t               errsz);

/** calc_select_prog_assigns() - dispatch bound variable assignment query. */
int calc_select_prog_assigns(const void *p, int i);

/** calc_select_prog_run() - dispatch compiled calculator program frame execution. */
int calc_select_prog_run(void *p, char *err, size_t errsz);

/** calc_select_torn_count() - dispatch cumulative torn SHM read count query. */
uint64_t calc_select_torn_count(void);

/** calc_select_prog_set_processinfo() - dispatch unit PROCESSINFO attachment. */
void calc_select_prog_set_processinfo(void *p, void *processinfo);

/** calc_select_prog_exec_once() - dispatch one-shot program execution. */
int calc_select_prog_exec_once(const char          *text,
                               const mcalc_binding *vars,
                               int                  nvars,
                               unsigned char       *assigned_out,
                               char                *err,
                               size_t               errsz);

/** calc_select_prog_free() - dispatch compiled calculator program release. */
void calc_select_prog_free(void *p);

#endif /* MILK_CALC_SELECT_H */
