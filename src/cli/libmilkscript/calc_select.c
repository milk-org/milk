// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "calc_select.h"
#include "cli_calc_engines.h"
#include "cli_calc_scalar.h"

#ifndef MILK_BASH_CLIENT
#    include "CLIcore.h"
#    include "CLIcore_script.h"
#    include "cli_calc_internal.h"
#    include "cli_calc_parser.h"
#    include "cli_calc_prog.h"
#    include "mcalc_host.h"
#endif

static int g_engine_sel = -1;

int calc_select_use_new(void)
{
    if (g_engine_sel < 0)
    {
        const char *ev = getenv("MILK_CALC_ENGINE");
        g_engine_sel   = (ev != NULL && strcmp(ev, "old") == 0) ? 0 : 1;
    }
    return g_engine_sel;
}

int calc_select_local(const char *expr, int digits, char *out, size_t outsz, int *truth)
{
    if (calc_select_use_new())
    {
        return mcalc_local(expr, digits, out, outsz, truth);
    }
    return cli_calc_local(expr, digits, out, outsz, truth);
}

int calc_select_print_catalog(const char *pattern, int json_mode)
{
    if (calc_select_use_new())
    {
        return mcalc_print_catalog(pattern, json_mode);
    }
#ifdef MILK_BASH_CLIENT
    return cli_calc_print_engines(pattern, json_mode, NULL, 0);
#else
    return cli_calc_print_engines(pattern, json_mode, cli_calc_ext_get, cli_calc_ext_count());
#endif
}

#ifndef MILK_BASH_CLIENT

static int g_host_bound = 0;

static void ensure_mcalc_host(void)
{
    if (!g_host_bound)
    {
        mcalc_host_ops ops = { cli_var_get, cli_var_set };
        mcalc_set_host(&ops);
        g_host_bound = 1;
    }
    mcalc_out_mode     = cli_calc_out_mode;
    mcalc_float_digits = cli_float_digits;
}

int calc_select_eval_bash(const char *input, int is_test, int *out_rc)
{
    if (calc_select_use_new())
    {
        ensure_mcalc_host();
        int r               = mcalc_eval_bash(input, is_test, out_rc);
        cli_calc_test_image = mcalc_test_image;
        snprintf(cli_calc_lasterr, sizeof(cli_calc_lasterr), "%s", mcalc_lasterr);
        return r;
    }
    return cli_calc_eval_bash(input, is_test, out_rc);
}

int calc_select_eval_str(const char *input, int is_test, int *out_rc, char *out, size_t outsz)
{
    int saved_mode    = cli_calc_out_mode;
    cli_calc_out_mode = CLI_CALC_OUT_VALUES;
    if (calc_select_use_new())
    {
        ensure_mcalc_host();
        int r               = mcalc_eval_str(input, is_test, out_rc, out, outsz);
        cli_calc_test_image = mcalc_test_image;
        snprintf(cli_calc_lasterr, sizeof(cli_calc_lasterr), "%s", mcalc_lasterr);
        cli_calc_out_mode = saved_mode;
        return r;
    }
    int r             = cli_calc_eval_str(input, is_test, out_rc, out, outsz);
    cli_calc_out_mode = saved_mode;
    return r;
}

int calc_select_eval_math_to_val(const char *input, int *out_type, long *out_lval, double *out_dval)
{
    if (calc_select_use_new())
    {
        ensure_mcalc_host();
        return mcalc_eval_math_to_val(input, out_type, out_lval, out_dval);
    }
    return cli_calc_eval_math_to_val(input, out_type, out_lval, out_dval);
}

void calc_select_tmp_cleanup(void)
{
    if (calc_select_use_new())
    {
        mcalc_tmp_cleanup();
        return;
    }
    cli_calc_tmp_cleanup();
}

int calc_select_prog_explain(const char *text, FILE *out)
{
    if (calc_select_use_new())
    {
        ensure_mcalc_host();
        return mcalc_prog_explain(text, out);
    }
    return cli_calc_prog_explain(text, out);
}

int calc_select_ext_load_list(const char *libs_str, char *err, size_t errsz)
{
    if (calc_select_use_new())
    {
        return mcalc_ext_load_list(libs_str, err, errsz);
    }
    return cli_calc_ext_load_list(libs_str, err, errsz);
}

int calc_select_ext_count(void)
{
    return calc_select_use_new() ? mcalc_ext_count() : cli_calc_ext_count();
}

const milk_calc_ext_entry_t *calc_select_ext_get(int idx)
{
    return calc_select_use_new() ? mcalc_ext_get(idx) : cli_calc_ext_get(idx);
}

void calc_select_ext_scan_source(const char               *src,
                                 mcalc_ext_param_cb_t      param_cb,
                                 mcalc_ext_scalar_out_cb_t scalar_out_cb,
                                 void                     *ud)
{
    if (calc_select_use_new())
    {
        mcalc_ext_scan_source(src, param_cb, scalar_out_cb, ud);
        return;
    }
    cli_calc_ext_scan_source(src, param_cb, scalar_out_cb, ud);
}

void calc_select_prog_attach(void)
{
    if (calc_select_use_new())
    {
        mcalc_prog_attach();
        return;
    }
    cli_calc_prog_attach();
}

void *calc_select_prog_new(const char          *text,
                           const mcalc_binding *vars,
                           int                  nvars,
                           char                *err,
                           size_t               errsz)
{
    if (calc_select_use_new())
    {
        ensure_mcalc_host();
        return mcalc_prog_new(text, vars, nvars, err, errsz);
    }
    return cli_calc_prog_new(text, (const cli_calc_binding *) vars, nvars, err, errsz);
}

int calc_select_prog_assigns(const void *p, int i)
{
    if (calc_select_use_new())
    {
        return mcalc_prog_assigns((const mcalc_prog *) p, i);
    }
    return cli_calc_prog_assigns((const cli_calc_prog *) p, i);
}

int calc_select_prog_run(void *p, char *err, size_t errsz)
{
    if (calc_select_use_new())
    {
        return mcalc_prog_run((mcalc_prog *) p, err, errsz);
    }
    return cli_calc_prog_run((cli_calc_prog *) p, err, errsz);
}

uint64_t calc_select_torn_count(void)
{
    return calc_select_use_new() ? mcalc_torn_count() : cli_calc_torn_count();
}

void calc_select_prog_set_processinfo(void *p, void *processinfo)
{
    if (calc_select_use_new())
    {
        mcalc_prog_set_processinfo((mcalc_prog *) p, processinfo);
        return;
    }
    cli_calc_prog_set_processinfo((cli_calc_prog *) p, processinfo);
}

int calc_select_prog_exec_once(const char          *text,
                               const mcalc_binding *vars,
                               int                  nvars,
                               unsigned char       *assigned_out,
                               char                *err,
                               size_t               errsz)
{
    if (calc_select_use_new())
    {
        ensure_mcalc_host();
        return mcalc_prog_exec_once(text, vars, nvars, assigned_out, err, errsz);
    }
    return cli_calc_prog_exec_once(text, (const cli_calc_binding *) vars, nvars, assigned_out, err,
                                   errsz);
}

void calc_select_prog_free(void *p)
{
    if (calc_select_use_new())
    {
        mcalc_prog_free((mcalc_prog *) p);
        return;
    }
    cli_calc_prog_free((cli_calc_prog *) p);
}

#endif /* !MILK_BASH_CLIENT */
