// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milk_unit_calc.c
 * @brief milk-fpsexec-milk-calc (milk.calc): a compute unit that runs calculator statements on each frame.
 *
 * .unit.expr (or the file .unit.file names) is compiled once and run on each frame of
 * .unit.trigger, else on each tick of the delay loop; a new .unit.expr or .unit.file is compiled
 * at the next frame. .unit.params declares parameters, "name:type[=default][:help] ...", as
 * calculator variables that @set changes live; a statement that assigns one writes it back to the
 * FPS. Names in the statements are shared streams, and a new image target becomes one. An error
 * stops the loop and is kept in .unit.error.
 *
 * On the command line the program is given with unit options (-e -t -p -f -i -I), and the values
 * after the command word set the declared parameters, in declaration order, as for any fpsexec.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "milk_runnable.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include "cli_calc_engines.h"
#include "cli_calc_prog.h"
#include "fps_modulevars_resync.h"
#include "milk_argval.h"
#include "milk_unit.h"

#define CALC_NFIXED 8
#define CALC_ERROR 7 /* index of .unit.error */
#define CALC_MAXPAR 64
#define CALC_NBUILTIN 7
#define CALC_MAXTEXT 65536

static char expr[FUNCTION_PARAMETER_STRMAXLEN];
static char trigger[FUNCTION_PARAMETER_STRMAXLEN];
static char params[FUNCTION_PARAMETER_STRMAXLEN];
static char file[FUNCTION_PARAMETER_STRMAXLEN];
static char init_expr[FUNCTION_PARAMETER_STRMAXLEN];
static char initfile[FUNCTION_PARAMETER_STRMAXLEN];
static char libs[FUNCTION_PARAMETER_STRMAXLEN];
static char error[FUNCTION_PARAMETER_STRMAXLEN];

/* Fixed unit-definition entries under .unit.*, set by unit options (not positional values);
 * user-declared parameters are appended at runtime */
#define CALC_FIXED_PARAMS(X)                                                                      \
    X(".unit.expr", expr, FPTYPE_STRING, 0, FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN,               \
      "statements, ; between them (-e)")                                                          \
    X(".unit.trigger", trigger, FPTYPE_STREAMNAME, 0,                                             \
      FPFLAG_DEFAULT_INPUT | FPFLAG_TRIGGER_STREAM, "a frame of this stream runs them (-t)")      \
    X(".unit.params", params, FPTYPE_STRING, 0, FPFLAG_DEFAULT_INPUT, "declared parameters (-p)") \
    X(".unit.file", file, FPTYPE_STRING, 0, FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN,               \
      "statements from this file (-f)")                                                           \
    X(".unit.init", init_expr, FPTYPE_STRING, 0, FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN,          \
      "statements run once at start (-i)")                                                        \
    X(".unit.initfile", initfile, FPTYPE_STRING, 0, FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN,       \
      "init statements from this file (-I)")                                                      \
    X(".unit.libs", libs, FPTYPE_STRING, 0, FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN,               \
      "compiled C/SO extension libraries (-l)")                                                   \
    X(".unit.error", error, FPTYPE_STRING, 0, FPFLAG_DEFAULT_OUTPUT, "why the loop stopped")

static FPS_CLI_BINDING bindings[CALC_NFIXED + CALC_MAXPAR] = { CALC_FIXED_PARAMS(FPS_X_BINDING) };
static CLICMDARGDEF    calc_farg[]                         = { CALC_FIXED_PARAMS(FPS_X_FARG) };
MILK_RUNNABLE_RT_DEFINE(calc);
static int nbind = CALC_NFIXED;

static union
{
    int64_t  i;
    int32_t  i32;
    uint64_t u;
    uint32_t u32;
    double   f;
    float    f32;
    int32_t  b;
    char     s[FUNCTION_PARAMETER_STRMAXLEN];
} pval[CALC_MAXPAR];
static char             pkey[CALC_MAXPAR][FUNCTION_PARAMETER_KEYWORD_STRMAXLEN + 1];
static char             pdescr[CALC_MAXPAR][FUNCTION_PARAMETER_DESCR_STRMAXLEN];
static cli_calc_binding cvars[CALC_MAXPAR + CALC_NBUILTIN];
static int64_t          frame_n    = 0;
static double           frame_t    = 0.0;
static double           frame_dt   = 0.0;
static int64_t          frame_cnt0 = 0; /* _cnt0: the trigger frame's counter */
static double  frame_ts     = 0.0; /* _ts: the trigger frame's write time (s, CLOCK_REALTIME) */
static int64_t frame_missed = 0;   /* _missed: trigger frames skipped before this one */
static int64_t frame_torn   = 0; /* _torn: kept passes re-run because an input was being written */

static errno_t compute(const MILK_RUNNABLE *self);

static MILK_RUNNABLE unit_calc = {
    .kind = MILK_RUNNABLE_FPS,
    .app  = {
        .fps_name    = "milkcalc",
        .cmdkey      = "calc",
        .description = "run calculator statements on each frame",
        .description_long =
            "Runs calculator statements (-e, or the file -f names) on each frame of the trigger "
            "stream (-t), or once, or on each -loopd tick without one. -i or -I runs statements "
            "once before frame 0. -p declares parameters that the statements read: the values "
            "after the command word set them, in declaration order, and @set changes them while "
            "it runs. Built-in variables: _n (frame index), _t and _dt (seconds), _cnt0, _ts and "
            "_missed (the trigger frame's counter, write time and skipped frames), _torn (passes "
            "re-run because an input stream was being written). Statements: "
            "'if <scalar>: <statement>', 'if <cond> { ... } elif <cond> { ... } else { ... }', "
            "'for k = start:stop[:step] { ... }', 'local name = <expr>', 'def f(x, ...) = <expr>', "
            "'skip', 'post(name)', 'hist(s, N)', 'prev(s[, k])' and 'zeros(nx[, ny[, nz]])'.",
    },
    .srcfile     = __FILE__,
    .bindings    = bindings,
    .nb_bindings = CALC_NFIXED,
    .farg        = calc_farg,
    .nbarg       = (int) (sizeof(calc_farg) / sizeof(calc_farg[0])),
    .rt          = &calc_rt,
    .compute     = compute,
};

/**
 * now_sec - Read the current monotonic clock time in seconds
 *
 * Queries `CLOCK_MONOTONIC` to compute elapsed time `_t` and frame delta `_dt`
 * for calculator expressions without sensitivity to wall-clock adjustments.
 *
 * Return: Monotonic timestamp in floating-point seconds.
 */
static double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double) ts.tv_sec + (double) ts.tv_nsec * 1e-9;
}

static char *source_text(const char *inline_str, const char *fpath, char *err, size_t errsz);

/**
 * is_fixed - Check whether a parameter name collides with built-in variables or reserved subtrees
 * @name: Candidate parameter name (without leading dot)
 *
 * Rejects built-in frame variables (`_n`, `_t`, `_dt`, `_cnt0`, `_ts`, `_missed`, `_torn`)
 * and reserved FPS subtrees (`unit`, `conf`, `procinfo`).
 *
 * Return: 1 if @name is reserved, 0 otherwise.
 */
static int is_fixed(const char *name)
{
    return strcmp(name, "_n") == 0 || strcmp(name, "_t") == 0 || strcmp(name, "_dt") == 0 ||
           strcmp(name, "_cnt0") == 0 || strcmp(name, "_ts") == 0 || strcmp(name, "_missed") == 0 ||
           strcmp(name, "_torn") == 0 || strcmp(name, "unit") == 0 || strcmp(name, "conf") == 0 ||
           strcmp(name, "procinfo") == 0;
}

static int is_bound(const char *name)
{
    int npar = nbind - CALC_NFIXED;
    for (int k = 0; k < npar; k++)
    {
        if (strcmp(cvars[k].name, name) == 0)
        {
            return 1;
        }
    }
    return 0;
}

/** bind() - add declared parameter @name of @type, primary when writable (its value in pval[]
 * stays to be set). */
static int bind(const char *name, uint64_t type, uint64_t fpflag, const char *descr)
{
    int k       = nbind - CALC_NFIXED;
    int primary = (fpflag & FPFLAG_WRITE) ? 1 : 0;

    if (k == CALC_MAXPAR || is_fixed(name))
    {
        fprintf(stderr, "milk.calc: parameter '%s': %s\n", name,
                is_fixed(name) ? "reserved" : "too many");
        return -1;
    }
    memset(&pval[k], 0, sizeof(pval[k]));
    snprintf(pkey[k], sizeof(pkey[k]), ".%s", name);
    snprintf(pdescr[k], sizeof(pdescr[k]), "%s", (descr != NULL) ? descr : "");
    bindings[nbind++] =
        (FPS_CLI_BINDING) { pkey[k], &pval[k], type, primary, fpflag, pdescr[k], -2, -1 };
    cvars[k] = (cli_calc_binding) { pkey[k] + 1, type, &pval[k] };
    return 0;
}

static void on_ext_param(const char *label, const milk_calc_param_spec_t *spec, int is_op, void *ud)
{
    int *rc = (int *) ud;
    if (*rc != 0)
    {
        return;
    }
    char pname[128];
    snprintf(pname, sizeof(pname), "%s.%s", label, spec->name);
    if (is_bound(pname))
    {
        return;
    }
    uint64_t fpflag = spec->fpflags;
    if (fpflag == 0)
    {
        fpflag = FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN;
    }
    else if (!is_op && (fpflag & FPFLAG_DEFAULT_INPUT))
    {
        fpflag |= FPFLAG_WRITERUN;
    }
    if (bind(pname, spec->fptype, fpflag, spec->descr) != 0)
    {
        *rc = -1;
        return;
    }
    int k = nbind - CALC_NFIXED - 1;
    switch (spec->fptype)
    {
    case FPTYPE_INT64:
        pval[k].i = (int64_t) spec->def_num;
        break;
    case FPTYPE_INT32:
        pval[k].i32 = (int32_t) spec->def_num;
        break;
    case FPTYPE_UINT64:
        pval[k].u = (uint64_t) ((spec->def_num > 0.0) ? spec->def_num : 0.0);
        break;
    case FPTYPE_UINT32:
        pval[k].u32 = (uint32_t) ((spec->def_num > 0.0) ? spec->def_num : 0.0);
        break;
    case FPTYPE_FLOAT64:
        pval[k].f = spec->def_num;
        break;
    case FPTYPE_FLOAT32:
        pval[k].f32 = (float) spec->def_num;
        break;
    case FPTYPE_ONOFF:
        pval[k].b = (spec->def_num != 0.0) ? 1 : 0;
        break;
    default:
        pval[k].s[0] = '\0';
        break;
    }
}

static void on_ext_scalar_out(const char *var_name, const char *descr, void *ud)
{
    int *rc = (int *) ud;
    if (*rc != 0 || is_bound(var_name) || is_fixed(var_name))
    {
        return;
    }
    if (bind(var_name, FPTYPE_FLOAT64, FPFLAG_DEFAULT_OUTPUT, descr) != 0)
    {
        *rc = -1;
    }
}

/** declare() - bind the parameters of @text ("name:type[=default][:help]", blank-separated) and of -l. */
static int declare(const char *text)
{
    char *buf = strdup((text != NULL) ? text : "");
    char *save;
    int   rc = (buf == NULL) ? -1 : 0;

    for (char *t = strtok_r(buf, " \t\n", &save); t != NULL && rc == 0;
         t       = strtok_r(NULL, " \t\n", &save))
    {
        milk_argspec_t d;
        CLICMDARGDATA  ad;
        char           spec[FUNCTION_PARAMETER_STRMAXLEN * 4];
        snprintf(spec, sizeof(spec), "%s", t); /* milk_argspec_parse() cuts @t */
        memset(&ad, 0, sizeof(ad));
        if (milk_argspec_parse(t, &d) != 0 || (d.def != NULL && d.type != FPTYPE_STRING &&
                                               milk_argval_parse(d.type, d.def, &ad) != 0))
        {
            fprintf(
                stderr,
                "milk.calc: bad parameter '%s': name:type[=default][:help], type int uint float "
                "bool str stream fits\n",
                spec);
            rc = -1;
        }
        else if ((rc = bind(d.name, d.type, FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN, d.help)) == 0)
        {
            int k = nbind - CALC_NFIXED - 1;
            switch (d.type)
            {
            case FPTYPE_INT64:
                pval[k].i = ad.val.i64;
                break;
            case FPTYPE_UINT64:
                pval[k].u = ad.val.ui64;
                break;
            case FPTYPE_FLOAT64:
                pval[k].f = ad.val.f64;
                break;
            case FPTYPE_ONOFF:
                pval[k].b = (ad.val.i64 != 0);
                break;
            default:
                snprintf(pval[k].s, sizeof(pval[k].s), "%s", (d.def != NULL) ? d.def : "");
            }
        }
    }
    free(buf);

    if (rc == 0 && (libs[0] != '\0' || cli_calc_ext_count() > 0))
    {
        char err[512] = "";
        if (libs[0] != '\0' && cli_calc_ext_load_list(libs, err, sizeof(err)) != 0)
        {
            fprintf(stderr, "milk.calc: %s\n", err);
            return -1;
        }
        char *itext = (init_expr[0] != '\0' || initfile[0] != '\0')
                          ? source_text(init_expr, initfile, err, sizeof(err))
                          : NULL;
        char *etext =
            (expr[0] != '\0' || file[0] != '\0') ? source_text(expr, file, err, sizeof(err)) : NULL;
        if (itext == NULL && etext == NULL && libs[0] != '\0')
        {
            int next = cli_calc_ext_count();
            for (int i = 0; i < next && rc == 0; i++)
            {
                const milk_calc_ext_entry_t *e = cli_calc_ext_get(i);
                const milk_calc_param_set_t *ps =
                    (e != NULL && e->get_params != NULL) ? e->get_params() : NULL;
                if (ps != NULL)
                {
                    for (int p = 0; p < ps->nparams && rc == 0; p++)
                    {
                        on_ext_param(e->name, &ps->specs[p], e->kind == MILK_EXT_OP, &rc);
                    }
                }
            }
        }
        else
        {
            if (itext != NULL)
            {
                cli_calc_ext_scan_source(itext, on_ext_param, on_ext_scalar_out, &rc);
            }
            if (etext != NULL && rc == 0)
            {
                cli_calc_ext_scan_source(etext, on_ext_param, on_ext_scalar_out, &rc);
            }
        }
        free(itext);
        free(etext);
    }
    return rc;
}

/** find_params() - bind the declared parameters of FPS @name (its one- or two-level parameters that are not
 * fixed), with their current values. */
static int find_params(const char *name)
{
    FPS fps;
    int rc = 0;

    if (fps_connect(name, &fps, FPSCONNECT_SIMPLE) == -1)
    {
        return 0;
    }
    long pl = functionparameter_GetParamIndex(&fps, ".unit.libs");
    if (pl >= 0 && fps.parray[pl].val.string[0][0] != '\0')
    {
        char err[512] = "";
        if (cli_calc_ext_load_list(fps.parray[pl].val.string[0], err, sizeof(err)) != 0)
        {
            fprintf(stderr, "milk.calc: %s\n", err);
            fps_disconnect(&fps);
            return -1;
        }
    }
    for (long i = 0; i < fps.md->NBparamMAX && rc == 0; i++)
    {
        const FPS_PARAM *p = &fps.parray[i];
        if ((p->fpflag & FPFLAG_ACTIVE) && (p->keywordlevel == 2 || p->keywordlevel == 3) &&
            !is_fixed(p->keyword[1]) &&
            (p->type == FPTYPE_INT64 || p->type == FPTYPE_INT32 || p->type == FPTYPE_UINT64 ||
             p->type == FPTYPE_UINT32 || p->type == FPTYPE_FLOAT64 || p->type == FPTYPE_FLOAT32 ||
             p->type == FPTYPE_ONOFF || p->type == FPTYPE_STRING || p->type == FPTYPE_STREAMNAME ||
             p->type == FPTYPE_FITSFILENAME || p->type == FPTYPE_FILENAME))
        {
            char bname[128];
            if (p->keywordlevel == 2)
            {
                snprintf(bname, sizeof(bname), "%s", p->keyword[1]);
            }
            else
            {
                snprintf(bname, sizeof(bname), "%s.%s", p->keyword[1], p->keyword[2]);
            }
            rc = bind(bname, p->type, p->fpflag, p->description);
            if (rc == 0)
            {
                sync_fps_to_local(&fps, i, &bindings[nbind - 1]);
            }
        }
    }
    fps_disconnect(&fps);
    return rc;
}

/** source_text() - @fpath's content when @fpath is set, else @inline_str; NULL (@err) if it cannot be read. */
static char *source_text(const char *inline_str, const char *fpath, char *err, size_t errsz)
{
    if (fpath[0] == '\0')
    {
        return strdup(inline_str);
    }
    FILE *fp = fopen(fpath, "r");
    char *t  = (fp != NULL) ? calloc(1, CALC_MAXTEXT) : NULL;
    if (t != NULL && fread(t, 1, CALC_MAXTEXT - 1, fp) == CALC_MAXTEXT - 1)
    {
        free(t);
        t = NULL;
        snprintf(err, errsz, "%s is longer than %d bytes", fpath, CALC_MAXTEXT - 1);
    }
    else if (t == NULL)
    {
        snprintf(err, errsz, "cannot read %s", fpath);
    }
    if (fp != NULL)
    {
        fclose(fp);
    }
    return t;
}

/** report - Print @msg to stderr, set `.unit.error`, and write processinfo message. */
static void report(PROCESSINFO *pi, const char *msg)
{
    fprintf(stderr, "milk.calc %s: %s\n", dcfpsptr->md->name, msg);
    snprintf(error, sizeof(error), "%s", msg);
    functionparameter_SetParamValue_STRING(dcfpsptr, ".unit.error", error);
    if (pi != NULL)
    {
        processinfo_WriteMessage(pi, msg);
    }
}

typedef struct
{
    cli_calc_prog *prog;
    long           pe;
    long           pf;
    long           pi;
    long           pif;
    long           pl;
    uint64_t       built[2];
    uint64_t       init_built[2];
    uint64_t       libs_built;
    uint64_t       stream_built[CALC_MAXPAR];
    int            init_done;
    double         t_start;
    double         t_prev;
    int            npar;
} calc_loop_ctx;

/** run_unit_init_if_needed - Execute `.unit.init` / `.unit.initfile` when first run or modified. */
static int run_unit_init_if_needed(calc_loop_ctx *ctx, char *err, size_t errsz)
{
    uint64_t pi_cnt  = (ctx->pi >= 0) ? dcfpsptr->parray[ctx->pi].value_cnt : 0;
    uint64_t pif_cnt = (ctx->pif >= 0) ? dcfpsptr->parray[ctx->pif].value_cnt : 0;

    if (ctx->init_done && pi_cnt == ctx->init_built[0] && pif_cnt == ctx->init_built[1])
    {
        return 0;
    }

    ctx->t_start       = now_sec();
    ctx->t_prev        = ctx->t_start;
    frame_n            = 0;
    frame_t            = 0.0;
    frame_dt           = 0.0;
    ctx->init_done     = 1;
    ctx->init_built[0] = pi_cnt;
    ctx->init_built[1] = pif_cnt;
    cli_calc_prog_free(ctx->prog);
    ctx->prog = NULL;

    if (init_expr[0] != '\0' || initfile[0] != '\0')
    {
        unsigned char assigned[CALC_MAXPAR + CALC_NBUILTIN] = { 0 };
        char         *itext = source_text(init_expr, initfile, err, errsz);
        if (itext == NULL || cli_calc_prog_exec_once(itext, cvars, ctx->npar + CALC_NBUILTIN,
                                                     assigned, err, errsz) != 0)
        {
            free(itext);
            return -1;
        }
        free(itext);
        for (int k = 0; k < ctx->npar; k++)
        {
            if (assigned[k])
            {
                sync_local_to_fps(dcfpsptr, bindings[CALC_NFIXED + k]._fps_pindex,
                                  &bindings[CALC_NFIXED + k]);
            }
        }
    }
    return 0;
}

/** run_unit_frame_step - Update frame telemetry, recompile if needed, and run one frame. */
static int run_unit_frame_step(PROCESSINFO   *processinfo,
                               calc_loop_ctx *ctx,
                               char          *err,
                               size_t         errsz)
{
    double   t_now          = now_sec();
    int      stream_changed = 0;
    uint64_t pe_cnt         = (ctx->pe >= 0) ? dcfpsptr->parray[ctx->pe].value_cnt : 0;
    uint64_t pf_cnt         = (ctx->pf >= 0) ? dcfpsptr->parray[ctx->pf].value_cnt : 0;

    frame_t      = (frame_n == 0) ? 0.0 : (t_now - ctx->t_start);
    frame_dt     = (frame_n == 0) ? 0.0 : (t_now - ctx->t_prev);
    ctx->t_prev  = t_now;
    frame_cnt0   = frame_n;
    frame_ts     = 0.0;
    frame_missed = 0;
    frame_torn   = (int64_t) cli_calc_torn_count();

    if (processinfo != NULL && processinfo->trigger_image != NULL &&
        processinfo->trigger_image->md != NULL)
    {
        const IMAGE_METADATA *tmd = processinfo->trigger_image->md;
        frame_cnt0                = (int64_t) tmd->cnt0;
        frame_ts     = (double) tmd->writetime.tv_sec + 1e-9 * (double) tmd->writetime.tv_nsec;
        frame_missed = processinfo->triggermissedframe;
    }

    for (int k = 0; k < ctx->npar; k++)
    {
        long pidx = bindings[CALC_NFIXED + k]._fps_pindex;
        if (bindings[CALC_NFIXED + k].type == FPTYPE_STREAMNAME && pidx >= 0 &&
            dcfpsptr->parray[pidx].value_cnt != ctx->stream_built[k])
        {
            stream_changed = 1;
        }
    }
    if (ctx->prog == NULL || stream_changed || pe_cnt != ctx->built[0] || pf_cnt != ctx->built[1])
    {
        char *text = source_text(expr, file, err, errsz);
        cli_calc_prog_free(ctx->prog);
        ctx->prog = (text != NULL)
                        ? cli_calc_prog_new(text, cvars, ctx->npar + CALC_NBUILTIN, err, errsz)
                        : NULL;
        cli_calc_prog_set_processinfo(ctx->prog, processinfo);
        ctx->built[0] = pe_cnt;
        ctx->built[1] = pf_cnt;
        for (int k = 0; k < ctx->npar; k++)
        {
            long pidx = bindings[CALC_NFIXED + k]._fps_pindex;
            if (pidx >= 0)
            {
                ctx->stream_built[k] = dcfpsptr->parray[pidx].value_cnt;
            }
        }
        free(text);
    }
    if (ctx->prog == NULL || cli_calc_prog_run(ctx->prog, err, errsz) != 0)
    {
        return -1;
    }
    for (int k = 0; k < ctx->npar; k++)
    {
        if (cli_calc_prog_assigns(ctx->prog, k))
        {
            sync_local_to_fps(dcfpsptr, bindings[CALC_NFIXED + k]._fps_pindex,
                              &bindings[CALC_NFIXED + k]);
        }
    }
    frame_n++;
    return 0;
}

/**
 * compute - Main processinfo frame loop compiling and executing calculator expressions
 * @self: Pointer to the `MILK_RUNNABLE` descriptor for this compute unit
 *
 * Return: `RETURN_SUCCESS` on clean loop exit, `RETURN_FAILURE` if an expression error occurred.
 */
static MILK_HOT errno_t compute(const MILK_RUNNABLE *self)
{
    char          err[512] = "";
    calc_loop_ctx ctx      = { 0 };
    const char   *tname    = dcfpsptr->cmdset.triggerstreamname;

    ctx.pe   = functionparameter_GetParamIndex(dcfpsptr, ".unit.expr");
    ctx.pf   = functionparameter_GetParamIndex(dcfpsptr, ".unit.file");
    ctx.pi   = functionparameter_GetParamIndex(dcfpsptr, ".unit.init");
    ctx.pif  = functionparameter_GetParamIndex(dcfpsptr, ".unit.initfile");
    ctx.pl   = functionparameter_GetParamIndex(dcfpsptr, ".unit.libs");
    ctx.npar = nbind - CALC_NFIXED;

    cvars[ctx.npar + 0] = (cli_calc_binding) { "_n", FPTYPE_INT64, &frame_n };
    cvars[ctx.npar + 1] = (cli_calc_binding) { "_t", FPTYPE_FLOAT64, &frame_t };
    cvars[ctx.npar + 2] = (cli_calc_binding) { "_dt", FPTYPE_FLOAT64, &frame_dt };
    cvars[ctx.npar + 3] = (cli_calc_binding) { "_cnt0", FPTYPE_INT64, &frame_cnt0 };
    cvars[ctx.npar + 4] = (cli_calc_binding) { "_ts", FPTYPE_FLOAT64, &frame_ts };
    cvars[ctx.npar + 5] = (cli_calc_binding) { "_missed", FPTYPE_INT64, &frame_missed };
    cvars[ctx.npar + 6] = (cli_calc_binding) { "_torn", FPTYPE_INT64, &frame_torn };

    cli_calc_prog_attach();
    error[0] = '\0';
    functionparameter_SetParamValue_STRING(dcfpsptr, ".unit.error", "");
    if (tname[0] != '\0' && image_ID(tname, dcimg, dcnimg) == -1)
    {
        read_sharedmem_image(tname, dcimg, dcnimg); /* the processinfo trigger waits on it */
    }
    processinfo_CatchSignals(); /* Ctrl-C and runstop end the loop through its cleanup */

    MILK_PROCINFO_INIT(self)
    MILK_PROCINFO_LOOPSTART(self)
    {
        if (libs[0] != '\0' &&
            (!ctx.init_done ||
             (ctx.pl >= 0 && dcfpsptr->parray[ctx.pl].value_cnt != ctx.libs_built)))
        {
            ctx.libs_built = (ctx.pl >= 0) ? dcfpsptr->parray[ctx.pl].value_cnt : 0;
            if (cli_calc_ext_load_list(libs, err, sizeof(err)) != 0)
            {
                report(processinfo, err);
                processloopOK = 0;
            }
        }
        if (err[0] == '\0' && run_unit_init_if_needed(&ctx, err, sizeof(err)) != 0)
        {
            report(processinfo, err);
            processloopOK = 0;
        }
        if (err[0] == '\0' && run_unit_frame_step(processinfo, &ctx, err, sizeof(err)) != 0)
        {
            report(processinfo, err);
            processloopOK = 0;
        }
    }
    MILK_PROCINFO_END(self)

    cli_calc_prog_free(ctx.prog);
    return (err[0] != '\0') ? RETURN_FAILURE : RETURN_SUCCESS;
}

/**
 * main - Entry point for the milk-fpsexec-milk-calc standalone compute unit
 * @argc: Command-line argument count
 * @argv: Command-line argument vector
 *
 * The unit options -e -t -p -f -i -I set the unit definition (.unit.expr .unit.trigger
 * .unit.params .unit.file .unit.init .unit.initfile); milk_unit_main() creates or updates the FPS
 * and delegates to `milk_runnable_main()`.
 *
 * Return: 0 on success, 2 on a usage error, non-zero on other errors.
 */
int main(int argc, char **argv)
{
    static const milk_unit_opt opts[] = {
        { "-e", "--expr", 0, 0, "write the statements in a file (-f)" },
        { "-t", "--trigger", 1, 0, NULL },
        { "-p", "--params", 2, 0, NULL },
        { "-f", "--file", 3, 1, NULL },
        { "-i", "--init", 4, 0, "write the statements in a file (-I)" },
        { "-I", "--initfile", 5, 1, NULL },
        { "-l", "--lib", 6, 2, NULL },
    };
    static const milk_unit_desc desc = {
        .tag         = "milk.calc",
        .unit        = &unit_calc,
        .bindings    = bindings,
        .nfixed      = CALC_NFIXED,
        .nbind       = &nbind,
        .opts        = opts,
        .nopt        = (int) (sizeof(opts) / sizeof(opts[0])),
        .params_opt  = 2,
        .declare     = declare,
        .find_params = find_params,
        .help_options =
            "  -e, --expr EXPR       statements, ; between them (.unit.expr)\n"
            "  -t, --trigger STREAM  a frame of STREAM runs them (.unit.trigger)\n"
            "  -p, --params DECL     declare parameters (.unit.params):\n"
            "                        \"name:type[=default][:help] ...\",\n"
            "                        types int uint float bool str stream fits\n"
            "  -f, --file FILE       statements from FILE instead of -e (.unit.file)\n"
            "  -i, --init EXPR       statements run once before frame 0 (.unit.init)\n"
            "  -I, --initfile FILE   init statements from FILE instead of -i (.unit.initfile)\n"
            "  -l, --lib FILE        load user C (.c) or shared object (.so) extension "
            "(.unit.libs)\n"
            "      --list [PATTERN]  list pre-compiled bytecode engines, tokens & C extensions\n"
            "  fpsinit and exec create the FPS from them; on an existing FPS, -e -t -f -i -I -l\n"
            "  update .unit.* (a running unit recompiles at its next frame) and -p cannot\n"
            "  change. Why the loop stopped is recorded in .unit.error.\n"
            "  Example:\n"
            "    milk-fpsexec-milk-calc -n dsub -t cam -p \"gain:float=1 off:float=0\" \\\n"
            "        -e \"out = cam * gain + off\" exec 2.5 .off=-1\n",
    };

    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "--list") == 0 || strcmp(argv[i], "--engines") == 0 ||
            strcmp(argv[i], "--tokens") == 0)
        {
            int         json_mode = 0;
            const char *pat       = NULL;
            for (int j = 1; j < argc; j++)
            {
                if (j == i || strcmp(argv[j], "--list") == 0 || strcmp(argv[j], "--engines") == 0 ||
                    strcmp(argv[j], "--tokens") == 0)
                {
                    continue;
                }
                if (strcmp(argv[j], "--json") == 0)
                {
                    json_mode = 1;
                }
                else if (j + 1 < argc &&
                         (strcmp(argv[j], "-l") == 0 || strcmp(argv[j], "--lib") == 0))
                {
                    char err[512] = "";
                    if (cli_calc_ext_load_list(argv[++j], err, sizeof(err)) != 0)
                    {
                        fprintf(stderr, "milk.calc: %s\n", err);
                        return 1;
                    }
                }
                else if (pat == NULL && argv[j][0] != '-')
                {
                    pat = argv[j];
                }
            }
            return cli_calc_print_engines(pat, json_mode, cli_calc_ext_get, cli_calc_ext_count());
        }
    }

    return milk_unit_main(argc, argv, &desc);
}
