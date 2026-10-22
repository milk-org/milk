// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milk_unit_calc.c
 * @brief milk-fpsexec-milk-calc (milk.calc): a compute unit that runs calculator statements on each frame.
 *
 * .expr (or the file .file names) is compiled once and run on each frame of .trigger, else on each tick of
 * the delay loop; a new .expr or .file is compiled at the next frame. .params declares more parameters,
 * "name:type[=default][:help] ...", as calculator variables that @set changes live; a statement that
 * assigns one writes it back to the FPS. Names in the statements are shared streams, and a new image
 * target becomes one. An error stops the loop and is kept in .error.
 *
 * On the command line the program is given with unit options (-e -t -p -f -i -I), and the values after
 * the command word set the declared parameters, in declaration order, as for any fpsexec.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "milk_runnable.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include "cli_calc_prog.h"
#include "fps_modulevars_resync.h"
#include "milk_argval.h"
#include "milk_unit.h"

#define CALC_NFIXED 7
#define CALC_ERROR 6 /* index of .error */
#define CALC_MAXPAR 32
#define CALC_NBUILTIN 3
#define CALC_MAXTEXT 65536

static char expr[FUNCTION_PARAMETER_STRMAXLEN];
static char trigger[FUNCTION_PARAMETER_STRMAXLEN];
static char params[FUNCTION_PARAMETER_STRMAXLEN];
static char file[FUNCTION_PARAMETER_STRMAXLEN];
static char init_expr[FUNCTION_PARAMETER_STRMAXLEN];
static char initfile[FUNCTION_PARAMETER_STRMAXLEN];
static char error[FUNCTION_PARAMETER_STRMAXLEN];

/* Fixed parameters declared with X-macro, set by unit options (not positional values);
 * user-declared parameters are appended at runtime */
#define CALC_FIXED_PARAMS(X)                                                                   \
    X(".expr", expr, FPTYPE_STRING, 0, FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN,                 \
      "statements, ; between them (-e)")                                                       \
    X(".trigger", trigger, FPTYPE_STREAMNAME, 0, FPFLAG_DEFAULT_INPUT | FPFLAG_TRIGGER_STREAM, \
      "a frame of this stream runs them (-t)")                                                 \
    X(".params", params, FPTYPE_STRING, 0, FPFLAG_DEFAULT_INPUT, "declared parameters (-p)")   \
    X(".file", file, FPTYPE_STRING, 0, FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN,                 \
      "statements from this file (-f)")                                                        \
    X(".init", init_expr, FPTYPE_STRING, 0, FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN,            \
      "statements run once at start (-i)")                                                     \
    X(".initfile", initfile, FPTYPE_STRING, 0, FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN,         \
      "init statements from this file (-I)")                                                   \
    X(".error", error, FPTYPE_STRING, 0, FPFLAG_DEFAULT_OUTPUT, "why the loop stopped")

static FPS_CLI_BINDING bindings[CALC_NFIXED + CALC_MAXPAR] = { CALC_FIXED_PARAMS(FPS_X_BINDING) };
static CLICMDARGDEF    calc_farg[]                         = { CALC_FIXED_PARAMS(FPS_X_FARG) };
MILK_RUNNABLE_RT_DEFINE(calc);
static int nbind = CALC_NFIXED;

static union
{
    int64_t  i;
    uint64_t u;
    double   f;
    int32_t  b;
    char     s[FUNCTION_PARAMETER_STRMAXLEN];
} pval[CALC_MAXPAR];
static char             pkey[CALC_MAXPAR][FUNCTION_PARAMETER_KEYWORD_STRMAXLEN + 1];
static char             pdescr[CALC_MAXPAR][FUNCTION_PARAMETER_DESCR_STRMAXLEN];
static cli_calc_binding cvars[CALC_MAXPAR + CALC_NBUILTIN];
static int64_t          frame_n  = 0;
static double           frame_t  = 0.0;
static double           frame_dt = 0.0;

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
            "it runs. Built-in variables: _n (frame index), _t and _dt (seconds).",
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

/**
 * is_fixed - Check whether a parameter name collides with built-in or fixed unit parameters
 * @name: Candidate parameter name (without leading dot)
 *
 * Rejects built-in frame variables (`_n`, `_t`, `_dt`) and the unit's fixed FPS
 * parameters (`.expr`, `.trigger`, `.params`, `.file`, `.init`, `.initfile`, `.error`).
 *
 * Return: 1 if @name is reserved, 0 otherwise.
 */
static int is_fixed(const char *name)
{
    if (strcmp(name, "_n") == 0 || strcmp(name, "_t") == 0 || strcmp(name, "_dt") == 0)
    {
        return 1;
    }
    for (int i = 0; i < CALC_NFIXED; i++)
    {
        if (strcmp(bindings[i].fpskeyword + 1, name) == 0)
        {
            return 1;
        }
    }
    return 0;
}

/** bind() - add declared parameter @name of @type, primary: positional values set it (its value in pval[]
 * stays to be set). */
static int bind(const char *name, uint64_t type, uint64_t fpflag, const char *descr)
{
    int k = nbind - CALC_NFIXED;

    if (k == CALC_MAXPAR || is_fixed(name))
    {
        fprintf(stderr, "milk.calc: parameter '%s': %s\n", name,
                is_fixed(name) ? "reserved" : "too many");
        return -1;
    }
    memset(&pval[k], 0, sizeof(pval[k]));
    snprintf(pkey[k], sizeof(pkey[k]), ".%s", name);
    snprintf(pdescr[k], sizeof(pdescr[k]), "%s", (descr != NULL) ? descr : "");
    bindings[nbind++] = (FPS_CLI_BINDING) { pkey[k], &pval[k], type, 1, fpflag, pdescr[k], -2, -1 };
    cvars[k]          = (cli_calc_binding) { pkey[k] + 1, type, &pval[k] };
    return 0;
}

/** declare() - bind the parameters of @text ("name:type[=default][:help]", blank-separated). */
static int declare(const char *text)
{
    char *buf = strdup(text);
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
    return rc;
}

/** find_params() - bind the declared parameters of FPS @name (its one-level parameters that are not fixed),
 * with their current values. */
static int find_params(const char *name)
{
    FPS fps;
    int rc = 0;

    if (fps_connect(name, &fps, FPSCONNECT_SIMPLE) == -1)
    {
        return 0;
    }
    for (long i = 0; i < fps.md->NBparamMAX && rc == 0; i++)
    {
        const FPS_PARAM *p = &fps.parray[i];
        if ((p->fpflag & FPFLAG_ACTIVE) && p->keywordlevel == 2 && !is_fixed(p->keyword[1]) &&
            (p->type == FPTYPE_INT64 || p->type == FPTYPE_UINT64 || p->type == FPTYPE_FLOAT64 ||
             p->type == FPTYPE_ONOFF || p->type == FPTYPE_STRING || p->type == FPTYPE_STREAMNAME ||
             p->type == FPTYPE_FITSFILENAME))
        {
            rc = bind(p->keyword[1], p->type, p->fpflag, p->description);
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

/**
 * report - Record a runtime error message to stderr, the FPS `.error` parameter, and processinfo
 * @pi:  Active `PROCESSINFO` handle (may be NULL)
 * @msg: Error message string
 *
 * Prints @msg to `stderr`, stores it in the `.error` output parameter of the unit's
 * FPS, and writes it to the processinfo status message when @pi is active.
 *
 * Return: void
 */
static void report(PROCESSINFO *pi, const char *msg)
{
    fprintf(stderr, "milk.calc %s: %s\n", dcfpsptr->md->name, msg);
    snprintf(error, sizeof(error), "%s", msg);
    functionparameter_SetParamValue_STRING(dcfpsptr, ".error", error);
    if (pi != NULL)
    {
        processinfo_WriteMessage(pi, msg);
    }
}

/**
 * compute - Main processinfo frame loop compiling and executing calculator expressions
 * @self: Pointer to the `MILK_RUNNABLE` descriptor for this compute unit
 *
 * Runs `.init` / `.initfile` statements once before frame 0 (and whenever modified),
 * recompiles `.expr` / `.file` bytecode when changed, executes the compiled program
 * on each trigger frame or delay tick, and syncs any assigned user parameters back to
 * the FPS.
 *
 * Return: `RETURN_SUCCESS` on clean loop exit, `RETURN_FAILURE` if an expression error occurred.
 */
static MILK_HOT errno_t compute(const MILK_RUNNABLE *self)
{
    char           err[512]      = "";
    cli_calc_prog *prog          = NULL;
    long           pe            = functionparameter_GetParamIndex(dcfpsptr, ".expr");
    long           pf            = functionparameter_GetParamIndex(dcfpsptr, ".file");
    long           pi            = functionparameter_GetParamIndex(dcfpsptr, ".init");
    long           pif           = functionparameter_GetParamIndex(dcfpsptr, ".initfile");
    uint64_t       built[2]      = { 0, 0 }; /* value_cnt of .expr and .file compiled */
    uint64_t       init_built[2] = { 0, 0 }; /* value_cnt of .init and .initfile run */
    uint64_t       stream_built[CALC_MAXPAR] = { 0 };
    int            init_done                 = 0;
    double         t_start                   = 0.0;
    double         t_prev                    = 0.0;
    int            npar                      = nbind - CALC_NFIXED;
    const char    *tname                     = dcfpsptr->cmdset.triggerstreamname;

    cvars[npar + 0] = (cli_calc_binding) { "_n", FPTYPE_INT64, &frame_n };
    cvars[npar + 1] = (cli_calc_binding) { "_t", FPTYPE_FLOAT64, &frame_t };
    cvars[npar + 2] = (cli_calc_binding) { "_dt", FPTYPE_FLOAT64, &frame_dt };

    cli_calc_prog_attach();
    error[0] = '\0';
    functionparameter_SetParamValue_STRING(dcfpsptr, ".error", "");
    if (tname[0] != '\0' && image_ID(tname, dcimg, dcnimg) == -1)
    {
        read_sharedmem_image(tname, dcimg, dcnimg); /* the processinfo trigger waits on it */
    }
    processinfo_CatchSignals(); /* Ctrl-C and runstop end the loop through its cleanup */

    MILK_PROCINFO_INIT(self)
    MILK_PROCINFO_LOOPSTART(self)
    {
        if (!init_done || dcfpsptr->parray[pi].value_cnt != init_built[0] ||
            dcfpsptr->parray[pif].value_cnt != init_built[1])
        {
            t_start       = now_sec();
            t_prev        = t_start;
            frame_n       = 0;
            frame_t       = 0.0;
            frame_dt      = 0.0;
            init_done     = 1;
            init_built[0] = dcfpsptr->parray[pi].value_cnt;
            init_built[1] = dcfpsptr->parray[pif].value_cnt;
            cli_calc_prog_free(prog);
            prog = NULL;
            if (init_expr[0] != '\0' || initfile[0] != '\0')
            {
                unsigned char assigned[CALC_MAXPAR + CALC_NBUILTIN] = { 0 };
                char         *itext = source_text(init_expr, initfile, err, sizeof(err));
                if (itext == NULL || cli_calc_prog_exec_once(itext, cvars, npar + CALC_NBUILTIN,
                                                             assigned, err, sizeof(err)) != 0)
                {
                    free(itext);
                    report(processinfo, err);
                    processloopOK = 0;
                }
                else
                {
                    free(itext);
                    for (int k = 0; k < npar; k++)
                    {
                        if (assigned[k])
                        {
                            sync_local_to_fps(dcfpsptr, bindings[CALC_NFIXED + k]._fps_pindex,
                                              &bindings[CALC_NFIXED + k]);
                        }
                    }
                }
            }
        }
        if (err[0] == '\0')
        {
            double t_now          = now_sec();
            int    stream_changed = 0;
            frame_t               = (frame_n == 0) ? 0.0 : (t_now - t_start);
            frame_dt              = (frame_n == 0) ? 0.0 : (t_now - t_prev);
            t_prev                = t_now;

            for (int k = 0; k < npar; k++)
            {
                long pidx = bindings[CALC_NFIXED + k]._fps_pindex;
                if (bindings[CALC_NFIXED + k].type == FPTYPE_STREAMNAME && pidx >= 0 &&
                    dcfpsptr->parray[pidx].value_cnt != stream_built[k])
                {
                    stream_changed = 1;
                }
            }
            if (prog == NULL || stream_changed || dcfpsptr->parray[pe].value_cnt != built[0] ||
                dcfpsptr->parray[pf].value_cnt != built[1])
            {
                char *text = source_text(expr, file, err, sizeof(err));
                cli_calc_prog_free(prog);
                prog = (text != NULL)
                           ? cli_calc_prog_new(text, cvars, npar + CALC_NBUILTIN, err, sizeof(err))
                           : NULL;
                built[0] = dcfpsptr->parray[pe].value_cnt;
                built[1] = dcfpsptr->parray[pf].value_cnt;
                for (int k = 0; k < npar; k++)
                {
                    long pidx = bindings[CALC_NFIXED + k]._fps_pindex;
                    if (pidx >= 0)
                    {
                        stream_built[k] = dcfpsptr->parray[pidx].value_cnt;
                    }
                }
                free(text);
            }
            if (prog == NULL || cli_calc_prog_run(prog, err, sizeof(err)) != 0)
            {
                report(processinfo, err);
                processloopOK = 0;
            }
            for (int k = 0; prog != NULL && err[0] == '\0' && k < npar; k++)
            {
                if (cli_calc_prog_assigns(prog,
                                          k)) /* a result: back to the FPS, for @get and others */
                {
                    sync_local_to_fps(dcfpsptr, bindings[CALC_NFIXED + k]._fps_pindex,
                                      &bindings[CALC_NFIXED + k]);
                }
            }
            frame_n++;
        }
    }
    MILK_PROCINFO_END(self)

    cli_calc_prog_free(prog);
    return (err[0] != '\0') ? RETURN_FAILURE : RETURN_SUCCESS;
}

/**
 * main - Entry point for the milk-fpsexec-milk-calc standalone compute unit
 * @argc: Command-line argument count
 * @argv: Command-line argument vector
 *
 * The unit options -e -t -p -f -i -I set the fixed parameters (.expr .trigger .params .file .init
 * .initfile); milk_unit_main() creates or updates the FPS and delegates to `milk_runnable_main()`.
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
    };
    static const milk_unit_desc desc = {
        .tag         = "milk.calc",
        .unit        = &unit_calc,
        .bindings    = bindings,
        .nbind       = &nbind,
        .opts        = opts,
        .nopt        = (int) (sizeof(opts) / sizeof(opts[0])),
        .params_opt  = 2,
        .declare     = declare,
        .find_params = find_params,
        .help_options =
            "  -e, --expr EXPR       statements, ; between them\n"
            "  -t, --trigger STREAM  a frame of STREAM runs them (none: once, or per -loopd tick)\n"
            "  -p, --params DECL     declare parameters: \"name:type[=default][:help] ...\",\n"
            "                        types int uint float bool str stream fits\n"
            "  -f, --file FILE       statements from FILE instead of -e (# starts a comment)\n"
            "  -i, --init EXPR       statements run once before frame 0\n"
            "  -I, --initfile FILE   init statements from FILE instead of -i\n"
            "  fpsinit and exec create the FPS from them; on an existing FPS, -e -t -f -i -I\n"
            "  update it (a running unit recompiles at its next frame) and -p cannot change.\n"
            "  Example:\n"
            "    milk-fpsexec-milk-calc -n dsub -t cam -p \"gain:float=1 off:float=0\" \\\n"
            "        -e \"out = cam * gain + off\" exec 2.5 .off=-1\n",
    };

    return milk_unit_main(argc, argv, &desc);
}
