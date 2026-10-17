// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file CLIcore_checkargs.c
 *
 * @brief Command line argument validation and type checking
 *
 * Architecture Overview:
 * This file verifies that the arguments provided to a CLI command match the
 * expected parameters defined by the module's `FPS_CMDDEF` table. It ensures
 * type safety for CLI interactions by checking scalars, arrays, FPS parameter
 * formats, existing files, and shared memory streams before the underlying
 * C function is invoked.
 */

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "CLIcore.h"

#include "COREMOD_memory/COREMOD_memory.h"
#include "fps_globals.h"
#include "milk_cmd_args.h"

// keep processing if 1
static int argcheck_process_flag = 1;


// toggles to 1 if function help called


/**
 * @brief Convert a CLI argument type enum to string.
 */
static const char *CLIargtype_to_string(uint32_t type)
{
    switch (type)
    {
    case FPTYPE_FLOAT32:
        return "FLOAT32";
    case FPTYPE_FLOAT64:
        return "FLOAT64";
    case FPTYPE_ONOFF:
        return "ONOFF";
    case FPTYPE_INT32:
        return "INT32";
    case FPTYPE_UINT32:
        return "UINT32";
    case FPTYPE_INT64:
        return "INT64";
    case FPTYPE_UINT64:
        return "UINT64";
    case FPTYPE_STRING_NOT_STREAM:
        return "STR_NOT_IMG";
    case FPTYPE_STREAMNAME:
        return "STREAM";
    case FPTYPE_STRING:
        return "STRING";
    case FPTYPE_FILENAME:
        return "FILENAME";
    case FPTYPE_FITSFILENAME:
        return "FITSFILE";
    case FPTYPE_FPSNAME:
        return "FPSNAME";
    case FPTYPE_EXECFILENAME:
        return "EXECFILE";
    case FPTYPE_DIRNAME:
        return "DIRNAME";
    case FPTYPE_PID:
        return "PID";
    case FPTYPE_TIMESPEC:
        return "TIMESPEC";
    case CLIARG_MISSING:
        return "MISSING";
    default:
        return "UNKNOWN";
    }
}

/**
 * @brief Convert a command argument token type to string.
 */
static const char *CMDARGTOKEN_type_to_string(uint32_t type)
{
    switch (type)
    {
    case 0:
        return "UNSOLVED_TOKEN";
    case 1:
        return "FLOAT_TOKEN";
    case 2:
        return "INT_TOKEN";
    case 3:
        return "STRING_TOKEN";
    case 4:
        return "IMG_TOKEN";
    case 5:
        return "CMD_TOKEN";
    case 6:
        return "RAWSTRING_TOKEN";
    default:
        return CLIargtype_to_string(type);
    }
}


/* raw_number() - read raw-string token @i strictly as a @type number; 0, or 2 if it is not one */
static int raw_number(int i, uint64_t type)
{
    CLICMDARGDATA v;

    v.type = type;
    if (milk_argval_parse(type, data.cmdargtoken[i].val.string, &v) != 0)
    {
        return 2;
    }
    data.cmdargtoken[i].val.numf = (type == FPTYPE_FLOAT64) ? v.val.f64 : (double) v.val.i64;
    data.cmdargtoken[i].val.numl = (type == FPTYPE_FLOAT64) ? (long) v.val.f64 : (long) v.val.i64;
    return 0;
}


/**
 * @brief check that input CLI argument matches required function argument type
 *
 * @param CLIargnum    CLI argument / token index
 * @param funcargtype   function argument type
 * @param errmsg    error message printing flag (1 if printing errors)
 * @return int
 */
static int CLI_checkarg0(int CLIargnum, uint32_t funcargtype, int errmsg)
{
    DEBUG_TRACE_FSTART();

    int      rval  = 2; // Default to 'wrong type'
    uint32_t ftype = funcargtype;
    int      ttype = data.cmdargtoken[CLIargnum].type;

    // Normalization and conversion
    if (ttype == CMDARGTOKEN_TYPE_FLOAT)
    {
        data.cmdargtoken[CLIargnum].val.numl = (long) (data.cmdargtoken[CLIargnum].val.numf + 0.5);
    }
    if (ttype == CMDARGTOKEN_TYPE_LONG)
    {
        data.cmdargtoken[CLIargnum].val.numf = (double) data.cmdargtoken[CLIargnum].val.numl;
    }

    // Special conversion for ONOFF
    if (ftype == FPTYPE_ONOFF)
    {
        if (strcasecmp(data.cmdargtoken[CLIargnum].val.string, "on") == 0)
        {
            data.cmdargtoken[CLIargnum].val.numl = 1;
            data.cmdargtoken[CLIargnum].val.numf = 1.0;
            rval                                 = 0;
        }
        else if (strcasecmp(data.cmdargtoken[CLIargnum].val.string, "off") == 0)
        {
            data.cmdargtoken[CLIargnum].val.numl = 0;
            data.cmdargtoken[CLIargnum].val.numf = 0.0;
            rval                                 = 0;
        }
    }

    // Type matching logic
    if (rval == 2)
    {
        if (ftype == FPTYPE_FLOAT32 || ftype == FPTYPE_FLOAT64)
        {
            if (ttype == CMDARGTOKEN_TYPE_FLOAT || ttype == (CLIARG_FLOAT32 & 0x0000FFFF) ||
                ttype == CMDARGTOKEN_TYPE_LONG || ttype == (CLIARG_INT64 & 0x0000FFFF) ||
                ttype == 6)
            {
                rval = (ttype == 6) ? raw_number(CLIargnum, FPTYPE_FLOAT64) : 0;
            }
        }
        else if (ftype == FPTYPE_INT32 || ftype == FPTYPE_INT64 || ftype == FPTYPE_UINT32 ||
                 ftype == FPTYPE_UINT64 || ftype == FPTYPE_ONOFF || ftype == FPTYPE_PID ||
                 ftype == FPTYPE_TIMESPEC)
        {
            if (ttype == CMDARGTOKEN_TYPE_LONG || ttype == (CLIARG_INT64 & 0x0000FFFF) ||
                ttype == CMDARGTOKEN_TYPE_FLOAT || ttype == (CLIARG_FLOAT32 & 0x0000FFFF) ||
                ttype == 6)
            {
                rval = (ttype == 6) ? raw_number(CLIargnum, FPTYPE_INT64) : 0;
            }
        }
        else if (ftype == FPTYPE_STREAMNAME)
        {
            if (ttype == CMDARGTOKEN_TYPE_EXISTINGIMAGE || ftype == FPTYPE_STREAMNAME ||
                ttype == CMDARGTOKEN_TYPE_STRING || ftype == FPTYPE_STRING || ttype == 6)
            {
                rval = 0;
            }
        }
        else if (ftype == FPTYPE_STRING || ftype == FPTYPE_STRING_NOT_STREAM ||
                 ftype == FPTYPE_FILENAME || ftype == FPTYPE_FITSFILENAME ||
                 ftype == FPTYPE_FPSNAME || ftype == FPTYPE_DIRNAME || ftype == FPTYPE_EXECFILENAME)
        {
            if (ttype == CMDARGTOKEN_TYPE_STRING || ftype == FPTYPE_STRING ||
                ttype == CMDARGTOKEN_TYPE_EXISTINGIMAGE || ftype == FPTYPE_STREAMNAME ||
                ttype == CLIARG_STR_NOT_IMG || ttype == 6)
            {
                rval = 0;
            }
        }
    }

    // Check if it's a variable if not already resolved
    if (rval == 2)
    {
        imageID IDv = variable_ID(data.cmdargtoken[CLIargnum].val.string);
        if (IDv != -1)
        {
            if (ftype == FPTYPE_FLOAT32 || ftype == FPTYPE_FLOAT64)
            {
                data.cmdargtoken[CLIargnum].val.numf = (double) dcvar[IDv].value.f;
                data.cmdargtoken[CLIargnum].val.numl = (long) data.cmdargtoken[CLIargnum].val.numf;
                data.cmdargtoken[CLIargnum].type     = CLIARG_FLOAT64;
                rval                                 = 0;
            }
            else if (ftype == FPTYPE_INT32 || ftype == FPTYPE_INT64 || ftype == FPTYPE_UINT32 ||
                     ftype == FPTYPE_UINT64 || ftype == FPTYPE_ONOFF)
            {
                data.cmdargtoken[CLIargnum].val.numl = (long) dcvar[IDv].value.l;
                data.cmdargtoken[CLIargnum].val.numf =
                    (double) data.cmdargtoken[CLIargnum].val.numl;
                data.cmdargtoken[CLIargnum].type = CLIARG_INT64;
                rval                             = 0;
            }
        }
    }

    // Final result and error reporting
    if (rval == 2)
    {
        if (errmsg == 1)
        {
            printf("\033[1;31mERROR\033[0m"
                   " arg %d: wrong type"
                   " (expected %s, got %s)\n",
                   CLIargnum - 1, CLIargtype_to_string(funcargtype),
                   CMDARGTOKEN_type_to_string(data.cmdargtoken[CLIargnum].type));
        }
        rval = 1;
    }

    DEBUG_TRACE_FEXIT();
    return rval;
}


/**
 * @brief Check that input CLI argument matches required argument type
 *
 * @param CLIargnum
 * @param funcargtype
 * @return int
 */
int CLI_checkarg(int CLIargnum, uint32_t funcargtype)
{
    DEBUG_TRACE_FSTART();

    int rval;

    if (CLIargnum == 1)
    {
        argcheck_process_flag = 1;
    }

    if (argcheck_process_flag == 1)
    {
        rval = CLI_checkarg0(CLIargnum, funcargtype, 1);
    }
    else
    {
        rval = 1;
    }

    DEBUG_TRACE_FEXIT();
    return rval;
}

/**
 * @brief Check that input CLI argument matches required argument type - do not print error message
 *
 * @param CLIargnum
 * @param funcargtype
 * @return int
 */
int CLI_checkarg_noerrmsg(int CLIargnum, uint32_t funcargtype)
{
    DEBUG_TRACE_FSTART();

    int rval;

    if (CLIargnum == 1)
    {
        argcheck_process_flag = 1;
    }

    if (argcheck_process_flag == 1)
    {
        rval = CLI_checkarg0(CLIargnum, funcargtype, 0);
    }
    else
    {
        rval = 1;
    }

    DEBUG_TRACE_FEXIT();
    return rval;
}


/* sync_fps_to_argdata() - read FPS parameter @fp (type @ptype) into argument @ad. */
static void sync_fps_to_argdata(uint32_t ptype, FPS_PARAM *fp, CLICMDARGDATA *ad)
{
    switch (ptype)
    {
    case FPTYPE_FLOAT32:
        ad->val.f32 = fp->val.f32[0];
        break;
    case FPTYPE_FLOAT64:
        ad->val.f64 = fp->val.f64[0];
        break;
    case FPTYPE_INT32:
        ad->val.i32 = fp->val.i32[0];
        break;
    case FPTYPE_UINT32:
        ad->val.ui32 = fp->val.ui32[0];
        break;
    case FPTYPE_INT64:
        ad->val.i64 = fp->val.i64[0];
        break;
    case FPTYPE_UINT64:
        ad->val.ui64 = fp->val.ui64[0];
        break;
    case FPTYPE_ONOFF:
        ad->val.i64 = fp->val.i32[0];
        break;
    case FPTYPE_PID:
        ad->val.i64 = (int64_t) fp->val.pid[0];
        break;
    case FPTYPE_TIMESPEC:
        ad->val.f64 = (double) fp->val.ts[0].tv_sec + (double) fp->val.ts[0].tv_nsec * 1e-9;
        break;
    case FPTYPE_STRING:
    case FPTYPE_STREAMNAME:
    case FPTYPE_DIRNAME:
    case FPTYPE_FILENAME:
    case FPTYPE_FITSFILENAME:
    case FPTYPE_EXECFILENAME:
    case FPTYPE_FPSNAME:
    case FPTYPE_PROCESS:
    case FPTYPE_STRING_NOT_STREAM:
        strncpy(ad->val.s, fp->val.string[0], STRINGMAXLEN_CLICMDARG - 1);
        break;
    }
}

static const char *raw_arg(const milk_call_t *call, int t)
{
    return (call->raw != NULL && t < call->nraw) ? call->raw[t] : data.cmdargtoken[t].val.string;
}

/* is_expression() - @s has an operator or a call (the CLI parser evaluated it: `2*3`, `sqrt(2)`) */
static int is_expression(const char *s)
{
    if (s[0] == '(')
    {
        return 1;
    }
    for (const char *p = s + 1; *p != '\0'; p++)
    {
        if (strchr("+-*/^%()", *p) != NULL)
        {
            return 1;
        }
    }
    return 0;
}

/* token_value() - store positional token @t (@raw as typed) in @ad: a literal, an evaluated
 * expression or a milk variable. 0, or -1 if it is none of them. */
static int token_value(int t, const char *raw, CLICMDARGDATA *ad)
{
    const CMDARGTOKEN *tok = &data.cmdargtoken[t];

    if (!milk_argval_is_numeric(ad->type))
    {
        return milk_argval_parse(ad->type, tok->val.string,
                                 ad); /* a slice may name a temporary image */
    }
    if (milk_argval_parse(ad->type, raw, ad) == 0)
    {
        return 0;
    }
    if (tok->type == CMDARGTOKEN_TYPE_FLOAT && is_expression(raw))
    {
        return milk_argval_set_number(ad->type, tok->val.numf, ad);
    }
    if (tok->type == CMDARGTOKEN_TYPE_LONG && is_expression(raw))
    {
        return milk_argval_set_number(ad->type, (double) tok->val.numl, ad);
    }

    imageID IDv = variable_ID(raw);
    if (IDv != -1 && dcvar[IDv].type != 2)
    {
        return milk_argval_set_number(
            ad->type, (dcvar[IDv].type == 0) ? dcvar[IDv].value.f : (double) dcvar[IDv].value.l,
            ad);
    }
    return -1;
}

static errno_t usage_error(milk_call_t *call, const char *fmt, ...)
    __attribute__((format(printf, 2, 3)));

static errno_t usage_error(milk_call_t *call, const char *fmt, ...)
{
    const CMD *c = &data.cmd[call->cmdi];
    va_list    ap;

    fprintf(stderr, "milk: %s: ", c->key);
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fprintf(stderr, "\nusage: %s %s  (%s -h for help)\n", c->key, c->syntax, c->key);
    call->usage_error = 1;
    return RETURN_CLICHECKARGARRAY_FAILURE;
}

/* store_session_defaults() - `--default`: the values given become session defaults, `.` clears */
static void store_session_defaults(milk_call_t *call, int n)
{
    CLICMDARGDATA *ad  = data.cmd[call->cmdi].argdata;
    int            any = 0;
    char           v[STRINGMAXLEN_CLICMDARG + 32];

    for (int i = 0; i < n; i++)
    {
        int s = call->src[i] & ~MILK_ARGSRC_NAMED;

        if (s == MILK_ARGSRC_DOT)
        {
            milk_args_session_set(call->cmdi, i, NULL);
            any = 1;
        }
        else if (s == MILK_ARGSRC_ARG || s == MILK_ARGSRC_TAG || s == MILK_ARGSRC_PRESET)
        {
            milk_argval_format(&ad[i], v, sizeof(v));
            milk_args_session_set(call->cmdi, i, v);
            any = 1;
        }
    }
    if (!any)
    {
        milk_args_print_call(call->cmdi, 1, stdout);
    }
}

static errno_t resolve_args(milk_call_t *call, CLICMDARGDEF fa[], int n)
{
    CLICMDARGDATA *ad  = data.cmd[call->cmdi].argdata;
    unsigned char *src = call->src;
    int            pos[NB_ARG_MAX];
    int            npos = 0;

    for (int i = 0; i < n; i++)
    {
        src[i] =
            (milk_args_session_get(call->cmdi, i) != NULL) ? MILK_ARGSRC_SESSION : MILK_ARGSRC_DECL;
    }

    /* a call on a shared FPS (cmd:fpsname) starts from the instance's current values */
    if (dcfpsptr != NULL && dcfpsname[0] != '\0' && dcfpsname[0] != '_')
    {
        for (int i = 0; i < n; i++)
        {
            long pindex = functionparameter_GetParamIndex(dcfpsptr, fa[i].fpstag);
            if (pindex != -1)
            {
                sync_fps_to_argdata(dcfpsptr->parray[pindex].type, &dcfpsptr->parray[pindex],
                                    &ad[i]);
                src[i] = MILK_ARGSRC_FPS;
            }
        }
    }

    if (call->load[0] != '\0' && milk_args_preset_load(call) != 0)
    {
        call->usage_error = 1;
        return RETURN_CLICHECKARGARRAY_FAILURE;
    }

    /* .tag=value anywhere before "--"; the other tokens are positional */
    for (int t = 1; t < data.cmdNBarg; t++)
    {
        const char *r = raw_arg(call, t);

        if (t < call->posonly && milk_args_named(r))
        {
            const char *eq = strchr(r, '=');
            int         i  = milk_args_find(call->cmdi, r + 1, (size_t) (eq - r - 1));

            if (i < 0 || i >= n)
            {
                return usage_error(call, "unknown parameter %.*s", (int) (eq - r), r);
            }
            if (src[i] & MILK_ARGSRC_NAMED)
            {
                return usage_error(call, "parameter %.*s given twice", (int) (eq - r), r);
            }
            if (strcmp(eq + 1, ".") == 0)
            {
                src[i] = MILK_ARGSRC_DOT | MILK_ARGSRC_NAMED;
            }
            else if (milk_argval_parse(ad[i].type, eq + 1, &ad[i]) == 0)
            {
                src[i] = MILK_ARGSRC_TAG | MILK_ARGSRC_NAMED;
            }
            else
            {
                return usage_error(call, "invalid %s value '%s' for %.*s",
                                   CLIargtype_to_string(ad[i].type), eq + 1, (int) (eq - r), r);
            }
            continue;
        }
        if (t < call->posonly && r[0] == '.' && r[1] != '\0' &&
            milk_args_find(call->cmdi, r + 1, strlen(r + 1)) >= 0)
        {
            return usage_error(
                call, "'%s VALUE' was removed: use %s=VALUE (this call) or --default %s=VALUE", r,
                r, r);
        }
        if (npos < NB_ARG_MAX)
        {
            pos[npos++] = t;
        }
    }

    /* positional values fill the primary parameters not given by name, in order */
    int k     = 0;
    int iprim = 0;
    for (int i = 0; i < n; i++)
    {
        if (!(fa[i].fpflag & FPFLAG_PRIMARY_CLI_INPUT))
        {
            continue;
        }
        iprim++;
        if (src[i] & MILK_ARGSRC_NAMED)
        {
            continue;
        }
        if (k < npos)
        {
            int         t = pos[k++];
            const char *r = raw_arg(call, t);

            if (strcmp(r, ".") == 0)
            {
                src[i] = MILK_ARGSRC_DOT;
            }
            else if (token_value(t, r, &ad[i]) == 0)
            {
                src[i] = MILK_ARGSRC_ARG;
            }
            else
            {
                return usage_error(call, "argument %d (.%s): invalid %s value '%s'", iprim,
                                   milk_args_tag(fa[i].fpstag), CLIargtype_to_string(ad[i].type),
                                   r);
            }
        }
        else if (src[i] == MILK_ARGSRC_DECL && !call->set_default &&
                 !(fa[i].fpflag & FPFLAG_CLI_OPTIONAL))
        {
            return usage_error(call, "missing argument %d (.%s: %s)", iprim,
                               milk_args_tag(fa[i].fpstag), fa[i].descr);
        }
    }
    if (k < npos)
    {
        return usage_error(call, "unexpected argument '%s'", raw_arg(call, pos[k]));
    }

    if (call->set_default)
    {
        store_session_defaults(call, n);
        return RETURN_CLICHECKARGARRAY_HELP; /* nothing runs */
    }
    if (call->save[0] != '\0' && milk_args_preset_save(call->cmdi, call->save) != 0)
    {
        return RETURN_CLICHECKARGARRAY_FAILURE;
    }
    if (call->no_run)
    {
        milk_args_print_call(call->cmdi, 0, stdout);
        return RETURN_CLICHECKARGARRAY_HELP;
    }
    return RETURN_CLICHECKARGARRAY_SUCCESS;
}

/**
 * CLI_checkarg_array() - resolve the parameters of the command being called from its tokens.
 *
 * Each parameter takes the first value found among: the call (`.tag=value` anywhere before `--`,
 * then the positional values in order for the primary parameters; `.` stands for the default),
 * the `--load` preset, the current value of a shared FPS (`cmd:fpsname`), the session default and
 * the declared default. A missing, invalid or extra value is a usage error (exit status 2).
 */
errno_t CLI_checkarg_array(CLICMDARGDEF fpscliarg[], int nbarg)
{
    DEBUG_TRACE_FSTART();

    int cmdi = (int) data.cmdindex;
    int n    = (nbarg < data.cmd[cmdi].nbparam) ? nbarg : data.cmd[cmdi].nbparam;

    argcheck_process_flag = 1;
    if (n <= 0 || data.cmd[cmdi].argdata == NULL)
    {
        DEBUG_TRACE_FEXIT();
        return RETURN_CLICHECKARGARRAY_SUCCESS; /* no declared parameters: it reads its tokens */
    }

    milk_call_t  dflt = { .cmdi = cmdi, .posonly = data.cmdNBarg };
    milk_call_t *call = milk_args_call(cmdi);
    if (call == NULL)
    {
        dflt.src = calloc((size_t) n, 1);
        if (dflt.src == NULL)
        {
            DEBUG_TRACE_FEXIT();
            return RETURN_CLICHECKARGARRAY_FAILURE;
        }
        call = &dflt;
    }

    errno_t rc = resolve_args(call, fpscliarg, n);
    free(dflt.src);
    DEBUG_TRACE_FEXIT();
    return rc;
}
