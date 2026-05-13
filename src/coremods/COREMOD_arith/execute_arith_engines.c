/**
 * @file    execute_arith_engines.c
 * @brief   Execution engines for arithmetic parser
 */

#include <math.h>

#ifdef MILK_NO_CLI
#include "CLIcore_standalone.h"
#else
#include "libmilkdata/milkdata.h"
#include "milkDebugTools.h"
#endif

#include "libmilkcommon/milk_compiler.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include "imgid_arith_helpers.h"

#include "image_crop.h"
#include "image_dxdy.h"
#include "image_merge3D.h"
#include "image_stats.h"
#include "image_total.h"
#include "imfunctions.h"
#include "mathfuncs.h"
#include "image_arith__Cim_Cim__Cim.h"
#include "image_arith__im__im.h"
#include "image_arith__im_f__im.h"
#include "image_arith__im_f_f__im.h"
#include "image_arith__im_im__im.h"

#include "execute_arith_engines.h"

int exec_arith_binary(
    const char *op,
    int lt, const char *lw,
    int rt, const char *rw,
    char *name,
    int *type,
    int *tmp_name_index)
{
    int lvar = (lt == ARITHTOKENTYPE_VARIABLE);
    int rvar = (rt == ARITHTOKENTYPE_VARIABLE);
    int lim  = (lt == ARITHTOKENTYPE_IMAGE);
    int rim  = (rt == ARITHTOKENTYPE_IMAGE);
    double lval = lvar ? dcvar[variable_ID(lw)].value.f : 0.0;
    double rval = rvar ? dcvar[variable_ID(rw)].value.f : 0.0;
    char name1[STRINGMAXLEN_IMGNAME];

    if(strcmp(op, "+") == 0)
    {
        if(lvar && rvar)
        {
            create_variable_ID(name, lval + rval);
            (*tmp_name_index)++;
            *type = ARITHTOKENTYPE_VARIABLE;
        }
        else if(lvar && rim)
        {
            arith_image_cstadd(rw, lval, name);
            (*tmp_name_index)++;
            *type = ARITHTOKENTYPE_IMAGE;
        }
        else if(lim && rvar)
        {
            arith_image_cstadd(lw, rval, name);
            (*tmp_name_index)++;
            *type = ARITHTOKENTYPE_IMAGE;
        }
        else if(lim && rim)
        {
            arith_image_add(lw, rw, name);
            (*tmp_name_index)++;
            *type = ARITHTOKENTYPE_IMAGE;
        }
    }
    else if(strcmp(op, "-") == 0)
    {
        if(lvar && rvar)
        {
            create_variable_ID(name, lval - rval);
            (*tmp_name_index)++;
            *type = ARITHTOKENTYPE_VARIABLE;
        }
        else if(lvar && rim)
        {
            CREATE_IMAGENAME(name1, "_tmp1%d_%d", *tmp_name_index, (int) getpid());
            arith_image_cstsub(rw, lval, name1);
            arith_image_cstmult(name1, -1.0, name);
            delete_image_ID(name1, DELETE_IMAGE_ERRMODE_WARNING);
            (*tmp_name_index)++;
            *type = ARITHTOKENTYPE_IMAGE;
        }
        else if(lim && rvar)
        {
            arith_image_cstsub(lw, rval, name);
            (*tmp_name_index)++;
            *type = ARITHTOKENTYPE_IMAGE;
        }
        else if(lim && rim)
        {
            arith_image_sub(lw, rw, name);
            (*tmp_name_index)++;
            *type = ARITHTOKENTYPE_IMAGE;
        }
    }
    else if(strcmp(op, "*") == 0)
    {
        if(lvar && rvar)
        {
            create_variable_ID(name, lval * rval);
            (*tmp_name_index)++;
            *type = ARITHTOKENTYPE_VARIABLE;
        }
        else if(lvar && rim)
        {
            arith_image_cstmult(rw, lval, name);
            (*tmp_name_index)++;
            *type = ARITHTOKENTYPE_IMAGE;
        }
        else if(lim && rvar)
        {
            arith_image_cstmult(lw, rval, name);
            (*tmp_name_index)++;
            *type = ARITHTOKENTYPE_IMAGE;
        }
        else if(lim && rim)
        {
            arith_image_mult(lw, rw, name);
            (*tmp_name_index)++;
            *type = ARITHTOKENTYPE_IMAGE;
        }
    }
    else if(strcmp(op, "/") == 0)
    {
        if(lvar && rvar)
        {
            create_variable_ID(name, lval / rval);
            (*tmp_name_index)++;
            *type = ARITHTOKENTYPE_VARIABLE;
        }
        else if(lvar && rim)
        {
            arith_image_cstdiv1(rw, lval, name);
            (*tmp_name_index)++;
            *type = ARITHTOKENTYPE_IMAGE;
        }
        else if(lim && rvar)
        {
            arith_image_cstdiv(lw, rval, name);
            (*tmp_name_index)++;
            *type = ARITHTOKENTYPE_IMAGE;
        }
        else if(lim && rim)
        {
            arith_image_div(lw, rw, name);
            (*tmp_name_index)++;
            *type = ARITHTOKENTYPE_IMAGE;
        }
    }
    else if(strcmp(op, "^") == 0)
    {
        if(lvar && rvar)
        {
            double tmp_prec = (rval < 0) ? 1.0 / pow(lval, -rval) : pow(lval, rval);
            create_variable_ID(name, tmp_prec);
            (*tmp_name_index)++;
            *type = ARITHTOKENTYPE_VARIABLE;
        }
        else if(lvar && rim)
        {
            CREATE_IMAGENAME(name1, "_tmp1%d_%d", *tmp_name_index, (int) getpid());
            arith_image_cstadd(rw, lval, name1);
            arith_image_pow(name1, rw, name);
            delete_image_ID(name1, DELETE_IMAGE_ERRMODE_WARNING);
            (*tmp_name_index)++;
            *type = ARITHTOKENTYPE_IMAGE;
        }
        else if(lim && rvar)
        {
            arith_image_cstpow(lw, rval, name);
            (*tmp_name_index)++;
            *type = ARITHTOKENTYPE_IMAGE;
        }
        else if(lim && rim)
        {
            arith_image_pow(lw, rw, name);
            (*tmp_name_index)++;
            *type = ARITHTOKENTYPE_IMAGE;
        }
    }
    return 0;
}

int exec_arith_unary(
    const char *fname,
    int arg_wtype,
    const char *arg_word,
    char *name,
    int *type,
    int *tmp_name_index)
{
    /*
     * Unary function dispatch table.
     */
    typedef double  (*scalar_fn_t)(double);
    typedef int     (*image_fn_t)(const char *, const char *);
    typedef double  (*reducer_fn_t)(const char *);

    struct unary_dispatch_entry {
        const char   *name;
        scalar_fn_t   scalar_fn;
        image_fn_t    image_fn;
        reducer_fn_t  reducer_fn;
    };

    static const struct unary_dispatch_entry unary_dispatch[] =
    {
        /* name      scalar      image               reducer */
        { "acos",  acos,  arith_image_acos,     NULL },
        { "asin",  asin,  arith_image_asin,     NULL },
        { "atan",  atan,  arith_image_atan,     NULL },
        { "ceil",  ceil,  arith_image_ceil,     NULL },
        { "cos",   cos,   arith_image_cos,      NULL },
        { "cosh",  cosh,  arith_image_cosh,     NULL },
        { "exp",   exp,   arith_image_exp,      NULL },
        { "fabs",  fabs,  arith_image_fabs,     NULL },
        { "floor", floor, arith_image_floor,    NULL },
        { "ln",    log,   arith_image_ln,       NULL },
        { "log",   log10, arith_image_log,      NULL },
        { "sqrt",  sqrt,  arith_image_sqrt,     NULL },
        { "sin",   sin,   arith_image_sin,      NULL },
        { "sinh",  sinh,  arith_image_sinh,     NULL },
        { "tan",   tan,   arith_image_tan,      NULL },
        { "tanh",  tanh,  arith_image_tanh,     NULL },
        { "posi",  Ppositive, arith_image_positive, NULL },
        /* image reducers -> scalar */
        { "imedian", NULL, NULL, arith_image_median },
        { "itot",    NULL, NULL, arith_image_total  },
        { "imean",   NULL, NULL, arith_image_mean   },
        { "imin",    NULL, NULL, arith_image_min    },
        { "imax",    NULL, NULL, arith_image_max    },
        { NULL, NULL, NULL, NULL }
    };

    for(int ui = 0; unary_dispatch[ui].name != NULL; ui++)
    {
        if(strcmp(fname, unary_dispatch[ui].name) != 0)
        {
            continue;
        }

        /* Image reducer: image -> scalar */
        if(unary_dispatch[ui].reducer_fn != NULL)
        {
            if(arg_wtype == ARITHTOKENTYPE_IMAGE)
            {
                double tmp_prec = unary_dispatch[ui].reducer_fn(arg_word);
                create_variable_ID(name, tmp_prec);
                (*tmp_name_index)++;
                *type = ARITHTOKENTYPE_VARIABLE;
            }
            else
            {
                PRINT_ERROR("Function %s only applicable on images", fname);
                return RETURN_FAILURE;
            }
            break;
        }

        /* Transform: scalar -> scalar */
        if(arg_wtype == ARITHTOKENTYPE_VARIABLE)
        {
            if(unary_dispatch[ui].scalar_fn == NULL)
            {
                PRINT_ERROR("Function %s only applicable on images", fname);
                return RETURN_FAILURE;
            }
            double tmp_prec = unary_dispatch[ui].scalar_fn(dcvar[variable_ID(arg_word)].value.f);
            create_variable_ID(name, tmp_prec);
            (*tmp_name_index)++;
            *type = ARITHTOKENTYPE_VARIABLE;
        }

        /* Transform: image -> image */
        if(arg_wtype == ARITHTOKENTYPE_IMAGE)
        {
            unary_dispatch[ui].image_fn(arg_word, name);
            (*tmp_name_index)++;
            *type = ARITHTOKENTYPE_IMAGE;
        }
        break;
    }
    return 0;
}

int exec_arith_multfunc(
    const char *fn,
    int nbvarinput,
    int a1t, const char *a1w,
    int a2t, const char *a2w,
    int a3t, const char *a3w,
    char *name,
    int *type,
    int *tmp_name_index)
{
    int a1var = (a1t == ARITHTOKENTYPE_VARIABLE);
    int a2var = (a2t == ARITHTOKENTYPE_VARIABLE);
    int a3var = (a3t == ARITHTOKENTYPE_VARIABLE);
    int a1im  = (a1t == ARITHTOKENTYPE_IMAGE);
    int a2im  = (a2t == ARITHTOKENTYPE_IMAGE);
    double a1v = a1var ? dcvar[variable_ID(a1w)].value.f : 0.0;
    double a2v = a2var ? dcvar[variable_ID(a2w)].value.f : 0.0;
    double a3v = a3var ? dcvar[variable_ID(a3w)].value.f : 0.0;
    char name1[STRINGMAXLEN_IMGNAME];

    if(strcmp(fn, "fmod") == 0)
    {
        if(a1var && a2var)
        {
            create_variable_ID(name, fmod(a1v, a2v));
            (*tmp_name_index)++;
            *type = ARITHTOKENTYPE_VARIABLE;
        }
        else if(a1var && a2im)
        {
            PRINT_ERROR("Function fmod not available for VARIABLE x IMAGE inputs");
            return RETURN_FAILURE;
        }
        else if(a1im && a2var)
        {
            arith_image_cstfmod(a1w, a2v, name);
            (*tmp_name_index)++;
            *type = ARITHTOKENTYPE_IMAGE;
        }
        else if(a1im && a2im)
        {
            arith_image_fmod(a1w, a2w, name);
            (*tmp_name_index)++;
            *type = ARITHTOKENTYPE_IMAGE;
        }
    }
    else if(strcmp(fn, "min") == 0)
    {
        if(a1var && a2var)
        {
            create_variable_ID(name, (a1v < a2v) ? a1v : a2v);
            (*tmp_name_index)++;
            *type = ARITHTOKENTYPE_VARIABLE;
        }
        else if(a1var && a2im)
        {
            arith_image_cstminv(a2w, a1v, name);
            (*tmp_name_index)++;
            *type = ARITHTOKENTYPE_IMAGE;
        }
        else if(a1im && a2var)
        {
            arith_image_cstminv(a1w, a2v, name);
            (*tmp_name_index)++;
            *type = ARITHTOKENTYPE_IMAGE;
        }
        else if(a1im && a2im)
        {
            arith_image_minv(a1w, a2w, name);
            (*tmp_name_index)++;
            *type = ARITHTOKENTYPE_IMAGE;
        }
    }
    else if(strcmp(fn, "max") == 0)
    {
        if(a1var && a2var)
        {
            create_variable_ID(name, (a1v > a2v) ? a1v : a2v);
            (*tmp_name_index)++;
            *type = ARITHTOKENTYPE_VARIABLE;
        }
        else if(a1var && a2im)
        {
            arith_image_cstmaxv(a2w, a1v, name);
            (*tmp_name_index)++;
            *type = ARITHTOKENTYPE_IMAGE;
        }
        else if(a1im && a2var)
        {
            arith_image_cstmaxv(a1w, a2v, name);
            (*tmp_name_index)++;
            *type = ARITHTOKENTYPE_IMAGE;
        }
        else if(a1im && a2im)
        {
            arith_image_maxv(a1w, a2w, name);
            (*tmp_name_index)++;
            *type = ARITHTOKENTYPE_IMAGE;
        }
    }
    else if(strcmp(fn, "testlt") == 0)
    {
        if(a1var && a2var)
        {
            create_variable_ID(name, (a1v < a2v) ? 1.0 : 0.0);
            (*tmp_name_index)++;
            *type = ARITHTOKENTYPE_VARIABLE;
        }
        else if(a1var && a2im)
        {
            arith_image_csttestmt(a2w, a1v, name);
            (*tmp_name_index)++;
            *type = ARITHTOKENTYPE_IMAGE;
        }
        else if(a1im && a2var)
        {
            arith_image_csttestlt(a1w, a2v, name);
            (*tmp_name_index)++;
            *type = ARITHTOKENTYPE_IMAGE;
        }
        else if(a1im && a2im)
        {
            arith_image_testlt(a1w, a2w, name);
            (*tmp_name_index)++;
            *type = ARITHTOKENTYPE_IMAGE;
        }
        else
        {
            PRINT_ERROR("Wrong input to function testlt");
            return RETURN_FAILURE;
        }
    }
    else if(strcmp(fn, "testmt") == 0)
    {
        if(a1var && a2var)
        {
            create_variable_ID(name, (a1v > a2v) ? 1.0 : 0.0);
            (*tmp_name_index)++;
            *type = ARITHTOKENTYPE_VARIABLE;
        }
        else if(a1var && a2im)
        {
            arith_image_csttestlt(a2w, a1v, name);
            (*tmp_name_index)++;
            *type = ARITHTOKENTYPE_IMAGE;
        }
        else if(a1im && a2var)
        {
            arith_image_csttestmt(a1w, a2v, name);
            (*tmp_name_index)++;
            *type = ARITHTOKENTYPE_IMAGE;
        }
        else if(a1im && a2im)
        {
            arith_image_testmt(a1w, a2w, name);
            (*tmp_name_index)++;
            *type = ARITHTOKENTYPE_IMAGE;
        }
        else
        {
            PRINT_ERROR("Wrong input to function testmt");
            return RETURN_FAILURE;
        }
    }
    else if(strcmp(fn, "perc") == 0)
    {
        if(!a1im || !a2var)
        {
            PRINT_ERROR("Wrong input to function perc");
            return RETURN_FAILURE;
        }
        else
        {
            create_variable_ID(name, arith_image_percentile(a1w, a2v));
            (*tmp_name_index)++;
            *type = ARITHTOKENTYPE_VARIABLE;
        }
    }
    else if(strcmp(fn, "trunc") == 0)
    {
        if(a1im && a2var && a3var)
        {
            (*tmp_name_index)++;
            arith_image_trunc(a1w, a2v, a3v, name);
            *type = ARITHTOKENTYPE_IMAGE;
        }
        else
        {
            PRINT_ERROR("Syntax error with function trunc");
            return RETURN_FAILURE;
        }
    }

    return 0;
}
