import re

with open('execute_arith_engines.c', 'r') as f:
    content = f.read()

# Add missing includes
inc_target = """#include "imfunctions.h"
#include "mathfuncs.h"
#include "image_arith__Cim_Cim__Cim.h"
#include "image_arith__im__im.h"
#include "image_arith__im_f__im.h"
#include "image_arith__im_f_f__im.h"
#include "image_arith__im_im__im.h"

#include "execute_arith_engines.h\""""

inc_rep = """#include "image_crop.h"
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

#include "execute_arith_engines.h\""""

content = content.replace(inc_target, inc_rep)

mult_target = """int exec_arith_multfunc("""
mult_pattern = re.escape(mult_target) + r".*?return 0;\n}"
mult_match = re.search(mult_pattern, content, re.DOTALL)
if mult_match:
    mult_rep = """int exec_arith_multfunc(
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
}"""
    content = content[:mult_match.start()] + mult_rep + content[mult_match.end():]

with open('execute_arith_engines.c', 'w') as f:
    f.write(content)

