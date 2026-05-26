/**
 * @file    execute_arith.c
 * @brief   image arithmetic parser
 *
 *
 */

#include <ctype.h>
#include <math.h>

#ifdef MILK_NO_CLI
#    include "CLIcore_standalone.h"
#else
#    include "libmilkdata/milkdata.h"
#    include "milkDebugTools.h"
#endif

#include "COREMOD_memory/COREMOD_memory.h"
#include "COREMOD_memory/stream_slice.h"

#include "image_arith__Cim_Cim__Cim.h"
#include "image_arith__im__im.h"
#include "image_arith__im_f__im.h"
#include "image_arith__im_f_f__im.h"
#include "image_arith__im_im__im.h"
#include "image_crop.h"
#include "image_dxdy.h"
#include "image_merge3D.h"
#include "image_stats.h"
#include "image_total.h"
#include "imfunctions.h"
#include "execute_arith_engines.h"

/* -------------------------------------------------------
 * Classifier tables
 *
 * These NULL-terminated arrays replace the former chains
 * of bare if(strcmp(word,"...")) blocks.  Lookups are
 * simple linear scans — the tables are small (< 30 entries)
 * and this path is not on the real-time hot path.
 * ------------------------------------------------------- */

/** Known single-character binary operators */
static const char *const operand_names[] = { "+", "-", "*", "/", "^", NULL };

/**
 * Known unary functions (token type ARITHTOKENTYPE_FUNCTION).
 * Functions beginning with 'i' that reduce an image to a
 * scalar (imedian, itot, imean, imin, imax) are included here;
 * they are handled with a separate image_reducer sub-table
 * in the dispatch step.
 */
static const char *const unary_func_names[] = { "acos", "asin",  "atan", "ceil",  "cos",
                                                "cosh", "exp",   "fabs", "floor", "imedian",
                                                "itot", "imean", "imin", "imax",  "ln",
                                                "log",  "sqrt",  "sin",  "sinh",  "tan",
                                                "tanh", "posi",  "imdx", "imdy",  NULL };

/**
 * Multi-arg functions: returns the number of input arguments
 * (2 or 3), or 0 if the name is not a multi-arg function.
 */
struct multfunc_entry
{
    const char *name;
    int         nargs; /* number of scalar/image arguments */
};

static const struct multfunc_entry multfunc_table[] = { { "fmod", 2 },   { "trunc", 3 },
                                                        { "perc", 2 },   { "min", 2 },
                                                        { "max", 2 },    { "testlt", 2 },
                                                        { "testmt", 2 }, { NULL, 0 } };

/**
 * isoperand - test whether a token is a binary operator.
 * @word: token string
 * Return: 1 if the token is +, -, *, /, or ^; 0 otherwise.
 */
static int isoperand(const char *word)
{
    for (int i = 0; operand_names[i] != NULL; i++)
    {
        if (strcmp(word, operand_names[i]) == 0)
        {
            return 1;
        }
    }
    return 0;
}

/**
 * isfunction - test whether a token is a unary function name.
 * @word: token string
 * Return: 1 if recognised, 0 otherwise.
 */
static int isfunction(const char *word)
{
    for (int i = 0; unary_func_names[i] != NULL; i++)
    {
        if (strcmp(word, unary_func_names[i]) == 0)
        {
            return 1;
        }
    }
    return 0;
}

/**
 * isfunction_sev_var - return arg count for multi-arg functions.
 * @word: token string
 * Return: number of arguments (2 or 3), or 0 if not recognised.
 */
static int isfunction_sev_var(const char *word)
{
    for (int i = 0; multfunc_table[i].name != NULL; i++)
    {
        if (strcmp(word, multfunc_table[i].name) == 0)
        {
            return multfunc_table[i].nargs;
        }
    }
    return 0;
}

int isanumber(const char *word)
{
    DEBUG_TRACE_FSTART();

    int                            value = 1; // 1 if number, 0 otherwise
    char                          *endptr;
    __attribute__((unused)) double v1;

    v1 = strtod(word, &endptr);
    if ((long) (endptr - word) == (long) strlen(word))
    {
        value = 1;
    }
    else
    {
        value = 0;
    }

    DEBUG_TRACE_FEXIT();
    return (value);
}

imageID arith_make_slopexy(const char *ID_name, uint32_t l1, uint32_t l2, double sx, double sy)
{
    DEBUG_TRACE_FSTART();

    imageID  ID;
    uint32_t naxes[2];
    double   coeff;

    create_2Dimage_ID(ID_name, l1, l2, &ID);
    naxes[0] = dcimg[ID].md[0].size[0];
    naxes[1] = dcimg[ID].md[0].size[1];

    coeff = sx * (naxes[0] / 2) + sy * (naxes[1] / 2);

    for (uint32_t jj = 0; jj < naxes[1]; jj++)
    {
        for (uint32_t ii = 0; ii < naxes[0]; ii++)
        {
            dcimg[ID].array.F[jj * naxes[0] + ii] = sx * ii + sy * jj - coeff;
        }
    }

    DEBUG_TRACE_FEXIT();
    return ID;
}


/**
 * execute_arith - evaluate an arithmetic expression on
 *   images and scalar variables.
 * @cmd1: null-terminated expression string
 *
 * Tokens are classified as numbers, variables, images,
 * operands (+,-,*,/,^), unary functions (sin, cos, ...)
 * or multi-arg functions (fmod, min, max, ...).  The
 * expression is reduced iteratively by executing the
 * highest-priority unresolved operation at each step.
 *
 * Return: 0. Parse/runtime errors are reported through
 * logging/error handling paths, and fatal errors may
 * terminate execution instead of being returned.
 */
int execute_arith(const char *cmd1)
{
    char word[100][100];
    int  nbword = 0;
    int  word_type[100];
    int  par_level[100];
    int  intr_priority[100]; /* 0 (+,-)  1 (*,/)  2 (functions) */

    int OKea = 1;

    int Debug = 0;

    //  if( Debug > 0 )   fprintf(stdout, "[execute_arith]\n");
    //  if( Debug > 0 )   fprintf(stdout, "[execute_arith] str: [%s]\n", cmd1);

    for (int i = 0; i < 100; i++)
    {
        word_type[i]     = 0;
        par_level[i]     = 0;
        intr_priority[i] = 0;
    }

    /*
       Pre-process string:
       - remove any spaces in cmd1
       - replace "=-" by "=0-" and "=+" by "="
       copy result into cmd */
    {
        int  CMDBUFFSIZE = 1000;
        char cmd[CMDBUFFSIZE];
        int  j = 0;

        for (int i = 0; i < (int) (strlen(cmd1)); i++)
        {
            if ((cmd1[i] == '=') && (cmd1[i + 1] == '-'))
            {
                cmd[j] = '=';
                j++;
                cmd[j] = '0';
                j++;
            }
            else if ((cmd1[i] == '=') && (cmd1[i + 1] == '+'))
            {
                cmd[j] = '=';
                j++;
                i++;
            }
            else if (cmd1[i] != ' ')
            {
                cmd[j] = cmd1[i];
                j++;
            }
        }
        cmd[j] = '\0';
        //  if( Debug > 0 )   fprintf(stdout, "[execute_arith] preprocessed str %s -> %s\n", cmd1, cmd);

        /*
        * cmd is first broken into words.
        * The spacing between words is operands (+,-,/,*), equal (=),
        * space ,comma and braces
        */
        int w             = 0;
        int l             = 0;
        int bracket_depth = 0; /* track [...] nesting */
        for (int i = 0; i < (signed) strlen(cmd); i++)
        {
            /* Inside brackets: everything is part of
         * the current word. Used for slice syntax
         * like im[0:19,10:29]. */
            if (cmd[i] == '[')
            {
                bracket_depth++;
                word[w][l] = cmd[i];
                l++;
                continue;
            }
            if (cmd[i] == ']')
            {
                if (bracket_depth > 0)
                {
                    bracket_depth--;
                }
                word[w][l] = cmd[i];
                l++;
                continue;
            }
            if (bracket_depth > 0)
            {
                word[w][l] = cmd[i];
                l++;
                continue;
            }

            switch (cmd[i])
            {
            case '+':
            case '-':
                if ((i > 1) && ((cmd[i - 1] == 'e') || (cmd[i - 1] == 'E')) &&
                    (isdigit(cmd[i - 2])) && (isdigit(cmd[i + 1])))
                {
                    // + or - is part of exponent
                    word[w][l] = cmd[i];
                    l++;
                }
                else
                {
                    if (l > 0)
                    {
                        word[w][l] = '\0';
                        w++;
                    }
                    l          = 0;
                    word[w][l] = cmd[i];
                    word[w][1] = '\0';
                    if (i < (signed) (strlen(cmd) - 1))
                    {
                        w++;
                    }
                    l = 0;
                }
                break;

            case '*':
            case '/':
            case '^':
            case '(':
            case ')':
            case '=':
            case ',':
                if (l > 0)
                {
                    word[w][l] = '\0';
                    w++;
                }
                l          = 0;
                word[w][l] = cmd[i];
                word[w][1] = '\0';
                if (i < (signed) (strlen(cmd) - 1))
                {
                    w++;
                }
                l = 0;
                break;

            case ' ':
                word[w][l] = '\0';
                w++;
                l = 0;

                /*word[w][l] = '\0';
                                              w++;
                                              l = 0;*/
                break;

            default:
                word[w][l] = cmd[i];
                l++;
                break;
            }
        }

        if (l > 0)
        {
            word[w][l] = '\0';
        }
        nbword = w + 1;
    }

    //  printf("number of words is %d\n",nbword);

    for (int i = 0; i < nbword; i++)
    {
        if (Debug > 0)
        {
            printf("TESTING WORD %d = %s\n", i, word[i]);
        }
        word_type[i]        = ARITHTOKENTYPE_UNKNOWN;
        int found_word_type = 0;
        if ((isanumber(word[i]) == 1) && (found_word_type == 0))
        {
            word_type[i]    = ARITHTOKENTYPE_NUMBER;
            found_word_type = 1;
        }
        if ((isfunction(word[i]) == 1) && (found_word_type == 0))
        {
            word_type[i]    = ARITHTOKENTYPE_FUNCTION;
            found_word_type = 1;
        }
        if ((isfunction_sev_var(word[i]) != 0) && (found_word_type == 0))
        {
            word_type[i]    = ARITHTOKENTYPE_MULTFUNC;
            found_word_type = 1;
        }
        if ((isoperand(word[i]) == 1) && (found_word_type == 0))
        {
            word_type[i]    = ARITHTOKENTYPE_OPERAND;
            found_word_type = 1;
        }
        if ((strcmp(word[i], "=") == 0) && (found_word_type == 0))
        {
            word_type[i]    = ARITHTOKENTYPE_EQUAL;
            found_word_type = 1;
        }
        if ((strcmp(word[i], ",") == 0) && (found_word_type == 0))
        {
            word_type[i]    = ARITHTOKENTYPE_COMA;
            found_word_type = 1;
        }
        if ((i < nbword - 1) && (found_word_type == 0))
        {
            if ((strcmp(word[i + 1], "(") == 0) && (isfunction(word[i]) == 1))
            {
                word_type[i]    = ARITHTOKENTYPE_FUNCTION;
                found_word_type = 1;
            }
        }
        if ((strcmp(word[i], "(") == 0) && (found_word_type == 0))
        {
            word_type[i]    = ARITHTOKENTYPE_OPENPAR;
            found_word_type = 1;
        }
        if ((strcmp(word[i], ")") == 0) && (found_word_type == 0))
        {
            word_type[i]    = ARITHTOKENTYPE_CLOSEPAR;
            found_word_type = 1;
        }
        if ((variable_ID(word[i]) != -1) && (found_word_type == 0))
        {
            word_type[i]    = ARITHTOKENTYPE_VARIABLE;
            found_word_type = 1;
        }
        if ((imgid_exists(word[i])) && (found_word_type == 0))
        {
            word_type[i]    = ARITHTOKENTYPE_IMAGE;
            found_word_type = 1;
        }
        if (found_word_type == 0)
        {
            word_type[i] = ARITHTOKENTYPE_NOTEXIST;
        }
        if (Debug > 0)
        {
            printf("word %d is  \"%s\" word type is %d\n", i, word[i], word_type[i]);
        }
    }

    /* checks for obvious errors */

    int passedequ = 0;
    for (int i = (nbword - 1); i > -1; i--)
    {
        if (passedequ == 1)
        {
            if (word_type[i] == ARITHTOKENTYPE_EQUAL)
            {
                PRINT_WARNING("line has multiple \"=\"");
                OKea = 0;
            }
            if (word_type[i] == ARITHTOKENTYPE_OPERAND)
            {
                PRINT_WARNING("operand on left side of \"=\"");
                OKea = 0;
            }
            if (word_type[i] == ARITHTOKENTYPE_OPENPAR)
            {
                PRINT_WARNING("\"(\" on left side of \"=\"");
                OKea = 0;
            }
            if (word_type[i] == ARITHTOKENTYPE_CLOSEPAR)
            {
                PRINT_WARNING("\")\" on left side of \"=\"");
                OKea = 0;
            }
        }
        if (word_type[i] == ARITHTOKENTYPE_EQUAL)
        {
            passedequ = 1;
        }
        if ((passedequ == 0) &&
            (word_type[i] == ARITHTOKENTYPE_NOTEXIST)) /* non-existing variable or image as input */
        {
            PRINT_WARNING("%s is a non-existing variable or image", word[i]);
            OKea = 0;
        }
    }

    for (int i = 0; i < nbword - 1; i++)
    {
        if ((word_type[i] == ARITHTOKENTYPE_OPERAND) &&
            (word_type[i + 1] == ARITHTOKENTYPE_OPERAND))
        {
            PRINT_WARNING("consecutive operands");
            OKea = 0;
        }
        if ((word_type[i + 1] == ARITHTOKENTYPE_OPENPAR) &&
            (!((word_type[i] == ARITHTOKENTYPE_OPENPAR) ||
               (word_type[i] == ARITHTOKENTYPE_FUNCTION) ||
               (word_type[i] == ARITHTOKENTYPE_MULTFUNC) ||
               (word_type[i] == ARITHTOKENTYPE_EQUAL) || (word_type[i] == ARITHTOKENTYPE_OPERAND))))
        {
            PRINT_WARNING("\"(\" should be preceeded by \"=\", \"(\", operand or "
                          "function");
            OKea = 0;
        }
    }

    long cntP = 0;
    for (int i = 0; i < nbword; i++)
    {
        if (word_type[i] == ARITHTOKENTYPE_OPENPAR)
        {
            cntP++;
        }
        if (word_type[i] == ARITHTOKENTYPE_CLOSEPAR)
        {
            cntP--;
        }
        if (cntP < 0)
        {
            PRINT_WARNING("parentheses error");
            OKea = 0;
        }
    }
    if (cntP != 0)
    {
        PRINT_WARNING("parentheses error");
        OKea = 0;
    }

    if (OKea == 1)
    {
        int tmp_name_index = 0;

        /* numbers are saved into variables */
        for (int i = 0; i < nbword; i++)
        {
            if (word_type[i] == ARITHTOKENTYPE_NUMBER)
            {
                char name[STRINGMAXLEN_IMGNAME];
                CREATE_IMAGENAME(name, "_tmp%d_%d", tmp_name_index, (int) getpid());

                create_variable_ID(name, 1.0 * strtod(word[i], NULL));
                snprintf(word[i], sizeof(word[i]), "%s", name);
                word_type[i] = ARITHTOKENTYPE_VARIABLE;
                tmp_name_index++;
            }
        }

        /* Sliced images are materialized into
         * temporary images so the rest of the
         * evaluator can work with plain names. */
        for (int i = 0; i < nbword; i++)
        {
            if (word_type[i] != ARITHTOKENTYPE_IMAGE)
            {
                continue;
            }
            if (strchr(word[i], '[') == NULL)
            {
                continue;
            }

            IMGID simg = imgid_make_from_name(word[i]);
            resolveIMGID(&simg, ERRMODE_NULL, dcimg, dcnimg);
            if (simg.ID < 0)
            {
                imgid_free(&simg);
                continue;
            }

            /* Materialize the slice */
            if (imgid_slice_materialize(&simg) != 0)
            {
                PRINT_WARNING("slice materialize failed"
                              " for %s",
                              word[i]);
                imgid_free(&simg);
                continue;
            }

            IMAGE   *slc    = simg.slice_im;
            int      snaxis = (int) slc->md[0].naxis;
            uint32_t ssz[3] = { 1, 1, 1 };
            for (int a = 0; a < snaxis; a++)
            {
                ssz[a] = slc->md[0].size[a];
            }

            /* Create temp image with slice
             * dimensions */
            char name[STRINGMAXLEN_IMGNAME];
            CREATE_IMAGENAME(name, "_slice%d_%d", tmp_name_index, (int) getpid());

            imageID tid = -1;
            create_image_ID(name, snaxis, ssz, slc->md[0].datatype, 0, /* not shared */
                            IMGID_NB_KEYWO_MAX, 0,                     /* CBsize */
                            &tid);

            if (tid >= 0)
            {
                uint64_t nbytes =
                    slc->md[0].nelement * (uint64_t) ImageStreamIO_typesize(slc->md[0].datatype);
                __builtin_memcpy(dcimg[tid].array.raw, slc->array.raw, nbytes);
            }

            snprintf(word[i], sizeof(word[i]), "%s", name);
            tmp_name_index++;
            imgid_free(&simg);
        }

        /* computing the number of to-be-processed words */
        int passedequ   = 0;
        int nb_tbp_word = 0;
        for (int i = (nbword - 1); i > -1; i--)
        {
            if (word_type[i] == ARITHTOKENTYPE_EQUAL)
            {
                passedequ = 1;
            }
            if (passedequ == 0)
            {
                nb_tbp_word++;
            }
        }

        /* main loop starts here */
        while (nb_tbp_word > 1)
        {
            /* non necessary braces are removed
             */
            for (int i = 0; i < nbword - 2; i++)
            {
                if ((word_type[i] == ARITHTOKENTYPE_OPENPAR) &&
                    (word_type[i + 2] == ARITHTOKENTYPE_CLOSEPAR))
                {
                    snprintf(word[i], sizeof(word[i]), "%s", word[i + 1]);
                    word_type[i] = word_type[i + 1];
                    for (int j = i + 1; j < nbword - 2; j++)
                    {
                        snprintf(word[j], sizeof(word[j]), "%s", word[j + 2]);
                        word_type[j] = word_type[j + 2];
                    }
                    nbword = nbword - 2;
                }
            }

            for (int i = 0; i < nbword - 3; i++)
            {
                if ((word_type[i] == ARITHTOKENTYPE_OPENPAR) &&
                    (word_type[i + 3] == ARITHTOKENTYPE_CLOSEPAR) &&
                    (strcmp(word[i + 1], "-") == 0))
                {
                    dcvar[variable_ID(word[i + 2])].value.f =
                        -dcvar[variable_ID(word[i + 2])].value.f;
                    snprintf(word[i], sizeof(word[i]), "%s", word[i + 2]);
                    word_type[i] = word_type[i + 2];
                    for (int j = i + 2; j < nbword - 3; j++)
                    {
                        snprintf(word[j], sizeof(word[j]), "%s", word[j + 3]);
                        word_type[j] = word_type[j + 3];
                    }
                    nbword = nbword - 3;
                }
            }

            /* now the priorities are given */

            int parlevel = 0;
            for (int i = 0; i < nbword; i++)
            {
                if (word_type[i] == ARITHTOKENTYPE_OPENPAR)
                {
                    parlevel++;
                }
                if (word_type[i] == ARITHTOKENTYPE_CLOSEPAR)
                {
                    parlevel--;
                }
                if ((word_type[i] == 4) || (word_type[i] == 8) ||
                    (word_type[i] == ARITHTOKENTYPE_MULTFUNC))
                {
                    par_level[i] = parlevel;
                    if (word_type[i] == ARITHTOKENTYPE_FUNCTION)
                    {
                        intr_priority[i] = 2;
                    }
                    if (word_type[i] == ARITHTOKENTYPE_MULTFUNC)
                    {
                        intr_priority[i] = 2;
                    }
                    if (word_type[i] == ARITHTOKENTYPE_OPERAND)
                    {
                        if ((strcmp(word[i], "+") == 0) || (strcmp(word[i], "-") == 0))
                        {
                            intr_priority[i] = 0;
                        }
                        if ((strcmp(word[i], "*") == 0) || (strcmp(word[i], "/") == 0))
                        {
                            intr_priority[i] = 1;
                        }
                    }
                }
            }

            /* the highest priority operation is executed */
            int highest_parlevel       = 0;
            int highest_intr_priority  = -1;
            int highest_priority_index = -1;

            for (int i = 0; i < nbword; i++)
            {
                if ((word_type[i] == ARITHTOKENTYPE_OPERAND) ||
                    (word_type[i] == ARITHTOKENTYPE_FUNCTION) ||
                    (word_type[i] == ARITHTOKENTYPE_MULTFUNC))
                {
                    /*printf("operation \"%s\" (%d,%d)\n",word[i],par_level[i],intr_priority[i]);*/
                    if (par_level[i] > highest_parlevel)
                    {
                        highest_priority_index = i;
                        highest_parlevel       = par_level[i];
                        highest_intr_priority  = 0;
                    }
                    else
                    {
                        if ((par_level[i] == highest_parlevel) &&
                            (intr_priority[i] > highest_intr_priority))
                        {
                            highest_priority_index = i;
                            highest_intr_priority  = intr_priority[i];
                        }
                    }
                }
            }

            /*      printf("executing operation  %s\n",word[highest_priority_index]);*/

            /*      printf("before : ");
              for (j=0;j<nbword;j++)
              {
              if(j==i)
              printf(">>");
              if(variable_ID(word[j])!=-1)
              printf(" %s(%f) ",word[j],dcvar[variable_ID(word[j])].value.f);
              else
              printf(" %s ",word[j]);
              }
              printf("\n");
            */
            if (word_type[highest_priority_index] == ARITHTOKENTYPE_OPERAND)
            {
                int  hpi = highest_priority_index;
                char name[STRINGMAXLEN_IMGNAME];
                CREATE_IMAGENAME(name, "_tmp%d_%d", tmp_name_index, (int) getpid());

                int type = 0;
                if (exec_arith_binary(word[hpi], word_type[hpi - 1], word[hpi - 1],
                                      word_type[hpi + 1], word[hpi + 1], name, &type,
                                      &tmp_name_index) != 0)
                {
                    return RETURN_FAILURE;
                }

                snprintf(word[hpi - 1], sizeof(word[hpi - 1]), "%s", name);
                word_type[hpi - 1] = type;
                for (int j = hpi; j < nbword - 2; j++)
                {
                    snprintf(word[j], sizeof(word[j]), "%s", word[j + 2]);
                    word_type[j] = word_type[j + 2];
                }
                nbword = nbword - 2;
            }


            if (word_type[highest_priority_index] == ARITHTOKENTYPE_FUNCTION)
            {
                char name[STRINGMAXLEN_IMGNAME];
                CREATE_IMAGENAME(name, "_tmp%d_%d", tmp_name_index, (int) getpid());

                int hpi  = highest_priority_index;
                int type = 0;
                if (exec_arith_unary(word[hpi], word_type[hpi + 1], word[hpi + 1], name, &type,
                                     &tmp_name_index) != 0)
                {
                    return RETURN_FAILURE;
                }

                snprintf(word[highest_priority_index], sizeof(word[highest_priority_index]), "%s",
                         name);
                word_type[highest_priority_index] = type;
                for (int j = highest_priority_index + 1; j < nbword - 1; j++)
                {
                    snprintf(word[j], sizeof(word[j]), "%s", word[j + 1]);
                    word_type[j] = word_type[j + 1];
                }
                nbword = nbword - 1;
            }


            if (word_type[highest_priority_index] == ARITHTOKENTYPE_MULTFUNC)
            {
                int  nbvarinput = isfunction_sev_var(word[highest_priority_index]);
                char name[STRINGMAXLEN_IMGNAME];
                CREATE_IMAGENAME(name, "_tmp%d_%d", tmp_name_index, (int) getpid());

                int         hpi = highest_priority_index;
                int         a1t = 0, a2t = 0, a3t = 0;
                const char *a1w = NULL, *a2w = NULL, *a3w = NULL;

                if (nbvarinput >= 1)
                {
                    a1t = word_type[hpi + 2];
                    a1w = word[hpi + 2];
                }
                if (nbvarinput >= 2)
                {
                    a2t = word_type[hpi + 4];
                    a2w = word[hpi + 4];
                }
                if (nbvarinput >= 3)
                {
                    a3t = word_type[hpi + 6];
                    a3w = word[hpi + 6];
                }

                int type = 0;
                if (exec_arith_multfunc(word[hpi], nbvarinput, a1t, a1w, a2t, a2w, a3t, a3w, name,
                                        &type, &tmp_name_index) != 0)
                {
                    return RETURN_FAILURE;
                }

                snprintf(word[highest_priority_index], sizeof(word[highest_priority_index]), "%s",
                         name);
                word_type[highest_priority_index] = type;
                for (int j = highest_priority_index + 1; j < nbword - (nbvarinput * 2 + 1); j++)
                {
                    snprintf(word[j], sizeof(word[j]), "%s", word[j + (nbvarinput * 2 + 1)]);
                    word_type[j] = word_type[j + (nbvarinput * 2 + 1)];
                }
                nbword = nbword - nbvarinput * 2 - 1;
            }


            /*      printf("after : ");
              for (i=0;i<nbword;i++)
              {
              if(variable_ID(word[i])!=-1)
              printf(" %s(%f) ",word[i],dcvar[variable_ID(word[i])].value.f);
              else
              printf(" %s ",word[i]);
              }
              printf("\n");
            */
            /* computing the number of to-be-processed words */
            int passedequ = 0;
            nb_tbp_word   = 0;
            for (int i = (nbword - 1); i > -1; i--)
            {
                if (word_type[i] == ARITHTOKENTYPE_EQUAL)
                {
                    passedequ = 1;
                }
                if (passedequ == 0)
                {
                    nb_tbp_word++;
                }
            }
        }

        if (nbword > 2)
        {
            if (word_type[1] == ARITHTOKENTYPE_EQUAL)
            {
                if (variable_ID(word[0]) != -1)
                {
                    delete_variable_ID(word[0]);
                }
                if (imgid_exists(word[0]))
                {
                    delete_image_ID(word[0], DELETE_IMAGE_ERRMODE_WARNING);
                }

                if (word_type[2] == ARITHTOKENTYPE_VARIABLE)
                {
                    create_variable_ID(word[0], dcvar[variable_ID(word[2])].value.f);
                    printf("%.20g\n", dcvar[variable_ID(word[2])].value.f);
                }
                if (word_type[2] == ARITHTOKENTYPE_IMAGE)
                {
                    chname_image_ID(word[2], word[0]);
                }
            }
        }
        else
        {
            printf("%.20g\n", dcvar[variable_ID(word[0])].value.f);
        }

        for (int i = 0; i < tmp_name_index; i++)
        {
            char name[STRINGMAXLEN_IMGNAME];
            CREATE_IMAGENAME(name, "_tmp%d_%d", i, (int) getpid());
            if (variable_ID(name) != -1)
            {
                delete_variable_ID(name);
            }
            if (imgid_exists(name))
            {
                delete_image_ID(name, DELETE_IMAGE_ERRMODE_WARNING);
            }
        }
    }

    return (0);
}
