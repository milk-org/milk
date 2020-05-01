/**
 * @file    execute_arith.c
 * @brief   image arithmetic parser
 *
 *
 */

#include <math.h>
#include <ctype.h>

#include "CommandLineInterface/CLIcore.h"
#include "COREMOD_memory/COREMOD_memory.h"

#include "set_pixel.h"
#include "image_crop.h"
#include "image_merge3D.h"
#include "image_total.h"
#include "image_stats.h"
#include "image_dxdy.h"
#include "imfunctions.h"
#include "image_arith__im__im.h"
#include "image_arith__im_im__im.h"
#include "image_arith__Cim_Cim__Cim.h"
#include "image_arith__im_f__im.h"
#include "image_arith__im_f_f__im.h"



int isoperand(const char *word)
{
    int value = 0;

    if(strcmp(word, "+") == 0)
    {
        value = 1;
    }
    if(strcmp(word, "-") == 0)
    {
        value = 1;
    }
    if(strcmp(word, "/") == 0)
    {
        value = 1;
    }
    if(strcmp(word, "*") == 0)
    {
        value = 1;
    }
    if(strcmp(word, "^") == 0)
    {
        value = 1;
    }

    return(value);
}


int isfunction(const char *word)
{
    int value = 0;

    if(strcmp(word, "acos") == 0)
    {
        value = 1;
    }
    if(strcmp(word, "asin") == 0)
    {
        value = 1;
    }
    if(strcmp(word, "atan") == 0)
    {
        value = 1;
    }
    if(strcmp(word, "ceil") == 0)
    {
        value = 1;
    }
    if(strcmp(word, "cos") == 0)
    {
        value = 1;
    }
    if(strcmp(word, "cosh") == 0)
    {
        value = 1;
    }
    if(strcmp(word, "exp") == 0)
    {
        value = 1;
    }
    if(strcmp(word, "fabs") == 0)
    {
        value = 1;
    }
    if(strcmp(word, "floor") == 0)
    {
        value = 1;
    }
    if(strcmp(word, "imedian") == 0)
    {
        value = 1;
    }
    if(strcmp(word, "itot") == 0)
    {
        value = 1;
    }
    if(strcmp(word, "imean") == 0)
    {
        value = 1;
    }
    if(strcmp(word, "imin") == 0)
    {
        value = 1;
    }
    if(strcmp(word, "imax") == 0)
    {
        value = 1;
    }
    if(strcmp(word, "ln") == 0)
    {
        value = 1;
    }
    if(strcmp(word, "log") == 0)
    {
        value = 1;
    }
    if(strcmp(word, "sqrt") == 0)
    {
        value = 1;
    }
    if(strcmp(word, "sin") == 0)
    {
        value = 1;
    }
    if(strcmp(word, "sinh") == 0)
    {
        value = 1;
    }
    if(strcmp(word, "tan") == 0)
    {
        value = 1;
    }
    if(strcmp(word, "tanh") == 0)
    {
        value = 1;
    }
    if(strcmp(word, "posi") == 0)
    {
        value = 1;
    }
    if(strcmp(word, "imdx") == 0)
    {
        value = 1;
    }
    if(strcmp(word, "imdy") == 0)
    {
        value = 1;
    }


    /*  if (!strcmp(word,"pow"))
        value = 1;
      if (!strcmp(word,"max"))
        value = 1;
      if (!strcmp(word,"min"))
        value = 1;
      if (!strcmp(word,"median"))
        value = 1;
    */
    return(value);
}



int isfunction_sev_var(const char *word)
{
    int value = 0; /* number of input variables */

    if(strcmp(word, "fmod") == 0)
    {
        value = 2;
    }
    if(strcmp(word, "trunc") == 0)
    {
        value = 3;
    }
    if(strcmp(word, "perc") == 0)
    {
        value = 2;
    }
    if(strcmp(word, "min") == 0)
    {
        value = 2;
    }
    if(strcmp(word, "max") == 0)
    {
        value = 2;
    }
    if(strcmp(word, "testlt") == 0)
    {
        value = 2;
    }
    if(strcmp(word, "testmt") == 0)
    {
        value = 2;
    }


    return(value);
}


int isanumber(const char *word)
{
    int value = 1; // 1 if number, 0 otherwise
    char *endptr;
    __attribute__((unused)) double v1;

    v1 = strtod(word, &endptr);
    if((long)(endptr - word) == (long) strlen(word))
    {
        value = 1;
    }
    else
    {
        value = 0;
    }

    return(value);
}




imageID arith_make_slopexy(
    const char *ID_name,
    long l1,
    long l2,
    double sx,
    double sy
)
{
    imageID ID;
    uint32_t naxes[2];
    double coeff;

    create_2Dimage_ID(ID_name, l1, l2);
    ID = image_ID(ID_name);
    naxes[0] = data.image[ID].md[0].size[0];
    naxes[1] = data.image[ID].md[0].size[1];

    coeff = sx * (naxes[0] / 2) + sy * (naxes[1] / 2);

    for(uint32_t jj = 0; jj < naxes[1]; jj++)
        for(uint32_t ii = 0; ii < naxes[0]; ii++)
        {
            data.image[ID].array.F[jj * naxes[0] + ii] = sx * ii + sy * jj - coeff;
        }

    return ID;
}







/*^-----------------------------------------------------------------------------
| int
| execute_arith :
|     const char *cmd :
|
|
|    0 : unknown
|    1 : non-existing variable or image
|    2 : existing variable
|    3 : number
|    4 : operand
|    5 : opening brace
|    6 : closing brace
|    7 : coma
|    8 : function
|    9 : equal sign
|    10 : existing image
|    11 : function of several variables/images, returning one variable/image
|
|
+-----------------------------------------------------------------------------*/
int execute_arith(const char *cmd1)
{
    char word[100][100];
    int i, w, l, j;
    int nbword;
    int word_type[100];
    int par_level[100];
    int parlevel;
    int intr_priority[100]; /* 0 (+,-)  1 (*,/)  2 (functions) */


    int found_word_type;
    int highest_parlevel;
    int highest_intr_priority;
    int highest_priority_index;
    int passedequ;
    int tmp_name_index;
    double tmp_prec;
    int nb_tbp_word;
    int type = 0;
    int nbvarinput;

	int CMDBUFFSIZE = 1000;
    char cmd[CMDBUFFSIZE];
    long cntP;
    int OKea = 1;

    int Debug = 0;

    //  if( Debug > 0 )   fprintf(stdout, "[execute_arith]\n");
    //  if( Debug > 0 )   fprintf(stdout, "[execute_arith] str: [%s]\n", cmd1);

    for(i = 0; i < 100; i++)
    {
        word_type[i] = 0;
        par_level[i] = 0;
        intr_priority[i] = 0;
    }




    /*
       Pre-process string:
       - remove any spaces in cmd1
       - replace "=-" by "=0-" and "=+" by "="
       copy result into cmd */
    j = 0;

    for(i = 0; i < (int)(strlen(cmd1)); i++)
    {
        if((cmd1[i] == '=') && (cmd1[i + 1] == '-'))
        {
            cmd[j] = '=';
            j++;
            cmd[j] = '0';
            j++;
        }
        else if((cmd1[i] == '=') && (cmd1[i + 1] == '+'))
        {
            cmd[j] = '=';
            j++;
            i++;
        }
        else if(cmd1[i] != ' ')
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
    w = 0;
    l = 0;
    for(i = 0; i < (signed) strlen(cmd); i++)
    {
        switch(cmd[i])
        {

            case '+':
            case '-':
                if(((cmd[i - 1] == 'e') || (cmd[i - 1] == 'E')) && (isdigit(cmd[i - 2]))
                        && (isdigit(cmd[i + 1])))
                {
                    // + or - is part of exponent
                    word[w][l] = cmd[i];
                    l++;
                }
                else
                {
                    if(l > 0)
                    {
                        word[w][l] = '\0';
                        w++;
                    }
                    l = 0;
                    word[w][l] = cmd[i];
                    word[w][1] = '\0';
                    if(i < (signed)(strlen(cmd) - 1))
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
                if(l > 0)
                {
                    word[w][l] = '\0';
                    w++;
                }
                l = 0;
                word[w][l] = cmd[i];
                word[w][1] = '\0';
                if(i < (signed)(strlen(cmd) - 1))
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


    if(l > 0)
    {
        word[w][l] = '\0';
    }
    nbword = w + 1;


    //  printf("number of words is %d\n",nbword);

    for(i = 0; i < nbword; i++)
    {
        if(Debug > 0)
        {
            printf("TESTING WORD %d = %s\n", i, word[i]);
        }
        word_type[i] = 0;
        found_word_type = 0;
        if((isanumber(word[i]) == 1) && (found_word_type == 0))
        {
            word_type[i] = 3;
            found_word_type = 1;
        }
        if((isfunction(word[i]) == 1) && (found_word_type == 0))
        {
            word_type[i] = 8;
            found_word_type = 1;
        }
        if((isfunction_sev_var(word[i]) != 0) && (found_word_type == 0))
        {
            word_type[i] = 11;
            found_word_type = 1;
        }
        if((isoperand(word[i]) == 1) && (found_word_type == 0))
        {
            word_type[i] = 4;
            found_word_type = 1;
        }
        if((strcmp(word[i], "=") == 0) && (found_word_type == 0))
        {
            word_type[i] = 9;
            found_word_type = 1;
        }
        if((strcmp(word[i], ",") == 0) && (found_word_type == 0))
        {
            word_type[i] = 7;
            found_word_type = 1;
        }
        if((i < nbword - 1) && (found_word_type == 0))
        {
            if((strcmp(word[i + 1], "(") == 0) && (isfunction(word[i]) == 1))
            {
                word_type[i] = 8;
                found_word_type = 1;
            }
        }
        if((strcmp(word[i], "(") == 0) && (found_word_type == 0))
        {
            word_type[i] = 5;
            found_word_type = 1;
        }
        if((strcmp(word[i], ")") == 0) && (found_word_type == 0))
        {
            word_type[i] = 6;
            found_word_type = 1;
        }
        if((variable_ID(word[i]) != -1) && (found_word_type == 0))
        {
            word_type[i] = 2;
            found_word_type = 1;
        }
        if((image_ID(word[i]) != -1) && (found_word_type == 0))
        {
            word_type[i] = 10;
            found_word_type = 1;
        }
        if(found_word_type == 0)
        {
            word_type[i] = 1;
        }
        if(Debug > 0)
        {
            printf("word %d is  \"%s\" word typ is %d\n", i, word[i], word_type[i]);
        }
    }









    /* checks for obvious errors */

    passedequ = 0;
    for(i = (nbword - 1); i > -1; i--)
    {
        if(passedequ == 1)
        {
            if(word_type[i] == 9)
            {
                PRINT_WARNING("line has multiple \"=\"");
                OKea = 0;
            }
            if(word_type[i] == 4)
            {
                PRINT_WARNING("operand on left side of \"=\"");
                OKea = 0;
            }
            if(word_type[i] == 5)
            {
                PRINT_WARNING("\"(\" on left side of \"=\"");
                OKea = 0;
            }
            if(word_type[i] == 6)
            {
                PRINT_WARNING( "\")\" on left side of \"=\"");
                OKea = 0;
            }
        }
        if(word_type[i] == 9)
        {
            passedequ = 1;
        }
        if((passedequ == 0)
                && (word_type[i] == 1)) /* non-existing variable or image as input */
        {
            PRINT_WARNING("%s is a non-existing variable or image", word[i]);
            OKea = 0;
        }
    }

    for(i = 0; i < nbword - 1; i++)
    {
        if((word_type[i] == 4) && (word_type[i + 1] == 4))
        {
            PRINT_WARNING("consecutive operands");
            OKea = 0;
        }
        if((word_type[i + 1] == 5) && (!((word_type[i] == 5) || (word_type[i] == 8)
                                         || (word_type[i] == 11) || (word_type[i] == 9) || (word_type[i] == 4))))
        {
            PRINT_WARNING("\"(\" should be preceeded by \"=\", \"(\", operand or function");
            OKea = 0;
        }
    }

    cntP = 0;
    for(i = 0; i < nbword; i++)
    {
        if(word_type[i] == 5)
        {
            cntP ++;
        }
        if(word_type[i] == 6)
        {
            cntP --;
        }
        if(cntP < 0)
        {
            PRINT_WARNING("parentheses error");
            OKea = 0;
        }
    }
    if(cntP != 0)
    {
        PRINT_WARNING("parentheses error");
        OKea = 0;
    }










    if(OKea == 1)
    {
        /* numbers are saved into variables */
        tmp_name_index = 0;
        for(i = 0; i < nbword; i++)
        {
            if(word_type[i] == 3)
            {
                CREATE_IMAGENAME(name, "_tmp%d_%d", tmp_name_index, (int) getpid());

                create_variable_ID(name, 1.0 * strtod(word[i], NULL));
                strcpy(word[i], name);
                word_type[i] = 2;
                tmp_name_index++;
            }
        }

        /* computing the number of to-be-processed words */
        passedequ = 0;
        nb_tbp_word = 0;
        for(i = (nbword - 1); i > -1; i--)
        {
            if(word_type[i] == 9)
            {
                passedequ = 1;
            }
            if(passedequ == 0)
            {
                nb_tbp_word++;
            }
        }

        /* main loop starts here */
        while(nb_tbp_word > 1)
        {
            /* non necessary braces are removed
             */
            for(i = 0; i < nbword - 2; i++)
                if((word_type[i] == 5) && (word_type[i + 2] == 6))
                {
                    strcpy(word[i], word[i + 1]);
                    word_type[i] = word_type[i + 1];
                    for(j = i + 1; j < nbword - 2; j++)
                    {
                        strcpy(word[j], word[j + 2]);
                        word_type[j] = word_type[j + 2];
                    }
                    nbword = nbword - 2;
                }
            for(i = 0; i < nbword - 3; i++)
                if((word_type[i] == 5) && (word_type[i + 3] == 6)
                        && (strcmp(word[i + 1], "-") == 0))
                {
                    data.variable[variable_ID(word[i + 2])].value.f = -data.variable[variable_ID(
                                word[i + 2])].value.f;
                    strcpy(word[i], word[i + 2]);
                    word_type[i] = word_type[i + 2];
                    for(j = i + 2; j < nbword - 3; j++)
                    {
                        strcpy(word[j], word[j + 3]);
                        word_type[j] = word_type[j + 3];
                    }
                    nbword = nbword - 3;
                }

            /* now the priorities are given */

            parlevel = 0;
            for(i = 0; i < nbword; i++)
            {
                if(word_type[i] == 5)
                {
                    parlevel++;
                }
                if(word_type[i] == 6)
                {
                    parlevel--;
                }
                if((word_type[i] == 4) || (word_type[i] == 8) || (word_type[i] == 11))
                {
                    par_level[i] = parlevel;
                    if(word_type[i] == 8)
                    {
                        intr_priority[i] = 2;
                    }
                    if(word_type[i] == 11)
                    {
                        intr_priority[i] = 2;
                    }
                    if(word_type[i] == 4)
                    {
                        if((strcmp(word[i], "+") == 0) || (strcmp(word[i], "-") == 0))
                        {
                            intr_priority[i] = 0;
                        }
                        if((strcmp(word[i], "*") == 0) || (strcmp(word[i], "/") == 0))
                        {
                            intr_priority[i] = 1;
                        }
                    }
                }
            }

            /* the highest priority operation is executed */
            highest_parlevel = 0;
            highest_intr_priority = -1;
            highest_priority_index = -1;

            for(i = 0; i < nbword; i++)
            {
                if((word_type[i] == 4) || (word_type[i] == 8) || (word_type[i] == 11))
                {
                    /*printf("operation \"%s\" (%d,%d)\n",word[i],par_level[i],intr_priority[i]);*/
                    if(par_level[i] > highest_parlevel)
                    {
                        highest_priority_index = i;
                        highest_parlevel = par_level[i];
                        highest_intr_priority = 0;
                    }
                    else
                    {
                        if((par_level[i] == highest_parlevel)
                                && (intr_priority[i] > highest_intr_priority))
                        {
                            highest_priority_index = i;
                            highest_intr_priority = intr_priority[i];
                        }
                    }
                }
            }

            /*      printf("executing operation  %s\n",word[highest_priority_index]);*/

            i = highest_priority_index;

            /*      printf("before : ");
              for (j=0;j<nbword;j++)
              {
              if(j==i)
              printf(">>");
              if(variable_ID(word[j])!=-1)
              printf(" %s(%f) ",word[j],data.variable[variable_ID(word[j])].value.f);
              else
              printf(" %s ",word[j]);
              }
              printf("\n");
            */
            if(word_type[i] == 4)
            {
				// name of image/variable where output is written
				CREATE_IMAGENAME(name, "_tmp%d_%d", tmp_name_index, (int) getpid());
				
                if(strcmp(word[i], "+") == 0)
                {
                    if((word_type[i - 1] == 2) && (word_type[i + 1] == 2))
                    {
                        tmp_prec = data.variable[variable_ID(word[i - 1])].value.f +
                                   data.variable[variable_ID(word[i + 1])].value.f;
                        create_variable_ID(name, tmp_prec);
                        tmp_name_index++;
                        type = 2;
                    }
                    if((word_type[i - 1] == 2) && (word_type[i + 1] == 10))
                    {
                        arith_image_cstadd(word[i + 1],
                                           (double) data.variable[variable_ID(word[i - 1])].value.f, name);
                        tmp_name_index++;
                        type = 10;
                    }
                    if((word_type[i - 1] == 10) && (word_type[i + 1] == 2))
                    {
                        arith_image_cstadd(word[i - 1],
                                           (double) data.variable[variable_ID(word[i + 1])].value.f, name);
                        tmp_name_index++;
                        type = 10;
                    }
                    if((word_type[i - 1] == 10) && (word_type[i + 1] == 10))
                    {
                        arith_image_add(word[i - 1], word[i + 1], name);
                        tmp_name_index++;
                        type = 10;
                    }
                }

                if(strcmp(word[i], "-") == 0)
                {
                    if((word_type[i - 1] == 2) && (word_type[i + 1] == 2))
                    {
                        tmp_prec = data.variable[variable_ID(word[i - 1])].value.f -
                                   data.variable[variable_ID(word[i + 1])].value.f;
                        create_variable_ID(name, tmp_prec);
                        tmp_name_index++;
                        type = 2;
                    }
                    if((word_type[i - 1] == 2) && (word_type[i + 1] == 10))
                    {
                        CREATE_IMAGENAME(name1, "_tmp1%d_%d", tmp_name_index, (int) getpid());
                        arith_image_cstsub(word[i + 1],
                                           (double) data.variable[variable_ID(word[i - 1])].value.f, name1);
                        arith_image_cstmult(name1, (double) -1.0, name);
                        delete_image_ID(name1);
                        tmp_name_index++;
                        type = 10;
                    }
                    if((word_type[i - 1] == 10) && (word_type[i + 1] == 2))
                    {
                        arith_image_cstsub(word[i - 1],
                                           (double) data.variable[variable_ID(word[i + 1])].value.f, name);
                        tmp_name_index++;
                        type = 10;
                    }
                    if((word_type[i - 1] == 10) && (word_type[i + 1] == 10))
                    {
                        arith_image_sub(word[i - 1], word[i + 1], name);
                        tmp_name_index++;
                        type = 10;
                    }
                }

                if(strcmp(word[i], "*") == 0)
                {
                    if((word_type[i - 1] == 2) && (word_type[i + 1] == 2))
                    {
                        tmp_prec = data.variable[variable_ID(word[i - 1])].value.f *
                                   data.variable[variable_ID(word[i + 1])].value.f;
                        create_variable_ID(name, tmp_prec);
                        tmp_name_index++;
                        type = 2;
                    }
                    if((word_type[i - 1] == 2) && (word_type[i + 1] == 10))
                    {
                        arith_image_cstmult(word[i + 1],
                                            (double) data.variable[variable_ID(word[i - 1])].value.f, name);
                        tmp_name_index++;
                        type = 10;
                    }
                    if((word_type[i - 1] == 10) && (word_type[i + 1] == 2))
                    {
                        arith_image_cstmult(word[i - 1],
                                            (double) data.variable[variable_ID(word[i + 1])].value.f, name);
                        tmp_name_index++;
                        type = 10;
                    }
                    if((word_type[i - 1] == 10) && (word_type[i + 1] == 10))
                    {
                        arith_image_mult(word[i - 1], word[i + 1], name);
                        tmp_name_index++;
                        type = 10;
                    }
                }

                if(strcmp(word[i], "/") == 0)
                {
                    if((word_type[i - 1] == 2) && (word_type[i + 1] == 2))
                    {
                        tmp_prec = data.variable[variable_ID(word[i - 1])].value.f /
                                   data.variable[variable_ID(word[i + 1])].value.f;
                        create_variable_ID(name, tmp_prec);
                        tmp_name_index++;
                        type = 2;
                    }
                    if((word_type[i - 1] == 2) && (word_type[i + 1] == 10))
                    {
                        //    printf("CASE 1\n");
                        arith_image_cstdiv1(word[i + 1],
                                            (double) data.variable[variable_ID(word[i - 1])].value.f, name);
                        tmp_name_index++;
                        type = 10;
                    }
                    if((word_type[i - 1] == 10) && (word_type[i + 1] == 2))
                    {
                        arith_image_cstdiv(word[i - 1],
                                           (double) data.variable[variable_ID(word[i + 1])].value.f, name);
                        tmp_name_index++;
                        type = 10;
                    }
                    if((word_type[i - 1] == 10) && (word_type[i + 1] == 10))
                    {
                        arith_image_div(word[i - 1], word[i + 1], name);
                        tmp_name_index++;
                        type = 10;
                    }
                }

                if(strcmp(word[i], "^") == 0)
                {
                    if((word_type[i - 1] == 2) && (word_type[i + 1] == 2))
                    {
                        if(data.variable[variable_ID(word[i + 1])].value.f < 0)
                        {
                            tmp_prec = pow(data.variable[variable_ID(word[i - 1])].value.f,
                                           -data.variable[variable_ID(word[i + 1])].value.f);
                            tmp_prec = 1.0 / tmp_prec;
                        }
                        else
                        {
                            tmp_prec = pow(data.variable[variable_ID(word[i - 1])].value.f,
                                           data.variable[variable_ID(word[i + 1])].value.f);
                        }
                        create_variable_ID(name, tmp_prec);
                        tmp_name_index++;
                        type = 2;
                    }
                    if((word_type[i - 1] == 2) && (word_type[i + 1] == 10))
                    {
                        CREATE_IMAGENAME(name1, "_tmp1%d_%d", tmp_name_index, (int) getpid());
                        arith_image_cstadd(word[i + 1],
                                           (double) data.variable[variable_ID(word[i - 1])].value.f, name1);
                        arith_image_pow(name1, word[i + 1], name);
                        delete_image_ID(name1);
                        tmp_name_index++;
                        type = 10;
                    }
                    if((word_type[i - 1] == 10) && (word_type[i + 1] == 2))
                    {
                        arith_image_cstpow(word[i - 1],
                                           (double) data.variable[variable_ID(word[i + 1])].value.f, name);
                        tmp_name_index++;
                        type = 10;
                    }
                    if((word_type[i - 1] == 10) && (word_type[i + 1] == 10))
                    {
                        arith_image_pow(word[i - 1], word[i + 1], name);
                        tmp_name_index++;
                        type = 10;
                    }
                }

                strcpy(word[i - 1], name);
                word_type[i - 1] = type;
                for(j = i; j < nbword - 2; j++)
                {
                    strcpy(word[j], word[j + 2]);
                    word_type[j] = word_type[j + 2];
                }
                nbword = nbword - 2;
            }



            if(word_type[i] == 8)
            {
				CREATE_IMAGENAME(name, "_tmp%d_%d", tmp_name_index, (int) getpid());
				
                if(strcmp(word[i], "acos") == 0)
                {
                    if(word_type[i + 1] == 2)
                    {
                        tmp_prec = acos(data.variable[variable_ID(word[i + 1])].value.f);
                        create_variable_ID(name, tmp_prec);
                        tmp_name_index++;
                        type = 2;
                    }
                    if(word_type[i + 1] == 10)
                    {
                        arith_image_acos(word[i + 1], name);
                        tmp_name_index++;
                        type = 10;
                    }
                }

                if(strcmp(word[i], "asin") == 0)
                {
                    if(word_type[i + 1] == 2)
                    {
                        tmp_prec = asin(data.variable[variable_ID(word[i + 1])].value.f);
                        create_variable_ID(name, tmp_prec);
                        tmp_name_index++;
                        type = 2;
                    }
                    if(word_type[i + 1] == 10)
                    {
                        arith_image_asin(word[i + 1], name);
                        tmp_name_index++;
                        type = 10;
                    }
                }

                if(strcmp(word[i], "atan") == 0)
                {
                    if(word_type[i + 1] == 2)
                    {
                        tmp_prec = atan(data.variable[variable_ID(word[i + 1])].value.f);
                        create_variable_ID(name, tmp_prec);
                        tmp_name_index++;
                        type = 2;
                    }
                    if(word_type[i + 1] == 10)
                    {
                        arith_image_atan(word[i + 1], name);
                        tmp_name_index++;
                        type = 10;
                    }
                }

                if(strcmp(word[i], "ceil") == 0)
                {
                    if(word_type[i + 1] == 2)
                    {
                        tmp_prec = (double) ceil(data.variable[variable_ID(word[i + 1])].value.f);
                        create_variable_ID(name, tmp_prec);
                        tmp_name_index++;
                        type = 2;
                    }
                    if(word_type[i + 1] == 10)
                    {
                        arith_image_ceil(word[i + 1], name);
                        tmp_name_index++;
                        type = 10;
                    }
                }

                if(strcmp(word[i], "cos") == 0)
                {
                    if(word_type[i + 1] == 2)
                    {
                        tmp_prec = cos(data.variable[variable_ID(word[i + 1])].value.f);
                        create_variable_ID(name, tmp_prec);
                        tmp_name_index++;
                        type = 2;
                    }
                    if(word_type[i + 1] == 10)
                    {
                        arith_image_cos(word[i + 1], name);
                        tmp_name_index++;
                        type = 10;
                    }
                }

                if(strcmp(word[i], "cosh") == 0)
                {
                    if(word_type[i + 1] == 2)
                    {
                        tmp_prec = cosh(data.variable[variable_ID(word[i + 1])].value.f);
                        create_variable_ID(name, tmp_prec);
                        tmp_name_index++;
                        type = 2;
                    }
                    if(word_type[i + 1] == 10)
                    {
                        arith_image_cosh(word[i + 1], name);
                        tmp_name_index++;
                        type = 10;
                    }
                }

                if(strcmp(word[i], "exp") == 0)
                {
                    if(word_type[i + 1] == 2)
                    {
                        tmp_prec = exp(data.variable[variable_ID(word[i + 1])].value.f);
                        create_variable_ID(name, tmp_prec);
                        tmp_name_index++;
                        type = 2;
                    }
                    if(word_type[i + 1] == 10)
                    {
                        arith_image_exp(word[i + 1], name);
                        tmp_name_index++;
                        type = 10;
                    }
                }

                if(strcmp(word[i], "fabs") == 0)
                {
                    if(word_type[i + 1] == 2)
                    {
                        tmp_prec = fabs(data.variable[variable_ID(word[i + 1])].value.f);
                        create_variable_ID(name, tmp_prec);
                        tmp_name_index++;
                        type = 2;
                    }
                    if(word_type[i + 1] == 10)
                    {
                        arith_image_fabs(word[i + 1], name);
                        tmp_name_index++;
                        type = 10;
                    }
                }

                if(strcmp(word[i], "floor") == 0)
                {
                    if(word_type[i + 1] == 2)
                    {
                        tmp_prec = floor(data.variable[variable_ID(word[i + 1])].value.f);
                        create_variable_ID(name, tmp_prec);
                        tmp_name_index++;
                        type = 2;
                    }
                    if(word_type[i + 1] == 10)
                    {
                        arith_image_floor(word[i + 1], name);
                        tmp_name_index++;
                        type = 10;
                    }
                }

                if(strcmp(word[i], "imedian") == 0)
                {
                    if(word_type[i + 1] == 10)
                    {
                        tmp_prec = arith_image_median(word[i + 1]);
                        create_variable_ID(name, tmp_prec);
                        tmp_name_index++;
                        type = 2;
                    }
                }

                if(strcmp(word[i], "itot") == 0)
                {
                    if(word_type[i + 1] == 10)
                    {
                        tmp_prec = arith_image_total(word[i + 1]);
                        create_variable_ID(name, tmp_prec);
                        tmp_name_index++;
                        type = 2;
                    }
                }

                if(strcmp(word[i], "imean") == 0)
                {
                    if(word_type[i + 1] == 10)
                    {
                        tmp_prec = arith_image_mean(word[i + 1]);
                        create_variable_ID(name, tmp_prec);
                        tmp_name_index++;
                        type = 2;
                    }
                }

                if(strcmp(word[i], "imin") == 0)
                {
                    if(word_type[i + 1] == 10)
                    {
                        tmp_prec = arith_image_min(word[i + 1]);
                        create_variable_ID(name, tmp_prec);
                        tmp_name_index++;
                        type = 2;
                    }
                }

                if(strcmp(word[i], "imax") == 0)
                {
                    if(word_type[i + 1] == 10)
                    {
                        tmp_prec = arith_image_max(word[i + 1]);
                        create_variable_ID(name, tmp_prec);
                        tmp_name_index++;
                        type = 2;
                    }
                }

                if(strcmp(word[i], "ln") == 0)
                {
                    if(word_type[i + 1] == 2)
                    {
                        tmp_prec = log(data.variable[variable_ID(word[i + 1])].value.f);
                        create_variable_ID(name, tmp_prec);
                        tmp_name_index++;
                        type = 2;
                    }
                    if(word_type[i + 1] == 10)
                    {
                        arith_image_ln(word[i + 1], name);
                        tmp_name_index++;
                        type = 10;
                    }
                }

                if(strcmp(word[i], "log") == 0)
                {
                    if(word_type[i + 1] == 2)
                    {
                        tmp_prec = log10(data.variable[variable_ID(word[i + 1])].value.f);
                        create_variable_ID(name, tmp_prec);
                        tmp_name_index++;
                        type = 2;
                    }
                    if(word_type[i + 1] == 10)
                    {
                        arith_image_log(word[i + 1], name);
                        tmp_name_index++;
                        type = 10;
                    }
                }

                if(strcmp(word[i], "sqrt") == 0)
                {
                    if(word_type[i + 1] == 2)
                    {
                        tmp_prec = sqrt(data.variable[variable_ID(word[i + 1])].value.f);
                        create_variable_ID(name, tmp_prec);
                        tmp_name_index++;
                        type = 2;
                    }
                    if(word_type[i + 1] == 10)
                    {
                        arith_image_sqrt(word[i + 1], name);
                        tmp_name_index++;
                        type = 10;
                    }
                }

                if(strcmp(word[i], "sin") == 0)
                {
                    if(word_type[i + 1] == 2)
                    {
                        tmp_prec = sin(data.variable[variable_ID(word[i + 1])].value.f);
                        create_variable_ID(name, tmp_prec);
                        tmp_name_index++;
                        type = 2;
                    }
                    if(word_type[i + 1] == 10)
                    {
                        arith_image_sin(word[i + 1], name);
                        tmp_name_index++;
                        type = 10;
                    }
                }

                if(strcmp(word[i], "sinh") == 0)
                {
                    if(word_type[i + 1] == 2)
                    {
                        tmp_prec = sinh(data.variable[variable_ID(word[i + 1])].value.f);
                        create_variable_ID(name, tmp_prec);
                        tmp_name_index++;
                        type = 2;
                    }
                    if(word_type[i + 1] == 10)
                    {
                        arith_image_sinh(word[i + 1], name);
                        tmp_name_index++;
                        type = 10;
                    }
                }

                if(strcmp(word[i], "tan") == 0)
                {
                    printf("LINE 4440\n");//TBE
                    if(word_type[i + 1] == 2)
                    {
                        tmp_prec = tan(data.variable[variable_ID(word[i + 1])].value.f);
                        create_variable_ID(name, tmp_prec);
                        tmp_name_index++;
                        type = 2;
                    }
                    if(word_type[i + 1] == 10)
                    {
                        arith_image_tan(word[i + 1], name);
                        tmp_name_index++;
                        type = 10;
                    }
                }

                if(strcmp(word[i], "tanh") == 0)
                {
                    if(word_type[i + 1] == 2)
                    {
                        tmp_prec = tanh(data.variable[variable_ID(word[i + 1])].value.f);
                        create_variable_ID(name, tmp_prec);
                        tmp_name_index++;
                        type = 2;
                    }
                    if(word_type[i + 1] == 10)
                    {
                        arith_image_tanh(word[i + 1], name);
                        tmp_name_index++;
                        type = 10;
                    }
                }

                if(strcmp(word[i], "posi") == 0)
                {
                    if(word_type[i + 1] == 2)
                    {
                        tmp_prec = Ppositive((double) data.variable[variable_ID(word[i + 1])].value.f);
                        create_variable_ID(name, tmp_prec);
                        tmp_name_index++;
                        type = 2;
                    }
                    if(word_type[i + 1] == 10)
                    {
                        arith_image_positive(word[i + 1], name);
                        tmp_name_index++;
                        type = 10;
                    }
                }

                if(strcmp(word[i], "imdx") == 0)
                {
                    if(word_type[i + 1] == 2)
                    {
                        PRINT_ERROR("Function imdx only applicable on images");
                        exit(0);
                    }
                    if(word_type[i + 1] == 10)
                    {
                        arith_image_dx(word[i + 1], name);
                        tmp_name_index++;
                        type = 10;
                    }
                }

                if(strcmp(word[i], "imdy") == 0)
                {
                    if(word_type[i + 1] == 2)
                    {
                        PRINT_ERROR("Function imdy only applicable on images");
                        exit(0);
                    }
                    if(word_type[i + 1] == 10)
                    {
                        arith_image_dy(word[i + 1], name);
                        tmp_name_index++;
                        type = 10;
                    }
                }


                strcpy(word[i], name);
                word_type[i] = type;
                for(j = i + 1; j < nbword - 1; j++)
                {
                    strcpy(word[j], word[j + 1]);
                    word_type[j] = word_type[j + 1];
                }
                nbword = nbword - 1;
            }




            if(word_type[i] == 11)
            {
                nbvarinput = isfunction_sev_var(word[i]);
                CREATE_IMAGENAME(name, "_tmp%d_%d", tmp_name_index, (int) getpid());

                if(strcmp(word[i], "fmod") == 0)
                {
                    if((word_type[i + 2] == 2) && (word_type[i + 4] == 2))
                    {
                        tmp_prec = fmod(data.variable[variable_ID(word[i + 2])].value.f,
                                        data.variable[variable_ID(word[i + 4])].value.f);                        
                        create_variable_ID(name, tmp_prec);
                        tmp_name_index++;
                        type = 2;
                    }
                    if((word_type[i + 2] == 2) && (word_type[i + 4] == 10))
                    {
                        printf("function not available\n");
                    }
                    if((word_type[i + 2] == 10) && (word_type[i + 4] == 2))
                    {
                        arith_image_cstfmod(word[i + 2],
                                            (double) data.variable[variable_ID(word[i + 4])].value.f, name);
                        tmp_name_index++;
                        type = 10;
                    }
                    if((word_type[i + 2] == 10) && (word_type[i + 4] == 10))
                    {
                        arith_image_fmod(word[i + 2], word[i + 4], name);
                        tmp_name_index++;
                        type = 10;
                    }
                }


                if(strcmp(word[i], "min") == 0)
                {
                    if((word_type[i + 2] == 2) && (word_type[i + 4] == 2))
                    {
                        if(data.variable[variable_ID(word[i + 2])].value.f < data.variable[variable_ID(
                                    word[i + 4])].value.f)
                        {
                            tmp_prec = data.variable[variable_ID(word[i + 2])].value.f;
                        }
                        else
                        {
                            tmp_prec = data.variable[variable_ID(word[i + 4])].value.f;
                        }
                        create_variable_ID(name, tmp_prec);
                        tmp_name_index++;
                        type = 2;
                    }
                    if((word_type[i + 2] == 2) && (word_type[i + 4] == 10))
                    {
                        arith_image_cstminv(word[i + 4],
                                            (double) data.variable[variable_ID(word[i + 2])].value.f, name);
                        tmp_name_index++;
                        type = 10;
                    }
                    if((word_type[i + 2] == 10) && (word_type[i + 4] == 2))
                    {
                        arith_image_cstminv(word[i + 2],
                                            (double) data.variable[variable_ID(word[i + 4])].value.f, name);
                        tmp_name_index++;
                        type = 10;
                    }
                    if((word_type[i + 2] == 10) && (word_type[i + 4] == 10))
                    {
                        arith_image_minv(word[i + 2], word[i + 4], name);
                        tmp_name_index++;
                        type = 10;
                    }
                }


                if(strcmp(word[i], "max") == 0)
                {
                    if((word_type[i + 2] == 2) && (word_type[i + 4] == 2))
                    {
                        if(data.variable[variable_ID(word[i + 2])].value.f > data.variable[variable_ID(
                                    word[i + 4])].value.f)
                        {
                            tmp_prec = data.variable[variable_ID(word[i + 2])].value.f;
                        }
                        else
                        {
                            tmp_prec = data.variable[variable_ID(word[i + 4])].value.f;
                        }
                        create_variable_ID(name, tmp_prec);
                        tmp_name_index++;
                        type = 2;
                    }
                    else if((word_type[i + 2] == 2) && (word_type[i + 4] == 10))
                    {
                        arith_image_cstmaxv(word[i + 4],
                                            (double) data.variable[variable_ID(word[i + 2])].value.f, name);
                        tmp_name_index++;
                        type = 10;
                    }
                    else if((word_type[i + 2] == 10) && (word_type[i + 4] == 2))
                    {
                        arith_image_cstmaxv(word[i + 2],
                                            (double) data.variable[variable_ID(word[i + 4])].value.f, name);
                        tmp_name_index++;
                        type = 10;
                    }
                    else if((word_type[i + 2] == 10) && (word_type[i + 4] == 10))
                    {
                        arith_image_maxv(word[i + 2], word[i + 4], name);
                        tmp_name_index++;
                        type = 10;
                    }
                }

                if(strcmp(word[i], "testlt") == 0)
                {
                    if((word_type[i + 2] == 2) && (word_type[i + 4] == 2))
                    {
                        if(data.variable[variable_ID(word[i + 2])].value.f > data.variable[variable_ID(
                                    word[i + 4])].value.f)
                        {
                            tmp_prec = 0.0;
                        }
                        else
                        {
                            tmp_prec = 1.0;
                        }
                        create_variable_ID(name, tmp_prec);
                        tmp_name_index++;
                        type = 2;
                    }
                    else if((word_type[i + 2] == 2) && (word_type[i + 4] == 10))
                    {
                        arith_image_csttestmt(word[i + 4],
                                              (double) data.variable[variable_ID(word[i + 2])].value.f, name);
                        tmp_name_index++;
                        type = 10;
                    }
                    else if((word_type[i + 2] == 10) && (word_type[i + 4] == 2))
                    {
                        arith_image_csttestlt(word[i + 2],
                                              (double) data.variable[variable_ID(word[i + 4])].value.f, name);
                        tmp_name_index++;
                        type = 10;
                    }
                    else if((word_type[i + 2] == 10) && (word_type[i + 4] == 10))
                    {
                        arith_image_testlt(word[i + 2], word[i + 4], name);
                        tmp_name_index++;
                        type = 10;
                    }
                    else
                    {
                        PRINT_ERROR("Wrong input to function testlt");
                    }
                }

                if(strcmp(word[i], "testmt") == 0)
                {
                    if((word_type[i + 2] == 2) && (word_type[i + 4] == 2))
                    {
                        if(data.variable[variable_ID(word[i + 2])].value.f > data.variable[variable_ID(
                                    word[i + 4])].value.f)
                        {
                            tmp_prec = 1.0;
                        }
                        else
                        {
                            tmp_prec = 0.0;
                        }
                        create_variable_ID(name, tmp_prec);
                        tmp_name_index++;
                        type = 2;
                    }
                    else if((word_type[i + 2] == 2) && (word_type[i + 4] == 10))
                    {
                        arith_image_csttestlt(word[i + 4],
                                              (double) data.variable[variable_ID(word[i + 2])].value.f, name);
                        tmp_name_index++;
                        type = 10;
                    }
                    else if((word_type[i + 2] == 10) && (word_type[i + 4] == 2))
                    {
                        arith_image_csttestmt(word[i + 2],
                                              (double) data.variable[variable_ID(word[i + 4])].value.f, name);
                        tmp_name_index++;
                        type = 10;
                    }
                    else if((word_type[i + 2] == 10) && (word_type[i + 4] == 10))
                    {
                        arith_image_testmt(word[i + 2], word[i + 4], name);
                        tmp_name_index++;
                        type = 10;
                    }
                    else
                    {
                        PRINT_ERROR("Wrong input to function testlt");
                    }
                }


                if(strcmp(word[i], "perc") == 0)
                {
                    //	      printf("%d %d\n",word_type[i+2],word_type[i+4]);
                    if((word_type[i + 2] != 10) || (word_type[i + 4] != 2))
                    {
                        PRINT_ERROR("Wrong input to function perc\n");
                    }
                    else
                    {
                        //		  printf("Running percentile args = %s %f\n",word[i+2],data.variable[variable_ID(word[i+4])].value.f);
                        tmp_prec = arith_image_percentile(word[i + 2],
                                                          (double) data.variable[variable_ID(word[i + 4])].value.f);
                        create_variable_ID(name, tmp_prec);
                        tmp_name_index++;
                        type = 2;
                    }
                }

                if(strcmp(word[i], "trunc") == 0)
                {
                    if((word_type[i + 2] == 10) && (word_type[i + 4] == 2)
                            && (word_type[i + 6] == 2))
                    {
                        tmp_name_index++;
                        arith_image_trunc(word[i + 2],
                                          (double) data.variable[variable_ID(word[i + 4])].value.f,
                                          (double) data.variable[variable_ID(word[i + 6])].value.f, name);
                        type = 10;
                    }
                    else
                    {
                        printf("Syntax error with function trunc\n");
                    }
                }

                strcpy(word[i], name);
                word_type[i] = type;
                for(j = i + 1; j < nbword - (nbvarinput * 2 + 1); j++)
                {
                    strcpy(word[j], word[j + (nbvarinput * 2 + 1)]);
                    word_type[j] = word_type[j + (nbvarinput * 2 + 1)];
                }
                nbword = nbword - nbvarinput * 2 - 1;
            }




            /*      printf("after : ");
              for (i=0;i<nbword;i++)
              {
              if(variable_ID(word[i])!=-1)
              printf(" %s(%f) ",word[i],data.variable[variable_ID(word[i])].value.f);
              else
              printf(" %s ",word[i]);
              }
              printf("\n");
            */
            /* computing the number of to-be-processed words */
            passedequ = 0;
            nb_tbp_word = 0;
            for(i = (nbword - 1); i > -1; i--)
            {
                if(word_type[i] == 9)
                {
                    passedequ = 1;
                }
                if(passedequ == 0)
                {
                    nb_tbp_word++;
                }
            }

        }

        if(nbword > 2)
        {
            if(word_type[1] == 9)
            {
                if(variable_ID(word[0]) != -1)
                {
                    delete_variable_ID(word[0]);
                }
                if(image_ID(word[0]) != -1)
                {
                    delete_image_ID(word[0]);
                }

                if(word_type[2] == 2)
                {
                    create_variable_ID(word[0], data.variable[variable_ID(word[2])].value.f);
                    printf("%.20g\n", data.variable[variable_ID(word[2])].value.f);
                }
                if(word_type[2] == 10)
                {
                    chname_image_ID(word[2], word[0]);
                }
            }
        }
        else
        {
            printf("%.20g\n", data.variable[variable_ID(word[0])].value.f);
        }

        for(i = 0; i < tmp_name_index; i++)
        {
            CREATE_IMAGENAME(name, "_tmp%d_%d", i, (int) getpid());
            if(variable_ID(name) != -1)
            {
                delete_variable_ID(name);
            }
            if(image_ID(name) != -1)
            {
                delete_image_ID(name);
            }
        }
    }

    return(0);
}
