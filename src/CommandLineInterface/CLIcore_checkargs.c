/**
 * @file CLIcore_checkargs.c
 * 
 * @brief Check CLI command line arguments
 *
 */

#include <stdio.h>

#include "CommandLineInterface/CLIcore.h"

#include "COREMOD_memory/COREMOD_memory.h"


// check that input CLI argument matches required argument type

int CLI_checkarg0(
    int argnum,
    int argtype,
    int errmsg
)
{
    int rval; // 0 if OK, 1 if not
    long IDv;

    rval = 2;

    switch(argtype)
    {

        case CLIARG_FLOAT:  // should be floating point
            switch(data.cmdargtoken[argnum].type)
            {
                case CLIARG_FLOAT:
                    rval = 0;
                    break;
                case CLIARG_LONG: // convert long to float
                    if(data.Debug > 0)
                    {
                        printf("Converting arg %d to floating point number\n", argnum);
                    }
                    data.cmdargtoken[argnum].val.numf = (double) data.cmdargtoken[argnum].val.numl;
                    data.cmdargtoken[argnum].type = 1;
                    rval = 0;
                    break;
                case CLIARG_STR_NOT_IMG:
                    IDv = variable_ID(data.cmdargtoken[argnum].val.string);
                    if(IDv == -1)
                    {
                        if(errmsg == 1)
                        {
                            printf("arg %d is string (=\"%s\"), but should be integer\n", argnum,
                                   data.cmdargtoken[argnum].val.string);
                        }
                        rval = 1;
                    }
                    else
                    {
                        switch(data.variable[IDv].type)
                        {
                            case 0: // double
                                data.cmdargtoken[argnum].val.numf = data.variable[IDv].value.f;
                                data.cmdargtoken[argnum].type = 1;
                                rval = 0;
                                break;
                            case 1: // long
                                data.cmdargtoken[argnum].val.numf = 1.0 * data.variable[IDv].value.l;
                                data.cmdargtoken[argnum].type = 1;
                                rval = 0;
                                break;
                            default:
                                if(errmsg == 1)
                                {
                                    printf("arg %d is string (=\"%s\"), but should be integer\n", argnum,
                                           data.cmdargtoken[argnum].val.string);
                                }
                                rval = 1;
                                break;
                        }
                    }
                    break;
                case CLIARG_IMG:
                    if(errmsg == 1)
                    {
                        printf("arg %d is image (=\"%s\"), but should be floating point number\n",
                               argnum, data.cmdargtoken[argnum].val.string);
                    }
                    rval = 1;
                    break;
                case CLIARG_STR:
                    if(errmsg == 1)
                    {
                        printf("arg %d is command (=\"%s\"), but should be floating point number\n",
                               argnum, data.cmdargtoken[argnum].val.string);
                    }
                    rval = 1;
                    break;
                case 6:
                    data.cmdargtoken[argnum].val.numf = atof(data.cmdargtoken[argnum].val.string);
                    data.cmdargtoken[argnum].type = 1;
                    rval = 0;
                    break;
            }
            break;

        case CLIARG_LONG:  // should be integer
            switch(data.cmdargtoken[argnum].type)
            {
                case CLIARG_FLOAT:
                    if(errmsg == 1)
                    {
                        printf("converting floating point arg %d to integer\n", argnum);
                    }
                    data.cmdargtoken[argnum].val.numl = (long)(data.cmdargtoken[argnum].val.numf +
                                                        0.5);
                    data.cmdargtoken[argnum].type = 2;
                    rval = 0;
                    break;
                case CLIARG_LONG:
                    rval = 0;
                    break;
                case CLIARG_STR_NOT_IMG:
                    IDv = variable_ID(data.cmdargtoken[argnum].val.string);
                    if(IDv == -1)
                    {
                        if(errmsg == 1)
                        {
                            printf("arg %d is string (=\"%s\"), but should be integer\n", argnum,
                                   data.cmdargtoken[argnum].val.string);
                        }
                        rval = 1;
                    }
                    else
                    {
                        switch(data.variable[IDv].type)
                        {
                            case 0: // double
                                data.cmdargtoken[argnum].val.numl = (long)(data.variable[IDv].value.f);
                                data.cmdargtoken[argnum].type = 2;
                                rval = 0;
                                break;
                            case 1: // long
                                data.cmdargtoken[argnum].val.numl = data.variable[IDv].value.l;
                                data.cmdargtoken[argnum].type = 2;
                                rval = 0;
                                break;
                            default:
                                if(errmsg == 1)
                                {
                                    printf("arg %d is string (=\"%s\"), but should be integer\n", argnum,
                                           data.cmdargtoken[argnum].val.string);
                                }
                                rval = 1;
                                break;
                        }
                    }
                    break;
                case CLIARG_IMG:
                    if(errmsg == 1)
                    {
                        printf("arg %d is image (=\"%s\"), but should be integer\n", argnum,
                               data.cmdargtoken[argnum].val.string);
                    }
                    rval = 1;
                    break;
                case CLIARG_STR:
                    if(errmsg == 1)
                    {
                        printf("arg %d is command (=\"%s\"), but should be integer\n", argnum,
                               data.cmdargtoken[argnum].val.string);
                    }
                    rval = 1;
                    break;
            }
            break;

        case 3:  // should be string, but not image
            switch(data.cmdargtoken[argnum].type)
            {
                case CLIARG_FLOAT:
                    if(errmsg == 1)
                    {
                        printf("arg %d is floating point, but should be string\n", argnum);
                    }
                    rval = 1;
                    break;
                case CLIARG_LONG:
                    if(errmsg == 1)
                    {
                        printf("arg %d is integer, but should be string\n", argnum);
                    }
                    rval = 1;
                    break;
                case CLIARG_STR_NOT_IMG:
                    rval = 0;
                    break;
                case CLIARG_IMG:
                    if(errmsg == 1)
                    {
                        printf("arg %d is existing image (=\"%s\"), but should be string\n", argnum,
                               data.cmdargtoken[argnum].val.string);
                    }
                    rval = 1;
                    break;
                case CLIARG_STR:
                    printf("arg %d is command (=\"%s\"), but should be string\n", argnum,
                           data.cmdargtoken[argnum].val.string);
                    rval = 1;
                    break;
                case 6:
                    rval = 0;
                    break;
            }
            break;

        case 4:  // should be existing image
            switch(data.cmdargtoken[argnum].type)
            {
                case CLIARG_FLOAT:
                    if(errmsg == 1)
                    {
                        printf("arg %d is floating point, but should be image\n", argnum);
                    }
                    rval = 1;
                    break;
                case CLIARG_LONG:
                    if(errmsg == 1)
                    {
                        printf("arg %d is integer, but should be image\n", argnum);
                    }
                    rval = 1;
                    break;
                case CLIARG_STR_NOT_IMG:
                    if(errmsg == 1)
                    {
                        printf("arg %d is string, but should be image\n", argnum);
                    }
                    rval = 1;
                    break;
                case CLIARG_IMG:
                    rval = 0;
                    break;
                case CLIARG_STR:
                    if(errmsg == 1)
                    {
                        printf("arg %d is command (=\"%s\"), but should be image\n", argnum,
                               data.cmdargtoken[argnum].val.string);
                    }
                    rval = 1;
                    break;
                case 6:
                    rval = 0;
                    break;
            }
            break;
        case 5: // should be string (image or not)
            switch(data.cmdargtoken[argnum].type)
            {
                case CLIARG_FLOAT:
                    if(errmsg == 1)
                    {
                        printf("arg %d is floating point, but should be string or image\n", argnum);
                    }
                    rval = 1;
                    break;
                case CLIARG_LONG:
                    if(errmsg == 1)
                    {
                        printf("arg %d is integer, but should be string or image\n", argnum);
                    }
                    rval = 1;
                    break;
                case CLIARG_STR_NOT_IMG:
                    rval = 0;
                    break;
                case CLIARG_IMG:
                    rval = 0;
                    break;
                case CLIARG_STR:
                    if(errmsg == 1)
                    {
                        printf("arg %d is command (=\"%s\"), but should be image\n", argnum,
                               data.cmdargtoken[argnum].val.string);
                    }
                    rval = 1;
                    break;
                case 6:
                    rval = 0;
                    break;
            }
            break;

    }


    if(rval == 2)
    {
        if(errmsg == 1)
        {
            printf("arg %d: wrong arg type %d :  %d\n", argnum, argtype,
                   data.cmdargtoken[argnum].type);
        }
        rval = 1;
    }


    return rval;
}



// check that input CLI argument matches required argument type
int CLI_checkarg(
    int argnum,
    int argtype
)
{
    int rval;

    rval = CLI_checkarg0(argnum, argtype, 1);
    return rval;
}



// check that input CLI argument matches required argument type - do not print error message
int CLI_checkarg_noerrmsg(
    int argnum,
    int argtype
)
{
    int rval;

    rval = CLI_checkarg0(argnum, argtype, 0);
    return rval;
}





/** @brief Check array of command line (CLI) arguments
 * 
 * Use list of arguments in fpscliarg[].
 * Skip arguments that have CLICMDARG_FLAG_NOCLI flag.
 * 
 * CLIarg keep count of argument position in CLI call
 * 
 */
errno_t CLI_checkarg_array(
    CLICMDARGDEF fpscliarg[],
    int nbarg
)
{
    printf("Number of args in list : %d\n", nbarg);

    int nberr = 0;
    int CLIarg = 0; // index of argument in CLI call
    for(int arg = 0; arg < nbarg; arg++)
    {
        if(! (fpscliarg[arg].flag & CLICMDARG_FLAG_NOCLI) )
        {
			printf("  arg %d  CLI %2d  [%2d]  %s\n", arg, CLIarg, fpscliarg[arg].type, fpscliarg[arg].fpstag);
            if(CLI_checkarg(CLIarg + 1, fpscliarg[arg].type) != 0)
            {
                nberr ++;
            }
            CLIarg++;
        }
        else
        { // argument not part of CLI
			printf("  arg %d  IGNORED [%2d]  %s\n", arg, fpscliarg[arg].type, fpscliarg[arg].fpstag);
		}
    }

    printf("Number of arg error(s): %d / %d\n", nberr, CLIarg);

    if(nberr == 0)
    {
        return RETURN_SUCCESS;
    }
    else
    {
        return RETURN_FAILURE;
    }
}







/** @brief Build FPS content from FPSCLIARG list
 * 
 * All CLI arguments converted to FPS parameters
 * 
 */
int CLIargs_to_FPSparams_setval(
    CLICMDARGDEF fpscliarg[],
    int nbarg,
    FUNCTION_PARAMETER_STRUCT *fps
)
{
    int NBarg_processed = 0;

    for(int arg = 0; arg < nbarg; arg++)
    {
        if( ! (fpscliarg[arg].flag & CLICMDARG_FLAG_NOFPS) )
        { // if argument is part of FPS
            switch(fpscliarg[arg].type)
            {
                case CLIARG_FLOAT:
                    functionparameter_SetParamValue_FLOAT64(fps, fpscliarg[arg].fpstag,
                                                            data.cmdargtoken[arg + 1].val.numl);
                    NBarg_processed++;
                    break;

                case CLIARG_LONG:
                    functionparameter_SetParamValue_INT64(fps, fpscliarg[arg].fpstag,
                                                          data.cmdargtoken[arg + 1].val.numl);
                    NBarg_processed++;
                    break;

                case CLIARG_STR_NOT_IMG:
                    functionparameter_SetParamValue_STRING(fps, fpscliarg[arg].fpstag,
                                                           data.cmdargtoken[arg + 1].val.string);
                    NBarg_processed++;
                    break;

                case CLIARG_IMG:
                    functionparameter_SetParamValue_STRING(fps, fpscliarg[arg].fpstag,
                                                           data.cmdargtoken[arg + 1].val.string);
                    NBarg_processed++;
                    break;

                case CLIARG_STR:
                    functionparameter_SetParamValue_STRING(fps, fpscliarg[arg].fpstag,
                                                           data.cmdargtoken[arg + 1].val.string);
                    NBarg_processed++;
                    break;

            }
        }
    }

    return NBarg_processed;
}


int CLIargs_to_FPSparams_create(
    CLICMDARGDEF fpscliarg[],
    int nbarg,
    FUNCTION_PARAMETER_STRUCT *fps
)
{
    int NBarg_processed = 0;

    for(int arg = 0; arg < nbarg; arg++)
    {
        if( ! (fpscliarg[arg].flag & CLICMDARG_FLAG_NOFPS) )
        { // if argument is part of FPS
            switch(fpscliarg[arg].type)
            {
                case CLIARG_FLOAT:
                    function_parameter_add_entry(fps, fpscliarg[arg].fpstag, fpscliarg[arg].descr,
                                                 FPTYPE_FLOAT64, FPFLAG_DEFAULT_INPUT, NULL);
                    NBarg_processed++;
                    break;

                case CLIARG_LONG:
                    function_parameter_add_entry(fps, fpscliarg[arg].fpstag, fpscliarg[arg].descr,
                                                 FPTYPE_INT64, FPFLAG_DEFAULT_INPUT, NULL);
                    NBarg_processed++;
                    break;

                case CLIARG_STR_NOT_IMG:
                    function_parameter_add_entry(fps, fpscliarg[arg].fpstag, fpscliarg[arg].descr,
                                                 FPTYPE_INT64, FPFLAG_DEFAULT_INPUT, "null");
                    NBarg_processed++;
                    break;

                case CLIARG_IMG:
                    function_parameter_add_entry(fps, fpscliarg[arg].fpstag, fpscliarg[arg].descr,
                                                 FPTYPE_STREAMNAME, FPFLAG_DEFAULT_INPUT_STREAM, "nullim");
                    NBarg_processed++;
                    break;

                case CLIARG_STR:
                    function_parameter_add_entry(fps, fpscliarg[arg].fpstag, fpscliarg[arg].descr,
                                                 FPTYPE_INT64, FPFLAG_DEFAULT_INPUT, "null");
                    NBarg_processed++;
                    break;

            }

        }
    }

    return NBarg_processed;
}



