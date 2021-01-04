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
    imageID IDv;

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
                        printf("Converting arg %d to floating point number\n", argnum - 1);
                    }
                    data.cmdargtoken[argnum].val.numf = (double) data.cmdargtoken[argnum].val.numl;
                    data.cmdargtoken[argnum].type = CLIARG_FLOAT;
                    rval = 0;
                    break;
                case CLIARG_STR_NOT_IMG:
                    IDv = variable_ID(data.cmdargtoken[argnum].val.string);
                    if(IDv == -1)
                    {
                        if(errmsg == 1)
                        {
                            printf("arg %d is string (=\"%s\"), but should be integer\n", argnum - 1,
                                   data.cmdargtoken[argnum].val.string);
                        }
                        rval = 1;
                    }
                    else
                    {
                        switch(data.variable[IDv].type)
                        {
                            case CLIARG_FLOAT:
                                data.cmdargtoken[argnum].val.numf = data.variable[IDv].value.f;
                                data.cmdargtoken[argnum].type = CLIARG_FLOAT;
                                rval = 0;
                                break;
                            case CLIARG_LONG:
                                data.cmdargtoken[argnum].val.numf = 1.0 * data.variable[IDv].value.l;
                                data.cmdargtoken[argnum].type = CLIARG_FLOAT;
                                rval = 0;
                                break;
                            default:
                                if(errmsg == 1)
                                {
                                    printf("  arg %d (string \"%s\") not an integer\n", argnum - 1,
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
                        printf("  arg %d (image \"%s\") not a floating point number\n",
                               argnum - 1, data.cmdargtoken[argnum].val.string);
                    }
                    rval = 1;
                    break;
                case CLIARG_STR:
                    if(errmsg == 1)
                    {
                        printf("  arg %d (command \"%s\") not a floating point number\n",
                               argnum - 1, data.cmdargtoken[argnum].val.string);
                    }
                    rval = 1;
                    break;
                case 6:
                    data.cmdargtoken[argnum].val.numf = atof(data.cmdargtoken[argnum].val.string);
                    data.cmdargtoken[argnum].type = CLIARG_FLOAT;
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
                        printf("converting floating point arg %d to integer\n", argnum - 1);
                    }
                    data.cmdargtoken[argnum].val.numl = (long)(data.cmdargtoken[argnum].val.numf +
                                                        0.5);
                    data.cmdargtoken[argnum].type = CLIARG_LONG;
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
                            printf("  arg %d (string \"%s\") not an integer\n", argnum - 1,
                                   data.cmdargtoken[argnum].val.string);
                        }
                        rval = 1;
                    }
                    else
                    {
                        switch(data.variable[IDv].type)
                        {
                            case CLIARG_FLOAT: // double
                                data.cmdargtoken[argnum].val.numl = (long)(data.variable[IDv].value.f);
                                data.cmdargtoken[argnum].type = CLIARG_LONG;
                                rval = 0;
                                break;
                            case CLIARG_LONG: // long
                                data.cmdargtoken[argnum].val.numl = data.variable[IDv].value.l;
                                data.cmdargtoken[argnum].type = CLIARG_LONG;
                                rval = 0;
                                break;
                            default:
                                if(errmsg == 1)
                                {
                                    printf("  arg %d (string \"%s\") not an integer\n", argnum - 1,
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
                        printf("  arg %d (image \"%s\") not an integer\n", argnum - 1,
                               data.cmdargtoken[argnum].val.string);
                    }
                    rval = 1;
                    break;
                case CLIARG_STR:
                    if(errmsg == 1)
                    {
                        printf("  arg %d (command \"%s\") not an integer\n", argnum - 1,
                               data.cmdargtoken[argnum].val.string);
                    }
                    rval = 1;
                    break;
            }
            break;

        case CLIARG_STR_NOT_IMG:  // should be string, but not image
            switch(data.cmdargtoken[argnum].type)
            {
                case CLIARG_FLOAT:
                    if(errmsg == 1)
                    {
                        printf("  arg %d (float %f) not a non-img-string\n", argnum - 1,
                               data.cmdargtoken[argnum].val.numf);
                    }
                    rval = 1;
                    break;
                case CLIARG_LONG:
                    if(errmsg == 1)
                    {
                        printf("  arg %d (integer %ld) not a non-img-string\n", argnum - 1,
                               data.cmdargtoken[argnum].val.numl);
                    }
                    rval = 1;
                    break;
                case CLIARG_STR_NOT_IMG:
                    rval = 0;
                    break;
                case CLIARG_IMG:
                    if(errmsg == 1)
                    {
                        printf("  arg %d (image %s) not a non-img-string\n", argnum - 1,
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

        case CLIARG_IMG:  // should be existing image
            switch(data.cmdargtoken[argnum].type)
            {
                case CLIARG_FLOAT:
                    if(errmsg == 1)
                    {
                        printf("  arg %d (float %f) not an image\n", argnum - 1,
                               data.cmdargtoken[argnum].val.numf);
                    }
                    rval = 1;
                    break;
                case CLIARG_LONG:
                    if(errmsg == 1)
                    {
                        printf("  arg %d (integer %ld) not an image\n", argnum - 1,
                               data.cmdargtoken[argnum].val.numl);
                    }
                    rval = 1;
                    break;
                case CLIARG_STR_NOT_IMG:
                    if(errmsg == 1)
                    {
                        printf("  arg %d (string \"%s\") not an image\n", argnum - 1,
                               data.cmdargtoken[argnum].val.string);
                    }
                    rval = 1;
                    break;
                case CLIARG_IMG:
                    rval = 0;
                    break;
                case CLIARG_STR:
                    if(errmsg == 1)
                    {
                        printf("  arg %d (string \"%s\") not an image\n", argnum - 1,
                               data.cmdargtoken[argnum].val.string);
                    }
                    rval = 1;
                    break;
                case 6:
                    rval = 0;
                    break;
            }
            break;
        case CLIARG_STR: // should be string (image or not)
            switch(data.cmdargtoken[argnum].type)
            {
                case CLIARG_FLOAT:
                    if(errmsg == 1)
                    {
                        printf("  arg %d (float %f) not a string or image\n", argnum - 1,
                               data.cmdargtoken[argnum].val.numf);
                    }
                    rval = 1;
                    break;
                case CLIARG_LONG:
                    if(errmsg == 1)
                    {
                        printf("  arg %d (integer %ld) not string or image\n", argnum - 1,
                               data.cmdargtoken[argnum].val.numl);
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
                        printf("  arg %d (command \"%s\") not string or image\n", argnum - 1,
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
    //printf("%d args in list\n", nbarg);


    //printf("arg 0: %s\n", data.cmdargtoken[1].val.string);
    int argindexmatch = -1;
    for(int arg = 0; arg < nbarg; arg++)
    {
        if(strcmp(data.cmdargtoken[1].val.string, fpscliarg[arg].fpstag) == 0)
        {
            argindexmatch = arg;
        }
    }
    if(argindexmatch != -1)
    {
        //printf("match to arg %s\n", fpscliarg[argindexmatch].fpstag);

        if(data.cmdargtoken[2].type == CLIARG_MISSING)
        {
            printf("Setting arg %s : input missing\n", fpscliarg[argindexmatch].fpstag);
            return RETURN_FAILURE;
        }

        if(CLI_checkarg(2, fpscliarg[argindexmatch].type) == 0)
        {
            int cmdi = data.cmdindex;
            switch(fpscliarg[argindexmatch].type)
            {
                case CLIARG_FLOAT:
                    data.cmd[cmdi].argdata[argindexmatch].val.f = data.cmdargtoken[2].val.numf;
                    break;
                case CLIARG_LONG:
                    data.cmd[cmdi].argdata[argindexmatch].val.l = data.cmdargtoken[2].val.numl;
                    break;
                case CLIARG_STR_NOT_IMG:
                    strcpy(data.cmd[cmdi].argdata[argindexmatch].val.s,
                           data.cmdargtoken[2].val.string);
                    break;
                case CLIARG_IMG:
                    strcpy(data.cmd[cmdi].argdata[argindexmatch].val.s,
                           data.cmdargtoken[2].val.string);
                    break;
                case CLIARG_STR:
                    strcpy(data.cmd[cmdi].argdata[argindexmatch].val.s,
                           data.cmdargtoken[2].val.string);
                    break;
            }
        }
        else
        {
            printf("Setting arg %s : Wrong type\n", fpscliarg[argindexmatch].fpstag);
            return RETURN_FAILURE;
        }


        printf("Argument %s value updated\n", fpscliarg[argindexmatch].fpstag);

        //printf("arg 1: [%d] %s %f %ld\n", data.cmdargtoken[2].type, data.cmdargtoken[2].val.string, data.cmdargtoken[2].val.numf, data.cmdargtoken[2].val.numl);
        return RETURN_FAILURE;
    }


    //printf("arg 1: %s %f %ld\n", data.cmdargtoken[2].val.string);



    int nberr = 0;
    int CLIarg = 0; // index of argument in CLI call
    for(int arg = 0; arg < nbarg; arg++)
    {
        char argtypestring[16];
        switch(fpscliarg[arg].type)
        {
            case CLIARG_FLOAT:
                strcpy(argtypestring, "FLOAT");
                break;
            case CLIARG_LONG:
                strcpy(argtypestring, "LONG");
                break;
            case CLIARG_STR_NOT_IMG:
                strcpy(argtypestring, "STRnIMG");
                break;
            case CLIARG_IMG:
                strcpy(argtypestring, "IMG");
                break;
            case CLIARG_STR:
                strcpy(argtypestring, "STRING");
                break;
        }




        if(!(fpscliarg[arg].flag & CLICMDARG_FLAG_NOCLI))
        {
            int cmdi = data.cmdindex;

            DEBUG_TRACEPOINT("  arg %d  CLI %2d  [%7s]  %s\n", arg, CLIarg, argtypestring,
                             fpscliarg[arg].fpstag);

            //printf("     input : %s\n", );data.cmdargtoken[argnum].type
            if(strcmp(data.cmdargtoken[CLIarg + 1].val.string, ".") == 0)
            {
                DEBUG_TRACEPOINT("ADOPTING DEFAULT/LAST VALUE");
                switch(fpscliarg[arg].type)
                {
                    case CLIARG_FLOAT:
                        data.cmdargtoken[CLIarg + 1].val.numf = data.cmd[cmdi].argdata[arg].val.f;
                        data.cmdargtoken[CLIarg + 1].type = CLIARG_FLOAT;
                        break;
                    case CLIARG_LONG:
                        data.cmdargtoken[CLIarg + 1].val.numl = data.cmd[cmdi].argdata[arg].val.l;
                        data.cmdargtoken[CLIarg + 1].type = CLIARG_LONG;
                        break;
                    case CLIARG_STR_NOT_IMG:
                        strcpy(data.cmdargtoken[CLIarg + 1].val.string,
                               data.cmd[cmdi].argdata[arg].val.s);
                        data.cmdargtoken[CLIarg + 1].type = CLIARG_STR_NOT_IMG;
                        break;
                    case CLIARG_IMG: // should be image
                        strcpy(data.cmdargtoken[CLIarg + 1].val.string,
                               data.cmd[cmdi].argdata[arg].val.s);
                        if(image_ID(data.cmd[cmdi].argdata[arg].val.s) != -1)
                        {
                            // if image exists
                            data.cmdargtoken[CLIarg + 1].type = CLIARG_IMG;
                        }
                        else
                        {
                            data.cmdargtoken[CLIarg + 1].type = CLIARG_STR_NOT_IMG;
                        }
                        //printf("arg %d IMG        : %s\n", CLIarg+1, data.cmdargtoken[CLIarg+1].val.string);
                        break;
                    case CLIARG_STR:
                        strcpy(data.cmdargtoken[CLIarg + 1].val.string,
                               data.cmd[cmdi].argdata[arg].val.s);
                        data.cmdargtoken[CLIarg + 1].type = CLIARG_STR;
                        break;
                }
            }



            if(CLI_checkarg(CLIarg + 1, fpscliarg[arg].type) == 0)
            {
                DEBUG_TRACEPOINT("successful parsing, update default to last");
                switch(fpscliarg[arg].type)
                {
                    case CLIARG_FLOAT:
                        data.cmd[cmdi].argdata[arg].val.f = data.cmdargtoken[CLIarg +
                                                            1].val.numf;
                        break;
                    case CLIARG_LONG:
                        data.cmd[cmdi].argdata[arg].val.l = data.cmdargtoken[CLIarg +
                                                            1].val.numl;
                        break;
                    case CLIARG_STR_NOT_IMG:
                        strcpy(data.cmd[cmdi].argdata[arg].val.s,
                               data.cmdargtoken[CLIarg + 1].val.string);
                        break;
                    case CLIARG_IMG:
                        strcpy(data.cmd[cmdi].argdata[arg].val.s,
                               data.cmdargtoken[CLIarg + 1].val.string);
                        break;
                    case CLIARG_STR:
                        strcpy(data.cmd[cmdi].argdata[arg].val.s,
                               data.cmdargtoken[CLIarg + 1].val.string);
                        break;
                }
            }
            else
            {
                nberr ++;
            }
            CLIarg++;
        }
        else
        {
            DEBUG_TRACEPOINT("argument not part of CLI");
            DEBUG_TRACEPOINT("  arg %d  IGNORED [%7s]  %s\n", arg, argtypestring,
                             fpscliarg[arg].fpstag);
        }
    }

    DEBUG_TRACEPOINT("Number of arg error(s): %d / %d\n", nberr, CLIarg);

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
        if(!(fpscliarg[arg].flag & CLICMDARG_FLAG_NOFPS))
        {
            // if argument is part of FPS
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



int CMDargs_to_FPSparams_create(
    FUNCTION_PARAMETER_STRUCT *fps
)
{
    int NBarg_processed = 0;

    printf("COMMAND [%ld] key : \"%s\"\n", data.cmdindex,
           data.cmd[data.cmdindex].key);

    for(int argi = 0; argi < data.cmd[data.cmdindex].nbarg; argi++)
    {
        if(!(data.cmd[data.cmdindex].argdata[argi].flag & CLICMDARG_FLAG_NOFPS))
        {
            // if argument is part of FPS
            double tmpvalf = 0.0;
            long tmpvall = 0;

            switch(data.cmd[data.cmdindex].argdata[argi].type)
            {
                case CLIARG_FLOAT:
                    tmpvalf = data.cmd[data.cmdindex].argdata[argi].val.f;
                    function_parameter_add_entry(fps, data.cmd[data.cmdindex].argdata[argi].fpstag,
                                                 data.cmd[data.cmdindex].argdata[argi].descr,
                                                 FPTYPE_FLOAT64, FPFLAG_DEFAULT_INPUT, &tmpvalf);
                    NBarg_processed++;
                    break;

                case CLIARG_LONG:
                    tmpvall = data.cmd[data.cmdindex].argdata[argi].val.l;
                    function_parameter_add_entry(fps, data.cmd[data.cmdindex].argdata[argi].fpstag,
                                                 data.cmd[data.cmdindex].argdata[argi].descr,
                                                 FPTYPE_INT64, FPFLAG_DEFAULT_INPUT, &tmpvall);
                    NBarg_processed++;
                    break;

                case CLIARG_STR_NOT_IMG:
                    function_parameter_add_entry(fps, data.cmd[data.cmdindex].argdata[argi].fpstag,
                                                 data.cmd[data.cmdindex].argdata[argi].descr,
                                                 FPTYPE_STRING, FPFLAG_DEFAULT_INPUT,
                                                 data.cmd[data.cmdindex].argdata[argi].val.s);
                    NBarg_processed++;
                    break;

                case CLIARG_IMG:
                    function_parameter_add_entry(fps, data.cmd[data.cmdindex].argdata[argi].fpstag,
                                                 data.cmd[data.cmdindex].argdata[argi].descr,
                                                 FPTYPE_STREAMNAME, FPFLAG_DEFAULT_INPUT,
                                                 data.cmd[data.cmdindex].argdata[argi].val.s);
                    NBarg_processed++;
                    break;

                case CLIARG_STR:
                    function_parameter_add_entry(fps, data.cmd[data.cmdindex].argdata[argi].fpstag,
                                                 data.cmd[data.cmdindex].argdata[argi].descr,
                                                 FPTYPE_STRING, FPFLAG_DEFAULT_INPUT,
                                                 data.cmd[data.cmdindex].argdata[argi].val.s);
                    NBarg_processed++;
                    break;

            }

        }
    }

    return NBarg_processed;
}



