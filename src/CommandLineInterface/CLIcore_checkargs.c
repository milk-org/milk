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

int CLI_checkarg0(int argnum, int argtype, int errmsg)
{
    int rval; // 0 if OK, 1 if not
    long IDv;

    rval = 2;

    switch(argtype)
    {

        case 1:  // should be floating point
            switch(data.cmdargtoken[argnum].type)
            {
                case 1:
                    rval = 0;
                    break;
                case 2: // convert long to float
                    if(data.Debug > 0)
                    {
                        printf("Converting arg %d to floating point number\n", argnum);
                    }
                    data.cmdargtoken[argnum].val.numf = (double) data.cmdargtoken[argnum].val.numl;
                    data.cmdargtoken[argnum].type = 1;
                    rval = 0;
                    break;
                case 3:
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
                case 4:
                    if(errmsg == 1)
                    {
                        printf("arg %d is image (=\"%s\"), but should be floating point number\n",
                               argnum, data.cmdargtoken[argnum].val.string);
                    }
                    rval = 1;
                    break;
                case 5:
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

        case 2:  // should be integer
            switch(data.cmdargtoken[argnum].type)
            {
                case 1:
                    if(errmsg == 1)
                    {
                        printf("converting floating point arg %d to integer\n", argnum);
                    }
                    data.cmdargtoken[argnum].val.numl = (long)(data.cmdargtoken[argnum].val.numf +
                                                        0.5);
                    data.cmdargtoken[argnum].type = 2;
                    rval = 0;
                    break;
                case 2:
                    rval = 0;
                    break;
                case 3:
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
                case 4:
                    if(errmsg == 1)
                    {
                        printf("arg %d is image (=\"%s\"), but should be integer\n", argnum,
                               data.cmdargtoken[argnum].val.string);
                    }
                    rval = 1;
                    break;
                case 5:
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
                case 1:
                    if(errmsg == 1)
                    {
                        printf("arg %d is floating point, but should be string\n", argnum);
                    }
                    rval = 1;
                    break;
                case 2:
                    if(errmsg == 1)
                    {
                        printf("arg %d is integer, but should be string\n", argnum);
                    }
                    rval = 1;
                    break;
                case 3:
                    rval = 0;
                    break;
                case 4:
                    if(errmsg == 1)
                    {
                        printf("arg %d is existing image (=\"%s\"), but should be string\n", argnum,
                               data.cmdargtoken[argnum].val.string);
                    }
                    rval = 1;
                    break;
                case 5:
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
                case 1:
                    if(errmsg == 1)
                    {
                        printf("arg %d is floating point, but should be image\n", argnum);
                    }
                    rval = 1;
                    break;
                case 2:
                    if(errmsg == 1)
                    {
                        printf("arg %d is integer, but should be image\n", argnum);
                    }
                    rval = 1;
                    break;
                case 3:
                    if(errmsg == 1)
                    {
                        printf("arg %d is string, but should be image\n", argnum);
                    }
                    rval = 1;
                    break;
                case 4:
                    rval = 0;
                    break;
                case 5:
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
                case 1:
                    if(errmsg == 1)
                    {
                        printf("arg %d is floating point, but should be string or image\n", argnum);
                    }
                    rval = 1;
                    break;
                case 2:
                    if(errmsg == 1)
                    {
                        printf("arg %d is integer, but should be string or image\n", argnum);
                    }
                    rval = 1;
                    break;
                case 3:
                    rval = 0;
                    break;
                case 4:
                    rval = 0;
                    break;
                case 5:
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
int CLI_checkarg(int argnum, int argtype)
{
    int rval;

    rval = CLI_checkarg0(argnum, argtype, 1);
    return rval;
}

// check that input CLI argument matches required argument type - do not print error message
int CLI_checkarg_noerrmsg(int argnum, int argtype)
{
    int rval;

    rval = CLI_checkarg0(argnum, argtype, 0);
    return rval;
}






