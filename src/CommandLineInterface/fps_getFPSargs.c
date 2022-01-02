/**
 * @file    fps_getFPSargs.c
 * @brief   read FPS args from CLI
 */

#include "CommandLineInterface/CLIcore.h"




/** @brief get FPS arguments from command line function call
 *
 * Write data.FPS_name and data.FPS_CMDCODE
 *
 * Reads FPS_CMDCODE from CLI argument 1
 *
 * Construct FPS_name from subsequent arguments
 *
 */
errno_t function_parameter_getFPSargs_from_CLIfunc(
    char     *fpsname_default
)
{
    // Check if function will be executed through FPS interface

    // set to 0 as default (no FPS, function will be processed according to CLI rules)
    data.FPS_CMDCODE = 0;

    // if using FPS implementation, FPSCMDCODE will be set to != 0
    DEBUG_TRACEPOINT("pre-processing CLI arg");

    int argpreprocess = 1; // by default, pre-process argument
    switch(data.cmdargtoken[1].type)
    {
        case CLIARG_FLOAT:
            argpreprocess = 0;
            break;

        case CLIARG_FLOAT32:
            argpreprocess = 0;
            break;

        case CLIARG_FLOAT64:
            argpreprocess = 0;
            break;

        case CLIARG_LONG:
            argpreprocess = 0;
            break;

        case CLIARG_INT32:
            argpreprocess = 0;
            break;

        case CLIARG_UINT32:
            argpreprocess = 0;
            break;

        case CLIARG_INT64:
            argpreprocess = 0;
            break;

        case CLIARG_UINT64:
            argpreprocess = 0;
            break;
    }

    if(argpreprocess == 1)
    {
        // modify function attribute

        if(strcmp(data.cmdargtoken[1].val.string, "..procinfo") == 0)
        {
            if(data.cmdargtoken[2].val.numl == 0)
            {
                printf("Command %ld: updating PROCINFO mode OFF\n", data.cmdindex);
                data.cmd[data.cmdindex].cmdsettings.flags &= ~CLICMDFLAG_PROCINFO;

            }
            else
            {
                printf("Command %ld: updating PROCINFO mode ON\n", data.cmdindex);
                data.cmd[data.cmdindex].cmdsettings.flags |= CLICMDFLAG_PROCINFO;
            }
            data.FPS_CMDCODE = FPSCMDCODE_IGNORE;
            return RETURN_SUCCESS;
        }

        if(strcmp(data.cmdargtoken[1].val.string, "..RTprio") == 0)
        {
            printf("Command %ld: updating RTprio to value %ld\n", data.cmdindex,
                   data.cmdargtoken[2].val.numl);
            data.cmd[data.cmdindex].cmdsettings.RT_priority =
                data.cmdargtoken[2].val.numl;
            data.FPS_CMDCODE = FPSCMDCODE_IGNORE;
            return RETURN_SUCCESS;
        }

        if(strcmp(data.cmdargtoken[1].val.string, "..loopcntMax") == 0)
        {
            printf("Command %ld: updating loopcntMax to value %ld\n", data.cmdindex,
                   data.cmdargtoken[2].val.numl);
            data.cmd[data.cmdindex].cmdsettings.procinfo_loopcntMax =
                data.cmdargtoken[2].val.numl;
            data.FPS_CMDCODE = FPSCMDCODE_IGNORE;
            return RETURN_SUCCESS;
        }

        if(strcmp(data.cmdargtoken[1].val.string, "..triggermode") == 0)
        {
            printf("Command %ld: updating triggermode to value %ld\n", data.cmdindex,
                   data.cmdargtoken[2].val.numl);
            data.cmd[data.cmdindex].cmdsettings.triggermode =
                data.cmdargtoken[2].val.numl;
            data.FPS_CMDCODE = FPSCMDCODE_IGNORE;
            return RETURN_SUCCESS;
        }


        if(strcmp(data.cmdargtoken[1].val.string, "..triggersname") == 0)
        {
            printf("Command %ld: updating triggerstreamname to value %s\n", data.cmdindex,
                   data.cmdargtoken[2].val.string);
            strcpy(data.cmd[data.cmdindex].cmdsettings.triggerstreamname,
                   data.cmdargtoken[2].val.string);
            data.FPS_CMDCODE = FPSCMDCODE_IGNORE;
            return RETURN_SUCCESS;
        }


        if(strcmp(data.cmdargtoken[1].val.string, "..triggerdelay") == 0)
        {
            double x = 0.0;
            switch (data.cmdargtoken[2].type)
            {
                case CMDARGTOKEN_TYPE_FLOAT :
                x = data.cmdargtoken[2].val.numf;
                break;

                case CMDARGTOKEN_TYPE_LONG :
                x = data.cmdargtoken[2].val.numl;
                break;

                default :
                printf("wrong argument type, should be float or int -> setting to zero\n");
            }
            printf("Command %ld: updating triggerdelay to value %f\n", data.cmdindex, x);
            x += 0.5e-9;
            long x_sec = (long) x;
            long x_nsec = (x - x_sec) * 1000000000L;

            data.cmd[data.cmdindex].cmdsettings.triggerdelay.tv_sec = x_sec;
            data.cmd[data.cmdindex].cmdsettings.triggerdelay.tv_nsec = x_nsec;
            data.FPS_CMDCODE = FPSCMDCODE_IGNORE;
            return RETURN_SUCCESS;
        }

        if(strcmp(data.cmdargtoken[1].val.string, "..triggertimeout") == 0)
        {
            printf("Command %ld: updating triggertimeout to value %f\n", data.cmdindex,
                   data.cmdargtoken[2].val.numf);
            double x = data.cmdargtoken[2].val.numf;
            x += 0.5e-9;
            long x_sec = (long) x;
            long x_nsec = (x - x_sec) * 1000000000L;

            data.cmd[data.cmdindex].cmdsettings.triggertimeout.tv_sec = x_sec;
            data.cmd[data.cmdindex].cmdsettings.triggertimeout.tv_nsec = x_nsec;
            data.FPS_CMDCODE = FPSCMDCODE_IGNORE;
            return RETURN_SUCCESS;
        }



        // check that first arg is a string
        // if it isn't, the non-FPS implementation should be called

        // check if recognized FPSCMDCODE
        if(strcmp(data.cmdargtoken[1].val.string,
                  "_FPSINIT_") == 0)    // Initialize FPS
        {
            data.FPS_CMDCODE = FPSCMDCODE_FPSINIT;
        }
        else if(strcmp(data.cmdargtoken[1].val.string,
                       "_CONFSTART_") == 0)     // Start conf process
        {
            data.FPS_CMDCODE = FPSCMDCODE_CONFSTART;
        }
        else if(strcmp(data.cmdargtoken[1].val.string,
                       "_CONFSTOP_") == 0)   // Stop conf process
        {
            data.FPS_CMDCODE = FPSCMDCODE_CONFSTOP;
        }
        else if(strcmp(data.cmdargtoken[1].val.string,
                       "_RUNSTART_") == 0)   // Run process
        {
            data.FPS_CMDCODE = FPSCMDCODE_RUNSTART;
        }
        else if(strcmp(data.cmdargtoken[1].val.string,
                       "_RUNSTOP_") == 0)   // Stop process
        {
            data.FPS_CMDCODE = FPSCMDCODE_RUNSTOP;
        }
        else if(strcmp(data.cmdargtoken[1].val.string,
                       "_TMUXSTART_") == 0)   // Start tmux session
        {
            data.FPS_CMDCODE = FPSCMDCODE_TMUXSTART;
        }
        else if(strcmp(data.cmdargtoken[1].val.string,
                       "_TMUXSTOP_") == 0)   // Stop tmux session
        {
            data.FPS_CMDCODE = FPSCMDCODE_TMUXSTOP;
        }
    }


    // if recognized FPSCMDCODE, use FPS implementation
    if((data.FPS_CMDCODE != 0) && (data.FPS_CMDCODE != FPSCMDCODE_IGNORE))
    {
        // ===============================
        //     SET FPS INTERFACE NAME
        // ===============================

        // if main CLI process has been named with -n option, then use the process name to construct fpsname
        if(data.processnameflag == 1)
        {
            // Automatically set fps name to be process name up to first instance of character '.'
            strcpy(data.FPS_name, data.processname0);
        }
        else   // otherwise, construct name as follows
        {
            // Adopt default name for fpsname
            int slen = snprintf(data.FPS_name,
                                FUNCTION_PARAMETER_STRMAXLEN,
                                "%s",
                                fpsname_default);
            if(slen < 1)
            {
                PRINT_ERROR("snprintf wrote <1 char");
                abort(); // can't handle this error any other way
            }
            if(slen >= FUNCTION_PARAMETER_STRMAXLEN)
            {
                PRINT_ERROR("snprintf string truncation.\n"
                            "Full string  : %s\n"
                            "Truncated to : %s",
                            fpsname_default,
                            data.FPS_name);
                abort(); // can't handle this error any other way
            }


            // By convention, if there are optional arguments,
            // they should be appended to the default fps name
            //
            int argindex = 2; // start at arg #2
            while(strlen(data.cmdargtoken[argindex].val.string) > 0)
            {
                char fpsname1[FUNCTION_PARAMETER_STRMAXLEN];

                int slen = snprintf(fpsname1,
                                    FUNCTION_PARAMETER_STRMAXLEN,
                                    "%s-%s",
                                    data.FPS_name,
                                    data.cmdargtoken[argindex].val.string);
                if(slen < 1)
                {
                    PRINT_ERROR("snprintf wrote <1 char");
                    abort(); // can't handle this error any other way
                }
                if(slen >= FUNCTION_PARAMETER_STRMAXLEN)
                {
                    PRINT_ERROR("snprintf string truncation.\n"
                                "Full string  : %s-%s\n"
                                "Truncated to : %s",
                                data.FPS_name,
                                data.cmdargtoken[argindex].val.string,
                                fpsname1);
                    abort(); // can't handle this error any other way
                }

                strncpy(data.FPS_name, fpsname1, FUNCTION_PARAMETER_STRMAXLEN - 1);
                argindex ++;
            }
        }
        //printf(">>>>> %s >>>>>>> FPS name : %s\n", fpsname_default, data.FPS_name);
    }

    return RETURN_SUCCESS;
}
