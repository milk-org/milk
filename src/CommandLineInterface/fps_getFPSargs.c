/**
 * @file    fps_getFPSargs.c
 * @brief   read FPS args from CLI
 */

#include "CommandLineInterface/CLIcore.h"




/** @brief get FPS arguments from command line function call
 *
 * write data.FPS_name and data.FPS_CMDCODE
 *
 */
errno_t function_parameter_getFPSargs_from_CLIfunc(
    char     *fpsname_default
)
{

#ifndef STANDALONE
    // Check if function will be executed through FPS interface
    // set to 0 as default (no FPS)
    data.FPS_CMDCODE = 0;

    // if using FPS implementation, FPSCMDCODE will be set to != 0
    if(CLI_checkarg(1, CLIARG_STR) == 0)
    {
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
    if(data.FPS_CMDCODE != 0)
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
            int slen = snprintf(data.FPS_name, FUNCTION_PARAMETER_STRMAXLEN, "%s",
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

                int slen = snprintf(fpsname1, FUNCTION_PARAMETER_STRMAXLEN,
                                    "%s-%s", data.FPS_name, data.cmdargtoken[argindex].val.string);
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

                strncpy(data.FPS_name, fpsname1, FUNCTION_PARAMETER_STRMAXLEN);
                argindex ++;
            }
        }
    }

#endif
    return RETURN_SUCCESS;
}

