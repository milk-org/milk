/**
 * @file    fps_processcmdline_file.c
 * @brief   FPS process command file batch execution
 */

#ifndef _GNU_SOURCE
#    define _GNU_SOURCE
#endif


#include "fps_processcmdline.h"

/**
 * @brief Process a batch of FPS commands from a file.
 *
 * Reads lines from the specified file and dispatches
 * each to the interactive command processor.
 */
int functionparameter_FPSprocess_cmdfile(char                 *infname,
                                         FPS                  *fps,
                                         KEYWORD_TREE_NODE    *keywnode,
                                         FPSCTRL_TASK_QUEUE   *fpsctrlqueuelist,
                                         FPSCTRL_PROCESS_VARS *fpsCTRLvar)
{
    FILE *fpinputcmd;
    fpinputcmd = fopen(infname, "r");

    if (fpinputcmd == NULL)
    {
        PRINT_ERROR("ERROR: cannot open command file %s", infname);
        return RETURN_FAILURE;
    }
    if (fpinputcmd != NULL)
    {
        char   *FPScmdline = NULL;
        size_t  len        = 0;
        ssize_t read;

        while ((read = getline(&FPScmdline, &len, fpinputcmd)) != -1)
        {
            uint64_t taskstatus = 0;
            printf("Processing line : %s\n", FPScmdline);
            functionparameter_FPSprocess_cmdline(FPScmdline, fpsctrlqueuelist, keywnode, fpsCTRLvar,
                                                 fps, &taskstatus);
        }
        free(FPScmdline);
        fclose(fpinputcmd);
    }

    return RETURN_SUCCESS;
}
