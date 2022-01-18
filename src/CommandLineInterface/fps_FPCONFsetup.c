/**
 * @file    fps_FPCONFsetup.c
 * @brief   FPS config setup
 */

#include "COREMOD_memory/fps_create.h"
#include "CommandLineInterface/CLIcore.h"

#include "fps_connect.h"
#include "fps_disconnect.h"

/** @brief FPS config setup
 *
 * called by conf and run functions
 *
 */
FUNCTION_PARAMETER_STRUCT function_parameter_FPCONFsetup(const char *fpsname, uint32_t CMDmode)
{
    long NBparamMAX = FUNCTION_PARAMETER_NBPARAM_DEFAULT;
    uint32_t FPSCONNECTFLAG;

    FUNCTION_PARAMETER_STRUCT fps;

    fps.CMDmode = CMDmode;
    fps.SMfd = -1;

    // record timestamp
    struct timespec tnow;
    clock_gettime(CLOCK_REALTIME, &tnow);
    data.FPS_TIMESTAMP = tnow.tv_sec;

    strcpy(data.FPS_PROCESS_TYPE, "UNDEF");
    //	char ptstring[STRINGMAXLEN_FPSPROCESSTYPE];

    switch (CMDmode)
    {
    case FPSCMDCODE_CONFSTART:
        snprintf(data.FPS_PROCESS_TYPE, STRINGMAXLEN_FPSPROCESSTYPE, "confstart-%s", fpsname);
        break;

    case FPSCMDCODE_CONFSTOP:
        snprintf(data.FPS_PROCESS_TYPE, STRINGMAXLEN_FPSPROCESSTYPE, "confstop-%s", fpsname);
        break;

    case FPSCMDCODE_FPSINIT:
        snprintf(data.FPS_PROCESS_TYPE, STRINGMAXLEN_FPSPROCESSTYPE, "fpsinit-%s", fpsname);
        break;

    case FPSCMDCODE_FPSINITCREATE:
        snprintf(data.FPS_PROCESS_TYPE, STRINGMAXLEN_FPSPROCESSTYPE, "fpsinitcreate-%s", fpsname);
        break;

    case FPSCMDCODE_RUNSTART:
        snprintf(data.FPS_PROCESS_TYPE, STRINGMAXLEN_FPSPROCESSTYPE, "runstart-%s", fpsname);
        break;

    case FPSCMDCODE_RUNSTOP:
        snprintf(data.FPS_PROCESS_TYPE, STRINGMAXLEN_FPSPROCESSTYPE, "runstop-%s", fpsname);
        break;

    case FPSCMDCODE_TMUXSTART:
        snprintf(data.FPS_PROCESS_TYPE, STRINGMAXLEN_FPSPROCESSTYPE, "tmuxstart-%s", fpsname);
        break;

    case FPSCMDCODE_TMUXSTOP:
        snprintf(data.FPS_PROCESS_TYPE, STRINGMAXLEN_FPSPROCESSTYPE, "tmuxstop-%s", fpsname);
        break;
    }

    if (CMDmode & FPSCMDCODE_FPSINITCREATE) // (re-)create fps even if it exists
    {
        //printf("=== FPSINITCREATE NBparamMAX = %ld\n", NBparamMAX);
        function_parameter_struct_create(NBparamMAX, fpsname);
        function_parameter_struct_connect(fpsname, &fps, FPSCONNECT_SIMPLE);
    }
    else // load existing fps if exists
    {
        //printf("=== CHECK IF FPS EXISTS\n");

        FPSCONNECTFLAG = FPSCONNECT_SIMPLE;
        if (CMDmode & FPSCMDCODE_CONFSTART)
        {
            FPSCONNECTFLAG = FPSCONNECT_CONF;
        }

        if (function_parameter_struct_connect(fpsname, &fps, FPSCONNECTFLAG) == -1)
        {
            //printf("=== FPS DOES NOT EXISTS -> CREATE\n");
            function_parameter_struct_create(NBparamMAX, fpsname);
            function_parameter_struct_connect(fpsname, &fps, FPSCONNECTFLAG);
        }
        /*        else
        {
            printf("=== FPS EXISTS\n");
        }*/
    }

    if (CMDmode & FPSCMDCODE_CONFSTOP) // stop conf
    {
        fps.md->signal &= ~FUNCTION_PARAMETER_STRUCT_SIGNAL_CONFRUN;
        function_parameter_struct_disconnect(&fps);
        fps.localstatus &= ~FPS_LOCALSTATUS_CONFLOOP; // stop loop
    }
    else
    {
        fps.localstatus |= FPS_LOCALSTATUS_CONFLOOP;
    }

    if ((CMDmode & FPSCMDCODE_FPSINITCREATE) || (CMDmode & FPSCMDCODE_FPSINIT) || (CMDmode & FPSCMDCODE_CONFSTOP))
    {
        fps.localstatus &= ~FPS_LOCALSTATUS_CONFLOOP; // do not start conf
    }

    if (CMDmode & FPSCMDCODE_CONFSTART)
    {
        fps.localstatus |= FPS_LOCALSTATUS_CONFLOOP;
    }

    return fps;
}
