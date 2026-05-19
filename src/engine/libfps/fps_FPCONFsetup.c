/**
 * @file    fps_FPCONFsetup.c
 * @brief   FPS config setup
 */

#include "fps.h"
#include "fps_internal.h"
#include "fps_globals.h"



/** @brief FPS config setup
 *
 * called by conf and run functions
 *
 */
FPS function_parameter_FPCONFsetup_sized(
    const char *fpsname,
    uint32_t   CMDmode,
    long        NBparamMAX
)
{
    uint32_t FPSCONNECTFLAG;

    FPS fps = {0};

    fps.CMDmode = CMDmode;
    fps.SMfd    = -1;

    // Set defaults
    fps.cmdset.procinfo_loopcntMax    = 1;
    fps.cmdset.procinfo_MeasureTiming = 1;


    FPS_TIMESTAMP = 0;
    snprintf(FPS_PROCESS_TYPE,
             STRINGMAXLEN_FPSPROCESSTYPE, "UNDEF");


    if(CMDmode & FPSCMDCODE_FPSINITCREATE)  // (re-)create fps even if it exists
    {
        if(getenv("FPS_DEBUG"))
            printf("=== FPSINITCREATE "
                   "NBparamMAX = %ld\n",
                   NBparamMAX);
        function_parameter_struct_create(NBparamMAX, fpsname);
        fps_connect(fpsname, &fps, FPSCONNECT_SIMPLE);
    }
    else // load existing fps if exists
    {
        //printf("=== CHECK IF FPS EXISTS\n");

        FPSCONNECTFLAG = FPSCONNECT_SIMPLE;
        if(CMDmode & FPSCMDCODE_CONFSTART)
        {
            FPSCONNECTFLAG = FPSCONNECT_CONF;
        }

        if(fps_connect(fpsname, &fps, FPSCONNECTFLAG) ==
                -1)
        {
            if(getenv("FPS_DEBUG"))
                printf("DEBUG: [%s:%d] "
                       "FPS DOES NOT EXIST "
                       "-> CREATE\n",
                       __FILE__, __LINE__);
            int ret = function_parameter_struct_create(NBparamMAX, fpsname);
            if(getenv("FPS_DEBUG"))
                printf("DEBUG: [%s:%d] "
                       "CREATE RETURNED %d\n",
                       __FILE__, __LINE__, ret);
            fps_connect(fpsname, &fps, FPSCONNECTFLAG);
        }
        else
        {
            if(getenv("FPS_DEBUG"))
            {
                printf("DEBUG: FPS EXISTS\n");
            }
        }
    }

    if(CMDmode & FPSCMDCODE_CONFSTOP)  // stop conf
    {
        fps.md->signal &= ~FUNCTION_PARAMETER_STRUCT_SIGNAL_CONFRUN;
        fps_disconnect(&fps);
        fps.localstatus &= ~FPS_LOCALSTATUS_CONFLOOP; // stop loop
    }
    else
    {
        fps.localstatus |= FPS_LOCALSTATUS_CONFLOOP;
    }

    if((CMDmode & FPSCMDCODE_FPSINITCREATE) ||
            (CMDmode & FPSCMDCODE_FPSINIT) || (CMDmode & FPSCMDCODE_CONFSTOP))
    {
        fps.localstatus &= ~FPS_LOCALSTATUS_CONFLOOP; // do not start conf
    }

    if(CMDmode & FPSCMDCODE_CONFSTART)
    {
        fps.localstatus |= FPS_LOCALSTATUS_CONFLOOP;
    }

    return fps;
}


/**
 * @brief Set up the FPS configuration process.
 *
 * Scans fpslist files, connects to all listed FPS
 * instances, builds the keyword tree, and opens
 * the command FIFO. Called once at fpsCTRL startup.
 */
FPS function_parameter_FPCONFsetup(
    const char *fpsname,
    uint32_t    CMDmode
)
{
    return function_parameter_FPCONFsetup_sized(fpsname, CMDmode,
            FUNCTION_PARAMETER_NBPARAM_DEFAULT);
}
