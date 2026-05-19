/**
 * @file    fps_list.c
 * @brief   list function parameter structures
 *
 * Uses FPS V2 framework.
 */

#include <dirent.h>

#ifdef MILK_NO_CLI
#include "CLIcore_standalone.h"
#else
#include "CLIcore.h"
#endif
#include "fps.h"
#include "COREMOD_memory/COREMOD_memory.h"


/* ================================================================
 * 1.  FPS COMPONENT IDENTITY
 * ============================================================= */

static FPS_APP_INFO FPS_app_info =
{
    .fps_name    = "fpslist",
    .cmdkey      = "fpslist",
    .description =
    "list function parameter structures",
    .description_long =
    "List all FPS (Function Parameter Structure) instances currently active in shared memory. Shows FPS name, status, and associated process."
};


/* ================================================================
 * 2.  LOCAL PARAMETER VARIABLES
 * ============================================================= */

/* (none — zero-arg command) */


/* ================================================================
 * 3.  UNIFIED PARAMETER TABLE (X-Macro)
 * ============================================================= */

#define FPS_PARAMS(X)  /* empty */


/* ================================================================
 * 4.  COMPUTATION LOGIC
 * ============================================================= */

errno_t fps_list()
{

    long fpscnt = 0;

    int NBchar_fpsID   = 5;
    int NBchar_fpsname = 12;
    int NBchar_NBparam = 4;

    for(long fpsID = 0; fpsID < dcnfps; fpsID++)
    {
        if(dcfpsarr[fpsID].SMfd > -1)
        {
            if(fpscnt == 0)
            {
                printf(
                    "FPSs currently connected :\n");
            }
            printf(
                "%*ld  %*s  %*ld/%*ld entries\n",
                NBchar_fpsID,
                fpsID,
                NBchar_fpsname,
                dcfpsarr[fpsID].md[0].name,
                NBchar_NBparam,
                dcfpsarr[fpsID].NBparamActive,
                NBchar_NBparam,
                dcfpsarr[fpsID].NBparam);

            fpscnt++;
        }
    }
    if(fpscnt == 0)
    {
        printf("No FPS currently connected\n");
    }

    printf(
        "FPSs in system shared memory (%s):\n",
        dcshmdir);

    struct dirent *de;
    DIR           *dr = opendir(dcshmdir);
    if(dr == NULL)
    {
        printf("Could not open current directory");
        return RETURN_FAILURE;
    }

    fpscnt = 0;
    while((de = readdir(dr)) != NULL)
    {
        if(strstr(de->d_name, ".fps.shm")
                != NULL)
        {
            char fpsname[100];
            int  slen1 =
                100 - strlen(".fps.shm");

            strncpy(fpsname, de->d_name, slen1);
            fpsname[slen1] = '\0';
            printf("%*ld  %*s\n",
                   NBchar_fpsID,
                   fpscnt,
                   NBchar_fpsname,
                   fpsname);
            fpscnt++;
        }
    }
    closedir(dr);

    return RETURN_SUCCESS;
}


/* ================================================================
 * 5.  BINDINGS, FARG, AND CLI DATA
 * ============================================================= */

#ifdef FPS_STANDALONE
CLICMDDATA CLIcmddata =
{
#else
static CLICMDDATA CLIcmddata =
{
#endif
    "",
    "",
    CLICMD_FIELDS_NOPARAM
};

FPS_CMDSETTINGS_INIT(dft, CLIcmddata, FPS_app_info)


/* ================================================================
 * 6.  COMPUTE WRAPPER
 * ============================================================= */

static MILK_HOT errno_t __attribute__((unused)) compute_function()
{
    DEBUG_TRACE_FSTART();

    FUNC_CHECK_RETURN(fps_list());

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}


/* ================================================================
 * 7.  MILK MODULE REGISTRATION
 * ============================================================= */

#if !defined(FPS_STANDALONE) && !defined(MILK_NO_CLI)
static errno_t CLIfunction(void)
{
    return safe_fps_generic_CLIfunction(
               &FPS_app_info, NULL, &CLIcmddata,
               NULL, 0,
               compute_function);
}

errno_t
CLIADDCMD_COREMOD_memory__fps_list()
{
    int cmdi = RegisterCLIcmd(
                   CLIcmddata, CLIfunction);
    CLIcmddata.cmdsettings =
        &data.cmd[cmdi].cmdsettings;

    return RETURN_SUCCESS;
}
#endif
