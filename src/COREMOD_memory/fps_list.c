/**
 * @file    fps_list.c
 * @brief   list function parameter structure
 */

#include <dirent.h>

#ifdef MILK_NO_CLI
#include "CLIcore_standalone.h"
#else
#include "CLIcore.h"
#endif

// ==========================================
// Forward declaration(s)
// ==========================================

errno_t fps_list();

// ==========================================
#ifndef MILK_NO_CLI
// Command line interface wrapper function(s)
// ==========================================

// ==========================================
// Register CLI command(s)
// ==========================================

errno_t fps_list_addCLIcmd()
{

    RegisterCLIcommand("fpslist",
                       __FILE__,
                       fps_list,
                       "list function parameter structures (FPSs)",
                       "no argument",
                       "fpslist",
                       "errno_t fps_list()");

    return RETURN_SUCCESS;
}
#endif /* MILK_NO_CLI */
errno_t fps_list()
{
    long fpsID;
    long fpscnt = 0;

    int NBchar_fpsID   = 5;
    int NBchar_fpsname = 12;
    int NBchar_NBparam = 4;

    for(fpsID = 0; fpsID < dcnfps; fpsID++)
    {
        if(dcfpsarr[fpsID].SMfd > -1)
        {

            if(fpscnt == 0)
            {
                printf("FPSs currently connected :\n");
            }
            // connected
            printf("%*ld  %*s  %*ld/%*ld entries\n",
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

    //printf("\n %ld FPS(s) currently loaded\n\n", fpscnt);
    //printf("\n");

    printf("FPSs in system shared memory (%s):\n", dcshmdir);

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
        if(strstr(de->d_name, ".fps.shm") != NULL)
        {
            char fpsname[100];
            int  slen1 = 100 - strlen(".fps.shm");

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

