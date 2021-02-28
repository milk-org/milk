/**
 * @file    fps_list.c
 * @brief   list function parameter structure
 */



#include <dirent.h>

#include "CommandLineInterface/CLIcore.h"





// ==========================================
// Forward declaration(s)
// ==========================================

errno_t fps_list();



// ==========================================
// Command line interface wrapper function(s)
// ==========================================





// ==========================================
// Register CLI command(s)
// ==========================================

errno_t fps_list_addCLIcmd()
{

    RegisterCLIcommand(
        "fpslist",
        __FILE__,
        fps_list,
        "list function parameter structures (FPSs)",
        "no argument",
        "fpslist",
        "errno_t fps_list()");

    return RETURN_SUCCESS;
}






errno_t fps_list()
{
    long fpsID;
    long fpscnt = 0;

    int NBchar_fpsID = 5;
    int NBchar_fpsname = 12;
    int NBchar_NBparam = 4;

    printf("FPSs currently connected :\n");
    /*printf("%*s  %*s  %*s\n",
           NBchar_fpsID, "ID",
           NBchar_fpsname, "name",
           NBchar_NBparam, "#par"
          );*/

    for(fpsID = 0; fpsID < data.NB_MAX_FPS; fpsID++)
    {
        if(data.fpsarray[fpsID].SMfd > -1)
        {
            // connected
            printf("%*ld  %*s  %*ld/%*ld entries\n",
                   NBchar_fpsID, fpsID,
                   NBchar_fpsname, data.fpsarray[fpsID].md[0].name,
                   NBchar_NBparam, data.fpsarray[fpsID].NBparamActive,
                   NBchar_NBparam, data.fpsarray[fpsID].NBparam
                  );

            fpscnt++;
        }
    }

    //printf("\n %ld FPS(s) currently loaded\n\n", fpscnt);
    printf("\n");

    printf("FPSs in system shared memory (%s):\n", data.shmdir);

    struct dirent *de;
    DIR *dr = opendir(data.shmdir);
    if(dr == NULL)
    {
        printf("Could not open current directory");
        return RETURN_FAILURE;
    }

    while((de = readdir(dr)) != NULL)
    {
        if(strstr(de->d_name, ".fps.shm") != NULL)
        {
            printf("    %s\n", de->d_name);
        }
    }
    closedir(dr);

    return RETURN_SUCCESS;
}




