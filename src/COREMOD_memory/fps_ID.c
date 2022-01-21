/**
 * @file    fps_ID.c
 * @brief   find fps ID(s) from name
 */

#include "CommandLineInterface/CLIcore.h"

/* ID number corresponding to a name */
long fps_ID(const char *name)
{
    long i;
    int  loopOK;
    long tmpID = 0;

    i      = 0;
    loopOK = 1;
    while (loopOK == 1)
    {

        if (data.fpsarray[i].SMfd >= 0)
        {
            // fps in use

            if ((strncmp(name, data.fpsarray[i].md->name, strlen(name)) == 0) &&
                (data.fpsarray[i].md->name[strlen(name)] == '\0'))
            {
                loopOK = 0;
                tmpID  = i;
            }
        }

        i++;

        if (i == data.NB_MAX_FPS)
        {
            loopOK = 0;
            tmpID  = -1;
        }
    }

    return tmpID;
}

/* next available ID number */
long next_avail_fps_ID()
{
    long i;
    long ID = -1;

#ifdef _OPENMP
#pragma omp critical
    {
#endif
        for (i = 0; i < data.NB_MAX_FPS; i++)
        {
            if (data.fpsarray[i].SMfd < 0)
            {
                // fps is unused, lets grab it
                ID = i;
                break;
            }
        }
#ifdef _OPENMP
    }
#endif

    if (ID == -1)
    {
        printf("ERROR: ran out of FPS IDs - cannot allocate new ID\n");
        printf("NB_MAX_FPS should be increased above current value (%ld)\n",
               data.NB_MAX_FPS);
        exit(0);
    }

    return ID;
}
