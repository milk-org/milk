/**
 * @file    fps_ID.c
 * @brief   find fps ID(s) from name
 */

#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#include "fps.h"
#include "fps_internal.h"
#include "fps_globals.h"

/* ID number corresponding to a name */
long fps_ID(const char *name)
{
    long i;
    int  loopOK;
    long tmpID = 0;

    if(fpsarray == NULL) {
        return -1;
    }

    i      = 0;
    loopOK = 1;
    while(loopOK == 1)
    {

        if(fpsarray[i].SMfd >= 0)
        {
            // fps in use

            if((strncmp(name, fpsarray[i].md->name, strlen(name)) == 0) &&
                    (fpsarray[i].md->name[strlen(name)] == '\0'))
            {
                loopOK = 0;
                tmpID  = i;
            }
        }

        i++;

        if(i == NB_FPS_MAX)
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

    if(fpsarray == NULL) {
        return -1;
    }

    for(i = 0; i < NB_FPS_MAX; i++)
    {
        if(fpsarray[i].SMfd < 0)
        {
            // fps is unused, lets grab it
            ID = i;
            break;
        }
    }

    if(ID == -1)
    {
        printf("ERROR: ran out of FPS IDs - cannot allocate new ID\n");
        printf("NB_FPS_MAX should be increased above current value (%d)\n",
               NB_FPS_MAX);
        exit(0);
    }

    return ID;
}
