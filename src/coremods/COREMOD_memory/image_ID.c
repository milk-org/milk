// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    image_ID.c
 * @brief   find image ID(s) from name
 */

#include <string.h>

#ifdef MILK_NO_CLI
#    include "CLIcore_standalone.h"
#    include "COREMOD_memory/COREMOD_memory.h"
#else
#    include "libmilkdata/milkdata.h"
#    include "milkDebugTools.h"
#endif

/* ID number corresponding to a name */
imageID image_ID(const char *name, IMAGE *imagearray, long NB_images)
{
    DEBUG_TRACE_FSTART();

    if (imagearray == NULL || name == NULL || NB_images <= 0)
    {
        DEBUG_TRACE_FEXIT();
        return -1;
    }

    size_t namelen = strlen(name);
    for (imageID i = 0; i < NB_images; i++)
    {
        if (imagearray[i].used == 1)
        {
            if ((strncmp(name, imagearray[i].name, namelen) == 0) &&
                (imagearray[i].name[namelen] == '\0'))
            {
                clock_gettime(CLOCK_MILK, &imagearray[i].md[0].lastaccesstime);
                DEBUG_TRACEPOINT("FOUT %s -> %ld", name, i);
                DEBUG_TRACE_FEXIT();
                return i;
            }
        }
    }

    DEBUG_TRACEPOINT("FOUT %s -> -1", name);
    DEBUG_TRACE_FEXIT();
    return -1;
}

/* ID number corresponding to a name */
MILK_PURE imageID image_ID_noaccessupdate(const char *name, IMAGE *imagearray, long NB_images)
{
    DEBUG_TRACE_FSTART();

    if (imagearray == NULL || name == NULL || NB_images <= 0)
    {
        DEBUG_TRACE_FEXIT();
        return -1;
    }

    size_t namelen = strlen(name);
    for (imageID i = 0; i < NB_images; i++)
    {
        if (imagearray[i].used == 1)
        {
            if ((strncmp(name, imagearray[i].name, namelen) == 0) &&
                (imagearray[i].name[namelen] == '\0'))
            {
                DEBUG_TRACE_FEXIT();
                return i;
            }
        }
    }

    DEBUG_TRACE_FEXIT();
    return -1;
}

/* next available ID number */
imageID next_avail_image_ID(imageID preferredID)
{
    DEBUG_TRACE_FSTART();

    imageID i;
    imageID ID = -1;

#ifdef _OPENMP
#    pragma omp critical
    {
#endif
        if ((preferredID > -1) && (preferredID < dcnimg) && (dcimg[preferredID].used == 0))
        {
            ID             = preferredID;
            dcimg[ID].used = 1;
        }
        else
        {
            for (i = 0; i < dcnimg; i++)
            {
                if (dcimg[i].used == 0)
                {
                    ID             = i;
                    dcimg[ID].used = 1;
                    break;
                }
            }
        }
#ifdef _OPENMP
    }
#endif
    if (ID == -1)
    {
        PRINT_ERROR("ran out of image IDs"
                    " (NB_MAX_IMAGE=%ld)",
                    dcnimg);
    }

    DEBUG_TRACEPOINT("FOUT ID : %ld", ID);

    DEBUG_TRACE_FEXIT();
    return ID;
}
