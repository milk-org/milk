/**
 * @file    fps_GetParamIndex.h
 * @brief   Get index of parameter
 */

#include "fps.h"

/**
 * @brief Look up a parameter index by its dot-separated
 * keyword name.
 *
 * Searches the FPS parameter array for a matching
 * keyword chain. Returns -1 if not found.
 */
int functionparameter_GetParamIndex(FPS *fps, const char *paramname)
{
    long index  = -1;
    long pindex = 0;

    long NBparamMAX = fps->md->NBparamMAX;

    int found = 0;
    for (pindex = 0; pindex < NBparamMAX; pindex++)
    {
        if (found == 0)
        {
            if (fps->parray[pindex].fpflag & FPFLAG_ACTIVE)
            {
                if (strstr(fps->parray[pindex].keywordfull, paramname) != NULL)
                {
                    index = pindex;
                    found = 1;
                }
            }
        }
    }

    /*
    if(index == -1)
    {
        printf("ERROR: cannot find parameter \"%s\" in structure\n", paramname);
        fflush(stdout);
        exit(0);
    }
    */

    return index;
}
