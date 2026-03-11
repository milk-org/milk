/**
 * @file    fps_GetParamIndex.h
 * @brief   Get index of parameter
 */

#include "fps.h"
#include "fps_internal.h"

int functionparameter_GetParamIndex(FUNCTION_PARAMETER_STRUCT *fps,
                                    const char                *paramname)
{
    long index  = -1;
    long pindex = 0;

    long NBparamMAX = fps->md->NBparamMAX;

    int found = 0;
    for(pindex = 0; pindex < NBparamMAX; pindex++)
    {
        if(found == 0)
        {
            if(fps->parray[pindex].fpflag & FPFLAG_ACTIVE)
            {
                if(strstr(fps->parray[pindex].keywordfull, paramname) != NULL)
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
