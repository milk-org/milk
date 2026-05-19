/**
 * @file    fps_GetFileName.c
 * @brief   get FPS filename for entry
 */

#include "fps.h"


/** @brief get parameter file name
 *
 * Most recent parameter value stored in this file
 *
 */
int functionparameter_GetFileName(
    FPS       *fps,
    FPS_PARAM *fparam,
    char      *outfname,
    char      *tagname)
{
    char ffname[STRINGMAXLEN_FULLFILENAME];

    // Create FPS directory
    {
        WRITE_DIRNAME(ffname, "%s/%s/fps/", fps->md->workdir, fps->md->datadir);
        EXECUTE_SYSTEM_COMMAND_NOCHECK("mkdir -p %s", ffname);
    }

    // Build up directory name, construct final filename, and copy to output buffer
    {
        char fname1[STRINGMAXLEN_FILENAME];
        
        // Loop index `ll` is declared outside the loop because it is needed
        // after the loop to access the final keyword level for the filename.
        int  ll;

        for(ll = 0; ll < fparam->keywordlevel - 1; ll++)
        {
            if(snprintf(fname1, STRINGMAXLEN_FILENAME, "%s.", fparam->keyword[ll]) <
                    0)
            {
                PRINT_ERROR("snprintf error");
            }
            strncat(ffname, fname1, STRINGMAXLEN_DIRNAME - 1);
        }

        if(snprintf(fname1,
                    STRINGMAXLEN_FILENAME,
                    "%s.%s.txt",
                    fparam->keyword[ll],
                    tagname) < 0)
        {
            PRINT_ERROR("snprintf error");
        }

        char ffname1[STRINGMAXLEN_FULLFILENAME]; // full filename
        snprintf(ffname1, STRINGMAXLEN_FULLFILENAME, "%s", ffname);
        strncat(ffname1, fname1, STRINGMAXLEN_FULLFILENAME - strlen(ffname1) - 1);

        strncpy(outfname, ffname1,
                STRINGMAXLEN_FULLFILENAME - 1);
        outfname[STRINGMAXLEN_FULLFILENAME - 1] = '\0';
    }

    return 0;
}
