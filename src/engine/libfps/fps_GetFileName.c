/**
 * @file    fps_GetFileName.c
 * @brief   get FPS filename for entry
 */

#include "fps.h"
#include "fps_internal.h"


/** @brief get parameter file name
 *
 * Most recent parameter value stored in this file
 *
 */
int functionparameter_GetFileName(
    FPS *fps,
    FPS_PARAM        *fparam,
    char                      *outfname,
    char                      *tagname)
{
    char ffname[STRINGMAXLEN_FULLFILENAME];
    char fname1[STRINGMAXLEN_FILENAME];
    int  l;
    //char fpsdatadirname[STRINGMAXLEN_DIRNAME];

    WRITE_DIRNAME(ffname, "%s/%s/fps/", fps->md->workdir, fps->md->datadir);
    EXECUTE_SYSTEM_COMMAND_NOCHECK("mkdir -p %s", ffname);

    // build up directory name
    for(l = 0; l < fparam->keywordlevel - 1; l++)
    {
        if(snprintf(fname1, STRINGMAXLEN_FILENAME, "%s.", fparam->keyword[l]) <
                0)
        {
            PRINT_ERROR("snprintf error");
        }
        strncat(ffname, fname1, STRINGMAXLEN_DIRNAME - 1);
    }

    if(snprintf(fname1,
                STRINGMAXLEN_FILENAME,
                "%s.%s.txt",
                fparam->keyword[l],
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

    return 0;
}
