/**
 * @file    fps_GetFileName.c
 * @brief   get FPS filename for entry
 */

#include "CommandLineInterface/CLIcore.h"


/** @brief get parameter file name
 *
 * Most recent parameter value stored in this file
 *
 */
int functionparameter_GetFileName(
    FUNCTION_PARAMETER_STRUCT *fps,
    FUNCTION_PARAMETER        *fparam,
    char                      *outfname,
    char                      *tagname)
{
    char ffname[STRINGMAXLEN_FULLFILENAME];
    char fname1[STRINGMAXLEN_FILENAME];
    int  l;
    //char fpsdatadirname[STRINGMAXLEN_DIRNAME];

    WRITE_DIRNAME(ffname, "%s/%s/fps/", fps->md->workdir, fps->md->datadir);
    EXECUTE_SYSTEM_COMMAND("mkdir -p %s", ffname);

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
    snprintf(ffname1, STRINGMAXLEN_FULLFILENAME, "%s%s", ffname, fname1);

    strcpy(outfname, ffname1);

    return 0;
}
