/**
 * @file    fps_GetFileName.c
 * @brief   get FPS filename for entry
 */

#include "CommandLineInterface/CLIcore.h"

int functionparameter_GetFileName(FUNCTION_PARAMETER_STRUCT *fps, FUNCTION_PARAMETER *fparam, char *outfname,
                                  char *tagname)
{
    int stringmaxlen = STRINGMAXLEN_DIRNAME / 2;
    char ffname[STRINGMAXLEN_FULLFILENAME]; // full filename
    char fname1[stringmaxlen];
    int l;
    char fpsconfdirname[STRINGMAXLEN_DIRNAME];

    if (snprintf(fpsconfdirname, stringmaxlen, "%s/fpsconf", fps->md->workdir) < 0)
    {
        PRINT_ERROR("snprintf error");
    }

    EXECUTE_SYSTEM_COMMAND("mkdir -p %s", fpsconfdirname);

    // build up directory name
    for (l = 0; l < fparam->keywordlevel - 1; l++)
    {
        if (snprintf(fname1, stringmaxlen, "/%s", fparam->keyword[l]) < 0)
        {
            PRINT_ERROR("snprintf error");
        }
        strncat(fpsconfdirname, fname1, STRINGMAXLEN_DIRNAME - 1);

        EXECUTE_SYSTEM_COMMAND("mkdir -p %s", fpsconfdirname);
    }

    if (snprintf(fname1, stringmaxlen, "/%s.%s.txt", fparam->keyword[l], tagname) < 0)
    {
        PRINT_ERROR("snprintf error");
    }

    snprintf(ffname, STRINGMAXLEN_FULLFILENAME, "%s%s", fpsconfdirname, fname1);

    strcpy(outfname, ffname);

    return 0;
}
