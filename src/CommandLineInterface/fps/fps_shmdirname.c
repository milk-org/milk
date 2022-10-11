/**
 * @file    fps_shmdirname.c
 * @brief   create FPS shared memory directory name
 */

#include <dirent.h>

#include "CommandLineInterface/CLIcore.h"

#define SHAREDSHMDIR data.shmdir

errno_t function_parameter_struct_shmdirname(char *shmdname)
{
    int                  shmdirOK = 0;
    DIR                 *tmpdir;
    static unsigned long functioncnt = 0;
    static char          shmdname_static[STRINGMAXLEN_SHMDIRNAME];

    if(functioncnt == 0)
    {
        functioncnt++; // ensure we only run this once, and then retrieve stored result from shmdname_static

        // first, we try the env variable if it exists
        char *MILK_SHM_DIR = getenv("MILK_SHM_DIR");
        if(MILK_SHM_DIR != NULL)
        {
            DEBUG_TRACEPOINT("MILK_SHM_DIR is '%s'\n", MILK_SHM_DIR);

            {
                int slen = snprintf(shmdname,
                                    STRINGMAXLEN_SHMDIRNAME,
                                    "%s",
                                    MILK_SHM_DIR);
                if(slen < 1)
                {
                    PRINT_ERROR("snprintf wrote <1 char");
                    abort(); // can't handle this error any other way
                }
                if(slen >= STRINGMAXLEN_SHMDIRNAME)
                {
                    PRINT_ERROR("snprintf string truncation");
                    abort(); // can't handle this error any other way
                }
            }

            // does this direcory exist ?
            tmpdir = opendir(shmdname);
            if(tmpdir)  // directory exits
            {
                shmdirOK = 1;
                closedir(tmpdir);
            }
            else
            {
                abort();
            }
        }

        // second, we try SHAREDSHMDIR default
        if(shmdirOK == 0)
        {
            tmpdir = opendir(SHAREDSHMDIR);
            if(tmpdir)  // directory exits
            {
                {
                    int slen = snprintf(shmdname,
                                        STRINGMAXLEN_SHMDIRNAME,
                                        "%s",
                                        SHAREDSHMDIR);
                    if(slen < 1)
                    {
                        PRINT_ERROR("snprintf wrote <1 char");
                        abort(); // can't handle this error any other way
                    }
                    if(slen >= STRINGMAXLEN_SHMDIRNAME)
                    {
                        PRINT_ERROR("snprintf string truncation");
                        abort(); // can't handle this error any other way
                    }
                }

                shmdirOK = 1;
                closedir(tmpdir);
            }
        }

        // if all above fails, set to /tmp
        if(shmdirOK == 0)
        {
            tmpdir = opendir("/tmp");
            if(!tmpdir)
            {
                exit(EXIT_FAILURE);
            }
            else
            {
                sprintf(shmdname, "/tmp");
                shmdirOK = 1;
                closedir(tmpdir);
            }
        }

        {
            int slen = snprintf(shmdname_static,
                                STRINGMAXLEN_SHMDIRNAME,
                                "%s",
                                shmdname); // keep it memory
            if(slen < 1)
            {
                PRINT_ERROR("snprintf wrote <1 char");
                abort(); // can't handle this error any other way
            }
            if(slen >= STRINGMAXLEN_SHMDIRNAME)
            {
                PRINT_ERROR("snprintf string truncation");
                abort(); // can't handle this error any other way
            }
        }
    }
    else
    {
        {
            int slen = snprintf(shmdname,
                                STRINGMAXLEN_SHMDIRNAME,
                                "%s",
                                shmdname_static);
            if(slen < 1)
            {
                PRINT_ERROR("snprintf wrote <1 char");
                abort(); // can't handle this error any other way
            }
            if(slen >= STRINGMAXLEN_SHMDIRNAME)
            {
                PRINT_ERROR("snprintf string truncation");
                abort(); // can't handle this error any other way
            }
        }
    }

    return RETURN_SUCCESS;
}
