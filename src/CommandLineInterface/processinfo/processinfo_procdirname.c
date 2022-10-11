#include <dirent.h>

#include "CLIcore.h"
#include <processtools.h>


#define SHAREDPROCDIR data.shmdir


errno_t processinfo_procdirname(char *procdname)
{
    int  procdirOK = 0;
    DIR *tmpdir;

    // first, we try the env variable if it exists
    char *MILK_PROC_DIR = getenv("MILK_PROC_DIR");
    if(MILK_PROC_DIR != NULL)
    {
        printf(" [ MILK_PROC_DIR ] '%s'\n", MILK_PROC_DIR);

        {
            int slen = snprintf(procdname,
                                STRINGMAXLEN_FULLFILENAME,
                                "%s",
                                MILK_PROC_DIR);
            if(slen < 1)
            {
                PRINT_ERROR("snprintf wrote <1 char");
                abort(); // can't handle this error any other way
            }
            if(slen >= STRINGMAXLEN_FULLFILENAME)
            {
                PRINT_ERROR("snprintf string truncation");
                abort(); // can't handle this error any other way
            }
        }

        // does this direcory exist ?
        tmpdir = opendir(procdname);
        if(tmpdir)  // directory exits
        {
            procdirOK = 1;
            closedir(tmpdir);
        }
        else
        {
            printf(" [ WARNING ] '%s' does not exist\n", MILK_PROC_DIR);
        }
    }

    // second, we try SHAREDPROCDIR default
    if(procdirOK == 0)
    {
        tmpdir = opendir(SHAREDPROCDIR);
        if(tmpdir)  // directory exits
        {
            sprintf(procdname, "%s", SHAREDPROCDIR);
            procdirOK = 1;
            closedir(tmpdir);
        }
    }

    // if all above fails, set to /tmp
    if(procdirOK == 0)
    {
        tmpdir = opendir("/tmp");
        if(!tmpdir)
        {
            exit(EXIT_FAILURE);
        }
        else
        {
            sprintf(procdname, "/tmp");
            procdirOK = 1;
        }
    }

    return RETURN_SUCCESS;
}
