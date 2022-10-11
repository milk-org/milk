#include <dirent.h>
#include <stdio.h>
#include <string.h>

#include "CommandLineInterface/CLIcore.h"

/** @brief Set shared memory directory
 *
 * SHM directory to store shared memory
 *
 * If MILK_SHM_DIR environment variable exists, use it.
 * If fails, print warning, use SHAREDMEMDIR defined in ImageStruct.h
 * If fails -> use /tmp
 *
 */
errno_t setSHMdir()
{
    char shmdirname[200];
    int  shmdirOK = 0; // toggles to 1 when directory is found
    DIR *tmpdir;

    // first, we try the env variable if it exists
    char *MILK_SHM_DIR = getenv("MILK_SHM_DIR");
    if(MILK_SHM_DIR != NULL)
    {
        if(data.quiet == 0)
        {
            printf("        MILK_SHM_DIR '%s'\n", MILK_SHM_DIR);
        }
        sprintf(shmdirname, "%s", MILK_SHM_DIR);

        // does this direcory exist ?
        tmpdir = opendir(shmdirname);
        if(tmpdir)  // directory exits
        {
            shmdirOK = 1;
            closedir(tmpdir);
            if(data.quiet == 0)
            {
                printf("        Using SHM directory %s\n", shmdirname);
            }
        }
        else
        {
            printf("%c[%d;%dm", (char) 27, 1, 31); // set color red
            printf("    ERROR: Directory %s : %s\n",
                   shmdirname,
                   strerror(errno));
            printf("%c[%d;m", (char) 27, 0); // unset color red
            exit(EXIT_FAILURE);
        }
    }
    else
    {
        if(data.quiet == 0)
        {
            printf("%c[%d;%dm", (char) 27, 1, 31); // set color red
            printf(
                "    WARNING: Environment variable MILK_SHM_DIR not "
                "specified -> falling back to default %s\n",
                SHAREDMEMDIR);
            printf(
                "    BEWARE : Other milk users may be using the same "
                "SHM directory on this machine, and could see "
                "your milk session data and temporary files\n");
            printf(
                "    BEWARE : Some scripts may rely on MILK_SHM_DIR to "
                "find/access shared memory and temporary "
                "files, and WILL not run.\n");
            printf(
                "             Please set MILK_SHM_DIR and restart CLI "
                "to set up user-specific shared memory and "
                "temporary files\n");
            printf(
                "             Example: Add \"export "
                "MILK_SHM_DIR=/milk/shm\" to .bashrc\n");
            printf("%c[%d;m", (char) 27, 0); // unset color red
        }
    }

    // second, we try SHAREDMEMDIR default
    if(shmdirOK == 0)
    {
        tmpdir = opendir(SHAREDMEMDIR);
        if(tmpdir)  // directory exits
        {
            sprintf(shmdirname, "%s", SHAREDMEMDIR);
            shmdirOK = 1;
            closedir(tmpdir);
            if(data.quiet == 0)
            {
                printf("        Using SHM directory %s\n", shmdirname);
            }
        }
        else
        {
            if(data.quiet == 0)
            {
                printf("        Directory %s : %s\n",
                       SHAREDMEMDIR,
                       strerror(errno));
            }
        }
    }

    // if all above fails, set to /tmp
    if(shmdirOK == 0)
    {
        tmpdir = opendir("/tmp");
        if(!tmpdir)
        {
            printf("        ERROR: Directory %s : %s\n",
                   shmdirname,
                   strerror(errno));
            exit(EXIT_FAILURE);
        }
        else
        {
            sprintf(shmdirname, "/tmp");
            shmdirOK = 1;
            if(data.quiet == 0)
            {
                printf("        Using SHM directory %s\n", shmdirname);

                printf(
                    "        NOTE: Consider creating tmpfs "
                    "directory and setting env var MILK_SHM_DIR "
                    "for improved "
                    "performance :\n");
                printf(
                    "        $ echo \"tmpfs %s tmpfs "
                    "rw,nosuid,nodev\" | sudo tee -a /etc/fstab\n",
                    SHAREDMEMDIR);
                printf("        $ sudo mkdir -p %s\n", SHAREDMEMDIR);
                printf("        $ sudo mount %s\n", SHAREDMEMDIR);
            }
        }
    }

    sprintf(data.shmdir, "%s", shmdirname);

    // change / to . and write to shmsemdirname
    unsigned int stri;
    for(stri = 0; stri < strlen(shmdirname); stri++)
        if(shmdirname[stri] == '/')  // replace '/' by '.'
        {
            shmdirname[stri] = '.';
        }

    sprintf(data.shmsemdirname, "%s", shmdirname);
    if(data.quiet == 0)
    {
        printf("        semaphore naming : /dev/shm/sem.%s.<sname>_sem<xx>\n",
               data.shmsemdirname);
    }

    return RETURN_SUCCESS;
}
