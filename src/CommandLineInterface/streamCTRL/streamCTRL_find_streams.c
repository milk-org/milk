#define _GNU_SOURCE
#include <string.h>

#include <dirent.h>
#include <sys/types.h>
#include <sys/stat.h>


#include <ncurses.h>


#include "CLIcore.h"

#include "streamCTRL_TUI.h"


// default location of file mapped semaphores, can be over-ridden by env variable MILK_SHM_DIR
#define SHAREDSHMDIR  data.shmdir



/** @brief find shared memory streams on system
 *
 * If filter is set to 1, require stream name to contain namefilter string
 * streaminfo needs to be pre-allocated
 *
 */

int find_streams(
    STREAMINFO *streaminfo,
    int         filter,
    const char * __restrict namefilter
)
{
    int            NBstream = 0;
    DIR           *d;
    struct dirent *dir;

    d = opendir(SHAREDSHMDIR);
    if(d)
    {
        int sindex = 0;
        while(((dir = readdir(d)) != NULL))
        {
            int   scanentryOK = 1;
            char *pch         = strstr(dir->d_name, ".im.shm");

            int matchOK = 1;

            // check that .im.shm terminates the string
            if( pch - dir->d_name != strlen(dir->d_name) - 7 )
            {
                matchOK = 0;
            }

            // name filtering (first pass, not exclusive to stream name, includes path and extension
            if(filter == 1)
            {
                if(strstr(dir->d_name, namefilter) == NULL)
                {
                    matchOK = 0;
                }
            }


            if((pch) && (matchOK == 1))
            {
                // is file sym link ?
                struct stat buf;
                int         retv;
                char        fullname[STRINGMAXLEN_FULLFILENAME];

                WRITE_FULLFILENAME(fullname,
                                   "%s/%s",
                                   SHAREDSHMDIR,
                                   dir->d_name);
                retv = lstat(fullname, &buf);
                if(retv == -1)
                {
                    endwin();
                    printf("File \"%s\"", dir->d_name);
                    perror("Error running lstat on file ");
                    exit(EXIT_FAILURE);
                }

                if(S_ISLNK(buf.st_mode))  // resolve link name
                {
                    char  fullname[STRINGMAXLEN_FULLFILENAME];
                    char *linknamefull;
                    char  linkname[STRINGMAXLEN_FULLFILENAME];
                    int   pathOK = 1;

                    streaminfo[sindex].SymLink = 1;
                    WRITE_FULLFILENAME(fullname,
                                       "%s/%s",
                                       SHAREDSHMDIR,
                                       dir->d_name);
                    linknamefull = realpath(fullname, NULL);

                    if(linknamefull == NULL)
                    {
                        pathOK = 0;
                    }
                    else if(access(linknamefull,
                                   R_OK)) // file cannot be read
                    {
                        pathOK = 0;
                    }

                    if(pathOK == 0)  // file cannot be read
                    {
                        scanentryOK = 0;
                    }
                    else
                    {
                        strcpy(linkname, basename(linknamefull));

                        int          lOK = 1;
                        unsigned int ii  = 0;
                        while((lOK == 1) && (ii < strlen(linkname)))
                        {
                            if(linkname[ii] == '.')
                            {
                                linkname[ii] = '\0';
                                lOK          = 0;
                            }
                            ii++;
                        }
                        strncpy(streaminfo[sindex].linkname,
                                linkname,
                                STRINGMAXLEN_STREAMINFO_NAME - 1);
                    }

                    if(linknamefull != NULL)
                    {
                        free(linknamefull);
                    }
                }
                else
                {
                    streaminfo[sindex].SymLink = 0;
                }

                // get stream name
                if(scanentryOK == 1)
                {
                    int strlencp1 = STRINGMAXLEN_STREAMINFO_NAME;
                    int strlencp  = strlen(dir->d_name) - strlen(".im.shm");
                    if(strlencp < strlencp1)
                    {
                        strlencp1 = strlencp;
                    }
                    strncpy(streaminfo[sindex].sname, dir->d_name, strlencp1);
                    streaminfo[sindex].sname[strlen(dir->d_name) - strlen(".im.shm")] = '\0';

                    if(filter == 1)
                    {
                        if(strstr(streaminfo[sindex].sname, namefilter) != NULL)
                        {
                            sindex++;
                        }
                    }
                    else
                    {
                        sindex++;
                    }
                }
            }
        }

        NBstream = sindex;
    }
    closedir(d);

    return NBstream;
}
