#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <string.h>

#include <dirent.h>
#include <sys/types.h>
#include <sys/stat.h>


#include "streamCTRL_defs.h"

#include "streamCTRL_TUI.h"


/** @brief find shared memory streams on system
 *
 * If filter is set to 1, require stream name to contain namefilter string
 * streaminfo needs to be pre-allocated
 *
 */

int find_streams(
    STREAMINFO *streaminfo,
    int         filter,
    const char *__restrict namefilter
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
            if((long)(pch - dir->d_name) != (long)(strlen(dir->d_name) - 7))
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

                snprintf(fullname, STRINGMAXLEN_FULLFILENAME,
                         "%.700s/%.255s",
                         SHAREDSHMDIR,
                         dir->d_name);
                retv = lstat(fullname, &buf);
                if(retv == -1)
                {
                    printf("File \"%s\"", dir->d_name);
                    PRINT_ERROR("Error running lstat on file: %s", strerror(errno));
                    exit(EXIT_FAILURE);
                }


                if(S_ISLNK(buf.st_mode))  // resolve link name
                {
                    char  fullname_lnk[STRINGMAXLEN_FULLFILENAME];
                    char  linkbuf[STRINGMAXLEN_FULLFILENAME];
                    int   pathOK = 1;

                    streaminfo[sindex].SymLink = 1;
                    snprintf(fullname_lnk, sizeof(fullname_lnk),
                             "%.700s/%.255s",
                             SHAREDSHMDIR,
                             dir->d_name);

                    /* stat() follows the symlink: one syscall to check
                     * reachability, avoiding the expensive realpath(). */
                    struct stat target_buf;
                    if(stat(fullname_lnk, &target_buf) == -1)
                    {
                        pathOK = 0;  /* broken or inaccessible symlink */
                    }

                    if(pathOK == 1)
                    {
                        /* readlink() gives the raw target — one syscall. */
                        ssize_t llen = readlink(fullname_lnk,
                                                linkbuf,
                                                sizeof(linkbuf) - 1);
                        if(llen <= 0)
                        {
                            pathOK = 0;
                        }
                        else
                        {
                            linkbuf[llen] = '\0';
                            char *bn = basename(linkbuf);

                            /* Strip trailing ".im.shm" from basename. */
                            int          lOK = 1;
                            unsigned int ii  = 0;
                            while((lOK == 1) && (ii < strlen(bn)))
                            {
                                if(bn[ii] == '.')
                                {
                                    bn[ii] = '\0';
                                    lOK    = 0;
                                }
                                ii++;
                            }
                            strncpy(streaminfo[sindex].linkname,
                                    bn,
                                    STRINGMAXLEN_STREAMINFO_NAME - 1);
                            streaminfo[sindex].linkname[
                                STRINGMAXLEN_STREAMINFO_NAME - 1] = '\0';
                        }
                    }

                    if(pathOK == 0)
                    {
                        scanentryOK = 0;
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
