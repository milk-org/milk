
#define _GNU_SOURCE
#include <string.h>

#include <stdlib.h>
#include <malloc.h>
#include <sys/types.h>

#include <unistd.h>

#include <dirent.h>

#include <sys/stat.h>

#include <ncurses.h>


#include "CLIcore.h"
#include "streamCTRL_TUI.h"
#include "streamCTRL_find_streams.h"
#include "streamCTRL_utilfuncs.h"




// default location of file mapped semaphores, can be over-ridden by env variable MILK_SHM_DIR
#define SHAREDSHMDIR  data.shmdir



void *streamCTRL_scan(
    void *argptr
)
{
    long NBsindex = 0;
    long sindex   = 0;
    long scancnt  = 0;

    STREAMINFO *streaminfo;
    char      **PIDname_array;

    // timing
    static int             firstIter = 1;
    static struct timespec t0;
    struct timespec        t1;
    double                 tdiffv;
    struct timespec        tdiff;

    streamCTRLarg_struct *streamCTRLdata =
        (streamCTRLarg_struct *) argptr;

    STREAMINFOPROC *streaminfoproc = streamCTRLdata->streaminfoproc;
    IMAGE          *images         = streamCTRLdata->images;
    streaminfo    = streamCTRLdata->sinfo;

    PIDname_array = streaminfoproc->PIDtable;

    streaminfoproc->loopcnt = 0;

    // if set, write file list to file on first scan
    //int WriteFlistToFile = 1;

    FILE *fpfscan;

    while(streaminfoproc->loop == 1)
    {

        // timing measurement
        clock_gettime(CLOCK_MILK, &t1);
        if(firstIter == 1)
        {
            tdiffv = 0.1;
        }
        else
        {
            tdiff  = timespec_diff(t0, t1);
            tdiffv = 1.0 * tdiff.tv_sec + 1.0e-9 * tdiff.tv_nsec;
        }
        clock_gettime(CLOCK_MILK, &t0);
        streaminfoproc->dtscan = tdiffv;

        int mode = 0;
        if(mode == 0)  // preferred mode
        {
            NBsindex = find_streams(streaminfo,
                                    streaminfoproc->filter,
                                    streaminfoproc->namefilter);

            // write stream list to file if applicable
            // ususally used for debugging only
            //
            if(streaminfoproc->WriteFlistToFile == 1)
            {
                fpfscan = fopen("streamCTRL_filescan.dat", "w");
                fprintf(fpfscan, "# stream scan result\n");
                fprintf(fpfscan,
                        "filter: %d %s\n",
                        streaminfoproc->filter,
                        streaminfoproc->namefilter);
                fprintf(fpfscan, "NBsindex = %ld\n", NBsindex);

                for(sindex = 0; sindex < NBsindex; sindex++)
                {
                    //fprintf(fpfscan, "%4ld  %20s ", sindex, dir->d_name);

                    if(streaminfo[sindex].SymLink == 1)
                    {
                        fprintf(fpfscan,
                                "| %12s -> [ %12s ] ",
                                streaminfo[sindex].sname,
                                streaminfo[sindex].linkname);
                    }
                    else
                    {
                        fprintf(fpfscan,
                                "| %12s -> [ %12s ] ",
                                streaminfo[sindex].sname,
                                " ");
                    }
                    fprintf(fpfscan, "\n");
                }
                fclose(fpfscan);
            }

            // Load into memory
            for(sindex = 0; sindex < NBsindex; sindex++)
            {
                imageID ID;

                ID = image_ID_from_images(images, streaminfo[sindex].sname);

                // connect to stream
                if(ID == -1)
                {
                    ID = image_get_first_ID_available_from_images(images);
                    if(ID < 0)
                    {
                        return NULL;
                    }
                    ImageStreamIO_read_sharedmem_image_toIMAGE(
                        streaminfo[sindex].sname,
                        &images[ID]);
                    streaminfo[sindex].deltacnt0          = 1;
                    streaminfo[sindex].updatevalue        = 1.0;
                    streaminfo[sindex].updatevalue_frozen = 1.0;
                }
                else
                {
                    float gainv = 1.0;
                    if(firstIter == 0)
                    {
                        streaminfo[sindex].deltacnt0 =
                            images[ID].md[0].cnt0 - streaminfo[sindex].cnt0;
                        streaminfo[sindex].updatevalue =
                            (1.0 - gainv) * streaminfo[sindex].updatevalue +
                            gainv *
                            (1.0 * streaminfo[sindex].deltacnt0 / tdiffv);
                    }

                    streaminfo[sindex].cnt0 =
                        images[ID].md[0].cnt0; // keep memory of cnt0
                    streaminfo[sindex].ID       = ID;
                    streaminfo[sindex].datatype = images[ID].md[0].datatype;
                }
            }
        }
        else // candidate for removal
        {
            //---------------------------------------
            DIR           *d;
            struct dirent *dir;

            d = opendir(SHAREDSHMDIR);
            if(d)
            {
                sindex = 0;

                while(((dir = readdir(d)) != NULL))
                {
                    int   scanentryOK = 1;
                    char *pch         = strstr(dir->d_name, ".im.shm");

                    int matchOK = 0;

                    // name filtering
                    if(streaminfoproc->filter == 1)
                    {
                        if(strstr(dir->d_name, streaminfoproc->namefilter) !=
                                NULL)
                        {
                            matchOK = 1;
                        }
                    }
                    else
                    {
                        matchOK = 1;
                    }

                    if((pch) && (matchOK == 1))
                    {
                        imageID ID;

                        // is file sym link ?
                        struct stat buf;
                        int         retv;
                        char        fullname[STRINGMAXLEN_FULLFILENAME];

                        if(streaminfoproc->WriteFlistToFile == 1)
                        {
                            fprintf(fpfscan,
                                    "%4ld  %20s ",
                                    sindex,
                                    dir->d_name);
                        }

                        WRITE_FULLFILENAME(fullname,
                                           "%s/%s",
                                           SHAREDSHMDIR,
                                           dir->d_name);
                        retv = lstat(fullname, &buf);
                        if(retv == -1)
                        {
                            endwin();
                            printf("File \"%s\"", dir->d_name);
                            perror(
                                "Error running lstat "
                                "on file ");
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
                            //                        readlink (fullname, linknamefull, 200-1);
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
                                if(streaminfoproc->WriteFlistToFile == 1)
                                {
                                    fprintf(fpfscan,
                                            " %s "
                                            "<-> "
                                            "LINK "
                                            "%s "
                                            "CANNOT"
                                            " BE "
                                            "READ "
                                            "-> "
                                            "off",
                                            fullname,
                                            linknamefull);
                                }
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

                        // get stream name and ID
                        if(scanentryOK == 1)
                        {
                            int strlencp1 = STRINGMAXLEN_STREAMINFO_NAME;
                            int strlencp =
                                strlen(dir->d_name) - strlen(".im.shm");
                            if(strlencp < strlencp1)
                            {
                                strlencp1 = strlencp;
                            }
                            strncpy(streaminfo[sindex].sname,
                                    dir->d_name,
                                    strlencp1);
                            streaminfo[sindex].sname[strlen(dir->d_name) -
                                                     strlen(".im.shm")] = '\0';

                            if(streaminfoproc->WriteFlistToFile == 1)
                            {
                                if(streaminfo[sindex].SymLink == 1)
                                {
                                    fprintf(fpfscan,
                                            "| "
                                            "%12s "
                                            "-> [ "
                                            "%12s "
                                            "] ",
                                            streaminfo[sindex].sname,
                                            streaminfo[sindex].linkname);
                                }
                                else
                                {
                                    fprintf(fpfscan,
                                            "| "
                                            "%12s "
                                            "-> [ "
                                            "%12s "
                                            "] ",
                                            streaminfo[sindex].sname,
                                            " ");
                                }
                            }
                        }

                        if(scanentryOK == 1)
                        {

                            ID = image_ID_from_images(images,
                                                      streaminfo[sindex].sname);

                            // connect to stream
                            if(ID == -1)
                            {
                                ID = image_get_first_ID_available_from_images(
                                         images);
                                if(ID < 0)
                                {
                                    return NULL;
                                }
                                ImageStreamIO_read_sharedmem_image_toIMAGE(
                                    streaminfo[sindex].sname,
                                    &images[ID]);
                                streaminfo[sindex].deltacnt0          = 1;
                                streaminfo[sindex].updatevalue        = 1.0;
                                streaminfo[sindex].updatevalue_frozen = 1.0;
                            }
                            else
                            {
                                float gainv = 1.0;
                                if(firstIter == 0)
                                {
                                    streaminfo[sindex].deltacnt0 =
                                        images[ID].md[0].cnt0 -
                                        streaminfo[sindex].cnt0;
                                    streaminfo[sindex].updatevalue =
                                        (1.0 - gainv) *
                                        streaminfo[sindex].updatevalue +
                                        gainv * (1.0 *
                                                 streaminfo[sindex].deltacnt0 /
                                                 tdiffv);
                                }

                                streaminfo[sindex].cnt0 =
                                    images[ID]
                                    .md[0]
                                    .cnt0; // keep memory of cnt0
                                streaminfo[sindex].ID = ID;
                                streaminfo[sindex].datatype =
                                    images[ID].md[0].datatype;

                                sindex++;
                            }
                        }
                        if(streaminfoproc->WriteFlistToFile == 1)
                        {
                            fprintf(fpfscan, "\n");
                        }
                    }
                }

                NBsindex = sindex;
            }
            closedir(d);

            if(streaminfoproc->WriteFlistToFile == 1)
            {
                fclose(fpfscan);
            }
            //---------------------------------------
        }

        streaminfoproc->WriteFlistToFile = 0;

        firstIter = 0;

        if(streaminfoproc->fuserUpdate == 1)
        {
            FILE *fp;
            int   STRINGMAXLEN_LINE = 2000;
            char  plistoutline[STRINGMAXLEN_LINE];
            char  command[STRINGMAXLEN_COMMAND];

            int NBpid = 0;

            //            sindexscan1 = ssindex[sindexscan];
            int sindexscan1 = streaminfoproc->sindexscan;

            if(streaminfoproc->sindexscan > NBsindex - 1)
            {
                streaminfoproc->fuserUpdate = 0;
            }
            else
            {
                int PReadMode = 1;

                if(PReadMode == 0)
                {
                    // popen option
                    {
                        int slen = snprintf(command,
                                            STRINGMAXLEN_COMMAND,
                                            "/bin/fuser %s/%s.im.shm "
                                            "2>/dev/null",
                                            SHAREDSHMDIR,
                                            streaminfo[sindexscan1].sname);
                        if(slen < 1)
                        {
                            PRINT_ERROR("snprintf wrote <1 char");
                            abort(); // can't handle this error any other way
                        }
                        if(slen >= STRINGMAXLEN_COMMAND)
                        {
                            PRINT_ERROR(
                                "snprintf string "
                                "truncation");
                            abort(); // can't handle this error any other way
                        }
                    }

                    fp = popen(command, "r");
                    if(fp == NULL)
                    {
                        streaminfo[sindexscan1].streamOpenPID_status =
                            2; // failed
                    }
                    else
                    {
                        streaminfo[sindexscan1].streamOpenPID_status = 1;

                        if(fgets(plistoutline, STRINGMAXLEN_LINE - 1, fp) ==
                                NULL)
                        {
                            snprintf(plistoutline, STRINGMAXLEN_LINE, " ");
                        }
                        pclose(fp);
                    }
                }
                else
                {
                    // filesystem option
                    char plistfname[STRINGMAXLEN_FULLFILENAME];
                    WRITE_FULLFILENAME(plistfname,
                                       "%s/%s.shmplist",
                                       SHAREDSHMDIR,
                                       streaminfo[sindexscan1].sname);

                    {
                        int slen = snprintf(command,
                                            STRINGMAXLEN_COMMAND,
                                            "/bin/fuser %s/%s.im.shm "
                                            "2>/dev/null > %s",
                                            SHAREDSHMDIR,
                                            streaminfo[sindexscan1].sname,
                                            plistfname);
                        if(slen < 1)
                        {
                            PRINT_ERROR("snprintf wrote <1 char");
                            abort(); // can't handle this error any other way
                        }
                        if(slen >= STRINGMAXLEN_COMMAND)
                        {
                            PRINT_ERROR(
                                "snprintf string "
                                "truncation");
                            abort(); // can't handle this error any other way
                        }
                    }

                    if(system(command) == -1)
                    {
                        perror("Command system() failed");
                        exit(EXIT_FAILURE);
                    }

                    fp = fopen(plistfname, "r");
                    if(fp == NULL)
                    {
                        streaminfo[sindexscan1].streamOpenPID_status = 2;
                    }
                    else
                    {
                        if(fgets(plistoutline, STRINGMAXLEN_LINE - 1, fp) ==
                                NULL)
                        {
                            snprintf(plistoutline, STRINGMAXLEN_LINE, " ");
                        }

                        fclose(fp);
                    }
                }

                if(streaminfo[sindexscan1].streamOpenPID_status != 2)
                {
                    char *pch;

                    pch = strtok(plistoutline, " ");

                    while(pch != NULL)
                    {
                        if(NBpid < streamOpenNBpid_MAX)
                        {
                            streaminfo[sindexscan1].streamOpenPID[NBpid] =
                                atoi(pch);
                            if(getpgid(streaminfo[sindexscan1]
                                       .streamOpenPID[NBpid]) >= 0)
                            {
                                NBpid++;
                            }
                        }
                        pch = strtok(NULL, " ");
                    }
                    streaminfo[sindexscan1].streamOpenPID_status = 1; // success
                }

                streaminfo[sindexscan1].streamOpenPID_cnt = NBpid;
                // Get PID names
                int pidIndex;
                int cnt1 = 0;
                for(pidIndex = 0;
                        pidIndex < streaminfo[sindexscan1].streamOpenPID_cnt;
                        pidIndex++)
                {
                    pid_t pid = streaminfo[sindexscan1].streamOpenPID[pidIndex];
                    if((getpgid(pid) >= 0) && (pid != getpid()))
                    {
                        char *pname = (char *) calloc(1024, sizeof(char));
                        get_process_name_by_pid(pid, pname);

                        if(PIDname_array[pid] == NULL)
                        {
                            PIDname_array[pid] = (char *) malloc(
                                                     sizeof(char) * (PIDnameStringLen + 1));
                        }
                        strncpy(PIDname_array[pid], pname, PIDnameStringLen);
                        free(pname);
                        cnt1++;
                    }
                }
                streaminfo[sindexscan1].streamOpenPID_cnt1 = cnt1;

                streaminfoproc->sindexscan++;
            }
        }

        streaminfoproc->fuserUpdate0 = 0;

        streaminfoproc->NBstream = NBsindex;
        streaminfoproc->loopcnt++;

        usleep(streaminfoproc->twaitus);

        scancnt++;
    }

    return NULL;
}
