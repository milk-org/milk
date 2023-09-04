
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
    static long scaniter = 0;
    static int  firstIter = 1;

    // get input pointers
    streamCTRLarg_struct *streamCTRLdata =
        (streamCTRLarg_struct *) argptr;

    STREAMINFOPROC *streaminfoproc = streamCTRLdata->streaminfoproc;
    IMAGE          *images         = streamCTRLdata->images;
    STREAMINFO *streaminfo         = streamCTRLdata->sinfo;
    char **PIDname_array           = streaminfoproc->PIDtable;

    streaminfoproc->loopcnt = 0;

    // if set, write file list to file on first scan
    //int WriteFlistToFile = 1;




    while(streaminfoproc->loop == 1)
    {
        EXECUTE_SYSTEM_COMMAND("echo \" \" >> IDlog.txt");
        EXECUTE_SYSTEM_COMMAND("echo \"[%ld] loopSTART\" >> IDlog.txt", scaniter);

        long NBsindex = 0;

        double  tdiffv;
        {
            // timing measurement

            static struct timespec t0;
            struct timespec        t1;
            struct timespec        tdiff;

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
        }


        // look for streams on filesystem
        // NBsindex is total nymber of streams found
        //
        NBsindex = find_streams(streaminfo,
                                streaminfoproc->filter,
                                streaminfoproc->namefilter);

        EXECUTE_SYSTEM_COMMAND("echo \"NBsindex = %ld\" >> IDlog.txt", NBsindex);

        // write stream list to file if applicable
        // ususally used for debugging only
        //
        if(streaminfoproc->WriteFlistToFile == 1)
        {
            FILE *fpfscan;
            fpfscan = fopen("streamCTRL_filescan.dat", "w");
            fprintf(fpfscan, "# stream scan result\n");
            fprintf(fpfscan,
                    "filter: %d %s\n",
                    streaminfoproc->filter,
                    streaminfoproc->namefilter);
            fprintf(fpfscan, "NBsindex = %ld\n", NBsindex);

            for(long sindex = 0; sindex < NBsindex; sindex++)
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
        for(long sindex = 0; sindex < NBsindex; sindex++)
        {
            imageID ID;

            //streaminfo[sindex].ISIOretval = IMAGESTREAMIO_FILEOPEN;

            // Check if already in memory
            //
            ID = image_ID_from_images(images, streaminfo[sindex].sname);
            EXECUTE_SYSTEM_COMMAND("echo \"  %ld %s : ID = %ld\" >> IDlog.txt",
                                   sindex,
                                   streaminfo[sindex].sname,
                                   ID);

            // if not in local memory, try to connect to stream
            //
            if(ID == -1)
            {
                // if not in memory, try to load
                //
                ID = image_get_first_ID_available_from_images(images);
                if(ID < 0)
                {
                    return NULL;
                }
                EXECUTE_SYSTEM_COMMAND("echo \"  %ld get ID = %ld\" >> IDlog.txt",
                                       sindex, ID);


                streaminfo[sindex].ISIOretval =
                    ImageStreamIO_read_sharedmem_image_toIMAGE(
                        streaminfo[sindex].sname,
                        &images[ID]);

                // images[ID] used to keep track of each stream, even if not successfully loaded
                // force used to be 1 even if load fails, so we can keep track of attempted loads
                images[ID].used = 1;
                // keep track of name
                strncpy(images[ID].name, streaminfo[sindex].sname, STRINGMAXLEN_IMAGE_NAME - 1);
                images[ID].name[STRINGMAXLEN_IMAGE_NAME-1] = '\0';

                streaminfo[sindex].deltacnt0          = 1;
                streaminfo[sindex].updatevalue        = 1.0;
                streaminfo[sindex].updatevalue_frozen = 1.0;
            }
            else
            {
                // if in memory, check if image data has been loaded
                //
                if ( images[ID].array.raw == NULL )
                {
                    streaminfo[sindex].ISIOretval = IMAGESTREAMIO_FILEOPEN;
                }
                else
                {
                    streaminfo[sindex].ISIOretval = IMAGESTREAMIO_SUCCESS;
                }

                if(streaminfo[sindex].ISIOretval == IMAGESTREAMIO_SUCCESS )
                {
                    EXECUTE_SYSTEM_COMMAND("echo \"  %ld  ISIO OK\" >> IDlog.txt",
                                           sindex);

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

                    // keep memory of cnt0
                    streaminfo[sindex].cnt0 = images[ID].md->cnt0;
                    streaminfo[sindex].datatype = images[ID].md->datatype;
                }
                else
                {
                    EXECUTE_SYSTEM_COMMAND("echo \"  %ld  ISIO NOTOK\" >> IDlog.txt",
                                           sindex);
                }
            }

            streaminfo[sindex].ID = ID;
        }
        DEBUG_TRACEPOINT(" ");

        // remove stale IDs




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

        EXECUTE_SYSTEM_COMMAND("echo \"[%ld] loopEND\" >> IDlog.txt", scaniter);
        scaniter++;


        usleep(streaminfoproc->twaitus);
    }

    return NULL;
}
