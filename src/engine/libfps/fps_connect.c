/**
 * @file    fps_connect.c
 * @brief   connect to FPS
 */

#include <fcntl.h>    // for open
#include <sys/mman.h> // mmap
#include <sys/stat.h> // fstat

#include "fps.h"
#include "fps_globals.h"



// #include "timeutils.h"


/**
 * @brief Connect to an existing FPS in shared memory.
 *
 * This function performs the mapping of a shared memory file containing
 * an FPS structure into the process's address space. It also handles
 * initial synchronization and optionally loads data streams specified
 * by FPS parameters.
 *
 * Logic flow:
 * 1.  Initialize global FPS timestamp and process type if not already set.
 * 2.  Construct the full path to the SHM file (e.g., /tmp/milk/fps/<name>.fps.shm).
 * 3.  Open the file and use `fstat` to determine its size.
 * 4.  `mmap` the entire file into memory.
 * 5.  Set the metadata (`fps->md`) and parameter array (`fps->parray`) pointers.
 * 6.  If connecting as CONF or RUN process, record the process PID and start time.
 * 7.  Parse the FPS name (which may contain indices separated by '-') into the metadata.
 * 8.  Count the number of active parameters.
 * 9.  If in CONF/RUN mode, iterate through all parameters and call
 *     `functionparameter_LoadStream` for any parameter of type `FPTYPE_STREAMNAME`.
 * 10. If in RUN mode, attempt to populate the `fps->cmdset` structure by looking
 *     up standard processinfo-related parameters (e.g., ".procinfo.enabled",
 *     ".procinfo.RTprio", etc.).
 */
long fps_connect(
    const char *name,
    FPS        *fps,
    int        fpsconnectmode)
{
    DEBUG_TRACEPOINT("Launching fps_connect for %s", name);

    int  stringmaxlen = 500;
    char SM_fname[stringmaxlen];
    int  SM_fd; // shared memory file descriptor
    long NBparamMAX;
    //    long NBparamActive;
    char *mapv;

    char shmdname[stringmaxlen];


    if(FPS_TIMESTAMP == 0)
    {
        {
            struct timespec tnow = {0};
            clock_gettime(CLOCK_REALTIME, &tnow);
            FPS_TIMESTAMP = tnow.tv_sec;
        }

        switch(fpsconnectmode)
        {

        case FPSCONNECT_CONF:
            snprintf(FPS_PROCESS_TYPE,
                     STRINGMAXLEN_FPSPROCESSTYPE,
                     "conf-%s",
                     name);
            break;

        case FPSCONNECT_RUN:
            snprintf(FPS_PROCESS_TYPE,
                     STRINGMAXLEN_FPSPROCESSTYPE,
                     "run-%s",
                     name);
            break;

        case FPSCONNECT_SIMPLE:
            snprintf(FPS_PROCESS_TYPE,
                     STRINGMAXLEN_FPSPROCESSTYPE,
                     "init-%s",
                     name);
            break;

        }


        functionparameter_outlog("CONNECTION", ">>>>");
        // functionparameter_outlog_namelink(); // This function depends on CLIcore
    }


    DEBUG_TRACEPOINT("Connect to fps %s\n", name);

    if(fps->SMfd > 2)
    {
        close(fps->SMfd);
        fps->SMfd = 0;
    }

    function_parameter_struct_shmdirname(shmdname);

    if(snprintf(SM_fname, stringmaxlen, "%s/%s.fps.shm", shmdname, name) < 0)
    {
        PRINT_ERROR("snprintf error");
    }
    DEBUG_TRACEPOINT("File: %s\n", SM_fname);
    SM_fd = open(SM_fname, O_RDWR);
    if(SM_fd == -1)
    {
        return (-1);
    }
    else
    {
        fps->SMfd = SM_fd;
    }
    DEBUG_TRACEPOINT("File: %s - attempting mapping\n", SM_fname);

    struct stat file_stat;
    fstat(SM_fd, &file_stat);

    fps->md = (FUNCTION_PARAMETER_STRUCT_MD *) mmap(0,
              file_stat.st_size,
              PROT_READ | PROT_WRITE,
              MAP_SHARED,
              SM_fd,
              0);
    if(fps->md == MAP_FAILED)
    {
        PRINT_ERROR("mmap failed: %s", strerror(errno));
        close(SM_fd);
        return -1;
    }

    DEBUG_TRACEPOINT("File: %s - attempting connect\n", SM_fname);

    if(fpsconnectmode == FPSCONNECT_CONF)
    {
        fps->md->confpid = getpid(); // write process PID into FPS
        clock_gettime(CLOCK_REALTIME, &fps->md->confpidstarttime);
    }

    if(fpsconnectmode == FPSCONNECT_RUN)
    {
        fps->md->runpid = getpid(); // write process PID into FPS
        clock_gettime(CLOCK_REALTIME, &fps->md->runpidstarttime);
    }

    mapv = (char *) fps->md;
    mapv += sizeof(FUNCTION_PARAMETER_STRUCT_MD);
    fps->parray = (FPS_PARAM *) mapv;

    //	NBparam = (int) (file_stat.st_size / sizeof(FPS_PARAM));
    NBparamMAX = fps->md->NBparamMAX;
    //printf("    Connected to %s, %ld entries\n", SM_fname, NBparamMAX);
    //fflush(stdout);

    DEBUG_TRACEPOINT("File: %s - successful connect.\n", SM_fname);

    // decompose full name into pname and indices
    int   NBi = 0;
    char  tmpstring[stringmaxlen];
    char  tmpstring1[stringmaxlen];
    char *pch;

    strncpy(tmpstring, name, stringmaxlen - 1);
    NBi = -1;
    pch = strtok(tmpstring, "-");
    while(pch != NULL)
    {
        strncpy(tmpstring1, pch, stringmaxlen - 1);

        if(NBi == -1)
        {
            //            strncpy(fps->md->pname, tmpstring1, stringmaxlen);
            if(snprintf(fps->md->pname,
                        FPS_PNAME_STRMAXLEN,
                        "%s",
                        tmpstring1) < 0)
            {
                PRINT_ERROR("snprintf error");
            }
        }

        if((NBi >= 0) && (NBi < 10))
        {
            if(snprintf(fps->md->nameindexW[NBi], 16, "%s", tmpstring1) < 0)
            {
                PRINT_ERROR("snprintf error");
            }
            //strncpy(fps->md->nameindexW[NBi], tmpstring1, 16);
        }

        NBi++;
        pch = strtok(NULL, "-");
    }

    DEBUG_TRACEPOINT("File: %s - Successful fps parse.\n", SM_fname);

    fps->md->NBnameindex = NBi;

    // count active parameters
    int pactivecnt = 0;
    for(int pindex = 0; pindex < NBparamMAX; pindex++)
    {
        if(fps->parray[pindex].fpflag & FPFLAG_ACTIVE)
        {
            pactivecnt++;
        }
    }
    fps->NBparamActive = pactivecnt;

    DEBUG_TRACEPOINT("File: %s - Successful parameter count.\n", SM_fname);
    //function_parameter_printlist(fps->parray, NBparamMAX);

    if((fpsconnectmode == FPSCONNECT_CONF) ||
            (fpsconnectmode == FPSCONNECT_RUN))
    {
        // load streams
        int pindex;
        for(pindex = 0; pindex < NBparamMAX; pindex++)
        {
            if((fps->parray[pindex].fpflag & FPFLAG_ACTIVE) &&
                    (fps->parray[pindex].fpflag & FPFLAG_USED) &&
                    (fps->parray[pindex].type & FPTYPE_STREAMNAME))
            {
                functionparameter_LoadStream(fps, pindex, fpsconnectmode);
            }
        }
    }
    DEBUG_TRACEPOINT("File: %s - Successful LoadStream.\n", SM_fname);

    // if available, get process settings from FPS entries
    if(fpsconnectmode == FPSCONNECT_RUN)
    {
        // update time
        //
        // set timestring if applicable
        //
        {
            int pindex =
                functionparameter_GetParamIndex(fps, ".conf.timestring");
            if(pindex > -1)
            {
                char timestring[100];
                mkUTtimestring_microsec(timestring, fps->md->runpidstarttime);
                if(snprintf(fps->parray[pindex].val.string[0],
                            FUNCTION_PARAMETER_STRMAXLEN,
                            "%s",
                            timestring) < 0)
                {
                    PRINT_ERROR("snprintf error");
                }
            }
        }

        {
            // check if processinfo is enabled
            int pindex =
                functionparameter_GetParamIndex(fps, ".procinfo.enabled");
            if(pindex > -1)
            {
                if(fps->parray[pindex].type == FPTYPE_ONOFF)
                {
                    if(fps->parray[pindex].fpflag & FPFLAG_ONOFF)
                    {
                        fps->cmdset.flags |= CLICMDFLAG_PROCINFO;
                    }
                    else
                    {
                        fps->cmdset.flags &= ~(CLICMDFLAG_PROCINFO);
                    }
                }
            }
        }

        {
            // procinfo_loopcntMax
            fps->cmdset.procinfo_loopcntMax_ptr = NULL;
            int pindex =
                functionparameter_GetParamIndex(fps, ".procinfo.loopcntMax");
            if(pindex > -1)
            {
                if(fps->parray[pindex].type == FPTYPE_INT64)
                {
                    fps->cmdset.procinfo_loopcntMax =
                        fps->parray[pindex].val.i64[0];

                    fps->cmdset.procinfo_loopcntMax_ptr = fps->parray[pindex].val.i64;
                }
            }
        }

        {
            // RT_priority
            int pindex =
                functionparameter_GetParamIndex(fps, ".procinfo.RTprio");
            if(pindex > -1)
            {
                if(fps->parray[pindex].type == FPTYPE_INT64)
                {
                    fps->cmdset.RT_priority = fps->parray[pindex].val.i64[0];
                }
            }
        }

        {
            // triggerstreamname
            int pindex =
                functionparameter_GetParamIndex(fps, ".procinfo.triggersname");
            if(pindex > -1)
            {
                if(fps->parray[pindex].type == FPTYPE_STREAMNAME)
                {
                    strncpy(
                        fps->cmdset.triggerstreamname,
                        fps->parray[pindex].val.string[0],
                        sizeof(fps->cmdset
                               .triggerstreamname)
                        - 1);
                    fps->cmdset.triggerstreamname[
                    sizeof(fps->cmdset
                               .triggerstreamname)
                    - 1] = '\0';
                }
            }
        }

        {
            // triggermode
            fps->cmdset.triggermodeptr = NULL;
            int pindex =
                functionparameter_GetParamIndex(fps, ".procinfo.triggermode");
            if(pindex > -1)
            {
                if(fps->parray[pindex].type == FPTYPE_INT64)
                {
                    fps->cmdset.triggermode = fps->parray[pindex].val.i64[0];

                    fps->cmdset.triggermodeptr = fps->parray[pindex].val.i64;
                }
            }
        }

        {
            // semindexrequested
            int pindex =
                functionparameter_GetParamIndex(fps,
                                                ".procinfo.semindexrequested");
            if(pindex > -1)
            {
                if(fps->parray[pindex].type == FPTYPE_INT64)
                {
                    fps->cmdset.semindexrequested =
                        fps->parray[pindex].val.i64[0];
                }
            }
        }

        {
            // triggerdelay
            fps->cmdset.triggerdelayptr = NULL;
            int pindex =
                functionparameter_GetParamIndex(fps, ".procinfo.triggerdelay");
            if(pindex > -1)
            {
                if(fps->parray[pindex].type == FPTYPE_TIMESPEC)
                {
                    fps->cmdset.triggerdelay.tv_sec =
                        fps->parray[pindex].val.ts[0].tv_sec;
                    fps->cmdset.triggerdelay.tv_nsec =
                        fps->parray[pindex].val.ts[0].tv_nsec;

                    fps->cmdset.triggerdelayptr = fps->parray[pindex].val.ts;
                }
            }
        }

        {
            // triggertimeout
            fps->cmdset.triggertimeoutptr = NULL;
            int pindex =
                functionparameter_GetParamIndex(fps,
                                                ".procinfo.triggertimeout");
            if(pindex > -1)
            {
                if(fps->parray[pindex].type == FPTYPE_TIMESPEC)
                {
                    fps->cmdset.triggertimeout.tv_sec =
                        fps->parray[pindex].val.ts[0].tv_sec;
                    fps->cmdset.triggertimeout.tv_nsec =
                        fps->parray[pindex].val.ts[0].tv_nsec;

                    fps->cmdset.triggertimeoutptr = fps->parray[pindex].val.ts;
                }
            }
        }
    }
    DEBUG_TRACEPOINT("File: %s - Successful termination of fps_connect.\n",
                     SM_fname);

    return (NBparamMAX);
}
