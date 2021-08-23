/**
 * @file    fps_connect.c
 * @brief   connect to FPS
 */



#include <fcntl.h> // for open
#include <sys/mman.h> // mmap
#include <sys/stat.h> // fstat


#include "CommandLineInterface/CLIcore.h"
#include "CommandLineInterface/timeutils.h"
#include "fps_shmdirname.h"
#include "fps_loadstream.h"
#include "fps_GetParamIndex.h"



/** @brief Connect to function parameter structure
 *
 *
 * ## Arguments
 *
 * fpsconnectmode can take following value
 *
 * FPSCONNECT_SIMPLE : simple connect, don't try load streams
 * FPSCONNECT_CONF   : connect as CONF process
 * FPSCONNECT_RUN    : connect as RUN process
 *
 */
long function_parameter_struct_connect(
    const char *name,
    FUNCTION_PARAMETER_STRUCT *fps,
    int fpsconnectmode
)
{
    int   stringmaxlen = 500;
    char  SM_fname[stringmaxlen];
    int   SM_fd; // shared memory file descriptor
    long  NBparamMAX;
    //    long NBparamActive;
    char *mapv;

    char shmdname[stringmaxlen];

    DEBUG_TRACEPOINT("Connect to fps %s\n", name);

    if(fps->SMfd > -1)
    {
        printf("[%s %s %d] ERROR: file descriptor already allocated : %d\n", __FILE__,
               __func__, __LINE__, fps->SMfd);
        //	exit(0);
    }


    function_parameter_struct_shmdirname(shmdname);

    if(snprintf(SM_fname, sizeof(SM_fname), "%s/%s.fps.shm", shmdname, name) < 0)
    {
        PRINT_ERROR("snprintf error");
    }
    DEBUG_TRACEPOINT("File : %s\n", SM_fname);
    SM_fd = open(SM_fname, O_RDWR);
    if(SM_fd == -1)
    {
        printf("ERROR [%s %s %d]: cannot connect to %s\n", __FILE__, __func__, __LINE__,
               SM_fname);
        return(-1);
    }
    else
    {
        fps->SMfd = SM_fd;
    }


    struct stat file_stat;
    fstat(SM_fd, &file_stat);


    fps->md = (FUNCTION_PARAMETER_STRUCT_MD *) mmap(0, file_stat.st_size,
              PROT_READ | PROT_WRITE, MAP_SHARED, SM_fd, 0);
    if(fps->md == MAP_FAILED)
    {
        close(SM_fd);
        perror("Error mmapping the file");
        printf("STEP %s %d\n", __FILE__, __LINE__);
        fflush(stdout);
        exit(EXIT_FAILURE);
    }

    if(fpsconnectmode == FPSCONNECT_CONF)
    {
        fps->md->confpid = getpid();    // write process PID into FPS
        clock_gettime(CLOCK_REALTIME, &fps->md->confpidstarttime);
    }

    if(fpsconnectmode == FPSCONNECT_RUN)
    {
        fps->md->runpid = getpid();    // write process PID into FPS
        clock_gettime(CLOCK_REALTIME, &fps->md->runpidstarttime);
    }



    mapv = (char *) fps->md;
    mapv += sizeof(FUNCTION_PARAMETER_STRUCT_MD);
    fps->parray = (FUNCTION_PARAMETER *) mapv;

    //	NBparam = (int) (file_stat.st_size / sizeof(FUNCTION_PARAMETER));
    NBparamMAX = fps->md->NBparamMAX;
    printf("    Connected to %s, %ld entries\n", SM_fname,
           NBparamMAX);
    fflush(stdout);


    // decompose full name into pname and indices
    int NBi = 0;
    char tmpstring[stringmaxlen];
    char tmpstring1[stringmaxlen];
    char *pch;


    strncpy(tmpstring, name, stringmaxlen-1);
    NBi = -1;
    pch = strtok(tmpstring, "-");
    while(pch != NULL)
    {
        strncpy(tmpstring1, pch, stringmaxlen-1);

        if(NBi == -1)
        {
            //            strncpy(fps->md->pname, tmpstring1, stringmaxlen);
            if(snprintf(fps->md->pname, FPS_PNAME_STRMAXLEN, "%s", tmpstring1) < 0)
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


    fps->md->NBnameindex = NBi;


    // count active parameters
    int pactivecnt = 0;
    for(int pindex = 0; pindex < NBparamMAX; pindex++)
    {
        if(fps->parray[pindex].fpflag & FPFLAG_ACTIVE)
        {
            pactivecnt ++;
        }
    }
    fps->NBparamActive = pactivecnt;



    //function_parameter_printlist(fps->parray, NBparamMAX);


    if((fpsconnectmode == FPSCONNECT_CONF) || (fpsconnectmode == FPSCONNECT_RUN))
    {
        // load streams
        int pindex;
        for(pindex = 0; pindex < NBparamMAX; pindex++)
        {
            if((fps->parray[pindex].fpflag & FPFLAG_ACTIVE)
                    && (fps->parray[pindex].fpflag & FPFLAG_USED)
                    && (fps->parray[pindex].type & FPTYPE_STREAMNAME))
            {
                functionparameter_LoadStream(fps, pindex, fpsconnectmode);
            }
        }
    }



    // if available, get process settings from FPS entries
    if(fpsconnectmode == FPSCONNECT_RUN)
    {
        // update time
        //
        // set timestring if applicable
        //
        {
            int pindex = functionparameter_GetParamIndex(fps, ".conf.timestring");
            if(pindex > -1)
            {
                char timestring[100];
                mkUTtimestring_microsec(timestring, fps->md->runpidstarttime);
                if(snprintf(fps->parray[pindex].val.string[0],
                            FUNCTION_PARAMETER_STRMAXLEN, "%s", timestring) < 0)
                {
                    PRINT_ERROR("snprintf error");
                }
            }
        }

        {   // check if processinfo is enabled
            int pindex = functionparameter_GetParamIndex(fps, ".procinfo.enabled");
            if(pindex > -1) {
                if(fps->parray[pindex].type == FPTYPE_ONOFF)
                {
                    if(fps->parray[pindex].fpflag & FPFLAG_ONOFF)
                    {
                        //printf("<<<<<<<<<<<<<<< FPS PROCESSINFO ENABLED\n");
                        fps->cmdset.flags |= CLICMDFLAG_PROCINFO;
                    }
                    else
                    {
                        //printf("<<<<<<<<<<<<<<< FPS PROCESSINFO DISABLED\n");
                        fps->cmdset.flags &= ~(CLICMDFLAG_PROCINFO);
                    }
                }
            }
        }


        {   // procinfo_loopcntMax
            int pindex = functionparameter_GetParamIndex(fps, ".procinfo.loopcntMax");
            if(pindex > -1) {
                if(fps->parray[pindex].type == FPTYPE_INT64)
                {
                    fps->cmdset.procinfo_loopcntMax = fps->parray[pindex].val.i64[0];
                }
            }
        }


        {   // RT_priority
            int pindex = functionparameter_GetParamIndex(fps, ".procinfo.RTprio");
            if(pindex > -1) {
                if(fps->parray[pindex].type == FPTYPE_INT64)
                {
                    fps->cmdset.RT_priority = fps->parray[pindex].val.i64[0];
                }
            }
        }

    }

    return(NBparamMAX);
}


