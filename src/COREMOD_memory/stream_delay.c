/** @file stream_delay,c
 */

#include <math.h>

#include "CommandLineInterface/CLIcore.h"
#include "create_image.h"
#include "delete_image.h"
#include "image_ID.h"
#include "stream_sem.h"

#include "COREMOD_tools/COREMOD_tools.h"
#include "CommandLineInterface/timeutils.h"

// Local variables pointers
static char     *inimname;
static char     *outimname;
static float    *delaysec;
static uint64_t *timebuffsize;

static int32_t *avemode;
static long     fpi_avemode;

static uint64_t *avedtns;
static long      fpi_timeavedtns;

static uint64_t *statusframelag;
static uint64_t *statuskkin;
static uint64_t *statuskkout;

static CLICMDARGDEF farg[] = {{
        CLIARG_IMG,
        ".in_name",
        "input image",
        "im1",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &inimname,
        NULL
    },
    {
        CLIARG_STR,
        ".out_name",
        "output image",
        "out1",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &outimname,
        NULL
    },
    {
        CLIARG_FLOAT32,
        ".delaysec",
        "delay [s]",
        "0.001",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &delaysec,
        NULL
    },
    {
        CLIARG_UINT64,
        ".timebuffsize",
        "time buffer size",
        "10000",
        CLIARG_HIDDEN_DEFAULT,
        (void **) &timebuffsize,
        NULL
    },
    {
        CLIARG_INT32,
        ".option.timeavemode",
        "Enable time window averaging (>0)",
        "0",
        CLIARG_HIDDEN_DEFAULT,
        (void **) &avemode,
        &fpi_avemode
    },
    {
        CLIARG_UINT64,
        ".option.timeavedtns",
        "Averaging time window width [ns]",
        "10000",
        CLIARG_HIDDEN_DEFAULT,
        (void **) &avedtns,
        &fpi_timeavedtns
    },
    {
        CLIARG_UINT64,
        ".status.framelag",
        "current time lag frame index",
        "100",
        CLIARG_OUTPUT_DEFAULT,
        (void **) &statusframelag,
        NULL
    },
    {
        CLIARG_UINT64,
        ".status.kkin",
        "input cube slice index",
        "100",
        CLIARG_OUTPUT_DEFAULT,
        (void **) &statuskkin,
        NULL
    },
    {
        CLIARG_UINT64,
        ".status.kkout",
        "output cube slice index",
        "100",
        CLIARG_OUTPUT_DEFAULT,
        (void **) &statuskkout,
        NULL
    }
};

static errno_t customCONFsetup()
{
    return RETURN_SUCCESS;
}

static errno_t customCONFcheck()
{
    if(data.fpsptr != NULL)
    {
        if(data.fpsptr->parray[fpi_avemode].val.i32[0] == 0)  // no ave mode
        {
            data.fpsptr->parray[fpi_timeavedtns].fpflag &= ~FPFLAG_USED;
            data.fpsptr->parray[fpi_timeavedtns].fpflag &= ~FPFLAG_VISIBLE;
        }
        else
        {
            data.fpsptr->parray[fpi_timeavedtns].fpflag |= FPFLAG_USED;
            data.fpsptr->parray[fpi_timeavedtns].fpflag |= FPFLAG_VISIBLE;
        }
    }

    return RETURN_SUCCESS;
}

static CLICMDDATA CLIcmddata = {"streamdelay",
                                "delay input stream to output stream",
                                CLICMD_FIELDS_DEFAULTS
                               };

// detailed help
static errno_t help_function()
{
    return RETURN_SUCCESS;
}




static errno_t streamdelay(IMGID            inimg,
                           IMGID            outimg,
                           IMGID            bufferimg,
                           struct timespec *tarray,
                           int             *warray,
                           int             *status)
{
    static uint64_t cnt0prev           = 0;
    static uint64_t bufferindex_input  = 0;
    static uint64_t bufferindex_output = 0;

    // get current time
    struct timespec tnow;
    clock_gettime(CLOCK_REALTIME, &tnow);

    // update circular buffer if new frame has arrived
    if(cnt0prev != inimg.md->cnt0)
    {
        //printf("cnt %8ld %8ld   CIRC BUFFER UPDATE -> index %8ld / %8ld\n",
        //       cnt0prev, inimg.md->cnt0, bufferindex_input, *timebuffsize);
        // new input frame

        // update counter for next detection loop
        cnt0prev = inimg.md->cnt0;

        // write current time to array
        tarray[bufferindex_input].tv_sec  = tnow.tv_sec;
        tarray[bufferindex_input].tv_nsec = tnow.tv_nsec;

        // copy image data to circular buffer
        char *destptr;
        destptr = (char *) bufferimg.im->array.raw;
        destptr += inimg.md->imdatamemsize * bufferindex_input;
        memcpy(destptr, inimg.im->array.raw, inimg.md->imdatamemsize);

        warray[bufferindex_input] = 0;

        bufferindex_input++;
        if(bufferindex_input == (*timebuffsize))
        {
            // end of circular buffer reached
            bufferindex_input = 0;
        }
    }

    // check if current time is past time array at output index + delay
    struct timespec tdiff  = timespec_diff(tarray[bufferindex_output], tnow);
    double          tdiffv = 1.0 * tdiff.tv_sec + 1.0e-9 * tdiff.tv_nsec;

    //printf("%8ld  %8ld [%d]  tdiffv = %lf sec    %lf\n",
    //       bufferindex_input, bufferindex_output, warray[bufferindex_output], tdiffv, (*delaysec));
    //fflush(stdout);

    int  updateflag              = 0;
    long bufferindex_output_last = 0;
    while((warray[bufferindex_output] == 0) && (tdiffv > (*delaysec)))
    {
        // update output frame
        updateflag                 = 1;
        warray[bufferindex_output] = 1;

        bufferindex_output_last = bufferindex_output;
        bufferindex_output++;
        if(bufferindex_output == (*timebuffsize))
        {
            // end of circular buffer reached
            bufferindex_output = 0;
        }

        //printf("    advance %8ld %8ld\n", bufferindex_input, bufferindex_output);
        tdiff  = timespec_diff(tarray[bufferindex_output], tnow);
        tdiffv = 1.0 * tdiff.tv_sec + 1.0e-9 * tdiff.tv_nsec;
    }

    if(updateflag == 1)
    {
        printf("     WRITE %8ld %8ld  :  %ld bytes\n",
               bufferindex_input,
               bufferindex_output_last,
               (long) inimg.md->imdatamemsize);
        // copy circular buffer frame to output
        char *srcptr;
        srcptr = (char *) bufferimg.im->array.raw;
        srcptr += inimg.md->imdatamemsize * bufferindex_output_last;
        memcpy(outimg.im->array.raw, srcptr, inimg.md->imdatamemsize);

        // frame has been processed
        *status = 1;
    }
    else
    {
        *status = 0;
    }

    return RETURN_SUCCESS;
}

static errno_t compute_function()
{
    DEBUG_TRACE_FSTART();

    IMGID inimg = mkIMGID_from_name(inimname);
    resolveIMGID(&inimg, ERRMODE_ABORT);

    IMGID outimg = mkIMGID_from_name(outimname);
    imcreatelikewiseIMGID(&outimg, &inimg);

    IMGID bufferimg    = makeIMGID_3D("streamdelaybuff",
                                      inimg.size[0],
                                      inimg.size[1],
                                      *timebuffsize);
    bufferimg.datatype = inimg.datatype;
    imcreateIMGID(&bufferimg);

    struct timespec *timeinarray;
    timeinarray =
        (struct timespec *) malloc(sizeof(struct timespec) * (*timebuffsize));
    // get current time
    struct timespec tnow;
    clock_gettime(CLOCK_REALTIME, &tnow);
    for(uint64_t i = 0; i < *timebuffsize; i++)
    {
        timeinarray[i].tv_sec  = tnow.tv_sec;
        timeinarray[i].tv_nsec = tnow.tv_nsec;
    }

    // write array
    // 0 if new, 1 if already sent to output
    int *warray;
    warray = (int *) malloc(sizeof(int) * (*timebuffsize));
    for(uint64_t i = 0; i < *timebuffsize; i++)
    {
        warray[i] = 1;
    }

    list_image_ID();
    int status = 0;

    INSERT_STD_PROCINFO_COMPUTEFUNC_INIT
    INSERT_STD_PROCINFO_COMPUTEFUNC_LOOPSTART

    streamdelay(inimg, outimg, bufferimg, timeinarray, warray, &status);
    // status is 0 if no update to output, 1 otherwise
    if(status != 0)
    {
        processinfo_update_output_stream(processinfo, outimg.ID);
    }

    INSERT_STD_PROCINFO_COMPUTEFUNC_END

    free(timeinarray);
    free(warray);

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}

INSERT_STD_FPSCLIfunctions

// Register function in CLI
errno_t
CLIADDCMD_COREMOD_memory__streamdelay()
{
    CLIcmddata.FPS_customCONFsetup = customCONFsetup;
    CLIcmddata.FPS_customCONFcheck = customCONFcheck;
    INSERT_STD_CLIREGISTERFUNC

    return RETURN_SUCCESS;
}

/*
    imageID             IDimc;
    imageID             IDin, IDout;
    uint32_t            xsize, ysize, xysize;
    //    long                cnt0old;
    long                ii;
    struct timespec    *t0array;
    struct timespec     tnow;
    double              tdiffv;
    struct timespec     tdiff;
    uint32_t           *arraytmp;
    long                cntskip = 0;
    long                kk;




    xsize = data.image[IDin].md[0].size[0];
    ysize = data.image[IDin].md[0].size[1];
    *zsize = (long)(2 * delayus / dtus);
    if(*zsize < 1)
    {
        *zsize = 1;
    }
    xysize = xsize * ysize;

    t0array = (struct timespec *) malloc(sizeof(struct timespec) * *zsize);

    create_3Dimage_ID("_tmpc", xsize, ysize, *zsize, &IDimc);



    IDout = image_ID(IDout_name);
    if(IDout == -1)   // CREATE IT
    {
        arraytmp = (uint32_t *) malloc(sizeof(uint32_t) * 2);
        arraytmp[0] = xsize;
        arraytmp[1] = ysize;
        create_image_ID(IDout_name, 2, arraytmp, _DATATYPE_FLOAT, 1, 0, 0, &IDout);
        COREMOD_MEMORY_image_set_createsem(IDout_name, IMAGE_NB_SEMAPHORE);
        free(arraytmp);
    }


    *kkin = 0;
    *kkout = 0;
//    cnt0old = data.image[IDin].md[0].cnt0;

    float *arraytmpf;
    arraytmpf = (float *) malloc(sizeof(float) * xsize * ysize);

    clock_gettime(CLOCK_REALTIME, &tnow);
    for(kk = 0; kk < *zsize; kk++)
    {
        t0array[kk] = tnow;
    }


    DEBUG_TRACEPOINT(" ");


    // Specify input stream trigger

    processinfo_waitoninputstream_init(processinfo, IDin,
                                       PROCESSINFO_TRIGGERMODE_DELAY, -1);
    processinfo->triggerdelay.tv_sec = 0;
    processinfo->triggerdelay.tv_nsec = (long)(dtus * 1000);
    while(processinfo->triggerdelay.tv_nsec > 1000000000)
    {
        processinfo->triggerdelay.tv_nsec -= 1000000000;
        processinfo->triggerdelay.tv_sec += 1;
    }


    // ===========================
    /// ### START LOOP
    // ===========================

    processinfo_loopstart(
        processinfo); // Notify processinfo that we are entering loop

    DEBUG_TRACEPOINT(" ");

    while(loopOK == 1)
    {
        int kkinscan;
        float normframes = 0.0;

        DEBUG_TRACEPOINT(" ");
        loopOK = processinfo_loopstep(processinfo);


        processinfo_waitoninputstream(processinfo);
        //usleep(dtus); // main loop wait

        processinfo_exec_start(processinfo);

        if(processinfo_compute_status(processinfo) == 1)
        {
            DEBUG_TRACEPOINT(" ");

            // has new frame arrived ?
//            cnt0 = data.image[IDin].md[0].cnt0;

//            if(cnt0 != cnt0old) { // new frame
            clock_gettime(CLOCK_REALTIME, &t0array[*kkin]);  // record time of input frame

            DEBUG_TRACEPOINT(" ");
            for(ii = 0; ii < xysize; ii++)
            {
                data.image[IDimc].array.F[(*kkin) * xysize + ii] = data.image[IDin].array.F[ii];
            }
            (*kkin) ++;
            DEBUG_TRACEPOINT(" ");

            if((*kkin) == (*zsize))
            {
                (*kkin) = 0;
            }



            clock_gettime(CLOCK_REALTIME, &tnow);
            DEBUG_TRACEPOINT(" ");


            cntskip = 0;
            tdiff = timespec_diff(t0array[*kkout], tnow);
            tdiffv = 1.0 * tdiff.tv_sec + 1.0e-9 * tdiff.tv_nsec;

            DEBUG_TRACEPOINT(" ");


            while((tdiffv > 1.0e-6 * delayus) && (cntskip < *zsize))
            {
                cntskip++;  // advance index until time condition is satisfied
                (*kkout) ++;
                if(*kkout == *zsize)
                {
                    *kkout = 0;
                }
                tdiff = timespec_diff(t0array[*kkout], tnow);
                tdiffv = 1.0 * tdiff.tv_sec + 1.0e-9 * tdiff.tv_nsec;
            }

            DEBUG_TRACEPOINT(" ");

            *framelag = *kkin - *kkout;
            if(*framelag < 0)
            {
                *framelag += *zsize;
            }


            DEBUG_TRACEPOINT(" ");


            switch(timeavemode)
            {

            case 0: // no time averaging - pick more recent frame that matches requirement
                DEBUG_TRACEPOINT(" ");
                if(cntskip > 0)
                {
                    char *ptr; // pointer address

                    data.image[IDout].md[0].write = 1;

                    ptr = (char *) data.image[IDimc].array.F;
                    ptr += SIZEOF_DATATYPE_FLOAT * xysize * *kkout;

                    memcpy(data.image[IDout].array.F, ptr,
                           SIZEOF_DATATYPE_FLOAT * xysize);  // copy time-delayed input to output

                    COREMOD_MEMORY_image_set_sempost_byID(IDout, -1);
                    data.image[IDout].md[0].cnt0++;
                    data.image[IDout].md[0].write = 0;
                }
                break;

            default : // strict time window (note: other modes will be coded in the future)
                normframes = 0.0;
                DEBUG_TRACEPOINT(" ");

                for(ii = 0; ii < xysize; ii++)
                {
                    arraytmpf[ii] = 0.0;
                }

                for(kkinscan = 0; kkinscan < *zsize; kkinscan++)
                {
                    tdiff = timespec_diff(t0array[kkinscan], tnow);
                    tdiffv = 1.0 * tdiff.tv_sec + 1.0e-9 * tdiff.tv_nsec;

                    if((tdiffv > 0) && (fabs(tdiffv - 1.0e-6 * delayus) < *avedt))
                    {
                        float coeff = 1.0;
                        for(ii = 0; ii < xysize; ii++)
                        {
                            arraytmpf[ii] += coeff * data.image[IDimc].array.F[kkinscan * xysize + ii];
                        }
                        normframes += coeff;
                    }
                }
                if(normframes < 0.0001)
                {
                    normframes = 0.0001;    // avoid division by zero
                }

                data.image[IDout].md[0].write = 1;
                for(ii = 0; ii < xysize; ii++)
                {
                    data.image[IDout].array.F[ii] = arraytmpf[ii] / normframes;
                }

                processinfo_update_output_stream(processinfo, IDout);
           */
