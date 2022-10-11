/** @file stream_pixmapdecode.c
 */

#include "CommandLineInterface/CLIcore.h"
#include "create_image.h"
#include "delete_image.h"
#include "image_ID.h"
#include "stream_sem.h"

#include "COREMOD_iofits/COREMOD_iofits.h"

// ==========================================
// Forward declaration(s)
// ==========================================

imageID COREMOD_MEMORY_PixMapDecode_U(const char *inputstream_name,
                                      uint32_t    xsizeim,
                                      uint32_t    ysizeim,
                                      const char *NBpix_fname,
                                      const char *IDmap_name,
                                      const char *IDout_name,
                                      const char *IDout_pixslice_fname,
                                      uint32_t    reverse);

// ==========================================
// Command line interface wrapper function(s)
// ==========================================

static errno_t COREMOD_MEMORY_PixMapDecode_U__cli()
{
    if(0 + CLI_checkarg(1, CLIARG_IMG) + CLI_checkarg(2, CLIARG_LONG) +
            CLI_checkarg(3, CLIARG_LONG) + CLI_checkarg(4, CLIARG_STR_NOT_IMG) +
            CLI_checkarg(5, CLIARG_IMG) + CLI_checkarg(6, CLIARG_STR_NOT_IMG) +
            CLI_checkarg(7, CLIARG_STR_NOT_IMG) +
            CLI_checkarg(8, CLIARG_LONG) ==
            0)
    {
        COREMOD_MEMORY_PixMapDecode_U(data.cmdargtoken[1].val.string,
                                      data.cmdargtoken[2].val.numl,
                                      data.cmdargtoken[3].val.numl,
                                      data.cmdargtoken[4].val.string,
                                      data.cmdargtoken[5].val.string,
                                      data.cmdargtoken[6].val.string,
                                      data.cmdargtoken[7].val.string,
                                      data.cmdargtoken[8].val.numl);
        return CLICMD_SUCCESS;
    }
    else
    {
        return CLICMD_INVALID_ARG;
    }
}

// ==========================================
// Register CLI command(s)
// ==========================================

errno_t stream_pixmapdecode_addCLIcmd()
{
    RegisterCLIcommand("impixdecodeU",
                       __FILE__,
                       COREMOD_MEMORY_PixMapDecode_U__cli,
                       "decode image stream",
                       "<in stream> <xsize [long]> <ysize [long]> <nbpix per "
                       "slice [ASCII file]> <decode map> <out "
                       "stream> <out image slice index [FITS]> <reverse mode>",
                       "impixdecodeU streamin 120 120 pixsclienb.txt decmap "
                       "outim outsliceindex.fits 0",
                       "COREMOD_MEMORY_PixMapDecode_U(const char "
                       "*inputstream_name, uint32_t xsizeim, uint32_t "
                       "ysizeim, const char* NBpix_fname, const char* "
                       "IDmap_name, const char *IDout_name, const char "
                       "*IDout_pixslice_fname, uint32_t reverse)");

    return RETURN_SUCCESS;
}

//
// pixel decode for unsigned short
// sem0, cnt0 gets updated at each full frame
// sem1 gets updated for each slice
// cnt1 contains the slice index that was just written
//
imageID COREMOD_MEMORY_PixMapDecode_U(const char *inputstream_name,
                                      uint32_t    xsizeim,
                                      uint32_t    ysizeim,
                                      const char *NBpix_fname,
                                      const char *IDmap_name,
                                      const char *IDout_name,
                                      const char *IDout_pixslice_fname,
                                      uint32_t    reverse)
{
    imageID            IDout = -1;
    imageID            IDin;
    imageID            IDmap;
    long               slice, sliceii;
    long               oldslice = 0;
    long               NBslice;
    long              *nbpixslice;
    uint32_t           xsizein;
    uint32_t           ysizein;
    uint32_t           nbpixout = xsizeim * ysizeim;
    FILE              *fp;
    uint32_t          *sizearray;
    imageID            IDout_pixslice;
    long               ii;
    unsigned long long cnt = 0;
    //    int RT_priority = 80; //any number from 0-99

    //    struct sched_param schedpar;
    struct timespec ts;
    long            scnt;
    int             semval;
    //    long long iter;
    //    int r;
    long tmpl0, tmpl1;
    int  semr;

    double          *dtarray;
    struct timespec *tarray;
    //    long slice1;

    PROCESSINFO *processinfo;

    IDin  = image_ID(inputstream_name);
    IDmap = image_ID(IDmap_name);
    // Size of IDmap is different depending if forward or reverse lookup !
    // Reverse = 0: same size as IDin
    // Reverse = 1: same size as IDout

    xsizein = data.image[IDin].md[0].size[0];
    ysizein = data.image[IDin].md[0].size[1];

    if(data.image[IDin].md[0].naxis > 2)
    {
        NBslice = data.image[IDin].md[0].size[2];
    }
    else
    {
        NBslice = 1;
    }

    char pinfoname[200]; // short name for the processinfo instance
    sprintf(pinfoname, "decode-%s-to-%s", inputstream_name, IDout_name);
    char pinfodescr[200];
    sprintf(pinfodescr,
            "%ldx%ldx%ld->%ldx%ld",
            (long) xsizein,
            (long) ysizein,
            NBslice,
            (long) xsizeim,
            (long) ysizeim);
    char msgstring[200];
    sprintf(msgstring, "%s->%s", inputstream_name, IDout_name);

    processinfo = processinfo_setup(
                      pinfoname, // short name for the processinfo instance, no spaces, no dot, name should be human-readable
                      pinfodescr, // description
                      msgstring,  // message on startup
                      __FUNCTION__,
                      __FILE__,
                      __LINE__);
    // OPTIONAL SETTINGS
    processinfo->MeasureTiming = 1; // Measure timing
    processinfo->RT_priority =
        20; // RT_priority, 0-99. Larger number = higher priority. If <0, ignore

    int loopOK = 1;

    processinfo_WriteMessage(processinfo, "Allocating memory");

    sizearray = (uint32_t *) malloc(sizeof(uint32_t) * 3);
    if(sizearray == NULL)
    {
        PRINT_ERROR("malloc error");
        abort();
    }

    int in_semwaitindex = ImageStreamIO_getsemwaitindex(&data.image[IDin], 0);

    if(reverse == 0 && (xsizein != data.image[IDmap].md[0].size[0] ||
                        ysizein != data.image[IDmap].md[0].size[1]))
    {
        printf(
            "ERROR: xsize, ysize for %s (%d, %d) does not match %s (%d, "
            "%d)\n",
            inputstream_name,
            xsizein,
            ysizein,
            IDmap_name,
            data.image[IDmap].md[0].size[0],
            data.image[IDmap].md[0].size[0]);
        exit(0);
    }
    if(reverse == 1 && (xsizeim != data.image[IDmap].md[0].size[0] ||
                        ysizeim != data.image[IDmap].md[0].size[1]))
    {
        printf(
            "ERROR: xsize, ysize for %s (%d, %d) does not match %s (%d, "
            "%d)\n",
            IDout_name,
            xsizein,
            ysizein,
            IDmap_name,
            data.image[IDmap].md[0].size[0],
            data.image[IDmap].md[0].size[0]);
        exit(0);
    }
    if(NBslice > 1 && reverse == 1)
    {
        printf(
            "ERROR: Cannot use reverse lookup decode with multiple "
            "slices\n");
    }

    sizearray[0] = xsizeim;
    sizearray[1] = ysizeim;
    create_image_ID(IDout_name,
                    2,
                    sizearray,
                    data.image[IDin].md[0].datatype,
                    1,
                    25,
                    0,
                    &IDout);

    // Copy the keywords over from IDin to IDout
    int NBkw = data.image[IDin].md[0].NBkw;
    for(int kw = 0; kw < NBkw; ++kw)
    {
        strcpy(data.image[IDout].kw[kw].name, data.image[IDin].kw[kw].name);
        data.image[IDout].kw[kw].type  = data.image[IDin].kw[kw].type;
        data.image[IDout].kw[kw].value = data.image[IDin].kw[kw].value;
        strcpy(data.image[IDout].kw[kw].comment,
               data.image[IDin].kw[kw].comment);
    }

    COREMOD_MEMORY_image_set_createsem(IDout_name, IMAGE_NB_SEMAPHORE);

    dtarray = (double *) malloc(sizeof(double) * NBslice);
    if(dtarray == NULL)
    {
        PRINT_ERROR("malloc error");
        abort();
    }

    tarray = (struct timespec *) malloc(sizeof(struct timespec) * NBslice);
    if(tarray == NULL)
    {
        PRINT_ERROR("malloc error");
        abort();
    }

    nbpixslice = (long *) malloc(sizeof(long) * NBslice);
    if(nbpixslice == NULL)
    {
        PRINT_ERROR("malloc error");
        abort();
    }

    if((fp = fopen(NBpix_fname, "r")) == NULL)
    {
        printf("ERROR : cannot open file \"%s\"\n", NBpix_fname);
        exit(0);
    }

    for(slice = 0; slice < NBslice; slice++)
    {
        int fscanfcnt =
            fscanf(fp, "%ld %ld %ld\n", &tmpl0, &nbpixslice[slice], &tmpl1);
        if(fscanfcnt == EOF)
        {
            if(ferror(fp))
            {
                perror("fscanf");
            }
            else
            {
                fprintf(stderr,
                        "Error: fscanf reached end of file, no "
                        "matching characters, no matching failure\n");
            }
            return RETURN_FAILURE;
        }
        else if(fscanfcnt != 3)
        {
            fprintf(stderr,
                    "Error: fscanf successfully matched and assigned "
                    "%i input items, 2 expected\n",
                    fscanfcnt);
            return RETURN_FAILURE;
        }
    }
    fclose(fp);

    for(slice = 0; slice < NBslice; slice++)
    {
        printf("Slice %5ld   : %5ld pix\n", slice, nbpixslice[slice]);
    }

    if(reverse == 0)  // Only for legacy mode
    {
        create_image_ID("outpixsl",
                        2,
                        sizearray,
                        _DATATYPE_UINT16,
                        0,
                        0,
                        0,
                        &IDout_pixslice);

        for(slice = 0; slice < NBslice; slice++)
        {
            sliceii = slice * data.image[IDmap].md[0].size[0] *
                      data.image[IDmap].md[0].size[1];
            for(ii = 0; ii < nbpixslice[slice]; ii++)
            {
                // ocam2kpixi files MUST now be in int32 - otherwise we'll overflow in 240x240
                data.image[IDout_pixslice]
                .array.UI16[data.image[IDmap].array.UI32[sliceii + ii]] =
                    (unsigned short)(1 + slice);
            }
        }

        save_fits("outpixsl", IDout_pixslice_fname);
        delete_image_ID("outpixsl", DELETE_IMAGE_ERRMODE_WARNING);
    }

    processinfo_WriteMessage(processinfo, "Starting loop");

    // ==================================
    // STARTING LOOP
    // ==================================
    processinfo_loopstart(
        processinfo); // Notify processinfo that we are entering loop

    // long loopcnt = 0;
    while(loopOK == 1)
    {
        loopOK = processinfo_loopstep(processinfo);

        if(data.image[IDin].md[0].sem == 0)
        {
            while(data.image[IDin].md[0].cnt0 ==
                    cnt) // test if new frame exists
            {
                usleep(5);
            }
            cnt = data.image[IDin].md[0].cnt0;
        }
        else
        {
            if(clock_gettime(CLOCK_REALTIME, &ts) == -1)
            {
                perror("clock_gettime");
                exit(EXIT_FAILURE);
            }
            ts.tv_sec += 1;

            semr = ImageStreamIO_semtimedwait(&data.image[IDin],
                                              in_semwaitindex,
                                              &ts);

            if(processinfo->loopcnt == 0)
            {
                sem_getvalue(data.image[IDin].semptr[in_semwaitindex], &semval);
                for(scnt = 0; scnt < semval; scnt++)
                {
                    sem_trywait(data.image[IDin].semptr[in_semwaitindex]);
                }
            }
        }

        processinfo_exec_start(processinfo);

        if(processinfo_compute_status(processinfo) == 1)
        {
            if(semr == 0)
            {
                slice = data.image[IDin].md[0].cnt1;
                if(slice > oldslice + 1)
                {
                    slice = oldslice + 1;
                }

                if(oldslice == NBslice - 1)
                {
                    slice = 0;
                }

                //   clock_gettime(CLOCK_REALTIME, &tarray[slice]);
                //  dtarray[slice] = 1.0*tarray[slice].tv_sec + 1.0e-9*tarray[slice].tv_nsec;
                data.image[IDout].md[0].write = 1;

                if(reverse == 0)  // legacy forward lookup mode
                {
                    if(slice < NBslice)
                    {
                        sliceii = slice * data.image[IDmap].md[0].size[0] *
                                  data.image[IDmap].md[0].size[1];
                        for(ii = 0; ii < nbpixslice[slice]; ii++)
                        {
                            data.image[IDout].array.UI16
                            [data.image[IDmap].array.UI32[sliceii + ii]] =
                                data.image[IDin].array.UI16[sliceii + ii];
                        }
                    }
                }
                else // reverse == 1, full image assumed (at least given how ocam is scrambled)
                {
                    for(ii = 0; ii < nbpixout; ++ii)
                    {
                        data.image[IDout].array.UI16[ii] =
                            data.image[IDin]
                            .array.UI16[data.image[IDmap].array.UI32[ii]];
                    }
                }

                // Copy the value of the keywords
                for(int kw = 0; kw < NBkw; ++kw)
                {
                    data.image[IDout].kw[kw].value =
                        data.image[IDin].kw[kw].value;
                }

                if(slice == NBslice - 1)
                {
                    COREMOD_MEMORY_image_set_sempost_byID(IDout, -1);
                    data.image[IDout].md[0].cnt0++;
                }

                data.image[IDout].md[0].cnt1 = slice;

                sem_getvalue(data.image[IDout].semptr[2], &semval);
                if(semval < SEMAPHORE_MAXVAL)
                {
                    sem_post(data.image[IDout].semptr[2]);
                }

                sem_getvalue(data.image[IDout].semptr[3], &semval);
                if(semval < SEMAPHORE_MAXVAL)
                {
                    sem_post(data.image[IDout].semptr[3]);
                }

                data.image[IDout].md[0].write = 0;

                oldslice = slice;
            }
        }

        processinfo_exec_end(processinfo);
    }

    // ==================================
    // ENDING LOOP
    // ==================================
    processinfo_cleanExit(processinfo);

    /*    if((data.processinfo == 1) && (processinfo->loopstat != 4)) {
            processinfo_cleanExit(processinfo);
        }*/

    free(nbpixslice);
    free(sizearray);
    free(dtarray);
    free(tarray);

    return IDout;
}
