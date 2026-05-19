/**
 * @file    stream_pixmapdecode.c
 * @brief   decode image stream via pixel map
 *
 * Uses FPS V2 framework.
 */

#include "ImageStreamIO/ImageStruct.h"

#ifdef MILK_NO_CLI
#include "CLIcore_standalone.h"
#else
#include "CLIcore.h"
#endif
#include "fps.h"
#include "COREMOD_memory/COREMOD_memory.h"

#include "create_image.h"
#include "delete_image.h"
#include "image_ID.h"
#include "stream_sem.h"

#ifdef USE_CFITSIO
#include "COREMOD_iofits/COREMOD_iofits.h"
#endif


/* ================================================================
 * 1.  FPS COMPONENT IDENTITY
 * ============================================================= */

static FPS_APP_INFO FPS_app_info = {
    .fps_name    = "impixdecodeU",
    .cmdkey      = "impixdecodeU",
    .description =
        "decode image stream",
    .description_long =
        "Decode a pixel map to reconstruct a 2D image from a 1D encoded stream. The pixel map specifies the mapping from linear to 2D coordinates."
};


/* ================================================================
 * 2.  LOCAL PARAMETER VARIABLES
 * ============================================================= */

static char p_instream[FUNCTION_PARAMETER_STRMAXLEN]
    = "streamin";

static long long p_xsizeim  = 120;
static long long p_ysizeim  = 120;

static char p_nbpix_fname[FUNCTION_PARAMETER_STRMAXLEN]
    = "pixsclienb.txt";

static char p_mapname[FUNCTION_PARAMETER_STRMAXLEN]
    = "decmap";

static char p_outname[FUNCTION_PARAMETER_STRMAXLEN]
    = "outim";

static char p_outslice[FUNCTION_PARAMETER_STRMAXLEN]
    = "outsliceindex.fits";

static long long p_reverse = 0;


/* ================================================================
 * 3.  UNIFIED PARAMETER TABLE (X-Macro)
 * ============================================================= */

#define FPS_PARAMS(X) \
    X(".in_stream", p_instream, \
      FPTYPE_STREAMNAME, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "input stream") \
    X(".xsizeim", &p_xsizeim, \
      FPTYPE_INT64, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "output x size") \
    X(".ysizeim", &p_ysizeim, \
      FPTYPE_INT64, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "output y size") \
    X(".nbpix_fname", p_nbpix_fname, \
      FPTYPE_STRING, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "nb pix per slice file") \
    X(".mapname", p_mapname, \
      FPTYPE_STREAMNAME, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "decode map") \
    X(".out_name", p_outname, \
      FPTYPE_STREAMNAME, 1, \
      FPFLAG_DEFAULT_OUTPUT, \
      "output stream") \
    X(".out_pixslice", p_outslice, \
      FPTYPE_STRING, 1, \
      FPFLAG_DEFAULT_OUTPUT, \
      "output slice index file") \
    X(".reverse", &p_reverse, \
      FPTYPE_INT64, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "reverse mode (0/1)")


/* ================================================================
 * 4.  COMPUTATION LOGIC — forward decl
 * ============================================================= */

imageID COREMOD_MEMORY_PixMapDecode_U(
    const char *inputstream_name,
    uint32_t   xsizeim,
    uint32_t   ysizeim,
    const char *NBpix_fname,
    const char *IDmap_name,
    const char *IDout_name,
    const char *IDout_pixslice_fname,
    uint32_t   reverse);
//
// pixel decode for unsigned short
// sem0, cnt0 gets updated at each full frame
// sem1 gets updated for each slice
// cnt1 contains the slice index that was just written
//
imageID COREMOD_MEMORY_PixMapDecode_U(
    const char *inputstream_name,
    uint32_t   xsizeim,
    uint32_t   ysizeim,
    const char *NBpix_fname,
    const char *IDmap_name,
    const char *IDout_name,
    const char *IDout_pixslice_fname,
    uint32_t    reverse
)
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

    IMGID img_in = imgid_make_from_name(inputstream_name);
    resolveIMGID(&img_in, ERRMODE_WARN, dcimg, dcnimg);
    IDin = img_in.ID;
    if (img_in.ID == -1) {
        return RETURN_FAILURE;
    }

    IMGID img_map = imgid_make_from_name(IDmap_name);
    resolveIMGID(&img_map, ERRMODE_WARN, dcimg, dcnimg);
    IDmap = img_map.ID;
    if (img_map.ID == -1) {
        return RETURN_FAILURE;
    }
    // Size of IDmap is different depending if forward or reverse lookup !
    // Reverse = 0: same size as IDin
    // Reverse = 1: same size as IDout

    xsizein = dcimg[IDin].md[0].size[0];
    ysizein = dcimg[IDin].md[0].size[1];

    if(dcimg[IDin].md[0].naxis > 2)
    {
        NBslice = dcimg[IDin].md[0].size[2];
    }
    else
    {
        NBslice = 1;
    }

    char pinfoname[200]; // short name for the processinfo instance
    snprintf(pinfoname, sizeof(pinfoname),
             "decode-%s-to-%s",
             inputstream_name, IDout_name);
    char pinfodescr[200];
    snprintf(pinfodescr, sizeof(pinfodescr),
             "%ldx%ldx%ld->%ldx%ld",
             (long) xsizein,
             (long) ysizein,
             NBslice,
             (long) xsizeim,
             (long) ysizeim);
    char msgstring[200];
    snprintf(msgstring, sizeof(msgstring),
             "%s->%s",
             inputstream_name, IDout_name);

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

    int in_semwaitindex = ImageStreamIO_getsemwaitindex(&dcimg[IDin], 0);

    if(reverse == 0 && (xsizein != dcimg[IDmap].md[0].size[0] ||
                        ysizein != dcimg[IDmap].md[0].size[1]))
    {
        PRINT_ERROR(
            "xsize,ysize for %s (%d,%d) "
            "does not match %s (%d,%d)",
            inputstream_name,
            xsizein, ysizein,
            IDmap_name,
            dcimg[IDmap].md[0].size[0],
            dcimg[IDmap].md[0].size[1]);
        free(sizearray);
        return RETURN_FAILURE;
    }
    if(reverse == 1 && (xsizeim != dcimg[IDmap].md[0].size[0] ||
                        ysizeim != dcimg[IDmap].md[0].size[1]))
    {
        PRINT_ERROR(
            "xsize,ysize for %s (%d,%d) "
            "does not match %s (%d,%d)",
            IDout_name,
            xsizein, ysizein,
            IDmap_name,
            dcimg[IDmap].md[0].size[0],
            dcimg[IDmap].md[0].size[1]);
        free(sizearray);
        return RETURN_FAILURE;
    }
    if(NBslice > 1 && reverse == 1)
    {
        printf(
            "ERROR: Cannot use reverse lookup decode with multiple "
            "slices\n");
    }

    sizearray[0] = xsizeim;
    sizearray[1] = ysizeim;
    {
        IMGID imgout_tmp =
            imgid_make_from_name(
                IDout_name);
        imgout_tmp.mdt->naxis = 2;
        imgout_tmp.mdt->size[0] =
            xsizeim;
        imgout_tmp.mdt->size[1] =
            ysizeim;
        imgout_tmp.mdt->datatype =
            dcimg[IDin].md->datatype;
        imgout_tmp.mdt->shared = 1;
        imgout_tmp.mdt->NBkw =
            dcimg[IDin].md->NBkw;
        imgout_tmp.im =
            (IMAGE *) calloc(
                1, sizeof(IMAGE));
        imgid_mkimage(&imgout_tmp);
        IDout = imgout_tmp.ID;
    }

    // Copy the keywords over from IDin to IDout
    int NBkw = dcimg[IDin].md[0].NBkw;
    for(int kw = 0; kw < NBkw; ++kw)
    {
        snprintf(dcimg[IDout].kw[kw].name,
                 KEYWORD_MAX_STRING,
                 "%s",
                 dcimg[IDin].kw[kw].name);
        dcimg[IDout].kw[kw].type  =
            dcimg[IDin].kw[kw].type;
        dcimg[IDout].kw[kw].value =
            dcimg[IDin].kw[kw].value;
        snprintf(dcimg[IDout].kw[kw].comment,
                 KEYWORD_MAX_COMMENT,
                 "%s",
                 dcimg[IDin].kw[kw].comment);
    }

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
        PRINT_ERROR(
            "cannot open file \"%s\"",
            NBpix_fname);
        free(nbpixslice);
        free(tarray);
        free(dtarray);
        free(sizearray);
        return RETURN_FAILURE;
    }

    for(slice = 0; slice < NBslice; slice++)
    {
        int fscanfcnt =
            fscanf(fp, "%ld %ld %ld\n", &tmpl0, &nbpixslice[slice], &tmpl1);
        if(fscanfcnt == EOF)
        {
            if(ferror(fp))
            {
                PRINT_ERROR("fscanf: %s", strerror(errno));
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
        IMGID imgpixsl =
            imgid_make_from_name(
                "outpixsl");
        imgpixsl.mdt->naxis = 2;
        imgpixsl.mdt->size[0] =
            sizearray[0];
        imgpixsl.mdt->size[1] =
            sizearray[1];
        imgpixsl.mdt->datatype =
            _DATATYPE_UINT16;
        imgpixsl.im =
            (IMAGE *) calloc(
                1, sizeof(IMAGE));
        imgid_mkimage(&imgpixsl);
        IDout_pixslice = imgpixsl.ID;

        for(slice = 0; slice < NBslice; slice++)
        {
            sliceii = slice * dcimg[IDmap].md[0].size[0] *
                      dcimg[IDmap].md[0].size[1];
            for(ii = 0; ii < nbpixslice[slice]; ii++)
            {
                // ocam2kpixi files MUST now be in int32 - otherwise we'll overflow in 240x240
                dcimg[IDout_pixslice]
                .array.UI16[dcimg[IDmap].array.UI32[sliceii + ii]] =
                    (unsigned short)(1 + slice);
            }
        }

#ifdef USE_CFITSIO
        save_fits("outpixsl", IDout_pixslice_fname);
#else
        (void) IDout_pixslice_fname;
        printf("WARNING: FITS save disabled"
               " (built without cfitsio)\n");
#endif
        delete_image_ID("outpixsl", DELETE_IMAGE_ERRMODE_WARNING);
    }

    processinfo->loopcntMax = -1;
    processinfo_WriteMessage(processinfo, "Starting loop");

    // ==================================
    // STARTING LOOP
    // ==================================
    processinfo_loopstart(
        processinfo); // Notify processinfo that we are entering loop

    printf("cnt0: %ld; loopOK %d\n", dcimg[IDin].md[0].cnt0, loopOK);
    fflush(stdout);

    // long loopcnt = 0;
    while(loopOK == 1)
    {
        loopOK = processinfo_loopstep(processinfo);
        //printf("cnt0: %ld; loopOK %d\n", dcimg[IDin].md[0].cnt0, loopOK);
        fflush(stdout);

        if(dcimg[IDin].md[0].sem == 0)
        {
            while(dcimg[IDin].md[0].cnt0 ==
                    cnt) // test if new frame exists
            {
                usleep(5);
            }
            cnt = dcimg[IDin].md[0].cnt0;
        }
        else
        {
            if(clock_gettime(CLOCK_MILK, &ts) == -1)
            {
                PRINT_ERROR("clock_gettime: %s", strerror(errno));
                exit(EXIT_FAILURE);
            }
            ts.tv_sec += 1;

            semr = ImageStreamIO_semtimedwait(&dcimg[IDin],
                                              in_semwaitindex,
                                              &ts);

            if(processinfo->loopcnt == 0)
            {
                semval = ImageStreamIO_semvalue(dcimg + IDin, in_semwaitindex);
                for(scnt = 0; scnt < semval; scnt++)
                {
                    ImageStreamIO_semtrywait(dcimg + IDin, in_semwaitindex);
                }
            }
        }

        processinfo_exec_start(processinfo);

        if(processinfo_compute_status(processinfo) == 1)
        {
            if(semr == 0)
            {
                slice = dcimg[IDin].md[0].cnt1;
                if(slice > oldslice + 1)
                {
                    slice = oldslice + 1;
                }

                if(oldslice == NBslice - 1)
                {
                    slice = 0;
                }

                //   clock_gettime(CLOCK_MILK, &tarray[slice]);
                //  dtarray[slice] = 1.0*tarray[slice].tv_sec + 1.0e-9*tarray[slice].tv_nsec;
                dcimg[IDout].md[0].write = 1;

                if(reverse == 0)  // legacy forward lookup mode
                {
                    if(slice < NBslice)
                    {
                        sliceii = slice * dcimg[IDmap].md[0].size[0] *
                                  dcimg[IDmap].md[0].size[1];
                        for(ii = 0; ii < nbpixslice[slice]; ii++)
                        {
                            dcimg[IDout].array.UI16
                            [dcimg[IDmap].array.UI32[sliceii + ii]] =
                                dcimg[IDin].array.UI16[sliceii + ii];
                        }
                    }
                }
                else // reverse == 1, full image assumed (at least given how ocam is scrambled)
                {
                    for(ii = 0; ii < nbpixout; ++ii)
                    {
                        dcimg[IDout].array.UI16[ii] =
                            dcimg[IDin]
                            .array.UI16[dcimg[IDmap].array.UI32[ii]];
                    }
                }

                // Copy the value of the keywords
                for(int kw = 0; kw < NBkw; ++kw)
                {
                    dcimg[IDout].kw[kw].value =
                        dcimg[IDin].kw[kw].value;
                }

                if(slice == NBslice - 1)
                {
                    processinfo_update_output_stream(processinfo,
                        &dcimg[IDout],
                        NULL);
                }

                dcimg[IDout].md[0].cnt1 = slice;

                // Whatever hacks these are to manage slicey business?
                semval = ImageStreamIO_semvalue(dcimg + IDout, 2);
                if(semval < SEMAPHORE_MAXVAL)
                {
                    ImageStreamIO_sempost(dcimg + IDout, 2);
                }

                semval = ImageStreamIO_semvalue(dcimg + IDout, 3);
                if(semval < SEMAPHORE_MAXVAL)
                {
                    ImageStreamIO_sempost(dcimg + IDout, 3);
                }

                dcimg[IDout].md[0].write = 0;

                oldslice = slice;
            }
        }

        processinfo_exec_end(processinfo);
    }

    // ==================================
    // ENDING LOOP
    // ==================================
    processinfo_cleanExit(processinfo);

    /*    if((dcprocinfo == 1) && (processinfo->loopstat != 4)) {
            processinfo_cleanExit(processinfo);
        }*/

    free(nbpixslice);
    free(sizearray);
    free(dtarray);
    free(tarray);

    return IDout;
}


/* ================================================================
 * 5.  BINDINGS, FARG, AND CLI DATA
 * ============================================================= */

FPS_V2_SECTION5(FPS_PARAMS)


/* ================================================================
 * 6.  COMPUTE WRAPPER
 * ============================================================= */

static MILK_HOT errno_t __attribute__((unused)) compute_function()
{
    DEBUG_TRACE_FSTART();

    INSERT_STD_PROCINFO_COMPUTEFUNC_START

    COREMOD_MEMORY_PixMapDecode_U(
        p_instream, p_xsizeim, p_ysizeim,
        p_nbpix_fname, p_mapname,
        p_outname, p_outslice, p_reverse);

    INSERT_STD_PROCINFO_COMPUTEFUNC_END

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}


/* ================================================================
 * 7.  MILK MODULE REGISTRATION
 * ============================================================= */

#if !defined(FPS_STANDALONE) && !defined(MILK_NO_CLI)
static errno_t CLIfunction(void)
{
    return safe_fps_generic_CLIfunction(
        &FPS_app_info, farg, &CLIcmddata,
        my_bindings, nb_bindings,
        compute_function);
}

errno_t
CLIADDCMD_COREMOD_memory__stream_pixmapdecode()
{
    safe_fps_fill_farg_examples(
        farg, my_bindings, nb_bindings);

    int cmdi = RegisterCLIcmd(
        CLIcmddata, CLIfunction);
    CLIcmddata.cmdsettings =
        &data.cmd[cmdi].cmdsettings;

    return RETURN_SUCCESS;
}
#endif
