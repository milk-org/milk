/**
 * @file    saveall.c
 * @brief   save all images/streams to disk
 *
 * Uses FPS V2 framework.
 */

#ifdef MILK_NO_CLI
#    include "CLIcore_standalone.h"
#else
#    include "CLIcore.h"
#endif
#include "fps.h"
#include "COREMOD_memory/COREMOD_memory.h"

#include "create_image.h"
#include "delete_image.h"
#include "image_ID.h"
#include "image_copy.h"
#include "list_image.h"

#ifdef USE_CFITSIO
#    include "COREMOD_iofits/COREMOD_iofits.h"
#endif

/* forward decls */
errno_t COREMOD_MEMORY_SaveAll_snapshot(const char *dirname);

errno_t COREMOD_MEMORY_SaveAll_sequ(const char *dirname,
                                    const char *IDtrig_name,
                                    long        semtrig,
                                    long        NBframes);


/* ================================================================
 *  CMD 1: imsaveallsnap (1 arg)
 * ============================================================= */

static FPS_APP_INFO FPS_app_info_snap = {
    .fps_name         = "imsaveallsnap",
    .cmdkey           = "imsaveallsnap",
    .description      = "save all images in directory",
    .description_long = "Save all images in the current process memory to FITS files on disk. Each "
                        "image is written as a separate file."
};

static char p_dirname_snap[FUNCTION_PARAMETER_STRMAXLEN] = "dir1";

#define FPS_PARAMS_snap(X) \
    X(".dirname", p_dirname_snap, FPTYPE_DIRNAME, 1, FPFLAG_DEFAULT_INPUT, "output directory")

static FPS_CLI_BINDING bindings_snap[] = { FPS_PARAMS_snap(FPS_X_BINDING) };

static const int __attribute__((unused)) nb_bindings_snap =
    sizeof(bindings_snap) / sizeof(FPS_CLI_BINDING);

static CLICMDARGDEF __attribute__((unused)) farg_snap[] = { FPS_PARAMS_snap(FPS_X_FARG) };

static CLICMDDATA CLIcmddata_snap = { "", "", CLICMD_FIELDS_NOPARAM };

FPS_CMDSETTINGS_INIT(snap, CLIcmddata_snap, FPS_app_info_snap)

static errno_t __attribute__((unused)) compute_snap()
{
    COREMOD_MEMORY_SaveAll_snapshot(p_dirname_snap);
    return RETURN_SUCCESS;
}


/* ================================================================
 *  CMD 2: imsaveallseq (4 args, primary)
 * ============================================================= */

static FPS_APP_INFO FPS_app_info = {
    .fps_name         = "imsaveallseq",
    .cmdkey           = "imsaveallseq",
    .description      = "save all images, sequence",
    .description_long = "Save all images in the current process memory to FITS files on disk. Each "
                        "image is written as a separate file."
};

static char      p_dirname[FUNCTION_PARAMETER_STRMAXLEN]  = "dir1";
static char      p_trigname[FUNCTION_PARAMETER_STRMAXLEN] = "im1";
static long long p_semtrig                                = 3;
static long long p_nbframes                               = 20;

#define FPS_PARAMS(X)                                                                       \
    X(".dirname", p_dirname, FPTYPE_DIRNAME, 1, FPFLAG_DEFAULT_INPUT, "output directory")   \
    X(".trigname", p_trigname, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "trigger image") \
    X(".semtrig", &p_semtrig, FPTYPE_INT64, 1, FPFLAG_DEFAULT_INPUT, "trigger semaphore")   \
    X(".nbframes", &p_nbframes, FPTYPE_INT64, 1, FPFLAG_DEFAULT_INPUT, "number of frames")

static FPS_CLI_BINDING my_bindings[] = { FPS_PARAMS(FPS_X_BINDING) };

static const int __attribute__((unused)) nb_bindings =
    sizeof(my_bindings) / sizeof(FPS_CLI_BINDING);

static CLICMDARGDEF farg[] = { FPS_PARAMS(FPS_X_FARG) };

static CLICMDDATA CLIcmddata = { "", "", CLICMD_FIELDS_DEFAULTS };

FPS_CMDSETTINGS_INIT(seq, CLIcmddata, FPS_app_info)

static MILK_HOT errno_t __attribute__((unused)) compute_function()
{
    DEBUG_TRACE_FSTART();
    INSERT_STD_PROCINFO_COMPUTEFUNC_START
    COREMOD_MEMORY_SaveAll_sequ(p_dirname, p_trigname, p_semtrig, p_nbframes);
    INSERT_STD_PROCINFO_COMPUTEFUNC_END DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}


/* ================================================================
 *  REGISTRATION
 * ============================================================= */

#if !defined(FPS_STANDALONE) && !defined(MILK_NO_CLI)

static errno_t CLIfunction_snap(void)
{
    return safe_fps_generic_CLIfunction(&FPS_app_info_snap, farg_snap, &CLIcmddata_snap,
                                        bindings_snap, nb_bindings_snap, compute_snap);
}

static errno_t CLIfunction(void)
{
    return safe_fps_generic_CLIfunction(&FPS_app_info, farg, &CLIcmddata, my_bindings, nb_bindings,
                                        compute_function);
}

errno_t CLIADDCMD_COREMOD_memory__saveall()
{
    {
        safe_fps_fill_farg_examples(farg_snap, bindings_snap, nb_bindings_snap);

        int cmdi                    = RegisterCLIcmd(CLIcmddata_snap, CLIfunction_snap);
        CLIcmddata_snap.cmdsettings = &data.cmd[cmdi].cmdsettings;
    }

    {
        safe_fps_fill_farg_examples(farg, my_bindings, nb_bindings);

        int cmdi               = RegisterCLIcmd(CLIcmddata, CLIfunction);
        CLIcmddata.cmdsettings = &data.cmd[cmdi].cmdsettings;
    }

    return RETURN_SUCCESS;
}
#endif


/* ================================================================
 *  COMPUTATION CODE
 * ============================================================= */

errno_t COREMOD_MEMORY_SaveAll_snapshot(const char *dirname)
{
    long imcnt = 0;

    for (long i = 0; i < dcnimg; i++)
    {
        if (dcimg[i].used == 1)
        {
            imcnt++;
        }
    }

    long *IDarray   = (long *) malloc(sizeof(long) * imcnt);
    long *IDarraycp = (long *) malloc(sizeof(long) * imcnt);

    imcnt = 0;
    for (int i = 0; i < dcnimg; i++)
    {
        if (dcimg[i].used == 1)
        {
            IDarray[imcnt] = i;
            imcnt++;
        }
    }

    EXECUTE_SYSTEM_COMMAND_NOCHECK("mkdir -p %s", dirname);

    for (int i = 0; i < imcnt; i++)
    {
        long ID = IDarray[i];
        char imnamecp[STRINGMAXLEN_IMGNAME];
        WRITE_IMAGENAME(imnamecp, "%s_cp", dcimg[ID].name);
        IDarraycp[i] = copy_image_ID(dcimg[ID].name, imnamecp, 0);
    }

    list_image_ID();

#ifdef USE_CFITSIO
    for (int i = 0; i < imcnt; i++)
    {
        long ID = IDarray[i];
        char imnamecp[STRINGMAXLEN_IMGNAME];
        char fnamecp[STRINGMAXLEN_FULLFILENAME];
        WRITE_IMAGENAME(imnamecp, "%s_cp", dcimg[ID].name);
        WRITE_FULLFILENAME(fnamecp, "./%s/%s.fits", dirname, dcimg[ID].name);
        save_fits(imnamecp, fnamecp);
    }
#else
    printf("WARNING: FITS save disabled"
           " (built without cfitsio)\n");
#endif

    free(IDarray);
    free(IDarraycp);

    return RETURN_SUCCESS;
}

errno_t COREMOD_MEMORY_SaveAll_sequ(const char *dirname,
                                    const char *IDtrig_name,
                                    long        semtrig,
                                    long        NBframes)
{
    long imcnt = 0;

    for (long i = 0; i < dcnimg; i++)
    {
        if (dcimg[i].used == 1)
        {
            imcnt++;
        }
    }

    imageID *IDarray    = (imageID *) malloc(sizeof(imageID) * imcnt);
    imageID *IDarrayout = (imageID *) malloc(sizeof(imageID) * imcnt);

    imcnt = 0;
    for (int i = 0; i < dcnimg; i++)
    {
        if (dcimg[i].used == 1)
        {
            IDarray[imcnt] = i;
            imcnt++;
        }
    }
    uint32_t *imsizearray = (uint32_t *) malloc(sizeof(uint32_t) * imcnt);

    EXECUTE_SYSTEM_COMMAND_NOCHECK("mkdir -p %s", dirname);

    imageID IDtrig;
    {
        IMGID imgtrig = imgid_make_from_name(IDtrig_name);
        resolveIMGID(&imgtrig, ERRMODE_NULL, dcimg, dcnimg);
        IDtrig = imgtrig.ID;
        if (IDtrig == -1)
        {
            fprintf(stderr, "ERROR: trigger stream \"%s\" not found\n", IDtrig_name);
            free(IDarray);
            free(IDarrayout);
            free(imsizearray);
            return RETURN_FAILURE;
        }
    }

    printf("Creating arrays\n");
    fflush(stdout);

    for (int i = 0; i < imcnt; i++)
    {
        char imnameout[200];
        snprintf(imnameout, sizeof(imnameout), "%s_out", dcimg[IDarray[i]].name);
        imsizearray[i] =
            sizeof(float) * dcimg[IDarray[i]].md[0].size[0] * dcimg[IDarray[i]].md[0].size[1];
        printf("Creating image %s"
               "  size %d x %d x %ld\n",
               imnameout, dcimg[IDarray[i]].md[0].size[0], dcimg[IDarray[i]].md[0].size[1],
               NBframes);
        fflush(stdout);
        create_3Dimage_ID(imnameout, dcimg[IDarray[i]].md[0].size[0],
                          dcimg[IDarray[i]].md[0].size[1], NBframes, &(IDarrayout[i]));
    }
    list_image_ID();

    printf("filling arrays\n");
    fflush(stdout);

    while (ImageStreamIO_semtrywait(dcimg + IDtrig, semtrig) == 0)
    {
    }

    long frame = 0;
    while (frame < NBframes)
    {
        ImageStreamIO_semwait(dcimg + IDtrig, semtrig);
        for (int i = 0; i < imcnt; i++)
        {
            imageID ID   = IDarray[i];
            char   *ptr0 = (char *) dcimg[IDarrayout[i]].array.F;
            char   *ptr1 = ptr0 + imsizearray[i] * frame;
            memcpy(ptr1, dcimg[ID].array.F, imsizearray[i]);
        }
        frame++;
    }

    printf("Saving images\n");
    fflush(stdout);

    list_image_ID();

#ifdef USE_CFITSIO
    for (int i = 0; i < imcnt; i++)
    {
        char    imnameout[200];
        char    fnameout[500];
        imageID ID = IDarray[i];
        snprintf(imnameout, sizeof(imnameout), "%s_out", dcimg[ID].name);
        snprintf(fnameout, sizeof(fnameout), "./%s/%s_out.fits", dirname, dcimg[ID].name);
        save_fits(imnameout, fnameout);
    }
#else
    printf("WARNING: FITS save disabled"
           " (built without cfitsio)\n");
#endif

    free(IDarray);
    free(IDarrayout);
    free(imsizearray);

    return RETURN_SUCCESS;
}
