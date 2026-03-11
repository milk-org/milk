/**
 * @file images2cube.c
 * @brief ==========================================
 */

/**
 * @file    images2cube.c
 */

#include "CLIcore.h"
#include "fps.h"
#include "COREMOD_memory/COREMOD_memory.h"

// ==========================================
// Forward declaration(s)
// ==========================================

errno_t images_to_cube(const char *restrict img_name,
                       long nbframes,
                       const char *restrict cube_name);

// ==========================================
// FPS V2
// ==========================================

static FPS_APP_INFO FPS_app_info = {
    .fps_name    = "imgs2cube",
    .cmdkey      = "imgs2cube",
    .description = "combine individual images into cube"
};

static char    *imgname;
static int64_t *nbframes;
static char    *cubename;

#define FPS_PARAMS(X) \
    X(".imgname", &imgname, \
      FPTYPE_STRING, 1, \
      FPFLAG_DEFAULT_INPUT, "input image name format") \
    X(".nbframes", &nbframes, \
      FPTYPE_INT64, 1, \
      FPFLAG_DEFAULT_INPUT, "number of frames") \
    X(".cubename", &cubename, \
      FPTYPE_STRING, 1, \
      FPFLAG_DEFAULT_INPUT, "output cube name")


FPS_V2_SECTION5(FPS_PARAMS)



static errno_t compute_function()
{
    return images_to_cube(imgname, *nbframes, cubename);
}


#if !defined(FPS_STANDALONE) && !defined(MILK_NO_CLI)
static errno_t CLIfunction(void)
{
    return safe_fps_generic_CLIfunction(
        &FPS_app_info, farg, &CLIcmddata,
        my_bindings, nb_bindings,
        compute_function);
}

// Register CLI command(s)
errno_t CLIADDCMD_COREMOD_iofits__images2cube()
{
    safe_fps_fill_farg_examples(
        farg, my_bindings, nb_bindings);

    INSERT_STD_CLIREGISTERFUNC

    return RETURN_SUCCESS;
}
#endif


#ifdef FPS_STANDALONE
FPS_MAIN_STANDALONE_V2(
    FPS_app_info,
    FPS_PARAMS,
    compute_function)
#endif


// ==========================================
// Compute code
// ==========================================

errno_t images_to_cube(const char *restrict img_name,
                       long nbframes,
                       const char *restrict cube_name)
{
    DEBUG_TRACE_FSTART();
    imageID  ID;
    imageID  ID1;
    long     frame;
    uint32_t naxes[2];
    uint32_t xsize, ysize;
    char     imname[STRINGMAXLEN_IMGNAME];

    frame = 0;

    CREATE_IMAGENAME(imname, "%s%05ld", img_name, frame);

    ID1 = image_ID(imname, dcimg, dcnimg);
    if(ID1 == -1)
    {
        PRINT_ERROR("Image \"%s\" does not exist", imname);
        exit(0);
    }
    naxes[0] = dcimg[ID1].md[0].size[0];
    naxes[1] = dcimg[ID1].md[0].size[1];
    xsize    = naxes[0];
    ysize    = naxes[1];

    printf("SIZE = %ld %ld %ld\n",
           (long) naxes[0],
           (long) naxes[1],
           (long) nbframes);
    fflush(stdout);

    FUNC_CHECK_RETURN(
        create_3Dimage_ID(cube_name, naxes[0], naxes[1], nbframes, &ID));

    for(uint32_t ii = 0; ii < naxes[0]; ii++)
        for(uint32_t jj = 0; jj < naxes[1]; jj++)
        {
            dcimg[ID]
            .array.F[frame * naxes[0] * naxes[1] + (jj * naxes[0] + ii)] =
                dcimg[ID1].array.F[jj * naxes[0] + ii];
        }

    for(frame = 1; frame < nbframes; frame++)
    {
        WRITE_IMAGENAME(imname, "%s%05ld", img_name, frame);
        printf("Adding image %s -> %ld/%ld ... ", img_name, frame, nbframes);
        fflush(stdout);

        ID1 = image_ID(imname, dcimg, dcnimg);
        if(ID1 == -1)
        {
            PRINT_ERROR("Image \"%s\" does not exist - skipping", imname);
        }
        else
        {
            naxes[0] = dcimg[ID1].md[0].size[0];
            naxes[1] = dcimg[ID1].md[0].size[1];
            if((xsize != naxes[0]) || (ysize != naxes[1]))
            {
                PRINT_ERROR("Image has wrong size");
                exit(0);
            }
            for(uint32_t ii = 0; ii < naxes[0]; ii++)
                for(uint32_t jj = 0; jj < naxes[1]; jj++)
                {
                    dcimg[ID].array.F[frame * naxes[0] * naxes[1] +
                                           (jj * naxes[0] + ii)] =
                                               dcimg[ID1].array.F[jj * naxes[0] + ii];
                }
        }
        printf("Done\n");
        fflush(stdout);
    }

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}
