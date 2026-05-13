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
                       uint32_t nbframes,
                       const char *restrict cube_name);

// ==========================================
// FPS V2
// ==========================================

static FPS_APP_INFO FPS_app_info = {
    .fps_name    = "imgs2cube",
    .cmdkey      = "imgs2cube",
    .description = "combine individual images into cube",
    .description_long =
        "Assemble multiple 2D FITS files into a single 3D cube. Input files must have identical dimensions. The z-axis corresponds to the file sequence."
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


static MILK_HOT errno_t __attribute__((unused)) compute_function()
{
    return images_to_cube(imgname, (uint32_t)(*nbframes), cubename);
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

errno_t images_to_cube(
    const char *restrict img_name,
    uint32_t nbframes,
    const char *restrict cube_name)
{
    DEBUG_TRACE_FSTART();
    char imname[STRINGMAXLEN_IMGNAME];

    uint32_t frame = 0;
    CREATE_IMAGENAME(imname, "%s%05u",
                     img_name, frame);

    IMGID img1 =
        imgid_make_from_name(imname);
    resolveIMGID(&img1, ERRMODE_WARN,
                 dcimg, dcnimg);
    if (img1.ID == -1) {
        return RETURN_FAILURE;
    }

    uint32_t xsize = img1.md->size[0];
    uint32_t ysize = img1.md->size[1];

    printf("SIZE = %u %u %u\n",
           xsize, ysize, nbframes);
    fflush(stdout);

    IMGID imgcube =
        imgid_make_from_name_3D(
            cube_name,
            xsize, ysize, nbframes);
    imgcube.mdt->shared = 0;
    imgcube.im = (IMAGE *) calloc(
        1, sizeof(IMAGE));
    imgid_mkimage(&imgcube);

    for(uint32_t ii = 0; ii < xsize; ii++)
        for(uint32_t jj = 0; jj < ysize; jj++)
        {
            imgcube.im->array.F[
                jj * xsize + ii] =
                img1.im->array.F[
                    jj * xsize + ii];
        }

    for(frame = 1; frame < nbframes; frame++)
    {
        WRITE_IMAGENAME(imname, "%s%05u",
                        img_name, frame);
        printf("Adding image %s -> %u/%u.."
               " ", img_name, frame,
               nbframes);
        fflush(stdout);

        img1 = imgid_make_from_name(imname);
        resolveIMGID(&img1, ERRMODE_NULL,
                     dcimg, dcnimg);
        if(img1.ID == -1)
        {
            PRINT_ERROR(
                "Image \"%s\" does not "
                "exist - skipping",
                imname);
        }
        else
        {
            if((xsize != img1.md->size[0])
                || (ysize
                    != img1.md->size[1]))
            {
                PRINT_ERROR(
                    "Image has wrong size");
                return RETURN_FAILURE;
            }
            for(uint32_t ii = 0;
                ii < xsize; ii++)
                for(uint32_t jj = 0;
                    jj < ysize; jj++)
                {
                    imgcube.im->array.F[
                        frame * xsize * ysize
                        + jj * xsize + ii] =
                        img1.im->array.F[
                            jj * xsize + ii];
                }
        }
        printf("Done\n");
        fflush(stdout);
    }

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}
