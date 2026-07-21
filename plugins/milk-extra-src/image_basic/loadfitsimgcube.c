// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file loadfitsimgcube.c
 * @brief Load images into a cube
 */

#ifdef MILK_NO_CLI
#    include "CLIcore_standalone.h"
#else
#    include "CLIcore.h"
#endif
#include "fps.h"

#include "COREMOD_iofits/COREMOD_iofits.h"
#include "COREMOD_memory/COREMOD_memory.h"

// Forward declaration
long load_fitsimages_cube(const char *__restrict strfilter, const char *__restrict ID_out_name);

static char p_pat[FUNCTION_PARAMETER_STRMAXLEN] = "im";
static char p_out[FUNCTION_PARAMETER_STRMAXLEN] = "out";

static FPS_APP_INFO FPS_app_info = { .fps_name    = "loadfitsimgcube",
                                     .cmdkey      = "loadfitsimgcube",
                                     .description = "load images into a single cube",
                                     .description_long =
                                         "Load multiple FITS image files from disk and assemble "
                                         "them into a single 3D cube in shared memory." };

#define FPS_PARAMS(X)                                                                 \
    X(".in_pattern", p_pat, FPTYPE_STRING, 1, FPFLAG_DEFAULT_INPUT, "string pattern") \
    X(".out_name", p_out, FPTYPE_STRING, 1, FPFLAG_DEFAULT_INPUT, "output cube")

FPS_V2_SECTION5(FPS_PARAMS)

static MILK_HOT errno_t compute_function()
{
    load_fitsimages_cube(p_pat, p_out);
    return RETURN_SUCCESS;
}

static errno_t __attribute__((unused)) CLIfunction(void)
{
    return safe_fps_generic_CLIfunction(&FPS_app_info, farg, &CLIcmddata, my_bindings, nb_bindings,
                                        compute_function);
}

errno_t CLIADDCMD_image_basic__loadfitsimgcube()
{
    safe_fps_fill_farg_examples(farg, my_bindings, nb_bindings);
    INSERT_STD_CLIREGISTERFUNC
    return RETURN_SUCCESS;
}

/**
 * Load all images matching strfilter
 * into a single 3D data cube.
 * Returns number of images loaded.
 */
long load_fitsimages_cube(const char *__restrict strfilter, const char *__restrict ID_out_name)
{
    long     cnt = 0;
    char     fname[STRINGMAXLEN_FILENAME];
    char     fname1[STRINGMAXLEN_FILENAME];
    FILE    *fp;
    uint32_t xsize, ysize;
    imageID  ID;

    printf("Filter = %s\n", strfilter);

    EXECUTE_SYSTEM_COMMAND_NOCHECK("ls %s > flist.tmp\n", strfilter);

    xsize = 0;
    ysize = 0;

    if ((fp = fopen("flist.tmp", "r")) == NULL)
    {
        C_ERRNO = errno;
        PRINT_ERROR("fopen() error");
        exit(0);
    }

    while (fgets(fname, STRINGMAXLEN_FILENAME, fp) != NULL)
    {
        fname[strlen(fname) - 1] = '\0';
        if (cnt == 0)
        {
            load_fits(fname, "imtmplfc", 1, &ID);
            xsize = dcimg[ID].md[0].size[0];
            ysize = dcimg[ID].md[0].size[1];
            delete_image_ID("imtmplfc", DELETE_IMAGE_ERRMODE_WARNING);
        }

        load_fits(fname, "imtmplfc", 1, &ID);
        if ((dcimg[ID].md[0].size[0] != xsize) || (dcimg[ID].md[0].size[1] != ysize))
        {
            fprintf(stderr, "ERROR in "
                            "load_fitsimages_cube:"
                            " not all images have"
                            " the same size\n");
            exit(0);
        }
        delete_image_ID("imtmplfc", DELETE_IMAGE_ERRMODE_WARNING);
        cnt++;
    }
    fclose(fp);

    printf("Creating 3D cube ... ");
    fflush(stdout);

    IMGID imgout       = imgid_make_from_name_3D(ID_out_name, xsize, ysize, cnt);
    imgout.mdt->shared = 0;
    imgout.im          = (IMAGE *) calloc(1, sizeof(IMAGE));
    imgid_mkimage(&imgout);
    printf("\n");
    fflush(stdout);

    cnt = 0;
    if ((fp = fopen("flist.tmp", "r")) == NULL)
    {
        C_ERRNO = errno;
        PRINT_ERROR("fopen() error");
        exit(0);
    }

    while (fgets(fname, STRINGMAXLEN_FILENAME, fp) != NULL)
    {
        fname[strlen(fname) - 1] = '\0';
        strncpy(fname1, fname, STRINGMAXLEN_FILENAME);
        fname1[strlen(fname) - 5] = '\0';
        load_fits(fname, fname1, 1, NULL);
        printf("Image %s loaded -> %s\n", fname, fname1);

        IMGID imgtmp = imgid_make_from_name(fname1);
        resolveIMGID(&imgtmp, ERRMODE_ABORT, dcimg, dcnimg);

        for (uint64_t ii = 0; ii < xsize * ysize; ii++)
        {
            imgout.im->array.F[xsize * ysize * cnt + ii] = imgtmp.im->array.F[ii];
        }
        delete_image_ID(fname1, DELETE_IMAGE_ERRMODE_WARNING);
        cnt++;
    }

    fclose(fp);

    printf("%ld images loaded into "
           "cube %s\n",
           cnt, ID_out_name);

    return cnt;
}
