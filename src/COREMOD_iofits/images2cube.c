// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    images2cube.c
 */

#include "CommandLineInterface/CLIcore.h"

#include "COREMOD_memory/COREMOD_memory.h"

// ==========================================
// Forward declaration(s)
// ==========================================

errno_t images_to_cube(const char *restrict img_name,
                       long nbframes,
                       const char *restrict cube_name);

// ==========================================
// Command line interface wrapper function(s)
// ==========================================

static char    *imgname;
static int64_t *nbframes;
static char    *cubename;

static CLICMDARGDEF farg[] = { { CLIARG_STR, ".imgname", "input image name format", "im_",
                                 CLIARG_VISIBLE_DEFAULT, (void **) &imgname, NULL },
                               { CLIARG_INT64, ".nbframes", "number of frames", "100",
                                 CLIARG_VISIBLE_DEFAULT, (void **) &nbframes, NULL },
                               { CLIARG_STR, ".cubename", "output cube name", "imc",
                                 CLIARG_VISIBLE_DEFAULT, (void **) &cubename, NULL } };

static CLICMDDATA CLIcmddata = { "imgs2cube", "combine individual images into cube",
                                 CLICMD_FIELDS_DEFAULTS };

static errno_t help_function()
{
    printf("Combine individual images into cube.\n");
    printf("Image name is prefix followed by 5 digits (e.g. im_00000, im_00001 "
           "...)\n");
    return RETURN_SUCCESS;
}

static errno_t compute_function()
{
    return images_to_cube(imgname, *nbframes, cubename);
}

INSERT_STD_FPSCLIfunctions

    // ==========================================
    // Register CLI command(s)
    // ==========================================

    errno_t images2cube_addCLIcmd()
{
    INSERT_STD_CLIREGISTERFUNC
    return RETURN_SUCCESS;
}

errno_t images_to_cube(const char *restrict img_name, long nbframes, const char *restrict cube_name)
{
    DEBUG_TRACE_FSTART();
    imageID  ID;
    imageID  ID1;
    long     frame;
    uint32_t naxes[2];
    uint32_t xsize, ysize;

    frame = 0;

    CREATE_IMAGENAME(imname, "%s%05ld", img_name, frame);

    ID1 = image_ID(imname);
    if (ID1 == -1)
    {
        PRINT_ERROR("Image \"%s\" does not exist", imname);
        exit(0);
    }
    naxes[0] = data.image[ID1].md[0].size[0];
    naxes[1] = data.image[ID1].md[0].size[1];
    xsize    = naxes[0];
    ysize    = naxes[1];

    printf("SIZE = %ld %ld %ld\n", (long) naxes[0], (long) naxes[1], (long) nbframes);
    fflush(stdout);

    FUNC_CHECK_RETURN(create_3Dimage_ID(cube_name, naxes[0], naxes[1], nbframes, &ID));

    for (uint32_t ii = 0; ii < naxes[0]; ii++)
    {
        for (uint32_t jj = 0; jj < naxes[1]; jj++)
        {
            data.image[ID].array.F[frame * naxes[0] * naxes[1] + (jj * naxes[0] + ii)] =
                data.image[ID1].array.F[jj * naxes[0] + ii];
        }
    }

    for (frame = 1; frame < nbframes; frame++)
    {
        WRITE_IMAGENAME(imname, "%s%05ld", img_name, frame);
        printf("Adding image %s -> %ld/%ld ... ", img_name, frame, nbframes);
        fflush(stdout);

        ID1 = image_ID(imname);
        if (ID1 == -1)
        {
            PRINT_ERROR("Image \"%s\" does not exist - skipping", imname);
        }
        else
        {
            naxes[0] = data.image[ID1].md[0].size[0];
            naxes[1] = data.image[ID1].md[0].size[1];
            if ((xsize != naxes[0]) || (ysize != naxes[1]))
            {
                PRINT_ERROR("Image has wrong size");
                exit(0);
            }
            for (uint32_t ii = 0; ii < naxes[0]; ii++)
            {
                for (uint32_t jj = 0; jj < naxes[1]; jj++)
                {
                    data.image[ID].array.F[frame * naxes[0] * naxes[1] + (jj * naxes[0] + ii)] =
                        data.image[ID1].array.F[jj * naxes[0] + ii];
                }
            }
        }
        printf("Done\n");
        fflush(stdout);
    }

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}
