// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file CR2toFITS.c
 * @brief Convert CR2 file to FITS
 */

#include <stdio.h>
#include <stdlib.h>

#include "milk_runnable.h"
#include "computeunit.h"

#include "COREMOD_iofits/COREMOD_iofits.h"
#include "COREMOD_memory/COREMOD_memory.h"

#include "readPGM.h"

static int CR2toFITS_NORM = 0;

// Forward declaration
imageID CR2toFITS(const char *__restrict fnameCR2, const char *__restrict fnameFITS);

/* =========================================
 *  V2 PARAMS
 * ======================================= */

static char p_in[FUNCTION_PARAMETER_STRMAXLEN]  = "im01.CR2";
static char p_out[FUNCTION_PARAMETER_STRMAXLEN] = "im01.fits";

#define FPS_PARAMS(X)                                                             \
    X(".in_name", p_in, FPTYPE_STRING, 1, FPFLAG_DEFAULT_INPUT, "input CR2 file") \
    X(".out_name", p_out, FPTYPE_STRING, 1, FPFLAG_DEFAULT_INPUT, "output FITS file")

MILK_PARAMS_DEFINE(CR2toFITS, FPS_PARAMS);

static MILK_HOT errno_t CR2toFITS_compute(const MILK_RUNNABLE *self)
{
    (void) self;
    CR2toFITS(p_in, p_out);
    return RETURN_SUCCESS;
}

/**
 * ## Purpose
 *
 *  Convert CR2 to FITS
 *
 * @note assumes dcraw is installed
 */
imageID CR2toFITS(const char *__restrict fnameCR2, const char *__restrict fnameFITS)
{
    FILE *fp;

    float   iso;
    float   shutter;
    float   aperture;
    imageID ID;
    long    xsize, ysize;
    long    ii;

    EXECUTE_SYSTEM_COMMAND_NOCHECK("dcraw -t 0 -D -4 -c %s > _tmppgm.pgm", fnameCR2);

    ID = read_PGMimage("_tmppgm.pgm", "tmpfits1");
    if (system("rm _tmppgm.pgm") != 0)
    {
        PRINT_ERROR("system() returns non-zero value");
    }

    if (CR2toFITS_NORM == 1)
    {
        EXECUTE_SYSTEM_COMMAND_NOCHECK("dcraw -i -v %s | grep \"ISO speed\"| awk '{print $3}' > "
                                       "iso_tmp.txt",
                                       fnameCR2);

        if ((fp = fopen("iso_tmp.txt", "r")) == NULL)
        {
            PRINT_ERROR("Cannot open file");
        }
        if (fscanf(fp, "%f\n", &iso) != 1)
        {
            PRINT_ERROR("fscanf returns value != 1");
        }
        fclose(fp);

        if (system("rm iso_tmp.txt") != 0)
        {
            PRINT_ERROR("system() returns non-zero value");
        }

        printf("iso = %f\n", iso);

        EXECUTE_SYSTEM_COMMAND_NOCHECK("dcraw -i -v %s | grep \"Shutter\"| awk '{print $2}' > "
                                       "shutter_tmp.txt",
                                       fnameCR2);

        if ((fp = fopen("shutter_tmp.txt", "r")) == NULL)
        {
            PRINT_ERROR("Cannot open file");
        }

        if (fscanf(fp, "%f\n", &shutter) != 1)
        {
            PRINT_ERROR("fscanf returns value != 1");
        }
        fclose(fp);

        if (system("rm shutter_tmp.txt") != 0)
        {
            PRINT_ERROR("system() returns non-zero value");
        }
        printf("shutter = %f\n", shutter);

        EXECUTE_SYSTEM_COMMAND_NOCHECK("dcraw -i -v %s | grep \"Aperture\"| awk '{print $2}' > "
                                       "aperture_tmp.txt",
                                       fnameCR2);

        if ((fp = fopen("aperture_tmp.txt", "r")) == NULL)
        {
            PRINT_ERROR("Cannot open file");
        }
        if (fscanf(fp, "f/%f\n", &aperture) != 1)
        {
            PRINT_ERROR("fscanf returns value != 1");
        }
        fclose(fp);

        if (system("rm aperture_tmp.txt") != 0)
        {
            PRINT_ERROR("system() returns non-zero value");
        }

        printf("aperture = %f\n", aperture);

        ID    = image_ID("tmpfits1", dcimg, dcnimg);
        xsize = dcimg[ID].md[0].size[0];
        ysize = dcimg[ID].md[0].size[1];

        for (ii = 0; ii < xsize * ysize; ii++)
        {
            dcimg[ID].array.F[ii] /= (shutter * aperture * aperture * iso);
        }
    }

    save_fl_fits("tmpfits1", fnameFITS);
    delete_image_ID("tmpfits1", DELETE_IMAGE_ERRMODE_WARNING);

    return ID;
}

/* ================================================================
 * RUNNABLE DESCRIPTOR
 * ============================================================= */

const MILK_RUNNABLE runnable_image_format_CR2toFITS = {
    .kind    = MILK_RUNNABLE_FPS,
    .app     = { .fps_name    = "cr2tofits",
                 .cmdkey      = "cr2tofits",
                 .description = "convert cr2 file to fits",
                 .description_long =
                     "Convert Canon CR2 raw camera files to FITS format. Extracts the raw sensor "
                     "data and writes it as a FITS image." },
    .srcfile = __FILE__,
    MILK_PARAMS_REF(CR2toFITS),
    .compute = CR2toFITS_compute,
};
