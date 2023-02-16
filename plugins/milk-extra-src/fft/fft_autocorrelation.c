/**
 * @file    fft_autocorrelation.c
 * @brief   Compute autocorrelation using FFT
 *
 *
 */

#include <math.h>

#include "CommandLineInterface/CLIcore.h"

#include "COREMOD_arith/COREMOD_arith.h"
#include "COREMOD_memory/COREMOD_memory.h"

#include "fft.h"

imageID autocorrelation(const char *IDin_name, const char *IDout_name)
{
    imageID  IDin;
    imageID  IDout;
    uint64_t nelement;

    IDin     = image_ID(IDin_name);
    nelement = data.image[IDin].md[0].nelement;

    char atmp1name[STRINGMAXLEN_IMGNAME];
    WRITE_IMAGENAME(atmp1name, "_atmp1_%d", (int) getpid());

    do2drfft(IDin_name, atmp1name);

    char aampname[STRINGMAXLEN_IMGNAME];
    WRITE_IMAGENAME(aampname, "_aamp_%d", (int) getpid());

    char aphaname[STRINGMAXLEN_IMGNAME];
    WRITE_IMAGENAME(aphaname, "_apha_%d", (int) getpid());

    mk_amph_from_complex(atmp1name, aampname, aphaname, 0);

    char sqaampname[STRINGMAXLEN_IMGNAME];
    WRITE_IMAGENAME(sqaampname, "_sqaamp_%d", (int) getpid());

    arith_image_mult(aampname, aampname, sqaampname);
    delete_image_ID(aampname, DELETE_IMAGE_ERRMODE_WARNING);
    delete_image_ID(aphaname, DELETE_IMAGE_ERRMODE_WARNING);
    delete_image_ID(atmp1name, DELETE_IMAGE_ERRMODE_WARNING);

    char sqaamp1name[STRINGMAXLEN_IMGNAME];
    WRITE_IMAGENAME(sqaamp1name, "_sqaamp1_%d", (int) getpid());

    arith_image_cstmult(sqaampname,
                        1.0 / sqrt(nelement) / (1.0 * nelement),
                        sqaamp1name);
    delete_image_ID(sqaampname, DELETE_IMAGE_ERRMODE_WARNING);

    do2drfft(sqaamp1name, atmp1name);
    mk_reim_from_complex(atmp1name, IDout_name, aphaname, 0);
    delete_image_ID(sqaamp1name, DELETE_IMAGE_ERRMODE_WARNING);
    delete_image_ID(atmp1name, DELETE_IMAGE_ERRMODE_WARNING);
    delete_image_ID(aphaname, DELETE_IMAGE_ERRMODE_WARNING);

    IDout = image_ID("IDout_name");

    return (IDout);
}
