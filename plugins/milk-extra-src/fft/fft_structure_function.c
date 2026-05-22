/**
 * @file    fft_structure_function.c
 * @brief   Compute structure function using FFT
 *
 *
 */

#include <math.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <stdbool.h>
#include <unistd.h>

#ifdef MILK_NO_CLI
#    include "CLIcore_standalone.h"
#else
#    include "libmilkdata/milkdata.h"
#    include "milkDebugTools.h"
#    include "fps.h"
#    include "ImageStreamIO/ImageStreamIO.h"
#endif

#include "COREMOD_memory/COREMOD_memory.h"

#include "fft_autocorrelation.h"

imageID fft_structure_function(const char *ID_in, const char *ID_out)
{
    imageID  IDout;
    double   value;
    uint64_t nelement;
    uint8_t  datatype;

    autocorrelation(ID_in, ID_out);
    IDout    = image_ID(ID_out, dcimg, dcnimg);
    nelement = dcimg[IDout].md[0].nelement;

    datatype = dcimg[IDout].md[0].datatype;

    if (datatype == _DATATYPE_FLOAT)
    {
        value = -dcimg[IDout].array.F[0];
        for (uint64_t ii = 0; ii < nelement; ii++)
        {
            dcimg[IDout].array.F[ii] += value;
            dcimg[IDout].array.F[ii] *= -2.0 / sqrt(nelement);
        }
    }
    if (datatype == _DATATYPE_DOUBLE)
    {
        value = -dcimg[IDout].array.D[0];
        for (uint64_t ii = 0; ii < nelement; ii++)
        {
            dcimg[IDout].array.D[ii] += value;
            dcimg[IDout].array.D[ii] *= -2.0 / sqrt(nelement);
        }
    }

    return IDout;
}
