/**
 * @file naninf2zero.c
 * @brief Naninf2zero module
 */

/** @file naninf2zero.c
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

/* set all nan and inf pixel values to zero */
int basic_naninf2zero(const char *ID_name)
{
    imageID  ID;
    uint32_t naxes[2];
    long     cnt = 0;

    ID       = image_ID(ID_name, dcimg, dcnimg);
    naxes[0] = dcimg[ID].md[0].size[0];
    naxes[1] = dcimg[ID].md[0].size[1];

    for (uint32_t jj = 0; jj < naxes[1]; jj++)
    {
        for (uint32_t ii = 0; ii < naxes[0]; ii++)
        {
            if (!(fabsf(dcimg[ID].array.F[jj * naxes[0] + ii]) < HUGE_VAL))
            {
                dcimg[ID].array.F[jj * naxes[0] + ii] = 0.0f;
                cnt++;
            }
        }
    }

    printf("%ld values replaced\n", cnt);

    return (0);
}
