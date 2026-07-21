// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file compute_image_memory.c
 * @brief Compute image memory module
 */

/**
 * @file    compute_image_memory.c
 */

#ifdef MILK_NO_CLI
#    include "CLIcore_standalone.h"
#    include "COREMOD_memory/COREMOD_memory.h"
#else
#    include "libmilkdata/milkdata.h"
#endif


/**
 * @brief Compute total memory used by all images.
 */
uint64_t compute_image_memory()
{
    uint64_t totalmem = 0;

    for (imageID i = 0; i < dcnimg; i++)
    {
        //printf("%5ld / %5ld  %d\n", i, dcnimg, dcimg[i].used);
        //	fflush(stdout);

        if (dcimg[i].used == 1)
        {
            totalmem += dcimg[i].md[0].nelement * ImageStreamIO_typesize(dcimg[i].md[0].datatype);
        }
    }

    return totalmem;
}
