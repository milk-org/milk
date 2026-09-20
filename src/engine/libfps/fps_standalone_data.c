// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file fps_standalone_data.c
 * @brief Fps standalone data module
 */

#define _GNU_SOURCE

/*
 * fitsio.h often expects LONGLONG to be defined.
 * It is typically defined by system headers if _GNU_SOURCE is present,
 * but we provide a fallback here just in case.
 */
#ifndef LONGLONG
#    define LONGLONG long long
#endif

#include "CLIcore.h"


#if !defined(FPS_STANDALONE_SKIP_STUBS) && !defined(MILK_NO_CLI)
/**
 * @brief Stub: image lookup by name for standalone builds.
 *
 * Linear search through the provided image array.
 *
 * @param name        Stream name to find
 * @param imagearray  Image array to search
 * @param NB_images   Number of entries in imagearray
 * @return imageID index or -1 if not found
 */
imageID image_ID(const char *name,
                 IMAGE      *imagearray __attribute__((unused)),
                 long        NB_images __attribute__((unused)))
{
    for (long ii = 0; ii < NB_images; ii++)
    {
        if (imagearray[ii].used == 1 &&
            strncmp(imagearray[ii].name, name, STRINGMAXLEN_IMAGE_NAME) == 0)
        {
            return ii;
        }
    }
    return -1;
}
#endif /* !FPS_STANDALONE_SKIP_STUBS && !MILK_NO_CLI */
