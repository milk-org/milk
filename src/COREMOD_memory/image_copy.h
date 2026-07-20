// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    image_copy.h
 */

errno_t image_copy_addCLIcmd();

imageID copy_image_ID(const char *name, const char *newname, int shared);
imageID copy_image_ID_IMGID(IMGID *imgin, IMGID *imgout, int shared);

imageID chname_image_ID(const char *ID_name, const char *new_name);
imageID chname_image_ID_IMGID(IMGID *imgin, const char *new_name);

errno_t COREMOD_MEMORY_cp2shm(const char *IDname, const char *IDshmname);
errno_t COREMOD_MEMORY_cp2shm_IMGID(IMGID *imgin, IMGID *imgout);
