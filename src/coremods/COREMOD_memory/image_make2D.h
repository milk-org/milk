// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file image_make2D.h
 * @brief Image make2d module
 */

#ifndef COREMOD_MEMORY_MK2DIMAGE_H
#define COREMOD_MEMORY_MK2DIMAGE_H

errno_t CLIADDCMD_COREMOD_memory__mk2Dim();

imageID make_2Dimage(const char *name, uint32_t xsize, uint32_t ysize);
imageID make_2Dimage_IMGID(IMGID *img);

#endif
