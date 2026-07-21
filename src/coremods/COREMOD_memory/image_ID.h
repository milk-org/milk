// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file image_ID.h
 * @brief Image id module
 */

/**
 * @file    image_ID.h
 */

imageID image_ID(const char *name, IMAGE *imagearray, long NB_images);

MILK_PURE imageID image_ID_noaccessupdate(const char *name, IMAGE *imagearray, long NB_images);

imageID next_avail_image_ID(imageID preferredID);
