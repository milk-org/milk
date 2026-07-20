// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    image_ID.h
 */

imageID image_ID(const char *name);

imageID image_ID_noaccessupdate(const char *name);

imageID next_avail_image_ID(imageID preferredID);
