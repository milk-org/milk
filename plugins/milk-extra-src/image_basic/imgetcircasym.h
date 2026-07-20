// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/** @file imgetcircasym.h
 */

errno_t __attribute__((cold)) imgetcircasym_addCLIcmd();

imageID
IMAGE_BASIC_get_circasym_component_byID(imageID ID,
                                        const char *__restrict ID_out_name,
                                        float       xcenter,
                                        float       ycenter,
                                        const char *options);

imageID IMAGE_BASIC_get_circasym_component(const char *__restrict ID_name,
        const char *__restrict ID_out_name,
        float       xcenter,
        float       ycenter,
        const char *options);
