// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef LINOPT_IMTOOLS__IMAGE_CONSTRUCT_H
#define LINOPT_IMTOOLS__IMAGE_CONSTRUCT_H

errno_t CLIADDCMD_linopt_imtools__image_construct();

errno_t linopt_imtools_image_construct(const char *IDmodes_name,
                                       const char *IDcoeff_name,
                                       const char *ID_name,
                                       imageID    *outID);

#endif
