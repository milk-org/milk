// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file makeCosRadModes.h
 * @brief Makecosradmodes module
 */

#ifndef LINOPT_IMTOOLS__MAKECOSRADMODES_H
#define LINOPT_IMTOOLS__MAKECOSRADMODES_H

errno_t CLIADDCMD_linopt_imtools__makeCosRadModes();

errno_t linopt_imtools_makeCosRadModes(const char *ID_name,
                                       long        size,
                                       long        kmax,
                                       float       radius,
                                       float       radfactlim,
                                       imageID    *outID);

#endif
