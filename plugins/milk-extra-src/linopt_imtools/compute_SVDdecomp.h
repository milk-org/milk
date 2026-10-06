// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file compute_SVDdecomp.h
 * @brief Compute svddecomp module
 */

#ifndef LINOPT_IMTOOLS__COMPUTE_SVDDECOMP_H
#define LINOPT_IMTOOLS__COMPUTE_SVDDECOMP_H

MILK_WEAK errno_t CLIADDCMD_linopt_imtools__compute_SVDdecomp() MILK_WEAK_CLIFUNCDEF;

MILK_WEAK errno_t linopt_compute_SVDdecomp(const char *IDin_name,
                                           const char *IDout_name,
                                           const char *IDcoeff_name,
                                           imageID    *outID) MILK_WEAK_FUNCDEF;

#endif
