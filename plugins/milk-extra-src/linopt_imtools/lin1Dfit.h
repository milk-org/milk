// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file lin1Dfit.h
 * @brief Lin1dfit module
 */

#ifndef LINOPT_IMTOOLS__LIN1DFIT_H
#define LINOPT_IMTOOLS__LIN1DFIT_H

errno_t CLIADDCMD_linopt_imtools__lin1Dfits();

errno_t linopt_compute_1Dfit(const char *fnamein,
                             long        NBpt,
                             long        MaxOrder,
                             const char *fnameout,
                             int         MODE,
                             imageID    *outID);

#endif
