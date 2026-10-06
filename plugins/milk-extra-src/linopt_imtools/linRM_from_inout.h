// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file linRM_from_inout.h
 * @brief Linrm from inout module
 */

#ifndef LINOPT_IMTOOLS__LINRM_FROM_INOUT_H
#define LINOPT_IMTOOLS__LINRM_FROM_INOUT_H

MILK_WEAK errno_t CLIADDCMD_linopt_imtools__linRM_from_inout() MILK_WEAK_CLIFUNCDEF;

MILK_WEAK errno_t linopt_compute_linRM_from_inout(const char *IDinput_name,
                                                  const char *IDinmask_name,
                                                  const char *IDoutput_name,
                                                  const char *IDRM_name,
                                                  imageID    *outID) MILK_WEAK_FUNCDEF;

#endif
