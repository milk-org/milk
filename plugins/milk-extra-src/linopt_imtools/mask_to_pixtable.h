// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file mask_to_pixtable.h
 * @brief Mask to pixtable module
 */

#ifndef LINOPT_IMTOOLS__MASK_TO_PIXTABLE_H
#define LINOPT_IMTOOLS__MASK_TO_PIXTABLE_H

errno_t CLIADDCMD_linopt_imtools__mask_to_pixtable();

errno_t linopt_imtools_mask_to_pixtable(const char *IDmask_name,
                                        const char *IDpixindex_name,
                                        const char *IDpixmult_name,
                                        long       *outNBpix);

#endif
