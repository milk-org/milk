// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file CFmeminit.h
 * @brief Cfmeminit module
 */

#ifndef CLUSTERING__CFMEMINIT_H
#define CLUSTERING__CFMEMINIT_H

#define CFMEMINIT_CFUPDATE 0x0001

errno_t CFmeminit(CLUSTERTREE *ctree, long CFindex, uint32_t mode);

#endif
