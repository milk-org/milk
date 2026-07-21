// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file ctree_init.h
 * @brief Ctree init module
 */

#ifndef CLUSTERING__CTREE_INIT_H
#define CLUSTERING__CTREE_INIT_H

errno_t ctree_init(CLUSTERTREE *ctree, double *datavector, long double ssqr);

#endif
