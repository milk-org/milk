// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file create_new_leaf.h
 * @brief Create new leaf module
 */

#ifndef CLUSTERING__CREATE_NEW_LEAF_H
#define CLUSTERING__CREATE_NEW_LEAF_H

errno_t create_new_leaf(CLUSTERTREE *ctree, double *datarray, long double ssqr, long *CFindex);

#endif
