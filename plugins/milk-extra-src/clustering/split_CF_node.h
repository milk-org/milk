// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file split_CF_node.h
 * @brief Split cf node module
 */

#ifndef CLUSTERING__SPLIT_CF_NODE_H
#define CLUSTERING__SPLIT_CF_NODE_H

errno_t split_CF_node(CLUSTERTREE *ctree, long CFindex, long *CFi0, long *CFi1);

#endif
