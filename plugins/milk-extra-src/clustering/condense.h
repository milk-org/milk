// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef CLUSTERING__CONDENSE_H
#define CLUSTERING__CONDENSE_H


errno_t ctree_condense_node(
    CLUSTERTREE *ctree,
    long CFindex,
    int *nbop
);

errno_t ctree_condense(CLUSTERTREE *ctree, int *nbop);

#endif
