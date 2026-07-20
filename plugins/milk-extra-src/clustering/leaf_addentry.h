// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef CLUSTERING__LEAF_ADDENTRY_H
#define CLUSTERING__LEAF_ADDENTRY_H

errno_t leaf_addentry(CLUSTERTREE *ctree,
                      double      *datavec,
                      long double  ssqr,
                      long         lCFindex,
                      int         *addOK,
                      double       distance);

#endif
