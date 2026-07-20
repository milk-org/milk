// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef CLUSTERING__ADDCF_TO_CF_H
#define CLUSTERING__ADDCF_TO_CF_H

errno_t addCF_to_CF(CLUSTERTREE *ctree,
                        CLUSTERING_CF CF,
                        long         CFindex,
                        int         *addOK);

errno_t subvector_to_CF(CLUSTERTREE *ctree,
                        CLUSTERING_CF CF,
                        long         CFindex);

#endif
