// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef LINALGEBRA_COMPSVD_MKM_H
#define LINALGEBRA_COMPSVD_MKM_H

errno_t SVDmkM(IMGID imgU, IMGID imgS, IMGID imgV, IMGID *imgM, int GPUdev);

errno_t CLIADDCMD_linalgebra__SVDmkM();

#endif
