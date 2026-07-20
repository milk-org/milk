// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef LINALGEBRA_QEXPAND_H
#define LINALGEBRA_QEXPAND_H

errno_t Qexpand(IMGID incoeffM, IMGID *outcoeffM, int axis);

errno_t CLIADDCMD_linalgebra__Qexpand();

#endif
