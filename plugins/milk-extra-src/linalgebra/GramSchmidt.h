// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef LINALGEBRA_GRAMSCHMIDT_H
#define LINALGEBRA_GRAMSCHMIDT_H


errno_t GramSchmidt(
    IMGID imginm,
    IMGID *imgoutm,
    int GPUdev
);

errno_t CLIADDCMD_linalgebra__GramSchmidt();

#endif
