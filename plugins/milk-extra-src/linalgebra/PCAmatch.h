// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef LINALGEBRA_PCAMATCH_H
#define LINALGEBRA_PCAMATCH_H

errno_t PCAmatch(
    IMGID    imgmodesA,
    IMGID    imgmodesB,
    IMGID    *imgoutcA,
    IMGID    *imgoutcB,
    IMGID    *imgoutimA,
    IMGID    *imgoutimB,
    int      GPUdev
);

errno_t CLIADDCMD_linalgebra__PCAmatch();

#endif
