// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/** @file Coeff2Map_Loop.h
 */

#ifdef HAVE_CUDA

errno_t Coeff2Map_Loop_addCLIcmd();

errno_t LINALGEBRA_Coeff2Map_Loop(
    const char *IDmodes_name,
    const char *IDcoeff_name,
    int         GPUindex,
    const char *IDoutmap_name,
    int         offsetmode,
    const char *IDoffset_name);

#endif
