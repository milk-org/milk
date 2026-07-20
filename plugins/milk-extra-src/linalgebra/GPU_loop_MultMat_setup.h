// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/** @file GPU_loop_MultMat_setup.c
 */

#ifdef HAVE_CUDA

/** @brief Setup memory and process for GPU-based matrix-vector multiply
 */
int GPU_loop_MultMat_setup(int         index,
                           const char *IDcontrM_name,
                           const char *IDwfsim_name,
                           const char *IDoutdmmodes_name,
                           long        NBGPUs,
                           int        *GPUdevice,
                           int         orientation,
                           int         USEsem,
                           int         initWFSref,
                           long        loopnb);

#endif
