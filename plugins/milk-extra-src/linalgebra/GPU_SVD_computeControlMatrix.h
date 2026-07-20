// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/** @file GPU_SVD_computeControlMatrix.h
 */

#ifdef HAVE_CUDA

errno_t GPU_SVD_computeControlMatrix(int         device,
                                     const char *ID_Rmatrix_name,
                                     const char *ID_Cmatrix_name,
                                     double      SVDeps,
                                     const char *ID_VTmatrix_name);

#endif
