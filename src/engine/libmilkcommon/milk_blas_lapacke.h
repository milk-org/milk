// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef INCLUDE_BLAS_LAPACKE_H
#define INCLUDE_BLAS_LAPACKE_H

// Provide BLAS from MKL or OPENBLAS
#ifdef HAVE_MKL
#    include "mkl.h"
#else
#    ifdef HAVE_OPENBLAS
#        include <cblas.h>
#    endif
#endif

// Provide LAPACKE from MKL or OPENBLAS or liblapacke
#ifdef HAVE_LAPACKE
#    ifdef HAVE_MKL
#        include "mkl_lapacke.h"
#    else
#        include <lapacke.h> // May come from OPENBLAS or LAPACKE standalone
#    endif
#endif


#endif // #ifndef INCLUDE_BLAS_LAPACKE_H
