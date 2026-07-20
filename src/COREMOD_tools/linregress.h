// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/** @file linregress.h
 */

errno_t lin_regress(double      *a,
                    double      *b,
                    double      *Xi2,
                    double      *x,
                    double      *y,
                    double      *sig,
                    unsigned int nb_points);
