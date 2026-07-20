// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef LINALGEBRA_MODALREMAP_H
#define LINALGEBRA_MODALREMAP_H

errno_t ModalRemap(IMGID imgM0, IMGID imgU0, IMGID imgU1, IMGID *imgM1, int GPUdev);

errno_t CLIADDCMD_linalgebra__ModalRemap();

#endif
