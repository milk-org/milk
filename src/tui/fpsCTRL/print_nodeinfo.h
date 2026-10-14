// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file print_nodeinfo.h
 * @brief Print nodeinfo module
 */

#ifndef FPS_CTRLSCREEN_PRINT_NODEINFO_H
#define FPS_CTRLSCREEN_PRINT_NODEINFO_H

void fpsCTRLscreen_print_nodeinfo(FPS               *fps,
                                  KEYWORD_TREE_NODE *keywnode,
                                  int                nodeSelected,
                                  int                fpsindexSelected,
                                  long               pindexSelected);

#endif
