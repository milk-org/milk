// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file streamCTRL_print_inode.h
 * @brief Streamctrl print inode module
 */

#ifndef _STREAMCTRL_PRINT_INODE_H
#define _STREAMCTRL_PRINT_INODE_H


int streamCTRL_print_inode(ino_t  inode,
                           ino_t *upstreaminode,
                           int    NBupstreaminode,
                           int    downstreamindex);

#endif
