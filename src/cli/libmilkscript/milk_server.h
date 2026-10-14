// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milk_server.h
 * @brief Milk command server and entity operations for bash integration.
 */

#ifndef MILK_SERVER_H
#define MILK_SERVER_H

#include <stddef.h>
#include <stdio.h>

/**
 * @brief Read a live milk entity or FITS keyword into @out.
 *
 * Supported paths (leading '@' optional):
 *   s.<name>.<prop>      Stream/image property (xsize, ysize, zsize, naxis,
 *                        dtype, cnt0, cnt1, nelem, sem, pid, shared)
 *   s.<name>.kw.<KEY>    Stream/image FITS keyword value
 *   f.<name>.<param>     FPS parameter value (or '*' for parameter list)
 *   p.<name>.<prop>      Processinfo property (pid, loopstat, loopcnt,
 *                        loopfreq, exectime, rtprio, ctrlval, statusmsg,
 *                        tmux, description, missedframes)
 *   q.<name>.<prop>      Sequencer property (status, tasks, errors, pid,
 *                        completed)
 *
 * @param spec   Entity path string
 * @param out    Output buffer
 * @param outsz  Size of @out in bytes
 * @return 0 on success, -1 if not found or invalid
 */
int milk_entity_get(const char *spec, char *out, size_t outsz);

/**
 * @brief Write a live milk entity or FITS keyword.
 *
 * Supported paths (leading '@' optional):
 *   f.<name>.<param>     FPS parameter value
 *   p.<name>.ctrlval     Processinfo control value (run/pause/step/stop/int)
 *   s.<name>.kw.<KEY>    Stream/image FITS keyword value (+ optional comment)
 *
 * @param spec     Entity path string (may also contain "=val")
 * @param val      Value string (or NULL if embedded in @spec, or "-" for stdin)
 * @param comment  Optional FITS keyword comment (may be NULL)
 * @return 0 on success, -1 on error
 */
int milk_entity_set(const char *spec, const char *val, const char *comment);

/**
 * @brief Check whether a milk entity exists (or is actively running).
 *
 * @param spec          Entity specification (e.g. "s.cam", "f.dm", "p.loop")
 * @param check_running If non-zero, check if the entity is actively running
 * @return 1 if exists/running, 0 otherwise
 */
int milk_entity_exists(const char *spec, int check_running);

/**
 * @brief Dump all registered canonical and unambiguous short command names.
 *
 * @param fp Output stream
 */
void milk_server_dump_commands(FILE *fp);

/**
 * @brief Run the milk UNIX domain socket command server loop.
 *
 * @param sockpath Socket path ("@abstract" or "/path/to/sock")
 * @param owner_fd Optional fd held by owning shell (-1 if none)
 * @param ready_fd Optional fd to write "READY\n" once listening (-1 if none)
 * @return 0 on clean shutdown, non-zero on error
 */
int milk_server_run(const char *sockpath, int owner_fd, int ready_fd);

#endif /* MILK_SERVER_H */
