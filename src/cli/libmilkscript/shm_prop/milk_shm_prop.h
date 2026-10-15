// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milk_shm_prop.h
 * @brief Unified live shared-memory entity resolver for milk-cli and bash.
 *
 * Role & Goal:
 *   Provides a single, canonical implementation for reading, writing, and
 *   querying the existence of live milk shared-memory entities across four
 *   namespaces:
 *     - Streams / local images : @s.<name>.<prop> / @stream.<name>.<prop>
 *     - Function Parameter Sys : @f.<name>.<param> / @fps.<name>.<param>
 *     - Processinfo telemetry  : @p.<name>.<prop> / @proc.<name>.<prop>
 *     - FPS Sequencers         : @q.<name>.<prop> / @seq.<name>.<prop>
 *
 *   Used by:
 *     - The bash engine's in-process C builtins (`milk_get`, `milk_set`,
 *       `milk_exists`) and the milk UNIX domain socket server.
 *     - The legacy `.milk` script `@` variable expander
 *       (`CLIcore_script_expand_fps.c`) and `fpsset` CLI command.
 */

#ifndef MILK_SHM_PROP_H
#define MILK_SHM_PROP_H

#include <stddef.h>

#include "ImageStreamIO/ImageStruct.h"

/**
 * @brief Locate an active local image or open a shared-memory stream.
 *
 * @param sname      Stream or image name
 * @param tmp_img    Caller-supplied temporary IMAGE struct for SHM open
 * @param need_close Set to 1 if @tmp_img was opened and must be closed
 * @return Pointer to IMAGE struct, or NULL if not found
 */
IMAGE *milk_shm_prop_find_or_open_stream(const char *sname, IMAGE *tmp_img, int *need_close);

/**
 * @brief Read a live milk entity property or FITS keyword into @out.
 *
 * Supported paths (leading '@' optional):
 *   s.<name>.<prop>      Stream/image property (xsize, ysize, zsize, naxis,
 *                        type, typeid, typename, dtype, cnt0, cnt1, nelement,
 *                        nelem, sem, pid, ownerPID, shared, NBkw)
 *   s.<name>.kw.<KEY>    Stream/image FITS keyword value (or kw.* for list)
 *   f.<name>.<param>     FPS parameter value (or '*' for parameter list)
 *   p.<name>.<prop>      Processinfo property (pid, loopstat, loopcnt,
 *                        loopfreq, exectime, rtprio, ctrlval, trigmode,
 *                        statusmsg, tmux, description, missedframes)
 *   q.<name>.<prop>      Sequencer property (status, tasks, errors, pid,
 *                        completed)
 *   <name>.<prop>        Legacy 1-dot fallback: tries FPS first, then procinfo
 *
 * @param spec   Entity path string
 * @param out    Output buffer
 * @param outsz  Size of @out in bytes
 * @return 0 on success, -1 if not found or invalid
 */
int milk_shm_prop_get(const char *spec, char *out, size_t outsz);

/**
 * @brief Write a live milk entity property or FITS keyword.
 *
 * Supported paths (leading '@' optional):
 *   f.<name>.<param>     FPS parameter value
 *   p.<name>.ctrlval     Processinfo control value (run/pause/step/stop/int)
 *   s.<name>.kw.<KEY>    Stream/image FITS keyword value (+ optional comment)
 *   <fpsname>.<param>    Legacy 1-dot FPS parameter write
 *
 * @param spec     Entity path string (may also contain "=val")
 * @param val      Value string (or NULL if embedded in @spec, or "-" for stdin)
 * @param comment  Optional FITS keyword comment (may be NULL)
 * @return 0 on success, -1 on error
 */
int milk_shm_prop_set(const char *spec, const char *val, const char *comment);

/**
 * @brief Check whether a milk entity exists (or is actively running).
 *
 * @param spec          Entity specification (e.g. "s.cam", "f.dm", "p.loop")
 * @param check_running If non-zero, verify that the entity is actively running
 * @return 1 if exists/running, 0 otherwise
 */
int milk_shm_prop_exists(const char *spec, int check_running);

/**
 * @brief Write a value string to a named FPS parameter.
 *
 * @param fpsname  FPS shared-memory name
 * @param pname    Parameter key inside FPS
 * @param valstr   Value to write (as string)
 * @return 0 on success, -1 on error
 */
int cli_fps_set_param(const char *fpsname, const char *pname, const char *valstr);

/**
 * @brief Export FPS parameters and stream FITS keywords to JSON (`@snapshot`).
 */
int milk_shm_prop_snapshot(int argc, char **argv);

/**
 * @brief Compare live FPS/stream state against a saved JSON snapshot (`@diff`).
 */
int milk_shm_prop_diff(int argc, char **argv);

/**
 * @brief Restore FPS parameters and stream FITS keywords from a JSON snapshot (`@restore`).
 */
int milk_shm_prop_restore(int argc, char **argv);

#endif /* MILK_SHM_PROP_H */
