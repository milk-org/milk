// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milk_shm_prop.h
 * @brief Reading, writing and testing live entities by path (`s.`, `l.`, `f.`, `p.`): the API of
 *        libmilkprop.
 *
 * Purpose:     one implementation of entity paths for every front end: `@get`, `@set` and
 *              `@exists` in a shell, the session server and the calculator read and write the
 *              same properties the same way, with or without a session.
 * Does:        reads a property as text (milk_shm_prop_get(), also with a glob or `json:`) or
 *              as a number (milk_shm_prop_get_num()); writes a value or runs a unit or process
 *              action (milk_shm_prop_set()); tells whether an entity exists or runs
 *              (milk_shm_prop_exists()); finds a local image or stream; saves, compares and
 *              restores FPS parameters and stream keywords (`@snapshot`, `@diff`, `@restore`).
 * Used by:     the shell client in libmilkbash.so (milk_client_ops.c, milk_client_stream.c,
 *              milk_waitfor.c); the server (milk_server.c, milk_server_dispatch.c,
 *              milk_server_eval.c, milk_server_stream.c, milk_server_complete.c); the
 *              calculator (mc_resolve.c, mc_scalar_eval.c, mc_scalar_ops.c); milk_cmd_args.c
 *              (an entity path as a command argument).
 * Uses:        ImageStreamIO, libfps, libcomputeunit (unit actions), libprocessinfo,
 *              libmilkdata (the local images), milk_shm_cache.h, milk_entity_ns.h, and
 *              COREMOD_arith's header-only statistics kernels (not linked).
 * See also:    milk_shm_prop.c; milk_shm_prop_internal.h (one file per namespace);
 *              src/prop/README.md; docs/guide/scripting.md, "Entities".
 */

#ifndef MILK_SHM_PROP_H
#define MILK_SHM_PROP_H

#include <stddef.h>

#include "ImageStreamIO/ImageStruct.h"
#include "libmilkdata/milkdata.h"

#ifndef MILK_LOCAL_IMAGES
#    ifdef MILK_BASH_CLIENT
#        define MILK_LOCAL_IMAGES ((IMAGE *) NULL)
#        define MILK_NB_LOCAL_IMAGES 0L
#    else
#        define MILK_LOCAL_IMAGES dcimg
#        define MILK_NB_LOCAL_IMAGES dcnimg
#    endif
#endif

/**
 * @brief Locate an active local image or open a shared-memory stream.
 *
 * @param sname      Stream or image name
 * @param tmp_img    Caller-supplied temporary IMAGE struct for SHM open
 * @param need_close Set to 1 if @tmp_img was opened and must be closed
 * @return Pointer to IMAGE struct, or NULL if not found
 */
IMAGE *milk_shm_prop_find_or_open_stream(const char *sname, IMAGE *tmp_img, int *need_close);

IMAGE *milk_shm_prop_find_stream(const char *sname);

/**
 * @brief Read a live milk entity property or FITS keyword into @out.
 *
 * Supported paths (leading '@' optional):
 *   s.<name>.<prop>      Stream/image property (milk_stream_props.h)
 *   s.<name>.kw.<KEY>    Stream/image FITS keyword value (or kw.* for list)
 *   f.<name>.<param>     FPS parameter value (or '*' for parameter list), or a
 *                        unit property (status, run, conf, ok, errors, msg, ...)
 *   p.<name>.<prop>      Processinfo property (pid, status, loopcnt, hz, dt,
 *                        dtmax, jitter, exectime, rtprio, trigmode, statusmsg,
 *                        unit, description, missedframes, timeouts, list)
 *   <name>.<prop>        Legacy 1-dot fallback: tries FPS first, then procinfo
 *
 * @param spec   Entity path string
 * @param out    Output buffer
 * @param outsz  Size of @out in bytes
 * @return 0 on success, -1 if not found or invalid
 */
int milk_shm_prop_get(const char *spec, char *out, size_t outsz);

/**
 * @brief Read a live milk entity property or FITS keyword directly as a number.
 *
 * Uses direct binary reads from cached SHM mappings for stream metadata, FITS
 * keywords, and numeric FPS parameters, falling back to milk_shm_prop_get() for
 * derived or text properties.
 *
 * @param spec    Entity path string (leading '@' optional)
 * @param out_d   Receives floating-point value
 * @param out_l   Receives integer value
 * @param is_long Set to 1 if the property has integer/boolean type, 0 if float
 * @return 0 on success, -1 if not found or non-numeric
 */
int milk_shm_prop_get_num(const char *spec, double *out_d, long *out_l, int *is_long);

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
 * @return 0 on success, -1 on error, MILK_SHM_PROP_ACTION_FAILED if a unit action ran here and failed
 */
int milk_shm_prop_set(const char *spec, const char *val, const char *comment);

/* milk_shm_prop_set(): a unit start or stop ran and failed; it must not run again elsewhere */
#define MILK_SHM_PROP_ACTION_FAILED (-3)

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
