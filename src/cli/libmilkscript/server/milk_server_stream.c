// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milk_server_stream.c
 * @brief Server-side stream synchronization and I/O commands (@waitfor, @grab, @streamcat, @streamwrite).
 *
 * Role & Goal:
 *   Implements the blocking and streaming primitives exposed to bash and
 *   `.milk` scripts:
 *   - `@waitfor`: Wait for N stream frames (via semaphore) or for an shm_prop/boolean
 *     condition to become true, with timeout and client Ctrl-C cancellation support.
 *   - `@grab`: Capture N consecutive frames from a stream into a 3D image cube.
 *   - `@streamcat`: Print all elements of a stream to stdout (one per line).
 *   - `@streamwrite`: Populate stream elements from command arguments or stdin
 *     and post output stream semaphores.
 */

#ifndef _GNU_SOURCE
#    define _GNU_SOURCE
#endif

#include <ctype.h>
#include <math.h>
#include <poll.h>
#include <semaphore.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include "CLIcore.h"
#ifdef USE_CFITSIO
#    include "COREMOD_iofits/COREMOD_iofits.h"
#endif
#include "COREMOD_memory/COREMOD_memory.h"
#include "ImageStreamIO/ImageStreamIO.h"
#include "cli_calc_parser.h"
#include "milk_shm_prop.h"
#include "milk_shm_prop_internal.h"
#include "milk_entity_ns.h"
#include "milk_proto.h"
#include "milk_server_internal.h"
#include "milk_waitfor.h"
#include "processinfo.h"

/**
 * @brief Non-blocking check whether the client sent `MILK_MSG_CANCEL` or disconnected.
 *
 * @param client_fd Client socket fd
 * @return 1 if cancelled or disconnected, 0 otherwise
 */
static int check_client_cancelled(int client_fd)
{
    if (client_fd < 0)
    {
        return 0;
    }
    struct pollfd pfd;
    pfd.fd      = client_fd;
    pfd.events  = POLLIN | POLLHUP | POLLERR;
    pfd.revents = 0;
    if (poll(&pfd, 1, 0) > 0)
    {
        if (pfd.revents & (POLLHUP | POLLERR))
        {
            return 1;
        }
        if (pfd.revents & POLLIN)
        {
            milk_frame_hdr_t hdr;
            uint8_t         *pl = NULL;
            if (milk_proto_recv(client_fd, &hdr, &pl, NULL) == 0)
            {
                free(pl);
                if (hdr.type == MILK_MSG_CANCEL)
                {
                    return 1;
                }
            }
            else
            {
                return 1;
            }
        }
    }
    return 0;
}

/** wait_cancelled() - milk_wait_ops_t callback: client fd in @ctx sent CANCEL or hung up. */
static int wait_cancelled(void *ctx)
{
    return check_client_cancelled(*(const int *) ctx);
}

static void wait_lock(void *ctx)
{
    (void) ctx;
    milk_server_cmd_lock(); /* may run in a worker thread */
}

static void wait_unlock(void *ctx)
{
    (void) ctx;
    milk_server_cmd_unlock();
}

/**
 * @brief Wait for a general boolean expression condition (`<`, `>`, `==`, `!=`, `&&`, `||`).
 */
static int waitfor_expr_cond(const char            *target,
                             double                 t_start,
                             double                 timeout_sec,
                             const milk_wait_ops_t *ops)
{
    for (int iter = 0;; milk_wait_backoff(&iter))
    {
        if (ops->cancelled(ops->ctx))
        {
            return MILK_WAIT_CANCEL;
        }
        char expanded[8192];
        int  eval_rc = 2;
        ops->lock(ops->ctx);
        milk_server_expand_entities_in_expr(target, expanded, sizeof(expanded));
        cli_calc_eval_bash(expanded, 1, &eval_rc);
        ops->unlock(ops->ctx);
        if (eval_rc == 0)
        {
            return 0;
        }
        if (timeout_sec > 0.0 && milk_wait_now() - t_start >= timeout_sec)
        {
            return MILK_WAIT_TIMEOUT;
        }
    }
}

/**
 * @brief Execute `@waitfor`: wait on a boolean expression, entity equality, or stream frame count.
 *
 * @param client_fd   Client socket fd (polled for cancellation)
 * @param target      Stream name, `entity=val`, `p.<name>`, or boolean expression
 * @param arg2        Frame count or process state, then optional `;sem=N;cnt=M`
 * @param timeout_sec Timeout in seconds (<= 0 for infinite)
 * @return 0 on success, 124 on timeout, 130 on Ctrl-C cancel, 1 on error
 */
int milk_server_waitfor(int client_fd, const char *target, const char *arg2, double timeout_sec)
{
    double          t_start = milk_wait_now();
    milk_wait_ops_t ops     = { wait_cancelled, wait_lock, wait_unlock, &client_fd };

    char state[64] = "";
    if (arg2 != NULL)
    {
        snprintf(state, sizeof(state), "%.*s", (int) strcspn(arg2, ";"), arg2);
    }
    milk_wait_target_t wt;
    switch (milk_waitfor_parse(target, state[0] ? state : NULL, &wt))
    {
    case MILK_WAIT_EXPR:
        while (*target == '@')
        {
            target++;
        }
        return waitfor_expr_cond(target, t_start, timeout_sec, &ops);
    case MILK_WAIT_EQ:
        return milk_waitfor_eq(wt.lhs, wt.expect, t_start, timeout_sec, &ops);
    case MILK_WAIT_STREAM:
        break;
    default:
        return 1;
    }

    const char *sp      = (arg2 != NULL) ? strstr(arg2, ";sem=") : NULL;
    const char *cp      = (arg2 != NULL) ? strstr(arg2, ";cnt=") : NULL;
    uint64_t    abs_cnt = (cp != NULL) ? (uint64_t) strtoull(cp + 5, NULL, 10) : 0;
    return milk_waitfor_stream(wt.stream, (arg2 != NULL) ? strtol(arg2, NULL, 10) : 1,
                               (sp != NULL) ? (int) strtol(sp + 5, NULL, 10) : -1,
                               (cp != NULL) ? &abs_cnt : NULL, 1, t_start, timeout_sec, &ops);
}

/**
 * @brief Read one scalar element at @idx from @im as `double`.
 */
static inline double read_typed_elem(const IMAGE *im, uint8_t dt, uint64_t idx)
{
    if (dt == _DATATYPE_FLOAT)
    {
        return (double) im->array.F[idx];
    }
    else if (dt == _DATATYPE_DOUBLE)
    {
        return im->array.D[idx];
    }
    else if (dt == _DATATYPE_UINT16)
    {
        return (double) im->array.UI16[idx];
    }
    else if (dt == _DATATYPE_INT16)
    {
        return (double) im->array.SI16[idx];
    }
    else if (dt == _DATATYPE_UINT32)
    {
        return (double) im->array.UI32[idx];
    }
    else if (dt == _DATATYPE_INT32)
    {
        return (double) im->array.SI32[idx];
    }
    else if (dt == _DATATYPE_INT64)
    {
        return (double) im->array.SI64[idx];
    }
    else if (dt == _DATATYPE_UINT8)
    {
        return (double) im->array.UI8[idx];
    }
    else if (dt == _DATATYPE_INT8)
    {
        return (double) im->array.SI8[idx];
    }
    else if (dt == _DATATYPE_UINT64)
    {
        return (double) im->array.UI64[idx];
    }
    return 0.0;
}

/**
 * @brief Write one scalar value @val into @im at element index @idx.
 */
static inline void write_typed_elem(IMAGE *im, uint8_t dt, uint64_t idx, double val)
{
    if (dt == _DATATYPE_FLOAT)
    {
        im->array.F[idx] = (float) val;
    }
    else if (dt == _DATATYPE_DOUBLE)
    {
        im->array.D[idx] = val;
    }
    else if (dt == _DATATYPE_UINT16)
    {
        im->array.UI16[idx] = (uint16_t) val;
    }
    else if (dt == _DATATYPE_INT16)
    {
        im->array.SI16[idx] = (int16_t) val;
    }
    else if (dt == _DATATYPE_UINT32)
    {
        im->array.UI32[idx] = (uint32_t) val;
    }
    else if (dt == _DATATYPE_INT32)
    {
        im->array.SI32[idx] = (int32_t) val;
    }
    else if (dt == _DATATYPE_INT64)
    {
        im->array.SI64[idx] = (int64_t) val;
    }
    else if (dt == _DATATYPE_UINT8)
    {
        im->array.UI8[idx] = (uint8_t) val;
    }
    else if (dt == _DATATYPE_INT8)
    {
        im->array.SI8[idx] = (int8_t) val;
    }
    else if (dt == _DATATYPE_UINT64)
    {
        im->array.UI64[idx] = (uint64_t) val;
    }
}

/** grab_image_alive() - index of the image @im was copied from, or -1 if it was deleted (lock held). */
static imageID grab_image_alive(const IMAGE *im)
{
    imageID id = image_ID(im->name, data.core.image, data.core.NB_MAX_IMAGE);
    return (id >= 0 && data.core.image[id].md == im->md) ? id : -1;
}

/**
 * @brief Execute `@grab`: capture @nframes consecutive frames from @sname_raw into @outname.
 *
 * When @nframes == 1 and input stream has naxis <= 2, @outname is created with matching
 * 1D/2D dimensions. When @nframes > 1, @outname is a 3D cube (`nx x ny x nframes`).
 *
 * @param client_fd   Client socket fd (polled for cancellation)
 * @param sname_raw   Input stream name (with optional `@` / `s.` prefix)
 * @param nframes     Number of frames to capture (>= 1)
 * @param outname     Destination image/stream name (supports `@L:` / `@T*:` modifiers)
 * @param timeout_sec Timeout in seconds (<= 0 for infinite; for nframes==1 without timeout, snapshots current frame)
 * @return 0 on success, 124 on timeout, 130 on Ctrl-C cancel, 1 on error
 */
int milk_server_grab(int         client_fd,
                     const char *sname_raw,
                     long        nframes,
                     const char *outname,
                     double      timeout_sec,
                     const char *fits_path,
                     const char *ts_path)
{
    if (sname_raw == NULL || outname == NULL || nframes < 1)
    {
        return 1;
    }
    if ((unsigned long) nframes > UINT32_MAX) // cube depth is a 32-bit axis
    {
        fprintf(stderr, "ERROR: @grab: too many frames (%ld)\n", nframes);
        return 1;
    }
    FPS_STREAMNAME_PARSED sp_in = fps_streamname_parse(sname_raw);
    const char           *sname =
        (!sp_in.error && sp_in.name != NULL && sp_in.name[0] != '\0') ? sp_in.name : sname_raw;

    /* Runs in a worker thread: milk state is touched only under the command lock. Between frames the
     * lock is released, so IMAGE structs (which may move) are re-checked by name; their md/array/sem
     * buffers do not move while the image exists. */
    milk_server_cmd_lock();
    IMAGE  tmp_in;
    int    need_close = 0;
    IMAGE *in_im      = milk_shm_prop_find_or_open_stream(sname, &tmp_in, &need_close);
    if (in_im == NULL || in_im->md == NULL)
    {
        milk_server_cmd_unlock();
        fprintf(stderr, "ERROR: @grab: stream '%s' not found\n", sname);
        return 1;
    }

    uint32_t nx      = in_im->md->size[0];
    uint32_t ny      = (in_im->md->naxis >= 2 && in_im->md->size[1] > 0) ? in_im->md->size[1] : 1U;
    uint64_t npix2d  = (uint64_t) nx * ny;
    uint8_t  in_dt   = in_im->md->datatype;
    size_t   in_elsz = ImageStreamIO_typesize(in_dt);

    char        tmp_cubename[64];
    int         is_temp_cube = 0;
    const char *out_spec     = outname;
    if (out_spec[0] == '\0')
    {
        snprintf(tmp_cubename, sizeof(tmp_cubename), "@L:_grab_tmp_%d", (int) getpid());
        out_spec     = tmp_cubename;
        is_temp_cube = 1;
    }

    IMGID out_img = imgid_make_from_name(out_spec);
    if (out_img.stream_parse_error)
    {
        imgid_free(&out_img);
        if (need_close)
        {
            ImageStreamIO_closeIm(&tmp_in);
        }
        milk_server_cmd_unlock();
        return 1;
    }
    if (nframes == 1 && in_im->md->naxis <= 2)
    {
        out_img.mdt->naxis   = (in_im->md->naxis >= 1) ? in_im->md->naxis : 2;
        out_img.mdt->size[0] = nx;
        out_img.mdt->size[1] = (out_img.mdt->naxis >= 2) ? ny : 0;
        out_img.mdt->size[2] = 0;
    }
    else
    {
        out_img.mdt->naxis   = 3;
        out_img.mdt->size[0] = nx;
        out_img.mdt->size[1] = ny;
        out_img.mdt->size[2] = (uint32_t) nframes;
    }
    out_img.mdt->datatype = (out_img.stream_datatype != 0) ? out_img.stream_datatype : in_dt;
    out_img.mdt->shared   = (out_img.stream_loc == 'L') ? 0 : 1;
    out_img.mdt->NBkw     = 10;
    out_img.mdt->CBsize   = 0;

    if (create_image_ID_IMGID(&out_img) != RETURN_SUCCESS || out_img.ID == -1)
    {
        imgid_free(&out_img);
        if (need_close)
        {
            ImageStreamIO_closeIm(&tmp_in);
        }
        milk_server_cmd_unlock();
        return 1;
    }

    IMAGE   cube     = data.core.image[out_img.ID]; /* shallow copies: see above */
    IMAGE   in       = *in_im;
    int     in_local = !need_close; /* in_im is an entry of data.core.image */
    uint8_t out_dt   = cube.md->datatype;
    size_t  out_elsz = ImageStreamIO_typesize(out_dt);
    imgid_free(&out_img);

    FILE *ts_fp = NULL;
    if (ts_path != NULL && ts_path[0] != '\0')
    {
        ts_fp = fopen(ts_path, "w");
        if (ts_fp == NULL)
        {
            fprintf(stderr, "ERROR: @grab: cannot open timestamps file '%s'\n", ts_path);
            if (is_temp_cube)
            {
                delete_image_ID(cube.name, DELETE_IMAGE_ERRMODE_IGNORE);
            }
            if (need_close)
            {
                ImageStreamIO_closeIm(&tmp_in);
            }
            milk_server_cmd_unlock();
            return 1;
        }
    }

    int wait_frames = (in.md->shared && (nframes > 1 || timeout_sec > 0.0)) ? 1 : 0;
    int semindex    = -1;
    if (wait_frames && in.md->sem > 0)
    {
        semindex = ImageStreamIO_getsemwaitindex(in_im, -1);
        if (semindex >= 0 && semindex < in.md->sem)
        {
            while (sem_trywait(in.semptr[semindex]) == 0)
            {
            }
        }
    }

    double   t_start  = milk_wait_now();
    int      has_timo = (timeout_sec > 0.0);
    uint64_t last_cnt = in.md->cnt0;
    int      rc       = 0;

    cube.md->write = 1;
    milk_server_cmd_unlock();
    for (long f = 0; f < nframes; f++)
    {
        if (wait_frames)
        {
            while (in.md->cnt0 == last_cnt)
            {
                if (check_client_cancelled(client_fd))
                {
                    rc = 130;
                    break;
                }
                if (has_timo && (milk_wait_now() - t_start) >= timeout_sec)
                {
                    rc = 124;
                    break;
                }
                if (semindex >= 0 && semindex < in.md->sem)
                {
                    struct timespec ts;
                    clock_gettime(CLOCK_REALTIME, &ts);
                    ts.tv_nsec += 20000000L;
                    if (ts.tv_nsec >= 1000000000L)
                    {
                        ts.tv_sec += 1;
                        ts.tv_nsec -= 1000000000L;
                    }
                    sem_timedwait(in.semptr[semindex], &ts);
                }
                else
                {
                    struct timespec slp = { 0, 2000000L };
                    nanosleep(&slp, NULL);
                }
            }
            if (rc != 0)
            {
                break;
            }
            last_cnt = in.md->cnt0;
        }

        milk_server_cmd_lock();
        if (grab_image_alive(&cube) < 0 || (in_local && grab_image_alive(&in) < 0))
        {
            milk_server_cmd_unlock();
            fprintf(stderr, "ERROR: @grab: image deleted during the grab\n");
            rc = 1;
            break;
        }

        if (ts_fp != NULL)
        {
            struct timespec ts_wall;
            struct timespec ts_mono;
            clock_gettime(CLOCK_REALTIME, &ts_wall);
            clock_gettime(CLOCK_MONOTONIC, &ts_mono);
            double wall_s = (double) ts_wall.tv_sec + 1.0e-9 * (double) ts_wall.tv_nsec;
            double mono_s = (double) ts_mono.tv_sec + 1.0e-9 * (double) ts_mono.tv_nsec;
            fprintf(ts_fp, "%ld %llu %.9f %.9f\n", f, (unsigned long long) in.md->cnt0, wall_s,
                    mono_s);
        }

        /* recopy if the writer touched the frame while we copied it (torn frame) */
        for (int tries = 0; tries < 3; tries++)
        {
            uint64_t c0 = in.md->cnt0;
            if (in_dt == out_dt)
            {
                __builtin_memcpy(((uint8_t *) cube.array.raw) + (size_t) f * npix2d * out_elsz,
                                 in.array.raw, npix2d * in_elsz);
            }
            else
            {
                uint64_t base = (uint64_t) f * npix2d;
                for (uint64_t p = 0; p < npix2d; p++)
                {
                    write_typed_elem(&cube, out_dt, base + p, read_typed_elem(&in, in_dt, p));
                }
            }
            if (in.md->cnt0 == c0 && in.md->write == 0)
            {
                break;
            }
        }
        milk_server_cmd_unlock();
        last_cnt = in.md->cnt0;
    } // for (long f = 0; f < nframes; f++)

    if (ts_fp != NULL)
    {
        fclose(ts_fp);
    }

    milk_server_cmd_lock();
    imageID cid = grab_image_alive(&cube);
    if (cid >= 0)
    {
        processinfo_update_output_stream(NULL, &data.core.image[cid], NULL);
    }

    if (rc == 0 && cid >= 0 && fits_path != NULL && fits_path[0] != '\0')
    {
#ifdef USE_CFITSIO
        const char *clean_fits = (fits_path[0] == '!') ? (fits_path + 1) : fits_path;
        if (save_fits(cube.name, clean_fits) != RETURN_SUCCESS)
        {
            rc = 1;
        }
#else
        fprintf(stderr, "ERROR: @grab: FITS support not compiled in (USE_CFITSIO=OFF)\n");
        rc = 1;
#endif
    }

    if (is_temp_cube && cid >= 0)
    {
        delete_image_ID(cube.name, DELETE_IMAGE_ERRMODE_IGNORE);
    }

    if (semindex >= 0 && semindex < in.md->sem && (!in_local || grab_image_alive(&in) >= 0))
    {
        in.semReadPID[semindex] = 0;
    }
    if (need_close)
    {
        ImageStreamIO_closeIm(&tmp_in);
    }
    milk_server_cmd_unlock();
    return rc;
}

/**
 * @brief Parse a datatype name (`F32`, `FLOAT`, `U16`, `UINT16`, etc.) into `_DATATYPE_*`.
 */
static uint8_t parse_dtype_name(const char *s)
{
    if (s == NULL || s[0] == '\0')
    {
        return _DATATYPE_FLOAT;
    }
    while (*s == '@' || *s == 'T')
    {
        s++;
    }
    if (strcasecmp(s, "F32") == 0 || strcasecmp(s, "FLT32") == 0 || strcasecmp(s, "FLOAT") == 0)
    {
        return _DATATYPE_FLOAT;
    }
    if (strcasecmp(s, "F64") == 0 || strcasecmp(s, "FLT64") == 0 || strcasecmp(s, "DOUBLE") == 0)
    {
        return _DATATYPE_DOUBLE;
    }
    if (strcasecmp(s, "F16") == 0 || strcasecmp(s, "FLT16") == 0 || strcasecmp(s, "HALF") == 0)
    {
        return _DATATYPE_HALF;
    }
    if (strcasecmp(s, "U8") == 0 || strcasecmp(s, "UI8") == 0 || strcasecmp(s, "UINT8") == 0)
    {
        return _DATATYPE_UINT8;
    }
    if (strcasecmp(s, "I8") == 0 || strcasecmp(s, "SI8") == 0 || strcasecmp(s, "INT8") == 0)
    {
        return _DATATYPE_INT8;
    }
    if (strcasecmp(s, "U16") == 0 || strcasecmp(s, "UI16") == 0 || strcasecmp(s, "UINT16") == 0)
    {
        return _DATATYPE_UINT16;
    }
    if (strcasecmp(s, "I16") == 0 || strcasecmp(s, "SI16") == 0 || strcasecmp(s, "INT16") == 0)
    {
        return _DATATYPE_INT16;
    }
    if (strcasecmp(s, "U32") == 0 || strcasecmp(s, "UI32") == 0 || strcasecmp(s, "UINT32") == 0)
    {
        return _DATATYPE_UINT32;
    }
    if (strcasecmp(s, "I32") == 0 || strcasecmp(s, "SI32") == 0 || strcasecmp(s, "INT32") == 0)
    {
        return _DATATYPE_INT32;
    }
    if (strcasecmp(s, "U64") == 0 || strcasecmp(s, "UI64") == 0 || strcasecmp(s, "UINT64") == 0)
    {
        return _DATATYPE_UINT64;
    }
    if (strcasecmp(s, "I64") == 0 || strcasecmp(s, "SI64") == 0 || strcasecmp(s, "INT64") == 0)
    {
        return _DATATYPE_INT64;
    }
    if (strcasecmp(s, "C64") == 0 || strcasecmp(s, "CPLX32") == 0 ||
        strcasecmp(s, "COMPLEX_FLOAT") == 0)
    {
        return _DATATYPE_COMPLEX_FLOAT;
    }
    if (strcasecmp(s, "C128") == 0 || strcasecmp(s, "CPLX64") == 0 ||
        strcasecmp(s, "COMPLEX_DOUBLE") == 0)
    {
        return _DATATYPE_COMPLEX_DOUBLE;
    }
    return 0;
}

/**
 * @brief Built-in `@streamcat`: print all elements of a stream to stdout (text or `--raw` binary).
 *
 * @param argc Argument count
 * @param argv Argument vector (`[--raw] <stream>`)
 * @return 0 on success, 1 on error
 */
int milk_server_streamcat(int argc, char **argv)
{
    int         raw_mode = 0;
    const char *sname    = NULL;
    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "-r") == 0 || strcmp(argv[i], "--raw") == 0)
        {
            raw_mode = 1;
        }
        else if (sname == NULL)
        {
            sname = argv[i];
        }
    }
    if (sname == NULL)
    {
        fprintf(stderr, "Usage: @streamcat [--raw] <stream>\n");
        return 1;
    }
    FPS_STREAMNAME_PARSED sp_cat = fps_streamname_parse(sname);
    if (!sp_cat.error && sp_cat.name != NULL && sp_cat.name[0] != '\0')
    {
        sname = sp_cat.name;
    }

    IMAGE  tmp_im;
    int    need_close = 0;
    IMAGE *im         = milk_shm_prop_find_or_open_stream(sname, &tmp_im, &need_close);
    if (im == NULL || im->md == NULL)
    {
        fprintf(stderr, "ERROR: @streamcat: stream '%s' not found\n", sname);
        return 1;
    }
    uint64_t nel = im->md->nelement;
    uint8_t  dt  = im->md->datatype;

    if (raw_mode)
    {
        size_t elsz = ImageStreamIO_typesize(dt);
        if (elsz > 0 && im->array.raw != NULL)
        {
            fwrite(im->array.raw, elsz, (size_t) nel, stdout);
        }
    }
    else
    {
        for (uint64_t i = 0; i < nel; i++)
        {
            if (dt == _DATATYPE_FLOAT)
            {
                printf("%.9g\n", (double) im->array.F[i]);
            }
            else if (dt == _DATATYPE_DOUBLE)
            {
                printf("%.17g\n", im->array.D[i]);
            }
            else if (dt == _DATATYPE_UINT16)
            {
                printf("%u\n", (unsigned) im->array.UI16[i]);
            }
            else if (dt == _DATATYPE_INT16)
            {
                printf("%d\n", (int) im->array.SI16[i]);
            }
            else if (dt == _DATATYPE_UINT32)
            {
                printf("%u\n", im->array.UI32[i]);
            }
            else if (dt == _DATATYPE_INT32)
            {
                printf("%d\n", (int) im->array.SI32[i]);
            }
            else if (dt == _DATATYPE_INT64)
            {
                printf("%ld\n", (long) im->array.SI64[i]);
            }
            else if (dt == _DATATYPE_UINT8)
            {
                printf("%u\n", (unsigned) im->array.UI8[i]);
            }
            else if (dt == _DATATYPE_INT8)
            {
                printf("%d\n", (int) im->array.SI8[i]);
            }
            else if (dt == _DATATYPE_UINT64)
            {
                printf("%lu\n", (unsigned long) im->array.UI64[i]);
            }
        }
    }
    fflush(stdout);
    if (need_close)
    {
        ImageStreamIO_closeIm(&tmp_im);
    }
    return 0;
}

/**
 * @brief Return 1 if `STDIN_FILENO` has readable data (from `memfd` or pipe).
 */
static int stdin_has_data(void)
{
    if (isatty(STDIN_FILENO))
    {
        return 0;
    }
    struct stat st;
    if (fstat(STDIN_FILENO, &st) == 0 && S_ISREG(st.st_mode))
    {
        off_t cur = lseek(STDIN_FILENO, 0, SEEK_CUR);
        return (cur >= 0 && st.st_size > cur) ? 1 : 0;
    }
    struct pollfd pfd;
    pfd.fd      = STDIN_FILENO;
    pfd.events  = POLLIN;
    pfd.revents = 0;
    if (poll(&pfd, 1, 0) > 0 && (pfd.revents & POLLIN))
    {
        return 1;
    }
    return 0;
}

/**
 * @brief Built-in `@streamwrite`: write elements into a stream from `stdin` or `argv`.
 *
 * Supports (stdin is read only when '-' is given):
 *   `... | @streamwrite [--raw] <stream> -`
 *   `... | @streamwrite [--raw] [--dtype <type>] <stream> <xsize> [ysize] [zsize] -`
 *   `@streamwrite <stream> <val1> <val2> ...`
 *   `@streamwrite [--dtype <type>] <new_stream> <xsize> [ysize] [zsize]`
 *
 * @param argc Argument count
 * @param argv Argument vector
 * @return 0 on success, 1 on error
 */
int milk_server_streamwrite(int argc, char **argv)
{
    int         raw_mode  = 0;
    int         has_dtype = 0;
    const char *dtype_str = NULL;
    const char *sname_raw = NULL;
    const char *pos[64];
    int         npos      = 0;
    int         use_stdin = 0;

    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "-r") == 0 || strcmp(argv[i], "--raw") == 0)
        {
            raw_mode = 1;
        }
        else if ((strcmp(argv[i], "-d") == 0 || strcmp(argv[i], "--dtype") == 0) && i + 1 < argc)
        {
            has_dtype = 1;
            dtype_str = argv[++i];
        }
        else if (strncmp(argv[i], "--dtype=", 8) == 0)
        {
            has_dtype = 1;
            dtype_str = argv[i] + 8;
        }
        else if (strcmp(argv[i], "-") == 0)
        {
            use_stdin = 1;
        }
        else if (sname_raw == NULL)
        {
            sname_raw = argv[i];
        }
        else if (npos < 64)
        {
            pos[npos++] = argv[i];
        }
    }

    if (sname_raw == NULL)
    {
        fprintf(
            stderr,
            "Usage: @streamwrite [--raw] [--dtype <type>] <stream> [xsize [ysize [zsize]]] [-]\n");
        return 1;
    }

    const char           *sname = sname_raw;
    FPS_STREAMNAME_PARSED sp_sw = fps_streamname_parse(sname_raw);
    const char           *sw_bare =
        (!sp_sw.error && sp_sw.name != NULL && sp_sw.name[0] != '\0') ? sp_sw.name : sname_raw;

    if (raw_mode && !use_stdin)
    {
        fprintf(stderr, "ERROR: @streamwrite: --raw reads stdin; add '-'\n");
        return 1;
    }
    int has_stdin = use_stdin && stdin_has_data();
    if (use_stdin && !has_stdin)
    {
        fprintf(stderr, "ERROR: @streamwrite: '-' given but stdin has no data\n");
        return 1;
    }
    IMAGE  tmp_im;
    int    need_close = 0;
    IMAGE *im         = NULL;

    /* Determine if pos[] are stream creation dimensions (<xsize> [ysize] [zsize]) */
    int pos_are_dims = 0;
    if (has_dtype && npos >= 1 && npos <= 3)
    {
        pos_are_dims = 1;
    }
    else
    {
        im = milk_shm_prop_find_or_open_stream(sw_bare, &tmp_im, &need_close);
        if ((im == NULL || im->md == NULL) && npos >= 1 && npos <= 3)
        {
            pos_are_dims = 1;
        }
        else if (has_stdin && npos >= 1 && npos <= 3)
        {
            pos_are_dims = 1;
        }
    }

    if (pos_are_dims)
    {
        if (need_close)
        {
            ImageStreamIO_closeIm(&tmp_im);
            need_close = 0;
        }
        uint8_t req_dt = _DATATYPE_FLOAT;
        if (has_dtype)
        {
            req_dt = parse_dtype_name(dtype_str);
            if (req_dt == 0)
            {
                fprintf(stderr, "ERROR: @streamwrite: invalid datatype '%s'\n", dtype_str);
                return 1;
            }
        }
        uint32_t nx = (uint32_t) strtoul(pos[0], NULL, 10);
        uint32_t ny = (npos >= 2) ? (uint32_t) strtoul(pos[1], NULL, 10) : 0;
        uint32_t nz = (npos >= 3) ? (uint32_t) strtoul(pos[2], NULL, 10) : 0;
        if (nx == 0)
        {
            fprintf(stderr, "ERROR: @streamwrite: invalid xsize '%s'\n", pos[0]);
            return 1;
        }

        IMGID out_img = imgid_make_from_name(sname);
        if (out_img.stream_parse_error)
        {
            imgid_free(&out_img);
            return 1;
        }
        out_img.mdt->naxis    = (nz > 0) ? 3 : ((ny > 0) ? 2 : 1);
        out_img.mdt->size[0]  = nx;
        out_img.mdt->size[1]  = ny;
        out_img.mdt->size[2]  = nz;
        out_img.mdt->datatype = (out_img.stream_datatype != 0) ? out_img.stream_datatype : req_dt;
        out_img.mdt->shared   = (out_img.stream_loc == 'L') ? 0 : 1;
        out_img.mdt->NBkw     = 10;
        out_img.mdt->CBsize   = 0;

        if (create_image_ID_IMGID(&out_img) != RETURN_SUCCESS || out_img.ID == -1)
        {
            imgid_free(&out_img);
            return 1;
        }
        im = &data.core.image[out_img.ID];
        imgid_free(&out_img);
        npos = 0;
    }

    if (im == NULL || im->md == NULL)
    {
        fprintf(stderr, "ERROR: @streamwrite: stream '%s' not found\n", sw_bare);
        return 1;
    }

    uint64_t nel  = im->md->nelement;
    uint8_t  dt   = im->md->datatype;
    im->md->write = 1;
    uint64_t idx  = 0;

    if (npos == 0 && !use_stdin && !pos_are_dims)
    {
        im->md->write = 0;
        if (need_close)
        {
            ImageStreamIO_closeIm(&tmp_im);
        }
        fprintf(stderr,
                "ERROR: @streamwrite: nothing to write (give values, or '-' to read stdin)\n");
        return 1;
    }

    if (raw_mode)
    {
        size_t   elsz        = ImageStreamIO_typesize(dt);
        size_t   total_bytes = (size_t) nel * elsz;
        size_t   off         = 0;
        uint8_t *dst         = (uint8_t *) im->array.raw;
        while (off < total_bytes)
        {
            ssize_t nr = read(STDIN_FILENO, dst + off, total_bytes - off);
            if (nr <= 0)
            {
                break;
            }
            off += (size_t) nr;
        }
    }
    else if (npos > 0)
    {
        for (int a = 0; a < npos && idx < nel; a++)
        {
            const char *p = pos[a];
            while (*p != '\0' && idx < nel)
            {
                while (*p != '\0' && isspace((unsigned char) *p))
                {
                    p++;
                }
                if (*p == '\0')
                {
                    break;
                }
                char  *endp = NULL;
                double val  = strtod(p, &endp);
                if (endp == p)
                {
                    break;
                }
                p = endp;
                write_typed_elem(im, dt, idx++, val);
            }
        }
    }
    else if (use_stdin)
    {
        int dup_fd = dup(STDIN_FILENO);
        if (dup_fd >= 0)
        {
            FILE *fp = fdopen(dup_fd, "r");
            if (fp != NULL)
            {
                double val = 0.0;
                for (; idx < nel; idx++)
                {
                    if (fscanf(fp, "%lf", &val) != 1)
                    {
                        break;
                    }
                    write_typed_elem(im, dt, idx, val);
                }
                fclose(fp);
            }
            else
            {
                close(dup_fd);
            }
        }
    }

    processinfo_update_output_stream(NULL, im, NULL);
    if (need_close)
    {
        ImageStreamIO_closeIm(&tmp_im);
    }
    return 0;
}

/**
 * @brief Built-in `@replay`: stream 2D slices from a 3D image cube or FITS file into a destination stream.
 *
 * Supports:
 *   `@replay [--fps <hz>] [--count <n>] [--loop] <src_cube_or_fits> <dst_stream>`
 *
 * @param argc Argument count
 * @param argv Argument vector
 * @return 0 on success, 130 on interrupt, 1 on error
 */
int milk_server_replay(int argc, char **argv)
{
    double      fps_hz       = 0.0;
    long        max_frames   = 0;
    int         loop_forever = 0;
    const char *src_raw      = NULL;
    const char *dst_raw      = NULL;

    for (int i = 1; i < argc; i++)
    {
        if ((strcmp(argv[i], "-r") == 0 || strcmp(argv[i], "--fps") == 0) && i + 1 < argc)
        {
            fps_hz = strtod(argv[++i], NULL);
        }
        else if (strncmp(argv[i], "--fps=", 6) == 0)
        {
            fps_hz = strtod(argv[i] + 6, NULL);
        }
        else if ((strcmp(argv[i], "-n") == 0 || strcmp(argv[i], "--count") == 0) && i + 1 < argc)
        {
            max_frames = strtol(argv[++i], NULL, 10);
        }
        else if (strncmp(argv[i], "--count=", 8) == 0)
        {
            max_frames = strtol(argv[i] + 8, NULL, 10);
        }
        else if (strcmp(argv[i], "-l") == 0 || strcmp(argv[i], "--loop") == 0)
        {
            loop_forever = 1;
        }
        else if (src_raw == NULL)
        {
            src_raw = argv[i];
        }
        else if (dst_raw == NULL)
        {
            dst_raw = argv[i];
        }
    }

    if (src_raw == NULL || dst_raw == NULL)
    {
        fprintf(stderr, "Usage: @replay [--fps <hz>] [--count <n>] [--loop] <src_cube_or_fits> "
                        "<dst_stream>\n");
        return 1;
    }

    FPS_STREAMNAME_PARSED sp_src = fps_streamname_parse(src_raw);
    const char           *src_clean =
        (!sp_src.error && sp_src.name != NULL && sp_src.name[0] != '\0') ? sp_src.name : src_raw;

    IMAGE  tmp_src;
    int    need_close_src  = 0;
    int    loaded_temp_src = 0;
    IMGID  loaded_imgid    = { 0 };
    IMAGE *src_im          = NULL;

    /* Prefer loading from FITS file if src_raw exists as a regular file on disk */
    struct stat st_src;
    if (stat(src_raw, &st_src) == 0 && S_ISREG(st_src.st_mode))
    {
#ifdef USE_CFITSIO
        char tmp_fits_im[64];
        snprintf(tmp_fits_im, sizeof(tmp_fits_im), "@L:_replay_src_%d", (int) getpid());
        loaded_imgid = imgid_make_from_name(tmp_fits_im);
        int saved_q  = dcquiet;
        dcquiet      = 1;
        errno_t lret = load_fits_IMGID(src_raw, &loaded_imgid, 0);
        dcquiet      = saved_q;
        if (lret == RETURN_SUCCESS && loaded_imgid.im != NULL && loaded_imgid.im->md != NULL)
        {
            src_im          = loaded_imgid.im;
            loaded_temp_src = 1;
        }
        else
        {
            imgid_free(&loaded_imgid);
        }
#endif
    }

    if (src_im == NULL)
    {
        src_im = milk_shm_prop_find_or_open_stream(src_clean, &tmp_src, &need_close_src);
    }
    if (src_im == NULL || src_im->md == NULL)
    {
        fprintf(stderr, "ERROR: @replay: source cube or FITS file '%s' not found\n", src_raw);
        return 1;
    }

    uint32_t nx = src_im->md->size[0];
    uint32_t ny = (src_im->md->naxis >= 2 && src_im->md->size[1] > 0) ? src_im->md->size[1] : 1U;
    uint32_t nz = (src_im->md->naxis >= 3 && src_im->md->size[2] > 0) ? src_im->md->size[2] : 1U;
    uint64_t npix2d      = (uint64_t) nx * ny;
    uint8_t  src_dt      = src_im->md->datatype;
    size_t   src_elsz    = ImageStreamIO_typesize(src_dt);
    size_t   slice_bytes = (size_t) npix2d * src_elsz;

    const char           *dst_spec = dst_raw;
    FPS_STREAMNAME_PARSED sp_dst   = fps_streamname_parse(dst_raw);
    const char           *dst_bare =
        (!sp_dst.error && sp_dst.name != NULL && sp_dst.name[0] != '\0') ? sp_dst.name : dst_raw;

    /* Reuse existing shared-memory stream if dimensions already match, preserving open readers */
    IMAGE  tmp_dst;
    int    need_close_dst = 0;
    IMAGE *dst_im         = NULL;
    if (sp_dst.loc != 'L')
    {
        IMAGE *cand = milk_shm_prop_find_or_open_stream(dst_bare, &tmp_dst, &need_close_dst);
        if (cand != NULL && cand->md != NULL)
        {
            uint32_t c_ny =
                (cand->md->naxis >= 2 && cand->md->size[1] > 0) ? cand->md->size[1] : 1U;
            if (cand->md->size[0] == nx && c_ny == ny && cand->md->naxis <= 2)
            {
                dst_im = cand;
            }
            else if (need_close_dst)
            {
                ImageStreamIO_closeIm(&tmp_dst);
                need_close_dst = 0;
            }
        }
    }

    if (dst_im == NULL)
    {
        IMGID dst_img = imgid_make_from_name(dst_spec);
        if (dst_img.stream_parse_error)
        {
            imgid_free(&dst_img);
            if (loaded_temp_src)
            {
                if (loaded_imgid.im != NULL)
                {
                    ImageStreamIO_destroyIm(loaded_imgid.im);
                    free(loaded_imgid.im);
                    loaded_imgid.im = NULL;
                }
                imgid_free(&loaded_imgid);
            }
            if (need_close_src)
            {
                ImageStreamIO_closeIm(&tmp_src);
            }
            return 1;
        }
        dst_img.mdt->naxis    = (src_im->md->naxis == 1) ? 1 : 2;
        dst_img.mdt->size[0]  = nx;
        dst_img.mdt->size[1]  = (dst_img.mdt->naxis >= 2) ? ny : 0;
        dst_img.mdt->size[2]  = 0;
        dst_img.mdt->datatype = (dst_img.stream_datatype != 0) ? dst_img.stream_datatype : src_dt;
        dst_img.mdt->shared   = (dst_img.stream_loc == 'L') ? 0 : 1;
        dst_img.mdt->NBkw     = 10;
        dst_img.mdt->CBsize   = 0;

        if (create_image_ID_IMGID(&dst_img) != RETURN_SUCCESS || dst_img.ID == -1)
        {
            imgid_free(&dst_img);
            if (loaded_temp_src)
            {
                if (loaded_imgid.im != NULL)
                {
                    ImageStreamIO_destroyIm(loaded_imgid.im);
                    free(loaded_imgid.im);
                    loaded_imgid.im = NULL;
                }
                imgid_free(&loaded_imgid);
            }
            if (need_close_src)
            {
                ImageStreamIO_closeIm(&tmp_src);
            }
            return 1;
        }
        dst_im = &data.core.image[dst_img.ID];
        imgid_free(&dst_img);
    }

    if (max_frames <= 0 && !loop_forever)
    {
        max_frames = (long) nz;
    }

    double  period_sec = (fps_hz > 0.0) ? (1.0 / fps_hz) : 0.0;
    uint8_t dst_dt     = dst_im->md->datatype;
    int     rc         = 0;

    for (long f = 0; loop_forever || f < max_frames; f++)
    {
        if (dcsigINT)
        {
            dcsigINT = 0;
            rc       = 130;
            break;
        }
        if (f > 0 && period_sec > 0.0)
        {
            double t0 = milk_wait_now();
            while ((milk_wait_now() - t0) < period_sec)
            {
                if (dcsigINT)
                {
                    break;
                }
                double rem = period_sec - (milk_wait_now() - t0);
                if (rem <= 0.0)
                {
                    break;
                }
                if (rem > 0.02)
                {
                    rem = 0.02;
                }
                struct timespec slp;
                slp.tv_sec  = (time_t) rem;
                slp.tv_nsec = (long) ((rem - (double) slp.tv_sec) * 1.0e9);
                nanosleep(&slp, NULL);
            }
            if (dcsigINT)
            {
                dcsigINT = 0;
                rc       = 130;
                break;
            }
        }

        uint32_t z        = (uint32_t) (f % (long) nz);
        dst_im->md->write = 1;
        if (dst_dt == src_dt)
        {
            __builtin_memcpy(dst_im->array.raw,
                             ((const uint8_t *) src_im->array.raw) + (size_t) z * slice_bytes,
                             slice_bytes);
        }
        else
        {
            uint64_t base = (uint64_t) z * npix2d;
            for (uint64_t p = 0; p < npix2d; p++)
            {
                write_typed_elem(dst_im, dst_dt, p, read_typed_elem(src_im, src_dt, base + p));
            }
        }
        processinfo_update_output_stream(NULL, dst_im, NULL);
    } // for (long f = 0; loop_forever || f < max_frames; f++)

    if (need_close_dst)
    {
        ImageStreamIO_closeIm(&tmp_dst);
    }
    if (loaded_temp_src)
    {
        if (loaded_imgid.im != NULL)
        {
            ImageStreamIO_destroyIm(loaded_imgid.im);
            free(loaded_imgid.im);
            loaded_imgid.im = NULL;
        }
        imgid_free(&loaded_imgid);
    }
    if (need_close_src)
    {
        ImageStreamIO_closeIm(&tmp_src);
    }
    return rc;
}
