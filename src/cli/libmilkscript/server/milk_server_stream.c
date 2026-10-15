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
#include "milk_proto.h"
#include "milk_server_internal.h"
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

/**
 * @brief Return current `CLOCK_MONOTONIC` timestamp in seconds.
 *
 * @return Monotonic time in seconds
 */
static double monotonic_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double) ts.tv_sec + 1.0e-9 * (double) ts.tv_nsec;
}

/**
 * @brief Wait for a general boolean expression condition (`<`, `>`, `==`, `!=`, `&&`, `||`).
 */
static int waitfor_expr_cond(int client_fd, const char *target, double t_start, double timeout_sec)
{
    int has_timo = (timeout_sec > 0.0);
    while (1)
    {
        if (check_client_cancelled(client_fd))
        {
            return 130;
        }
        char expanded[8192];
        milk_server_expand_entities_in_expr(target, expanded, sizeof(expanded));
        int eval_rc = 2;
        cli_calc_eval_bash(expanded, 1, &eval_rc);
        if (eval_rc == 0)
        {
            return 0;
        }
        if (has_timo && (monotonic_sec() - t_start) >= timeout_sec)
        {
            return 124;
        }
        struct timespec slp = { 0, 10000000L }; /* 10 ms */
        nanosleep(&slp, NULL);
    }
}

/**
 * @brief Wait for a simple equality condition `entity=expected`.
 */
static int waitfor_eq_cond(int         client_fd,
                           const char *target,
                           const char *eq,
                           double      t_start,
                           double      timeout_sec)
{
    char   lhs[256];
    size_t lhslen = (size_t) (eq - target);
    if (lhslen >= sizeof(lhs))
    {
        return 1;
    }
    memcpy(lhs, target, lhslen);
    lhs[lhslen]          = '\0';
    const char *expect   = eq + 1;
    int         has_timo = (timeout_sec > 0.0);

    while (1)
    {
        if (check_client_cancelled(client_fd))
        {
            return 130;
        }
        char val[256];
        if (milk_shm_prop_get(lhs, val, sizeof(val)) == 0)
        {
            if (strcmp(val, expect) == 0)
            {
                return 0;
            }
            char  *e1 = NULL;
            char  *e2 = NULL;
            double d1 = strtod(val, &e1);
            double d2 = strtod(expect, &e2);
            if (val[0] != '\0' && expect[0] != '\0' && *e1 == '\0' && *e2 == '\0' &&
                fabs(d1 - d2) <= 1.0e-9 * (1.0 + fabs(d2)))
            {
                return 0;
            }
        }
        if (has_timo && (monotonic_sec() - t_start) >= timeout_sec)
        {
            return 124;
        }
        struct timespec slp = { 0, 10000000L }; /* 10 ms */
        nanosleep(&slp, NULL);
    }
}

/**
 * @brief Execute `@waitfor`: wait on a boolean expression, entity equality, or stream frame count.
 *
 * @param client_fd   Client socket fd (polled for cancellation)
 * @param target      Stream name, `entity=val`, or boolean expression
 * @param arg2        Optional frame count / `;sem=N;cnt=M` string when @target is a stream
 * @param timeout_sec Timeout in seconds (<= 0 for infinite)
 * @return 0 on success, 124 on timeout, 130 on Ctrl-C cancel, 1 on error
 */
int milk_server_waitfor(int client_fd, const char *target, const char *arg2, double timeout_sec)
{
    while (*target == '@')
    {
        target++;
    }
    double t_start  = monotonic_sec();
    int    has_timo = (timeout_sec > 0.0);

    /* Case 1a: General boolean expression condition */
    if (strchr(target, '<') != NULL || strchr(target, '>') != NULL ||
        strstr(target, "==") != NULL || strstr(target, "!=") != NULL ||
        strstr(target, "&&") != NULL || strstr(target, "||") != NULL)
    {
        return waitfor_expr_cond(client_fd, target, t_start, timeout_sec);
    }

    /* Case 1b: Condition wait f.name.param=val or s.name.prop=val */
    const char *eq = strchr(target, '=');
    if (eq != NULL)
    {
        return waitfor_eq_cond(client_fd, target, eq, t_start, timeout_sec);
    }

    /* Case 2: Stream frame wait: s.cam [nframes] [-s sem] [--cnt target] */
    const char *sname = target;
    if (strncmp(sname, "s.", 2) == 0)
    {
        sname += 2;
    }
    else if (strncmp(sname, "stream.", 7) == 0)
    {
        sname += 7;
    }

    long     nframes     = 1;
    int      req_sem     = -1;
    uint64_t abs_cnt     = 0;
    int      has_abs_cnt = 0;
    if (arg2 != NULL && arg2[0] != '\0')
    {
        if (arg2[0] != ';')
        {
            nframes = strtol(arg2, NULL, 10);
            if (nframes < 1)
            {
                nframes = 1;
            }
        }
        const char *sp = strstr(arg2, ";sem=");
        if (sp != NULL)
        {
            req_sem = (int) strtol(sp + 5, NULL, 10);
        }
        const char *cp = strstr(arg2, ";cnt=");
        if (cp != NULL)
        {
            abs_cnt     = (uint64_t) strtoull(cp + 5, NULL, 10);
            has_abs_cnt = 1;
        }
    }

    IMAGE img;
    memset(&img, 0, sizeof(img));
    while (ImageStreamIO_openIm(&img, sname) != IMAGESTREAMIO_SUCCESS || img.md == NULL)
    {
        if (check_client_cancelled(client_fd))
        {
            return 130;
        }
        if (has_timo && (monotonic_sec() - t_start) >= timeout_sec)
        {
            return 124;
        }
        struct timespec slp = { 0, 10000000L };
        nanosleep(&slp, NULL);
    }

    uint64_t start_cnt0 = img.md->cnt0;
    uint64_t target_cnt = has_abs_cnt ? abs_cnt : (start_cnt0 + (uint64_t) nframes);
    int      semindex   = (req_sem >= 0) ? req_sem : ImageStreamIO_getsemwaitindex(&img, -1);

    if (semindex >= 0 && semindex < img.md->sem)
    {
        while (sem_trywait(img.semptr[semindex]) == 0)
        {
        }
    }

    int rc = 0;
    while (img.md->cnt0 < target_cnt)
    {
        if (check_client_cancelled(client_fd))
        {
            rc = 130;
            break;
        }
        if (has_timo && (monotonic_sec() - t_start) >= timeout_sec)
        {
            rc = 124;
            break;
        }
        if (semindex >= 0 && semindex < img.md->sem)
        {
            struct timespec ts;
            clock_gettime(CLOCK_REALTIME, &ts);
            ts.tv_nsec += 20000000L; /* 20 ms step */
            if (ts.tv_nsec >= 1000000000L)
            {
                ts.tv_sec += 1;
                ts.tv_nsec -= 1000000000L;
            }
            sem_timedwait(img.semptr[semindex], &ts);
        }
        else
        {
            struct timespec slp = { 0, 5000000L };
            nanosleep(&slp, NULL);
        }
    }

    if (req_sem < 0 && semindex >= 0 && semindex < img.md->sem)
    {
        img.semReadPID[semindex] = 0;
    }
    ImageStreamIO_closeIm(&img);
    return rc;
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
    const char *sname = sname_raw;
    while (*sname == '@')
    {
        sname++;
    }
    if (strncmp(sname, "s.", 2) == 0)
    {
        sname += 2;
    }
    else if (strncmp(sname, "stream.", 7) == 0)
    {
        sname += 7;
    }

    IMAGE  tmp_in;
    int    need_close = 0;
    IMAGE *in_im      = milk_shm_prop_find_or_open_stream(sname, &tmp_in, &need_close);
    if (in_im == NULL || in_im->md == NULL)
    {
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
    else if (strncmp(out_spec, "@s.", 3) == 0)
    {
        out_spec += 3;
    }
    else if (strncmp(out_spec, "@stream.", 8) == 0)
    {
        out_spec += 8;
    }
    else if (strncmp(out_spec, "s.", 2) == 0)
    {
        out_spec += 2;
    }
    else if (strncmp(out_spec, "stream.", 7) == 0)
    {
        out_spec += 7;
    }

    IMGID out_img = imgid_make_from_name(out_spec);
    if (out_img.stream_parse_error)
    {
        imgid_free(&out_img);
        if (need_close)
        {
            ImageStreamIO_closeIm(&tmp_in);
        }
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
    out_img.mdt->shared   = (out_img.stream_loc == 2) ? 0 : 1;
    out_img.mdt->NBkw     = 10;
    out_img.mdt->CBsize   = 0;

    if (create_image_ID_IMGID(&out_img) != RETURN_SUCCESS || out_img.ID == -1)
    {
        imgid_free(&out_img);
        if (need_close)
        {
            ImageStreamIO_closeIm(&tmp_in);
        }
        return 1;
    }

    IMAGE  *cube     = &data.core.image[out_img.ID];
    uint8_t out_dt   = cube->md->datatype;
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
                delete_image_ID(cube->name, DELETE_IMAGE_ERRMODE_IGNORE);
            }
            if (need_close)
            {
                ImageStreamIO_closeIm(&tmp_in);
            }
            return 1;
        }
    }

    int wait_frames = (in_im->md->shared && (nframes > 1 || timeout_sec > 0.0)) ? 1 : 0;
    int semindex    = -1;
    if (wait_frames && in_im->md->sem > 0)
    {
        semindex = ImageStreamIO_getsemwaitindex(in_im, -1);
        if (semindex >= 0 && semindex < in_im->md->sem)
        {
            while (sem_trywait(in_im->semptr[semindex]) == 0)
            {
            }
        }
    }

    double   t_start  = monotonic_sec();
    int      has_timo = (timeout_sec > 0.0);
    uint64_t last_cnt = in_im->md->cnt0;
    int      rc       = 0;

    cube->md->write = 1;
    for (long f = 0; f < nframes; f++)
    {
        if (wait_frames)
        {
            while (in_im->md->cnt0 == last_cnt)
            {
                if (check_client_cancelled(client_fd))
                {
                    rc = 130;
                    break;
                }
                if (has_timo && (monotonic_sec() - t_start) >= timeout_sec)
                {
                    rc = 124;
                    break;
                }
                if (semindex >= 0 && semindex < in_im->md->sem)
                {
                    struct timespec ts;
                    clock_gettime(CLOCK_REALTIME, &ts);
                    ts.tv_nsec += 20000000L;
                    if (ts.tv_nsec >= 1000000000L)
                    {
                        ts.tv_sec += 1;
                        ts.tv_nsec -= 1000000000L;
                    }
                    sem_timedwait(in_im->semptr[semindex], &ts);
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
            last_cnt = in_im->md->cnt0;
        }

        if (ts_fp != NULL)
        {
            struct timespec ts_wall;
            struct timespec ts_mono;
            clock_gettime(CLOCK_REALTIME, &ts_wall);
            clock_gettime(CLOCK_MONOTONIC, &ts_mono);
            double wall_s = (double) ts_wall.tv_sec + 1.0e-9 * (double) ts_wall.tv_nsec;
            double mono_s = (double) ts_mono.tv_sec + 1.0e-9 * (double) ts_mono.tv_nsec;
            fprintf(ts_fp, "%ld %llu %.9f %.9f\n", f, (unsigned long long) in_im->md->cnt0, wall_s,
                    mono_s);
        }

        if (in_dt == out_dt)
        {
            __builtin_memcpy(((uint8_t *) cube->array.raw) + (size_t) f * npix2d * out_elsz,
                             in_im->array.raw, npix2d * in_elsz);
        }
        else
        {
            uint64_t base = (uint64_t) f * npix2d;
            for (uint64_t p = 0; p < npix2d; p++)
            {
                write_typed_elem(cube, out_dt, base + p, read_typed_elem(in_im, in_dt, p));
            }
        }
    } // for (long f = 0; f < nframes; f++)

    if (ts_fp != NULL)
    {
        fclose(ts_fp);
    }

    processinfo_update_output_stream(NULL, cube, NULL);

    if (rc == 0 && fits_path != NULL && fits_path[0] != '\0')
    {
#ifdef USE_CFITSIO
        const char *clean_fits = (fits_path[0] == '!') ? (fits_path + 1) : fits_path;
        if (save_fits(cube->name, clean_fits) != RETURN_SUCCESS)
        {
            rc = 1;
        }
#else
        fprintf(stderr, "ERROR: @grab: FITS support not compiled in (USE_CFITSIO=OFF)\n");
        rc = 1;
#endif
    }

    if (is_temp_cube)
    {
        delete_image_ID(cube->name, DELETE_IMAGE_ERRMODE_IGNORE);
    }

    if (semindex >= 0 && semindex < in_im->md->sem)
    {
        in_im->semReadPID[semindex] = 0;
    }
    if (need_close)
    {
        ImageStreamIO_closeIm(&tmp_in);
    }
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
    while (*sname == '@')
    {
        sname++;
    }
    if (strncmp(sname, "s.", 2) == 0)
    {
        sname += 2;
    }
    else if (strncmp(sname, "stream.", 7) == 0)
    {
        sname += 7;
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
                printf("%d\n", im->array.SI32[i]);
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
 * Supports:
 *   `... | @streamwrite [--raw] <stream>`
 *   `... | @streamwrite [--raw] [--dtype <type>] <stream> <xsize> [ysize] [zsize]`
 *   `@streamwrite <stream> <val1> <val2> ...`
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
    int         npos = 0;

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
        fprintf(stderr,
                "Usage: @streamwrite [--raw] [--dtype <type>] <stream> [xsize [ysize [zsize]]]\n");
        return 1;
    }

    const char *sname = sname_raw;
    if (strncmp(sname, "@s.", 3) == 0)
    {
        sname += 3;
    }
    else if (strncmp(sname, "@stream.", 8) == 0)
    {
        sname += 8;
    }
    else if (strncmp(sname, "s.", 2) == 0)
    {
        sname += 2;
    }
    else if (strncmp(sname, "stream.", 7) == 0)
    {
        sname += 7;
    }
    else if (sname[0] == '@' && strchr(sname, ':') == NULL)
    {
        sname++;
    }

    int    has_stdin = stdin_has_data();
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
        im = milk_shm_prop_find_or_open_stream(sname, &tmp_im, &need_close);
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
        out_img.mdt->shared   = (out_img.stream_loc == 2) ? 0 : 1;
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
        fprintf(stderr, "ERROR: @streamwrite: stream '%s' not found\n", sname);
        return 1;
    }

    uint64_t nel  = im->md->nelement;
    uint8_t  dt   = im->md->datatype;
    im->md->write = 1;
    uint64_t idx  = 0;

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
    else
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

    const char *src_clean = src_raw;
    while (*src_clean == '@')
    {
        src_clean++;
    }
    if (strncmp(src_clean, "s.", 2) == 0)
    {
        src_clean += 2;
    }
    else if (strncmp(src_clean, "stream.", 7) == 0)
    {
        src_clean += 7;
    }

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

    const char *dst_spec = dst_raw;
    if (strncmp(dst_spec, "@s.", 3) == 0)
    {
        dst_spec += 3;
    }
    else if (strncmp(dst_spec, "@stream.", 8) == 0)
    {
        dst_spec += 8;
    }
    else if (strncmp(dst_spec, "s.", 2) == 0)
    {
        dst_spec += 2;
    }
    else if (strncmp(dst_spec, "stream.", 7) == 0)
    {
        dst_spec += 7;
    }

    /* Reuse existing shared-memory stream if dimensions already match, preserving open readers */
    IMAGE  tmp_dst;
    int    need_close_dst = 0;
    IMAGE *dst_im         = NULL;
    if (dst_spec[0] != '@')
    {
        IMAGE *cand = milk_shm_prop_find_or_open_stream(dst_spec, &tmp_dst, &need_close_dst);
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
        dst_img.mdt->shared   = (dst_img.stream_loc == 2) ? 0 : 1;
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
            double t0 = monotonic_sec();
            while ((monotonic_sec() - t0) < period_sec)
            {
                if (dcsigINT)
                {
                    break;
                }
                double rem = period_sec - (monotonic_sec() - t0);
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
