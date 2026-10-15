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
#include <time.h>

#include "CLIcore.h"
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
 * @param arg2        Optional frame count string when @target is a stream
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

    /* Case 2: Stream frame wait: s.cam [nframes] */
    const char *sname = target;
    if (strncmp(sname, "s.", 2) == 0)
    {
        sname += 2;
    }
    else if (strncmp(sname, "stream.", 7) == 0)
    {
        sname += 7;
    }

    long nframes = (arg2 != NULL && arg2[0] != '\0') ? strtol(arg2, NULL, 10) : 1;
    if (nframes < 1)
    {
        nframes = 1;
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
    uint64_t target_cnt = start_cnt0 + (uint64_t) nframes;
    int      semindex   = ImageStreamIO_getsemwaitindex(&img, -1);

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

    if (semindex >= 0 && semindex < img.md->sem)
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
 * @brief Execute `@grab`: capture @nframes consecutive frames from @sname_raw into 3D cube @outname.
 *
 * @param client_fd   Client socket fd (polled for cancellation)
 * @param sname_raw   Input stream name (with optional `@` / `s.` prefix)
 * @param nframes     Number of frames to capture (>= 1)
 * @param outname     Destination image/stream name (supports `@L:` / `@T*:` modifiers)
 * @param timeout_sec Timeout in seconds (<= 0 for infinite)
 * @return 0 on success, 124 on timeout, 130 on Ctrl-C cancel, 1 on error
 */
int milk_server_grab(int         client_fd,
                     const char *sname_raw,
                     long        nframes,
                     const char *outname,
                     double      timeout_sec)
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

    IMGID out_img = imgid_make_from_name(outname);
    if (out_img.stream_parse_error)
    {
        imgid_free(&out_img);
        if (need_close)
        {
            ImageStreamIO_closeIm(&tmp_in);
        }
        return 1;
    }
    out_img.mdt->naxis    = 3;
    out_img.mdt->size[0]  = nx;
    out_img.mdt->size[1]  = ny;
    out_img.mdt->size[2]  = (uint32_t) nframes;
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

    int semindex = -1;
    if (in_im->md->shared && in_im->md->sem > 0)
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
        if (in_im->md->shared)
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

    processinfo_update_output_stream(NULL, cube, NULL);

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
 * @brief Built-in `@streamcat`: print all elements of a stream to stdout.
 *
 * @param argc Argument count
 * @param argv Argument vector (`argv[1]` = stream name)
 * @return 0 on success, 1 on error
 */
int milk_server_streamcat(int argc, char **argv)
{
    if (argc < 2)
    {
        fprintf(stderr, "Usage: @streamcat <stream>\n");
        return 1;
    }
    const char *sname = argv[1];
    while (*sname == '@')
    {
        sname++;
    }
    if (strncmp(sname, "s.", 2) == 0)
    {
        sname += 2;
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
    fflush(stdout);
    if (need_close)
    {
        ImageStreamIO_closeIm(&tmp_im);
    }
    return 0;
}

/**
 * @brief Built-in `@streamwrite`: write elements into a stream from `argv[2..]` or `stdin`.
 *
 * @param argc Argument count
 * @param argv Argument vector (`argv[1]` = stream name, optional `argv[2..]` = values)
 * @return 0 on success, 1 on error
 */
int milk_server_streamwrite(int argc, char **argv)
{
    if (argc < 2)
    {
        fprintf(stderr, "Usage: @streamwrite <stream>\n");
        return 1;
    }
    const char *sname = argv[1];
    while (*sname == '@')
    {
        sname++;
    }
    if (strncmp(sname, "s.", 2) == 0)
    {
        sname += 2;
    }
    IMAGE  tmp_im;
    int    need_close = 0;
    IMAGE *im         = milk_shm_prop_find_or_open_stream(sname, &tmp_im, &need_close);
    if (im == NULL || im->md == NULL)
    {
        fprintf(stderr, "ERROR: @streamwrite: stream '%s' not found\n", sname);
        return 1;
    }
    uint64_t nel  = im->md->nelement;
    uint8_t  dt   = im->md->datatype;
    im->md->write = 1;
    uint64_t idx  = 0;

    if (argc >= 3)
    {
        for (int a = 2; a < argc && idx < nel; a++)
        {
            const char *p = argv[a];
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
        double val = 0.0;
        for (; idx < nel; idx++)
        {
            if (scanf("%lf", &val) != 1)
            {
                break;
            }
            write_typed_elem(im, dt, idx, val);
        }
    }
    processinfo_update_output_stream(NULL, im, NULL);
    if (need_close)
    {
        ImageStreamIO_closeIm(&tmp_im);
    }
    return 0;
}
