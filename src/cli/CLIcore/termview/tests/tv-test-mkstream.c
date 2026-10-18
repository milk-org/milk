// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file tv-test-mkstream.c
 * @brief Test-only stream generator for the milk-termview tests (not installed).
 *
 * Refuses to run unless MILK_SHM_DIR is set to an existing directory that the stream path resolves
 * under, and the stream name starts with "tvtest", so it can never replace a production stream.
 */

#include <math.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include "ImageStreamIO/ImageStreamIO.h"

static volatile sig_atomic_t stop_flag = 0;

static void on_signal(int sig)
{
    (void) sig;
    stop_flag = 1;
}

static void usage(void)
{
    fprintf(stderr,
            "usage: tv-test-mkstream -n NAME [-t u8|i16|u16|i32|f32|f64] [-s WxH[xZ]]\n"
            "         [-p ramp|noise|nan|inf|const] [-r RATE_HZ] [-d SECONDS] [-x]\n"
            "  -r 0 writes one frame; -d 0 runs until SIGTERM; -x deletes the stream at exit\n");
}

static int parse_type(const char *s, uint8_t *dt)
{
    static const struct
    {
        const char *name;
        uint8_t     dt;
    } types[] = {
        { "u8", _DATATYPE_UINT8 },  { "i16", _DATATYPE_INT16 }, { "u16", _DATATYPE_UINT16 },
        { "i32", _DATATYPE_INT32 }, { "f32", _DATATYPE_FLOAT }, { "f64", _DATATYPE_DOUBLE },
    };
    for (size_t i = 0; i < sizeof(types) / sizeof(types[0]); i++)
    {
        if (strcmp(s, types[i].name) == 0)
        {
            *dt = types[i].dt;
            return 0;
        }
    }
    return -1;
}

static uint64_t xorshift(uint64_t *s)
{
    uint64_t x = *s;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    return *s = x;
}

/** pixel_value() - Value of pixel @i (of @n) in frame @frame for @pattern. */
static double pixel_value(const char *pattern,
                          uint64_t    i,
                          uint64_t    n,
                          uint64_t    frame,
                          uint64_t   *rng)
{
    double ramp = (double) ((i + frame) % n) / (double) n * 1000.0;
    if (strcmp(pattern, "noise") == 0)
    {
        return (double) (xorshift(rng) >> 11) * (1000.0 / 9007199254740992.0);
    }
    if (strcmp(pattern, "const") == 0)
    {
        return 42.0;
    }
    if (strcmp(pattern, "nan") == 0 && i % 7 == 0)
    {
        return NAN;
    }
    if (strcmp(pattern, "inf") == 0 && i % 5 == 0)
    {
        return (i % 10 == 0) ? INFINITY : -INFINITY;
    }
    return ramp;
}

static void fill(IMAGE *img, const char *pattern, uint64_t frame, uint64_t *rng)
{
    uint64_t n = img->md->nelement;
    for (uint64_t i = 0; i < n; i++)
    {
        double v = pixel_value(pattern, i, n, frame, rng);
        switch (img->md->datatype)
        {
        case _DATATYPE_UINT8:
            img->array.UI8[i] = (uint8_t) (isfinite(v) ? fmod(v, 256.0) : 0);
            break;
        case _DATATYPE_INT16:
            img->array.SI16[i] = (int16_t) (isfinite(v) ? v - 500.0 : 0);
            break;
        case _DATATYPE_UINT16:
            img->array.UI16[i] = (uint16_t) (isfinite(v) ? v : 0);
            break;
        case _DATATYPE_INT32:
            img->array.SI32[i] = (int32_t) (isfinite(v) ? v * 1000.0 : 0);
            break;
        case _DATATYPE_FLOAT:
            img->array.F[i] = (float) v;
            break;
        default:
            img->array.D[i] = v;
            break;
        }
    }
}

int main(int argc, char **argv)
{
    const char *name    = NULL;
    const char *pattern = "ramp";
    uint8_t     dt      = _DATATYPE_FLOAT;
    uint32_t    size[3] = { 64, 64, 1 };
    long        naxis   = 2;
    double      rate    = 0.0;
    double      dur     = 0.0;
    int         del     = 0;
    int         c;

    while ((c = getopt(argc, argv, "n:t:s:p:r:d:x")) != -1)
    {
        switch (c)
        {
        case 'n':
            name = optarg;
            break;
        case 't':
            if (parse_type(optarg, &dt) != 0)
            {
                usage();
                return 2;
            }
            break;
        case 's':
        {
            unsigned w = 0, h = 0, z = 0;
            int      k = sscanf(optarg, "%ux%ux%u", &w, &h, &z);
            if (k < 2 || w == 0 || h == 0 || (k == 3 && z == 0))
            {
                usage();
                return 2;
            }
            size[0] = w;
            size[1] = h;
            size[2] = (k == 3) ? z : 1;
            naxis   = (k == 3) ? 3 : 2;
            break;
        }
        case 'p':
            pattern = optarg;
            break;
        case 'r':
            rate = atof(optarg);
            break;
        case 'd':
            dur = atof(optarg);
            break;
        case 'x':
            del = 1;
            break;
        default:
            usage();
            return 2;
        }
    }
    if (name == NULL || strncmp(name, "tvtest", 6) != 0)
    {
        fprintf(stderr, "tv-test-mkstream: the stream name must start with 'tvtest'\n");
        return 2;
    }

    const char *dir = getenv("MILK_SHM_DIR");
    struct stat st;
    if (dir == NULL || stat(dir, &st) != 0 || !S_ISDIR(st.st_mode))
    {
        fprintf(stderr, "tv-test-mkstream: MILK_SHM_DIR must name an existing directory\n");
        return 2;
    }
    char path[STRINGMAXLEN_FILE_NAME];
    if (ImageStreamIO_filename(path, sizeof(path), name) != IMAGESTREAMIO_SUCCESS ||
        strncmp(path, dir, strlen(dir)) != 0)
    {
        fprintf(stderr, "tv-test-mkstream: stream path does not resolve under MILK_SHM_DIR\n");
        return 2;
    }

    signal(SIGTERM, on_signal);
    signal(SIGINT, on_signal);

    IMAGE img;
    memset(&img, 0, sizeof(img));
    if (ImageStreamIO_createIm(&img, name, naxis, size, dt, 1, 0, 0) != IMAGESTREAMIO_SUCCESS)
    {
        fprintf(stderr, "tv-test-mkstream: cannot create %s\n", path);
        return 1;
    }

    uint64_t        rng   = 0x9E3779B97F4A7C15ULL;
    uint64_t        frame = 0;
    struct timespec t0, now;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    do
    {
        img.md->write = 1;
        fill(&img, pattern, frame++, &rng);
        ImageStreamIO_UpdateIm(&img);
        if (rate <= 0.0)
        {
            break;
        }
        struct timespec ts = { .tv_sec = 0, .tv_nsec = (long) (1e9 / rate) };
        if (rate < 1.0)
        {
            ts.tv_sec  = (time_t) (1.0 / rate);
            ts.tv_nsec = 0;
        }
        nanosleep(&ts, NULL);
        clock_gettime(CLOCK_MONOTONIC, &now);
    } while (!stop_flag &&
             (dur <= 0.0 || (now.tv_sec - t0.tv_sec) + (now.tv_nsec - t0.tv_nsec) / 1e9 < dur));

    if (rate <= 0.0 && !del)
    {
        ImageStreamIO_closeIm(&img);
        return 0;
    }
    if (rate <= 0.0)
    {
        while (!stop_flag)
        {
            pause();
        }
    }
    if (del)
    {
        ImageStreamIO_destroyIm(&img);
    }
    else
    {
        ImageStreamIO_closeIm(&img);
    }
    return 0;
}
