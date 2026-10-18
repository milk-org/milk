// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file tv_stream.c
 * @brief Read-only stream access: validate the header before mapping, never write to the stream.
 */

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "ImageStreamIO/ImageStreamIO.h"
#include "ImageStreamIO/ImageStreamIO_config.h"
#include "libmilkcommon/pixel_dispatch.h"
#include "termview_internal.h"

/* Exported by libImageStreamIO but not declared in its header; only reads image->md. */
errno_t ImageStreamIO_image_sizing(IMAGE *image, uint8_t *map);

static int tv_dtype_supported(uint8_t dt)
{
#define TV_DT_CASE(DT, ACC, CTYPE) \
    case DT:                       \
        return 1;
    switch (dt)
    {
        FOREACH_REAL_DATATYPE(TV_DT_CASE)
    default:
        return 0;
    }
#undef TV_DT_CASE
}

/** tv_stream_path() - Accept "name", "name.im.shm" or a path containing '/'. */
static int tv_stream_path(const char *arg, char *path, size_t sz)
{
    if (strchr(arg, '/') != NULL)
    {
        return (snprintf(path, sz, "%s", arg) < (int) sz) ? 0 : -1;
    }

    char        name[STRINGMAXLEN_IMAGE_NAME];
    size_t      len = strlen(arg);
    const char *ext = ".im.shm";
    size_t      el  = strlen(ext);
    if (len > el && strcmp(arg + len - el, ext) == 0)
    {
        len -= el;
    }
    if (len == 0 || len >= sizeof(name))
    {
        return -1;
    }
    memcpy(name, arg, len);
    name[len] = '\0';
    return (ImageStreamIO_filename(path, sz, name) == IMAGESTREAMIO_SUCCESS) ? 0 : -1;
}

static int tv_stream_check_md(const IMAGE_METADATA *md, char *err, size_t errsz)
{
    if (strncmp(md->version, IMAGESTRUCT_VERSION, sizeof(md->version)) != 0)
    {
        snprintf(err, errsz, "incompatible stream version (expected %s)", IMAGESTRUCT_VERSION);
        return -1;
    }
    if (md->shared != 1)
    {
        snprintf(err, errsz, "not a shared-memory stream");
        return -1;
    }
    if (md->location != -1)
    {
        snprintf(err, errsz, "GPU-resident stream (location %d) cannot be viewed", md->location);
        return -1;
    }
    if (md->naxis < 1 || md->naxis > 3)
    {
        snprintf(err, errsz, "unsupported number of axes (%d)", md->naxis);
        return -1;
    }
    uint64_t nelem = 1;
    for (int i = 0; i < md->naxis; i++)
    {
        if (md->size[i] < 1 || __builtin_mul_overflow(nelem, (uint64_t) md->size[i], &nelem))
        {
            snprintf(err, errsz, "invalid size along axis %d", i);
            return -1;
        }
    }
    if (nelem != md->nelement)
    {
        snprintf(err, errsz, "inconsistent element count");
        return -1;
    }
    if (!tv_dtype_supported(md->datatype))
    {
        snprintf(err, errsz, "unsupported datatype %d (real types only)", md->datatype);
        return -1;
    }
    uint64_t bytes;
    int      ts = ImageStreamIO_typesize(md->datatype);
    if (ts <= 0 || __builtin_mul_overflow(nelem, (uint64_t) ts, &bytes) ||
        bytes > md->imdatamemsize)
    {
        snprintf(err, errsz, "inconsistent data size");
        return -1;
    }
    return 0;
}

int tv_stream_open(tv_stream_t *s, const char *arg, char *err, size_t errsz)
{
    memset(s, 0, sizeof(*s));
    s->fd        = -1;
    s->img.shmfd = -1;

    if (tv_stream_path(arg, s->path, sizeof(s->path)) != 0)
    {
        snprintf(err, errsz, "invalid stream name '%s'", arg);
        return -1;
    }

    struct stat st;
    if (stat(s->path, &st) != 0)
    {
        snprintf(err, errsz, "stream '%s' not found (%s)", arg, s->path);
        return -1;
    }
    if (!S_ISREG(st.st_mode))
    {
        snprintf(err, errsz, "%s is not a regular file", s->path);
        return -1;
    }

    int fd = open(s->path, O_RDONLY | O_CLOEXEC);
    if (fd < 0)
    {
        snprintf(err, errsz, "cannot open %s: %s", s->path, strerror(errno));
        return -1;
    }

    IMAGE_METADATA md;
    if (fstat(fd, &st) != 0 || st.st_size <= (off_t) sizeof(md) ||
        pread(fd, &md, sizeof(md), 0) != (ssize_t) sizeof(md))
    {
        snprintf(err, errsz, "%s is too small to be a milk stream", s->path);
        close(fd);
        return -1;
    }
    if (tv_stream_check_md(&md, err, errsz) != 0)
    {
        close(fd);
        return -1;
    }

    void *map = mmap(NULL, (size_t) st.st_size, PROT_READ, MAP_SHARED, fd, 0);
    if (map == MAP_FAILED)
    {
        snprintf(err, errsz, "cannot map %s: %s", s->path, strerror(errno));
        close(fd);
        return -1;
    }
    s->img.md = (IMAGE_METADATA *) map;
    if (ImageStreamIO_image_sizing(&s->img, (uint8_t *) map) != IMAGESTREAMIO_SUCCESS ||
        s->img.memsize != (uint64_t) st.st_size || s->img.array.raw == NULL)
    {
        snprintf(err, errsz, "unexpected layout in %s", s->path);
        munmap(map, (size_t) st.st_size);
        close(fd);
        memset(&s->img, 0, sizeof(s->img));
        return -1;
    }

    s->fd       = fd;
    s->map      = map;
    s->mapsize  = (size_t) st.st_size;
    s->dev      = st.st_dev;
    s->ino      = st.st_ino;
    s->naxis    = md.naxis;
    s->datatype = md.datatype;
    s->xsize    = md.size[0];
    s->ysize    = (md.naxis >= 2) ? md.size[1] : 1;
    s->zsize    = (md.naxis == 3) ? md.size[2] : 1;

    size_t i = 0;
    for (; i < sizeof(s->name) - 1 && i < sizeof(md.name) && md.name[i] != '\0'; i++)
    {
        unsigned char c = (unsigned char) md.name[i];
        s->name[i]      = (c >= 0x20 && c < 0x7F) ? (char) c : '?';
    }
    s->name[i] = '\0';
    return 0;
}

void tv_stream_close(tv_stream_t *s)
{
    if (s->map != NULL)
    {
        munmap(s->map, s->mapsize);
    }
    if (s->fd >= 0)
    {
        close(s->fd);
    }
    s->map           = NULL;
    s->fd            = -1;
    s->img.md        = NULL;
    s->img.array.raw = NULL;
}
