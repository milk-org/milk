// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milk_shm_prop_stream.c
 * @brief Stream and in-memory image handler (`@s.` / `@stream.`) for entity ops.
 *
 * Role & Goal:
 *   Implements metadata inspection (`xsize`, `ysize`, `zsize`, `naxis`,
 *   `dtype`, `cnt0`, `cnt1`, `nelem`, `sem`, `pid`, `shared`, `NBkw`),
 *   FITS keyword reads/writes (`kw.<KEY>` and `kw.*`), and existence checks
 *   for both local CLI images (`data.core.image[]`) and `/dev/shm/*.im.shm`
 *   ImageStreamIO shared-memory streams.
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "CLIcore.h"
#include "ImageStreamIO/ImageStreamIO.h"
#include "milk_shm_prop.h"
#include "milk_shm_cache.h"
#include "milk_shm_prop_internal.h"
#include "libmilkdata/milk_type_dispatch.h"

/**
 * @brief Locate an active local image or open a shared-memory stream.
 *
 * @param sname      Stream or image name
 * @param tmp_img    Caller-supplied temporary IMAGE struct for SHM open
 * @param need_close Set to 1 if @tmp_img was opened and must be closed
 * @return Pointer to IMAGE struct, or NULL if not found
 */
IMAGE *milk_shm_prop_find_or_open_stream(const char *sname, IMAGE *tmp_img, int *need_close)
{
    *need_close = 0;
    if (MILK_LOCAL_IMAGES != NULL)
    {
        for (long i = 0; i < MILK_NB_LOCAL_IMAGES; i++)
        {
            if (MILK_LOCAL_IMAGES[i].used == 1 && strcmp(MILK_LOCAL_IMAGES[i].name, sname) == 0)
            {
                return &MILK_LOCAL_IMAGES[i];
            }
        }
    }

    char        shmpath[STRINGMAXLEN_DIRNAME + 256];
    const char *shmdir = milk_shm_prop_shmdir();
    snprintf(shmpath, sizeof(shmpath), "%s/%s.im.shm", shmdir, sname);
    if (access(shmpath, F_OK) != 0)
    {
        return NULL;
    }

    memset(tmp_img, 0, sizeof(*tmp_img));
    if (ImageStreamIO_openIm(tmp_img, sname) == IMAGESTREAMIO_SUCCESS && tmp_img->md != NULL)
    {
        *need_close = 1;
        return tmp_img;
    }
    return NULL;
}

/**
 * @brief Locate an active local image, or the cached mapping of a shared-memory stream.
 *
 * For short accesses under the command lock (or in the single-threaded client): the pointer is not
 * to be kept across another lookup. Long-lived users take their own mapping with
 * milk_shm_prop_find_or_open_stream().
 *
 * @param sname Stream or image name
 * @return Pointer to IMAGE struct, or NULL if not found
 */
IMAGE *milk_shm_prop_find_stream(const char *sname)
{
    if (MILK_LOCAL_IMAGES != NULL)
    {
        for (long i = 0; i < MILK_NB_LOCAL_IMAGES; i++)
        {
            if (MILK_LOCAL_IMAGES[i].used == 1 && strcmp(MILK_LOCAL_IMAGES[i].name, sname) == 0)
            {
                return &MILK_LOCAL_IMAGES[i];
            }
        }
    }
    return milk_shm_cache_stream(sname);
}

/**
 * @brief Map an ImageStreamIO datatype code to its uppercase symbolic name.
 *
 * @param dtype ImageStreamIO `_DATATYPE_*` code
 * @return Constant string such as "FLOAT", "UINT16", or "UNKNOWN"
 */
static const char *stream_dtype_name(uint8_t dtype)
{
    switch (dtype)
    {
    case _DATATYPE_UINT8:
        return "UINT8";
    case _DATATYPE_INT8:
        return "INT8";
    case _DATATYPE_UINT16:
        return "UINT16";
    case _DATATYPE_INT16:
        return "INT16";
    case _DATATYPE_UINT32:
        return "UINT32";
    case _DATATYPE_INT32:
        return "INT32";
    case _DATATYPE_UINT64:
        return "UINT64";
    case _DATATYPE_INT64:
        return "INT64";
    case _DATATYPE_HALF:
        return "HALF";
    case _DATATYPE_FLOAT:
        return "FLOAT";
    case _DATATYPE_DOUBLE:
        return "DOUBLE";
    case _DATATYPE_COMPLEX_FLOAT:
        return "COMPLEX_FLOAT";
    case _DATATYPE_COMPLEX_DOUBLE:
        return "COMPLEX_DOUBLE";
    default:
        return "UNKNOWN";
    }
}

/**
 * @brief Read a FITS keyword (`kw.<KEY>` or `kw.*`) from an open IMAGE.
 *
 * @param im    Pointer to open IMAGE
 * @param kname Keyword name (or "*" to list all keyword names)
 * @param out   Output buffer
 * @param outsz Size of @out in bytes
 * @return 1 if found, 0 if not found
 */
static int read_stream_keyword(const IMAGE *im, const char *kname, char *out, size_t outsz)
{
    if (im->kw == NULL)
    {
        return 0;
    }
    if (strcmp(kname, "*") == 0)
    {
        size_t pos = 0;
        for (uint16_t k = 0; k < im->md->NBkw; k++)
        {
            if (im->kw[k].type != 'N' && im->kw[k].type != '\0' && im->kw[k].name[0] != '\0')
            {
                int n =
                    snprintf(out + pos, outsz - pos, "%s%s", (pos > 0) ? " " : "", im->kw[k].name);
                if (n > 0 && pos + (size_t) n < outsz)
                {
                    pos += (size_t) n;
                }
            }
        }
        return 1;
    }

    if (strcmp(kname, "kv") == 0)
    {
        size_t pos = 0;
        for (uint16_t k = 0; k < im->md->NBkw; k++)
        {
            if (im->kw[k].type != 'N' && im->kw[k].type != '\0' && im->kw[k].name[0] != '\0')
            {
                char vbuf[128];
                if (im->kw[k].type == 'L')
                {
                    snprintf(vbuf, sizeof(vbuf), "%ld", (long) im->kw[k].value.numl);
                }
                else if (im->kw[k].type == 'D')
                {
                    milk_shm_prop_fmt_double(vbuf, sizeof(vbuf), im->kw[k].value.numf);
                }
                else
                {
                    snprintf(vbuf, sizeof(vbuf), "%s", im->kw[k].value.valstr);
                }
                int n = snprintf(out + pos, outsz - pos, "%s%s\t%s", (pos > 0) ? "\n" : "",
                                 im->kw[k].name, vbuf);
                if (n > 0 && pos + (size_t) n < outsz)
                {
                    pos += (size_t) n;
                }
            }
        }
        return 1;
    }

    for (uint16_t k = 0; k < im->md->NBkw; k++)
    {
        if (im->kw[k].type != 'N' && im->kw[k].type != '\0' && strcmp(im->kw[k].name, kname) == 0)
        {
            if (im->kw[k].type == 'L')
            {
                snprintf(out, outsz, "%ld", (long) im->kw[k].value.numl);
            }
            else if (im->kw[k].type == 'D')
            {
                milk_shm_prop_fmt_double(out, outsz, im->kw[k].value.numf);
            }
            else
            {
                snprintf(out, outsz, "%s", im->kw[k].value.valstr);
            }
            return 1;
        }
    }
    return 0;
}

/* Statistics of all the pixels of a stream, as the calculator computes them (imin, imean, istd, ...) */
typedef struct
{
    double min, max, tot, mean, var;
    double count; /* non-zero pixels, as icount() */
} stream_stats_t;

/** stream_stats() - fill @st from the pixels of @im (two passes, like arith_image_rms); -1 if complex. */
static int stream_stats(const IMAGE *im, stream_stats_t *st)
{
    uint64_t n  = im->md->nelement;
    uint8_t  dt = im->md->datatype;
    double   s2 = 0.0;

    memset(st, 0, sizeof(*st));
    if (n == 0)
    {
        return 0;
    }
    st->min = INFINITY;
    st->max = -INFINITY;
#define STATS_PASS1(MBR)                       \
    for (uint64_t i = 0; i < n; i++)           \
    {                                          \
        double v = (double) im->array.MBR[i];  \
        st->tot += v;                          \
        st->min = (v < st->min) ? v : st->min; \
        st->max = (v > st->max) ? v : st->max; \
        st->count += (v != 0.0);               \
    }
#define STATS_PASS2(MBR)                                 \
    for (uint64_t i = 0; i < n; i++)                     \
    {                                                    \
        double d = (double) im->array.MBR[i] - st->mean; \
        s2 += d * d;                                     \
    }
    if (dt < _DATATYPE_UINT8 || dt > _DATATYPE_DOUBLE) /* complex, half: not dispatched below */
    {
        return -1;
    }
    MILK_FOR_EACH_DATATYPE(dt, STATS_PASS1, STATS_PASS1(D))
    st->mean = st->tot / (double) n;
    MILK_FOR_EACH_DATATYPE(dt, STATS_PASS2, STATS_PASS2(D))
#undef STATS_PASS1
#undef STATS_PASS2
    st->var = s2 / (double) n;
    return 0;
}

/** stream_stat_prop() - write statistic @prop of @im ("min" ... "count", or "stats": all, `key<TAB>value`
 * lines); 1 if @prop is one, -1 if it cannot be computed (complex data), else 0. */
static int stream_stat_prop(const IMAGE *im, const char *prop, char *out, size_t outsz)
{
    static const char *const names[] = { "min", "max", "mean", "std", "var", "tot", "count", NULL };
    stream_stats_t           st;
    double                   v[7];
    int                      k;

    for (k = 0; names[k] != NULL && strcmp(prop, names[k]) != 0; k++)
    {
    }
    if (names[k] == NULL && strcmp(prop, "stats") != 0)
    {
        return 0;
    }
    if (stream_stats(im, &st) != 0)
    {
        return -1;
    }
    v[0] = st.min;
    v[1] = st.max;
    v[2] = st.mean;
    v[3] = sqrt(st.var);
    v[4] = st.var;
    v[5] = st.tot;
    v[6] = st.count;
    if (names[k] != NULL)
    {
        milk_shm_prop_fmt_double(out, outsz, v[k]);
        return 1;
    }
    size_t pos = 0;
    for (int i = 0; names[i] != NULL && pos < outsz; i++)
    {
        char num[64];
        milk_shm_prop_fmt_double(num, sizeof(num), v[i]);
        pos += (size_t) snprintf(out + pos, outsz - pos, "%s%s\t%s", (i > 0) ? "\n" : "", names[i],
                                 num);
    }
    return 1;
}

/**
 * @brief Read a stream or local image property into @out.
 *
 * @param name  Stream/image name
 * @param prop  Property name (e.g. "xsize", "dtype", "cnt0", "kw.EXPTIME")
 * @param out   Output buffer
 * @param outsz Size of @out in bytes
 * @return 0 on success, -1 if stream or property not found
 */
int milk_shm_prop_get_stream(const char *name, const char *prop, char *out, size_t outsz)
{
    IMAGE *im = milk_shm_prop_find_stream(name); /* cached mapping: never closed here */
    if (im == NULL || im->md == NULL)
    {
        return -1;
    }

    int found = stream_stat_prop(im, prop, out, outsz);
    if (found != 0)
    {
        return (found > 0) ? 0 : -1;
    }
    found = 1;
    if (strcmp(prop, "xsize") == 0 || strcmp(prop, "size0") == 0)
    {
        snprintf(out, outsz, "%u", im->md->size[0]);
    }
    else if (strcmp(prop, "ysize") == 0 || strcmp(prop, "size1") == 0)
    {
        snprintf(out, outsz, "%u",
                 (im->md->naxis > 1 && im->md->size[1] > 0) ? im->md->size[1] : 1U);
    }
    else if (strcmp(prop, "zsize") == 0 || strcmp(prop, "size2") == 0)
    {
        snprintf(out, outsz, "%u",
                 (im->md->naxis > 2 && im->md->size[2] > 0) ? im->md->size[2] : 1U);
    }
    else if (strcmp(prop, "naxis") == 0)
    {
        snprintf(out, outsz, "%u", (unsigned) im->md->naxis);
    }
    else if (strcmp(prop, "type") == 0 || strcmp(prop, "typeid") == 0)
    {
        snprintf(out, outsz, "%u", (unsigned) im->md->datatype);
    }
    else if (strcmp(prop, "typename") == 0)
    {
        const char *tn = ImageStreamIO_typename(im->md->datatype);
        snprintf(out, outsz, "%s", tn ? tn : "UNKNOWN");
    }
    else if (strcmp(prop, "dtype") == 0)
    {
        snprintf(out, outsz, "%s", stream_dtype_name(im->md->datatype));
    }
    else if (strcmp(prop, "cnt0") == 0)
    {
        snprintf(out, outsz, "%lu", (unsigned long) im->md->cnt0);
    }
    else if (strcmp(prop, "cnt1") == 0)
    {
        snprintf(out, outsz, "%lu", (unsigned long) im->md->cnt1);
    }
    else if (strcmp(prop, "sem") == 0)
    {
        snprintf(out, outsz, "%u", (unsigned) im->md->sem);
    }
    else if (strcmp(prop, "pid") == 0)
    {
        snprintf(out, outsz, "%d", (int) im->md->creatorPID);
    }
    else if (strcmp(prop, "ownerPID") == 0 || strcmp(prop, "owner_pid") == 0)
    {
        snprintf(out, outsz, "%d", (int) im->md->ownerPID);
    }
    else if (strcmp(prop, "nelement") == 0 || strcmp(prop, "nelem") == 0)
    {
        snprintf(out, outsz, "%lu", (unsigned long) im->md->nelement);
    }
    else if (strcmp(prop, "shared") == 0)
    {
        snprintf(out, outsz, "%u", (unsigned) im->md->shared);
    }
    else if (strcmp(prop, "NBkw") == 0)
    {
        snprintf(out, outsz, "%u", (unsigned) im->md->NBkw);
    }
    else if (strcmp(prop, "location") == 0)
    {
        snprintf(out, outsz, "%d", (int) im->md->location);
    }
    else if (strcmp(prop, "loc") == 0 || strcmp(prop, "storage") == 0)
    {
        if (im->md->location >= 0)
        {
            snprintf(out, outsz, "GPU%d", (int) im->md->location);
        }
        else
        {
            snprintf(out, outsz, "%s", im->md->shared ? "SHM" : "RAM");
        }
    }
    else if (strcmp(prop, "inode") == 0)
    {
        snprintf(out, outsz, "%lu", (unsigned long) im->md->inode);
    }
    else if (strcmp(prop, "age") == 0)
    {
        struct timespec now;
        clock_gettime(CLOCK_REALTIME, &now);
        struct timespec ref =
            (im->md->writetime.tv_sec > 0) ? im->md->writetime : im->md->creationtime;
        double age = 0.0;
        if (ref.tv_sec > 0)
        {
            age =
                (double) (now.tv_sec - ref.tv_sec) + 1.0e-9 * (double) (now.tv_nsec - ref.tv_nsec);
            if (age < 0.0)
            {
                age = 0.0;
            }
        }
        snprintf(out, outsz, "%.3f", age);
    }
    else if (strcmp(prop, "fps") == 0)
    {
        double fps_val = 0.0;
        if (im->md->writetime.tv_sec > 0 && im->md->creationtime.tv_sec > 0 && im->md->cnt0 > 0)
        {
            double dt =
                (double) (im->md->writetime.tv_sec - im->md->creationtime.tv_sec) +
                1.0e-9 * (double) (im->md->writetime.tv_nsec - im->md->creationtime.tv_nsec);
            if (dt > 1.0e-6)
            {
                fps_val = (double) im->md->cnt0 / dt;
            }
        }
        snprintf(out, outsz, "%.1f", fps_val);
    }
    else if (strcmp(prop, "summary") == 0)
    {
        if (im->md->naxis <= 1)
        {
            snprintf(out, outsz, "%s [%u] %s cnt0=%lu", name, im->md->size[0],
                     stream_dtype_name(im->md->datatype), (unsigned long) im->md->cnt0);
        }
        else if (im->md->naxis == 2)
        {
            snprintf(out, outsz, "%s [%ux%u] %s cnt0=%lu", name, im->md->size[0], im->md->size[1],
                     stream_dtype_name(im->md->datatype), (unsigned long) im->md->cnt0);
        }
        else
        {
            snprintf(out, outsz, "%s [%ux%ux%u] %s cnt0=%lu", name, im->md->size[0],
                     im->md->size[1], im->md->size[2], stream_dtype_name(im->md->datatype),
                     (unsigned long) im->md->cnt0);
        }
    }
    else if (strcmp(prop, "list") == 0 || strcmp(prop, "*") == 0)
    {
        snprintf(out, outsz,
                 "xsize ysize zsize naxis dtype typename type cnt0 cnt1 nelem sem pid ownerPID "
                 "shared NBkw location inode age fps summary min max mean std var tot count stats");
    }
    else if (strcmp(prop, "kv") == 0)
    {
        uint32_t    ysize = (im->md->naxis > 1 && im->md->size[1] > 0) ? im->md->size[1] : 1U;
        uint32_t    zsize = (im->md->naxis > 2 && im->md->size[2] > 0) ? im->md->size[2] : 1U;
        const char *tn    = ImageStreamIO_typename(im->md->datatype);
        const char *dt    = stream_dtype_name(im->md->datatype);

        struct timespec now;
        clock_gettime(CLOCK_REALTIME, &now);
        struct timespec ref =
            (im->md->writetime.tv_sec > 0) ? im->md->writetime : im->md->creationtime;
        double age = 0.0;
        if (ref.tv_sec > 0)
        {
            age =
                (double) (now.tv_sec - ref.tv_sec) + 1.0e-9 * (double) (now.tv_nsec - ref.tv_nsec);
            if (age < 0.0)
            {
                age = 0.0;
            }
        }
        double fps_val = 0.0;
        if (im->md->writetime.tv_sec > 0 && im->md->creationtime.tv_sec > 0 && im->md->cnt0 > 0)
        {
            double dts =
                (double) (im->md->writetime.tv_sec - im->md->creationtime.tv_sec) +
                1.0e-9 * (double) (im->md->writetime.tv_nsec - im->md->creationtime.tv_nsec);
            if (dts > 1.0e-6)
            {
                fps_val = (double) im->md->cnt0 / dts;
            }
        }
        char sum_buf[256];
        if (im->md->naxis <= 1)
        {
            snprintf(sum_buf, sizeof(sum_buf), "%s [%u] %s cnt0=%lu", name, im->md->size[0], dt,
                     (unsigned long) im->md->cnt0);
        }
        else if (im->md->naxis == 2)
        {
            snprintf(sum_buf, sizeof(sum_buf), "%s [%ux%u] %s cnt0=%lu", name, im->md->size[0],
                     im->md->size[1], dt, (unsigned long) im->md->cnt0);
        }
        else
        {
            snprintf(sum_buf, sizeof(sum_buf), "%s [%ux%ux%u] %s cnt0=%lu", name, im->md->size[0],
                     im->md->size[1], im->md->size[2], dt, (unsigned long) im->md->cnt0);
        }

        snprintf(out, outsz,
                 "xsize\t%u\nysize\t%u\nzsize\t%u\nnaxis\t%u\nnelement\t%lu\ndtype\t%s\n"
                 "typename\t%s\ntype\t%u\ncnt0\t%lu\ncnt1\t%lu\nsem\t%u\npid\t%d\n"
                 "owner_pid\t%d\nshared\t%u\nNBkw\t%u\nlocation\t%d\ninode\t%lu\n"
                 "age\t%.3f\nfps\t%.1f\nsummary\t%s",
                 im->md->size[0], ysize, zsize, (unsigned) im->md->naxis,
                 (unsigned long) im->md->nelement, dt, tn ? tn : "UNKNOWN",
                 (unsigned) im->md->datatype, (unsigned long) im->md->cnt0,
                 (unsigned long) im->md->cnt1, (unsigned) im->md->sem, (int) im->md->creatorPID,
                 (int) im->md->ownerPID, (unsigned) im->md->shared, (unsigned) im->md->NBkw,
                 (int) im->md->location, (unsigned long) im->md->inode, age, fps_val, sum_buf);
    }
    else if (strncmp(prop, "kw.", 3) == 0)
    {
        found = read_stream_keyword(im, prop + 3, out, outsz);
    }
    else
    {
        found = 0;
    }

    return found ? 0 : -1;
}

/**
 * @brief Write a FITS keyword (`kw.<KEY>`) on a stream or local image.
 *
 * Automatically infers keyword type: integer ('L'), floating-point ('D'),
 * or string ('S').
 *
 * @param name     Stream/image name
 * @param prop     Property path (must start with "kw.")
 * @param val      Value string to write
 * @param comment  Optional FITS keyword comment (may be NULL)
 * @return 0 on success, -1 on error
 */
int milk_shm_prop_set_stream(const char *name,
                             const char *prop,
                             const char *val,
                             const char *comment)
{
    if (strncmp(prop, "kw.", 3) != 0)
    {
        fprintf(stderr, "ERROR: only s.<name>.kw.<KEY> is writable on streams\n");
        return -1;
    }
    if (prop[3] == '\0')
    {
        return -1;
    }
    char kname[KEYWORD_MAX_STRING]; // stored width: look up by the name as it will be stored
    snprintf(kname, sizeof(kname), "%s", prop + 3);
    if (strlen(prop + 3) >= sizeof(kname))
    {
        fprintf(stderr, "WARNING: keyword name '%s' truncated to '%s'\n", prop + 3, kname);
    }

    IMAGE *im = milk_shm_prop_find_stream(name); /* cached mapping: never closed here */
    if (im == NULL || im->md == NULL || im->kw == NULL)
    {
        fprintf(stderr, "ERROR: stream '%s' not found or has no keyword table\n", name);
        return -1;
    }

    int slot = -1;
    for (uint16_t k = 0; k < im->md->NBkw; k++)
    {
        if (im->kw[k].type != 'N' && im->kw[k].type != '\0' && strcmp(im->kw[k].name, kname) == 0)
        {
            slot = (int) k;
            break;
        }
    }
    if (slot == -1)
    {
        for (uint16_t k = 0; k < im->md->NBkw; k++)
        {
            if (im->kw[k].type == 'N' || im->kw[k].type == '\0' || im->kw[k].name[0] == '\0')
            {
                slot = (int) k;
                break;
            }
        }
    }
    if (slot == -1)
    {
        fprintf(stderr, "ERROR: keyword table full on stream '%s'\n", name);
        return -1;
    }

    snprintf(im->kw[slot].name, KEYWORD_MAX_STRING, "%s", kname);
    char     *endp = NULL;
    long long lval = strtoll(val, &endp, 10);
    if (val[0] != '\0' && endp != val && *endp == '\0')
    {
        im->kw[slot].type       = 'L';
        im->kw[slot].value.numl = (int64_t) lval;
    }
    else
    {
        double dval = strtod(val, &endp);
        if (val[0] != '\0' && endp != val && *endp == '\0')
        {
            im->kw[slot].type       = 'D';
            im->kw[slot].value.numf = dval;
        }
        else
        {
            im->kw[slot].type = 'S';
            snprintf(im->kw[slot].value.valstr, KEYWORD_MAX_STRING, "%s", val);
        }
    }
    if (comment != NULL && comment[0] != '\0')
    {
        snprintf(im->kw[slot].comment, KEYWORD_MAX_COMMENT, "%s", comment);
    }
    im->kw[slot].cnt++;

    return 0;
}

/**
 * @brief Check whether a local image or shared-memory stream exists.
 *
 * @param name Stream or image name
 * @return 1 if present, 0 otherwise
 */
int milk_shm_prop_exists_stream(const char *name)
{
    if (MILK_LOCAL_IMAGES != NULL)
    {
        for (long i = 0; i < MILK_NB_LOCAL_IMAGES; i++)
        {
            if (MILK_LOCAL_IMAGES[i].used == 1 && strcmp(MILK_LOCAL_IMAGES[i].name, name) == 0)
            {
                return 1;
            }
        }
    }
    const char *shmdir = milk_shm_prop_shmdir();
    char        shmpath[STRINGMAXLEN_DIRNAME + 256];
    snprintf(shmpath, sizeof(shmpath), "%s/%s.im.shm", shmdir, name);
    return (access(shmpath, F_OK) == 0) ? 1 : 0;
}

/**
 * milk_shm_prop_exists_local_stream - Check whether a process-local non-shared image exists
 * @name: Local image name to look up in `MILK_LOCAL_IMAGES`
 *
 * Scans the process-local image table for an active entry matching @name that is
 * not backed by shared memory (`md->shared == 0`).
 *
 * Return: 1 if a matching local image exists, 0 otherwise.
 */
int milk_shm_prop_exists_local_stream(const char *name)
{
    if (MILK_LOCAL_IMAGES == NULL)
    {
        return 0;
    }
    for (long i = 0; i < MILK_NB_LOCAL_IMAGES; i++)
    {
        if (MILK_LOCAL_IMAGES[i].used == 1 && strcmp(MILK_LOCAL_IMAGES[i].name, name) == 0 &&
            (MILK_LOCAL_IMAGES[i].md == NULL || !MILK_LOCAL_IMAGES[i].md->shared))
        {
            return 1;
        }
    }
    return 0;
}

/**
 * milk_shm_prop_get_local_stream - Read a metadata property or keyword from a local image
 * @name:  Local image name (`@l.<name>.<prop>`)
 * @prop:  Property key or `kw.<FITSKEY>` specifier
 * @out:   Output buffer receiving the formatted property value
 * @outsz: Capacity of @out in bytes
 *
 * Verifies that @name is a process-local (non-shared) image before delegating property
 * extraction to milk_shm_prop_get_stream().
 *
 * Return: 0 on success, -1 if @name is not a local image or @prop is not found.
 */
int milk_shm_prop_get_local_stream(const char *name, const char *prop, char *out, size_t outsz)
{
    if (!milk_shm_prop_exists_local_stream(name))
    {
        return -1;
    }
    return milk_shm_prop_get_stream(name, prop, out, outsz);
}

/**
 * milk_shm_prop_set_local_stream - Write a FITS keyword on a process-local image
 * @name:    Local image name (`@l.<name>.kw.<FITSKEY>`)
 * @prop:    Property path (must start with `kw.`)
 * @val:     Value string to assign
 * @comment: Optional FITS keyword comment (may be NULL)
 *
 * Verifies that @name exists as a process-local (non-shared) image before delegating
 * the keyword update to milk_shm_prop_set_stream().
 *
 * Return: 0 on success, -1 if @name is not a local image or the write fails.
 */
int milk_shm_prop_set_local_stream(const char *name,
                                   const char *prop,
                                   const char *val,
                                   const char *comment)
{
    if (!milk_shm_prop_exists_local_stream(name))
    {
        return -1;
    }
    return milk_shm_prop_set_stream(name, prop, val, comment);
}
