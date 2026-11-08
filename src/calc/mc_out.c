// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file mc_out.c
 * @brief Temporary image pool, output image preparation, semaphore batching, and formatting.
 *
 * Manages image allocation, reuse, publication, and CLI result formatting for
 * the `libmilkcalc` execution engine:
 *
 * 1. **Temporary Image Pool (`_tmpcalcpool0..3`)**: Maintains a 4-slot, 256 MiB
 *    pool of local `IMAGE` buffers (`mc_tmp_image`, `mc_tmp_release`) so
 *    per-frame evaluations that need scratch images (aliased in-place clones,
 *    binned slices, compound reduction inputs) avoid repeated `malloc`/`free`
 *    and table scans. Sub-program slots (`_tmpcalcsub*`) stay pinned across
 *    evaluations until `mc_tmp_release_force()` or `mcalc_tmp_cleanup()`.
 * 2. **Output Preparation (`mc_out_prepare`)**: Resolves or creates the target
 *    `IMAGE` for a planned slot, enforcing `@S:` (shared) vs `@L:`/`local`
 *    storage rules, rejecting name collisions with scalar calculator/shell
 *    variables, validating shared stream dimensions, and setting `md[0].write = 1`
 *    prior to pixel writes. Uses `milk_table_generation()` (`checked_gen`) to
 *    bypass hash lookups on repeated loop iterations.
 * 3. **Deferred Stream Posting (`mc_out_post`, `mc_post_defer`, `mc_post_flush`)**:
 *    Allows multi-step programs (or `for` loops) to batch shared-memory stream
 *    updates (`processinfo_update_output_stream`) so downstream consumers are
 *    woken once at the end of a transaction rather than on every intermediate
 *    write.
 * 4. **CLI Output Formatting (`mc_format_image_summary`, `mc_format_pixel_line`)**:
 *    Formats human-readable image summaries (`name [dims] type`) and `-a`
 *    space-separated pixel dumps.
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ImageStreamIO/ImageStreamIO.h"
#include "libmilkcommon/milkDebugTools.h"
#include "libmilkdata/milk_type_dispatch.h"
#include "libmilkdata/milkdata.h"
#include "libprocessinfo/processinfo.h"
#include "COREMOD_memory/create_image.h"
#include "COREMOD_memory/delete_image.h"
#include "COREMOD_memory/image_ID.h"
#include "COREMOD_memory/variable_ID.h"
#include "mc_plan.h"
#include "mcalc.h"

#define MC_POOL_N 4
#define MC_POOL_BYTES ((size_t) 256 << 20)
#define MC_MAX_PIXELS 1048576

/* ========================================================================== */
/* Temporary Image Pool (`_tmpcalcpool0..3`)                                  */
/* ========================================================================== */

static imageID g_pool_ids[MC_POOL_N] = { -1, -1, -1, -1 };
static long    g_tmp_seq             = 1;
static int     g_post_defer          = 0;
static imageID g_post_ids[64];
static int     g_post_n = 0;

/**
 * im_bytes - Compute total pixel buffer size in bytes for an `IMAGE`
 * @im: Pointer to a valid `IMAGE`
 *
 * Return: Byte count (`nelement * typesize(datatype)`)
 */
static size_t im_bytes(const IMAGE *im)
{
    return (size_t) im->md[0].nelement * (size_t) ImageStreamIO_typesize(im->md[0].datatype);
}

/**
 * pool_im - Validate and return the `IMAGE` pointer for pool slot @k
 * @k: Pool slot index (`0 .. MC_POOL_N - 1`)
 *
 * Verifies that `g_pool_ids[k]` still refers to a live image named
 * `_tmpcalcpool<k>` in `milk_data.image[]`, resetting the slot to `-1` if it
 * was externally deleted or overwritten.
 *
 * Return: Pointer to the pooled `IMAGE`, or NULL if slot @k is empty/stale
 */
static IMAGE *pool_im(int k)
{
    char    name[32];
    imageID id = g_pool_ids[k];

    snprintf(name, sizeof(name), "_tmpcalcpool%d", k);
    if (milk_data.image == NULL || id < 0 || id >= milk_data.NB_MAX_IMAGE ||
        !milk_data.image[id].used || strcmp(milk_data.image[id].name, name) != 0)
    {
        g_pool_ids[k] = -1;
        return NULL;
    }
    return &milk_data.image[id];
}

/**
 * pool_put - Attempt to park a local temporary image in the 4-slot reuse pool
 * @id: `imageID` of the temporary image to recycle
 *
 * Rejects shared-memory images or images that would push total pooled memory
 * above `MC_POOL_BYTES` (256 MiB). On success, renames the image in place to
 * `_tmpcalcpool<slot>` so its pixel allocation stays warm.
 *
 * Return: 1 if parked in the pool, 0 if caller should delete @id instead
 */
static int pool_put(imageID id)
{
    if (milk_data.image == NULL || id < 0 || id >= milk_data.NB_MAX_IMAGE ||
        !milk_data.image[id].used)
    {
        return 0;
    }
    IMAGE *im    = &milk_data.image[id];
    size_t total = 0;
    int    slot  = -1;

    if (im->md == NULL || im->md[0].shared != 0 || im->array.raw == NULL)
    {
        return 0;
    }
    for (int k = 0; k < MC_POOL_N; k++)
    {
        IMAGE *pk = pool_im(k);
        if (pk != NULL)
        {
            total += im_bytes(pk);
        }
        else if (slot < 0)
        {
            slot = k;
        }
    }
    if (slot < 0 || total + im_bytes(im) > MC_POOL_BYTES)
    {
        return 0;
    }
    snprintf(im->name, STRINGMAXLEN_IMAGE_NAME, "_tmpcalcpool%d", slot);
    g_pool_ids[slot] = id;
    return 1;
}

/**
 * mc_tmp_image - Allocate or reuse a local temporary image `_tmpcalc<seq>`
 * @naxis: Number of axes (`1..3`)
 * @size:  Axis extents `[3]`
 * @dt:    ImageStreamIO datatype code (`_DATATYPE_*`)
 *
 * Scans `g_pool_ids[]` for a pooled image with matching @dt, @naxis, and @size.
 * On a pool hit, renames it to `_tmpcalc<seq>`, resets counters and keywords,
 * and returns its `imageID` without allocating memory. Otherwise creates a new
 * local image via `create_image_ID()`.
 *
 * Return: Valid `imageID` on success, or -1 on allocation failure
 */
imageID mc_tmp_image(uint8_t naxis, const uint32_t size[3], uint8_t dt)
{
    char tmpname[32];
    snprintf(tmpname, sizeof(tmpname), "_tmpcalc%ld", g_tmp_seq++);

    for (int k = 0; k < MC_POOL_N; k++)
    {
        IMAGE *im   = pool_im(k);
        int    same = (im != NULL && im->md[0].datatype == dt && im->md[0].naxis == naxis);
        for (uint8_t a = 0; same && a < naxis; a++)
        {
            same = (im->md[0].size[a] == size[a]);
        }
        if (!same)
        {
            continue;
        }
        snprintf(im->name, STRINGMAXLEN_IMAGE_NAME, "%s", tmpname);
        imageID id      = g_pool_ids[k];
        g_pool_ids[k]   = -1;
        im->md[0].cnt0  = 0;
        im->md[0].cnt1  = 0;
        im->md[0].write = 0;
        for (int i = 0; im->kw != NULL && i < im->md[0].NBkw; i++)
        {
            im->kw[i].type = 'N';
        }
        return id;
    }

    uint32_t sz[3] = { size[0], (naxis >= 2) ? size[1] : 1, (naxis >= 3) ? size[2] : 1 };
    imageID  id    = -1;
    create_image_ID(tmpname, naxis, sz, dt, 0, 10, 0, &id);
    return id;
}

/**
 * mc_tmp_release_force - Recycle or delete any temporary image (including `_tmpcalcsub*`)
 * @tid: `imageID` of the temporary image to release
 *
 * Called when tearing down a planned program or clearing the sub-program cache.
 *
 * Return: void
 */
void mc_tmp_release_force(imageID tid)
{
    if (milk_data.image == NULL || tid < 0 || tid >= milk_data.NB_MAX_IMAGE ||
        !milk_data.image[tid].used)
    {
        return;
    }
    if (!pool_put(tid))
    {
        delete_image_ID(milk_data.image[tid].name, DELETE_IMAGE_ERRMODE_WARNING);
    }
}

/**
 * mc_tmp_release - Return a transient `_tmpcalc*` image to the pool or delete it
 * @tid: `imageID` of the temporary image to release
 *
 * No-op for pinned sub-program output images (`_tmpcalcsub*`) so cached
 * reduction/constructor sub-programs retain their output buffer across frames.
 *
 * Return: void
 */
void mc_tmp_release(imageID tid)
{
    if (milk_data.image == NULL || tid < 0 || tid >= milk_data.NB_MAX_IMAGE ||
        !milk_data.image[tid].used || strncmp(milk_data.image[tid].name, "_tmpcalcsub", 11) == 0)
    {
        return;
    }
    mc_tmp_release_force(tid);
}

/**
 * mc_tmp_steal_slot - Detach or clone the image produced by a sub-program's final slot
 * @slot: Final output slot of a sub-program
 *
 * For pinned `_tmpcalcsub*` slots, returns `slot->im_id` directly (since
 * `mc_tmp_release()` will leave it pinned). For transient `_tmpcalc*` slots,
 * detaches ownership from @slot and assigns a fresh `_tmpcalc<seq>` name. For
 * user-named slots, clones the image into a new temporary so the user's named
 * image is not consumed.
 *
 * Return: Temporary `imageID` owned by caller, or -1 on error
 */
imageID mc_tmp_steal_slot(mc_slot_t *slot)
{
    if (slot == NULL || slot->im_id < 0 || slot->im_id >= milk_data.NB_MAX_IMAGE ||
        !milk_data.image[slot->im_id].used)
    {
        return -1;
    }
    if (strncmp(slot->bare, "_tmpcalcsub", 11) == 0)
    {
        return slot->im_id;
    }
    if (strncmp(slot->bare, "_tmpcalc", 8) != 0)
    {
        IMAGE  *src = &milk_data.image[slot->im_id];
        imageID tid = mc_tmp_image(src->md[0].naxis, src->md[0].size, src->md[0].datatype);
        if (tid >= 0)
        {
            memcpy(milk_data.image[tid].array.raw, src->array.raw, im_bytes(src));
        }
        return tid;
    }
    imageID id        = slot->im_id;
    slot->im_id       = -1;
    slot->checked_gen = 0;
    snprintf(milk_data.image[id].name, STRINGMAXLEN_IMAGE_NAME, "_tmpcalc%ld", g_tmp_seq++);
    return id;
}

/**
 * mc_ensure_pool_warm - Ensure at least one pooled buffer exists for complex passes
 * @naxis: Number of axes (`1..3`)
 * @size:  Axis extents `[3]`
 * @dt:    Complex datatype (`_DATATYPE_COMPLEX_FLOAT` or `_DATATYPE_COMPLEX_DOUBLE`)
 *
 * Pre-allocates and immediately recycles one temporary image when the pool is
 * completely empty after a complex pass.
 *
 * Return: void
 */
void mc_ensure_pool_warm(uint8_t naxis, const uint32_t size[3], uint8_t dt)
{
    for (int k = 0; k < MC_POOL_N; k++)
    {
        if (pool_im(k) != NULL)
        {
            return;
        }
    }
    imageID tid = mc_tmp_image(naxis, size, dt);
    if (tid != -1)
    {
        mc_tmp_release(tid);
    }
}

/**
 * mcalc_tmp_cleanup - Recycle or delete all non-pooled `_tmpcalc*` images in `milk_data`
 *
 * Sweeps `milk_data.image[]` after a top-level `mcalc_eval()` call to reclaim
 * any unreferenced `_tmpcalc*` buffers while preserving `_tmpcalcpool*` slots.
 *
 * Return: void
 */
void mcalc_tmp_cleanup(void)
{
    if (milk_data.image == NULL || milk_data.NB_MAX_IMAGE <= 0)
    {
        return;
    }
    for (long i = 0; i < milk_data.NB_MAX_IMAGE; i++)
    {
        if (milk_data.image[i].used && strncmp(milk_data.image[i].name, "_tmpcalc", 8) == 0 &&
            strncmp(milk_data.image[i].name, "_tmpcalcpool", 12) != 0)
        {
            if (!pool_put((imageID) i))
            {
                delete_image_ID(milk_data.image[i].name, DELETE_IMAGE_ERRMODE_WARNING);
            }
        }
    }
    g_tmp_seq = 1;
}

/* ========================================================================== */
/* Output Image Preparation & Deferred Stream Publication                     */
/* ========================================================================== */

/**
 * mc_out_prepare - Resolve, validate, or allocate the output `IMAGE` for a slot
 * @slot:    Destination program slot
 * @naxis:   Required number of axes (`1..3`)
 * @size:    Required axis extents `[3]`
 * @dt:      Required output datatype (`_DATATYPE_*`)
 * @in_unit: 1 if executing inside a persistent FPS unit (defaults outputs to shared)
 * @err:     Error message buffer
 * @errsz:   Size of @err in bytes
 *
 * Fast-paths slot resolution when `slot->checked_gen == milk_table_generation()`.
 * Verifies that @slot does not collide with an existing scalar variable. If the
 * target image already exists:
 * - Reuses shared streams in place if the element count matches (setting
 *   `md[0].write = 1`), or errors on dimension mismatch.
 * - Reuses local images in place if shape and datatype match, or deletes and
 *   recreates them when resized or re-typed.
 *
 * Return: Prepared `imageID` with `md[0].write = 1`, or -1 on error
 */
imageID mc_out_prepare(mc_slot_t     *slot,
                       uint8_t        naxis,
                       const uint32_t size[3],
                       uint8_t        dt,
                       int            in_unit,
                       char          *err,
                       size_t         errsz)
{
    const char *bare   = (slot->bare[0] != '\0') ? slot->bare : slot->name;
    uint64_t    tgen   = milk_table_generation();
    int         is_tmp = (strncmp(bare, "_tmpcalc", 8) == 0);
    int want_shm = (is_tmp || slot->loc == 2 || slot->is_local) ? 0 : (slot->loc == 1 || in_unit);
    imageID id   = -1;

    if (slot->im_id >= 0 && slot->im_id < milk_data.NB_MAX_IMAGE &&
        milk_data.image[slot->im_id].used && slot->checked_gen == tgen && tgen != 0 &&
        strcmp(milk_data.image[slot->im_id].name, bare) == 0)
    {
        id = slot->im_id;
    }
    else
    {
        id       = (slot->im_id >= 0 && slot->im_id < milk_data.NB_MAX_IMAGE &&
                    milk_data.image[slot->im_id].used &&
                    strcmp(milk_data.image[slot->im_id].name, bare) == 0)
                       ? slot->im_id
                   : is_tmp ? -1
                   : (slot->loc == 2 || slot->is_local)
                       ? image_ID(bare, milk_data.image, milk_data.NB_MAX_IMAGE)
                       : mc_resolve_or_autoconnect_image(slot->name);
        long vid = (is_tmp || id >= 0) ? -1 : variable_ID(bare); /* an image comes first */
        if (vid >= 0 && milk_data.variable[vid].type != 3)
        {
            if (err != NULL && errsz > 0)
            {
                if (strncmp(bare, "__g", 3) == 0)
                {
                    snprintf(err, errsz,
                             "an if or for needs a number, not an image: test "
                             "imin(...) for all pixels, imax(...) for any");
                }
                else
                {
                    snprintf(err, errsz, "'%s' is a variable (calculator or bash), not an image",
                             bare);
                }
            }
            return -1;
        }
    }

    uint64_t new_nel = 1;
    for (uint8_t a = 0; a < naxis; a++)
    {
        new_nel *= size[a];
    }

    if (id != -1)
    {
        IMAGE *im         = &milk_data.image[id];
        int    same_shape = (im->md[0].naxis == naxis && im->md[0].nelement == new_nel);
        for (uint8_t a = 0; same_shape && a < naxis; a++)
        {
            same_shape = (im->md[0].size[a] == size[a]);
        }

        if (!slot->is_local && im->md[0].shared == 1)
        {
            if (im->md[0].nelement != new_nel)
            {
                if (err != NULL && errsz > 0)
                {
                    snprintf(err, errsz, "Dimension mismatch in image assignment");
                }
                return -1;
            }
            im->md[0].write   = 1;
            slot->im_id       = id;
            slot->checked_gen = tgen;
            return id;
        }
        if (im->md[0].shared == 0 && same_shape && im->md[0].datatype == dt && !want_shm)
        {
            im->md[0].write   = 1;
            slot->im_id       = id;
            slot->checked_gen = tgen;
            return id;
        }
        if (im->md[0].shared == 1)
        {
            if (err != NULL && errsz > 0)
            {
                snprintf(err, errsz, "Dimension mismatch in image assignment");
            }
            return -1;
        }
        if (strncmp(bare, "_tmpcalc", 8) == 0)
        {
            mc_tmp_release(id);
        }
        else
        {
            delete_image_ID(bare, DELETE_IMAGE_ERRMODE_WARNING);
        }
        id = -1;
    }

    uint32_t sz[3] = { size[0], (naxis >= 2) ? size[1] : 1, (naxis >= 3) ? size[2] : 1 };
    if (!want_shm && strncmp(bare, "_tmpcalc", 8) == 0)
    {
        id = mc_tmp_image(naxis, sz, dt);
        if (id != -1 && strcmp(milk_data.image[id].name, bare) != 0)
        {
            snprintf(milk_data.image[id].name, STRINGMAXLEN_IMAGE_NAME, "%s", bare);
        }
    }
    else
    {
        create_image_ID(bare, naxis, sz, dt, want_shm, 10, 0, &id);
    }
    if (id == -1)
    {
        if (err != NULL && errsz > 0)
        {
            snprintf(err, errsz, "cannot create image '%s'", bare);
        }
        return -1;
    }
    if (!slot->existed_at_plan && !is_tmp && dt == slot->plan_dt &&
        (slot->plan_naxis == 0 ||
         (naxis == slot->plan_naxis && memcmp(sz, slot->plan_size, sizeof(sz)) == 0)))
    {
        slot->existed_at_plan = 1;
        slot->orig_dt         = dt;
    }
    milk_data.image[id].md[0].write = 1;
    slot->im_id                     = id;
    slot->checked_gen               = milk_table_generation();
    return id;
}

/**
 * mc_out_post - Finalize and publish an updated output image
 * @id: `imageID` of the modified image
 *
 * For local images, clears `md[0].write = 0` and increments `md[0].cnt0`. For
 * shared-memory streams, either calls `processinfo_update_output_stream()`
 * immediately or queues @id in `g_post_ids[]` when deferred posting is active.
 *
 * Return: void
 */
void mc_out_post(imageID id)
{
    if (milk_data.image == NULL || id < 0 || id >= milk_data.NB_MAX_IMAGE ||
        !milk_data.image[id].used)
    {
        return;
    }
    IMAGE *im = &milk_data.image[id];
    if (im->md == NULL || !im->md[0].shared)
    {
        if (im->md != NULL)
        {
            im->md[0].write = 0;
            im->md[0].cnt0++;
        }
        return;
    }
    if (!g_post_defer)
    {
        processinfo_update_output_stream(NULL, im, NULL);
        return;
    }
    for (int i = 0; i < g_post_n; i++)
    {
        if (g_post_ids[i] == id)
        {
            return;
        }
    }
    if (g_post_n < (int) (sizeof(g_post_ids) / sizeof(g_post_ids[0])))
    {
        g_post_ids[g_post_n++] = id;
    }
    else
    {
        processinfo_update_output_stream(NULL, im, NULL);
    }
}

/**
 * mc_post_defer - Enable or disable deferred shared-stream semaphore posting
 * @on: 1 to increment deferral nesting depth, 0 to reset deferral state
 *
 * Return: void
 */
void mc_post_defer(int on)
{
    if (on)
    {
        if (g_post_defer++ == 0)
        {
            g_post_n = 0;
        }
    }
    else
    {
        g_post_defer = 0;
        g_post_n     = 0;
    }
}

/**
 * mc_post_flush - Flush all queued shared-memory stream updates in a single batch
 * @processinfo: Optional `PROCESSINFO *` handle (may be NULL)
 *
 * Decrements `g_post_defer`; when the outermost deferral scope exits, calls
 * `processinfo_update_output_stream()` once for each unique queued `imageID`.
 *
 * Return: void
 */
void mc_post_flush(void *processinfo)
{
    if (g_post_defer > 1)
    {
        g_post_defer--;
        return;
    }
    g_post_defer = 0;
    for (int i = 0; i < g_post_n; i++)
    {
        imageID id = g_post_ids[i];
        if (milk_data.image != NULL && id >= 0 && id < milk_data.NB_MAX_IMAGE)
        {
            IMAGE *im = &milk_data.image[id];
            if (im->used == 1 && im->md != NULL && im->md[0].shared)
            {
                processinfo_update_output_stream((PROCESSINFO *) processinfo, im, NULL);
            }
        }
    }
    g_post_n = 0;
}

/* ========================================================================== */
/* CLI Image Summary & Pixel Line Formatting                                  */
/* ========================================================================== */

/**
 * mc_format_image_summary - Format a concise `name [d0 x d1 ...] type` summary
 * @id:          `imageID` to summarize
 * @assign_name: Optional LHS name override (omits `_tmpcalc*` prefix when NULL/empty)
 * @buf:         Output character buffer
 * @bufsz:       Size of @buf in bytes
 *
 * Return: void
 */
void mc_format_image_summary(imageID id, const char *assign_name, char *buf, size_t bufsz)
{
    if (milk_data.image == NULL || id < 0 || id >= milk_data.NB_MAX_IMAGE ||
        !milk_data.image[id].used)
    {
        snprintf(buf, bufsz, "%s", (assign_name != NULL) ? assign_name : "");
        return;
    }
    const IMAGE_METADATA *md   = &milk_data.image[id].md[0];
    size_t                dpos = 0;
    char                  dims[128];
    dims[0] = '\0';
    for (uint8_t a = 0; a < md->naxis && dpos + 24 < sizeof(dims); a++)
    {
        dpos += (size_t) snprintf(dims + dpos, sizeof(dims) - dpos, "%s%lu", (a > 0) ? " x " : "",
                                  (unsigned long) md->size[a]);
    }
    const char *tstr = "?";
    static const struct
    {
        uint8_t     dt;
        const char *name;
    } dt_names[] = {
        { _DATATYPE_FLOAT, "f32" },          { _DATATYPE_DOUBLE, "f64" },
        { _DATATYPE_UINT8, "u8" },           { _DATATYPE_INT8, "i8" },
        { _DATATYPE_UINT16, "u16" },         { _DATATYPE_INT16, "i16" },
        { _DATATYPE_UINT32, "u32" },         { _DATATYPE_INT32, "i32" },
        { _DATATYPE_UINT64, "u64" },         { _DATATYPE_INT64, "i64" },
        { _DATATYPE_COMPLEX_FLOAT, "cf32" }, { _DATATYPE_COMPLEX_DOUBLE, "cf64" },
        { _DATATYPE_HALF, "f16" },
    };
    for (size_t i = 0; i < sizeof(dt_names) / sizeof(dt_names[0]); i++)
    {
        if (dt_names[i].dt == md->datatype)
        {
            tstr = dt_names[i].name;
            break;
        }
    }
    const char *nm =
        (assign_name != NULL && assign_name[0] != '\0') ? assign_name : milk_data.image[id].name;
    if ((assign_name != NULL && assign_name[0] != '\0') || strncmp(nm, "_tmpcalc", 8) != 0)
    {
        snprintf(buf, bufsz, "%s [%s] %s", nm, dims, tstr);
    }
    else
    {
        snprintf(buf, bufsz, "[%s] %s", dims, tstr);
    }
}

/**
 * mc_format_pixel_line - Format all pixels of a real image as a space-separated string
 * @id: `imageID` of the real image to format (up to `MC_MAX_PIXELS` elements)
 *
 * Used by `milk-calc -a` to print 1D/2D array elements on a single line.
 * Rejects complex or half-precision images and images exceeding 1,048,576 pixels.
 *
 * Return: Newly heap-allocated NUL-terminated string (caller must `free()`), or NULL
 */
char *mc_format_pixel_line(imageID id)
{
    if (milk_data.image == NULL || id < 0 || id >= milk_data.NB_MAX_IMAGE ||
        !milk_data.image[id].used)
    {
        return NULL;
    }
    const IMAGE *im  = &milk_data.image[id];
    uint64_t     n   = im->md[0].nelement;
    uint8_t      dt  = im->md[0].datatype;
    char        *out = NULL;
    size_t       pos = 0;

    if (n > MC_MAX_PIXELS || (out = (char *) malloc(n * 26 + 1)) == NULL)
    {
        fprintf(stderr, "milk: calc error: %s has %lu pixels (-a takes %d at most)\n", im->name,
                (unsigned long) n, MC_MAX_PIXELS);
        return NULL;
    }
    out[0] = '\0';
#define PIX_INT(MBR)                                                                              \
    for (uint64_t i = 0; i < n; i++)                                                              \
    {                                                                                             \
        pos += (size_t) sprintf(out + pos, "%s%lld", i ? " " : "", (long long) im->array.MBR[i]); \
    }
#define PIX_REAL(MBR)                                                                  \
    for (uint64_t i = 0; i < n; i++)                                                   \
    {                                                                                  \
        pos += (size_t) sprintf(out + pos, "%s%.*g", i ? " " : "", mcalc_float_digits, \
                                (double) im->array.MBR[i]);                            \
    }
    if (dt < _DATATYPE_UINT8 || dt > _DATATYPE_DOUBLE)
    {
        fprintf(stderr, "milk: calc error: %s: -a takes real images\n", im->name);
        free(out);
        return NULL;
    }
    if (dt == _DATATYPE_FLOAT)
    {
        PIX_REAL(F)
    }
    else if (dt == _DATATYPE_DOUBLE)
    {
        PIX_REAL(D)
    }
    else
    {
        MILK_FOR_EACH_DATATYPE(dt, PIX_INT, PIX_INT(D))
    }
#undef PIX_INT
#undef PIX_REAL
    return out;
}
