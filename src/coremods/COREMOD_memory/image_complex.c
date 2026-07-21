// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    image_complex.c
 * @brief   complex number conversion
 */


#ifdef MILK_NO_CLI
#    include "CLIcore_standalone.h"
#    include "COREMOD_memory/COREMOD_memory.h"
#else
#    include "libmilkdata/milkdata.h"
#    include "milkDebugTools.h"
#endif
#include "create_image.h"
#include "delete_image.h"
#include "image_ID.h"
#include "stream_sem.h"

#include "image_complex.h"
#include "image_mk_complex_from_amph.h"
#include "image_mk_complex_from_reim.h"
#include "image_mk_amph_from_complex.h"
#include "image_mk_reim_from_complex.h"

errno_t mk_reim_from_amph_IMGID(IMGID *imgam, IMGID *imgph, IMGID *imgre, IMGID *imgim)
{
    DEBUG_TRACE_FSTART();

    IMGID imgC       = imgid_make_from_name("Ctmp");
    imgC.mdt->shared = 0;

    FUNC_CHECK_RETURN(mk_complex_from_amph_IMGID(imgam, imgph, &imgC));

    FUNC_CHECK_RETURN(mk_reim_from_complex_IMGID(&imgC, imgre, imgim));

    FUNC_CHECK_RETURN(delete_image_IMGID(&imgC, DELETE_IMAGE_ERRMODE_WARNING));
    imgid_free(&imgC);

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}

errno_t mk_reim_from_amph(const char *am_name,
                          const char *ph_name,
                          const char *re_out_name,
                          const char *im_out_name,
                          int         sharedmem)
{
    IMGID imgam       = imgid_make_from_name(am_name);
    IMGID imgph       = imgid_make_from_name(ph_name);
    IMGID imgre       = imgid_make_from_name(re_out_name);
    IMGID imgim       = imgid_make_from_name(im_out_name);
    imgre.mdt->shared = sharedmem;
    imgim.mdt->shared = sharedmem;

    errno_t ret = mk_reim_from_amph_IMGID(&imgam, &imgph, &imgre, &imgim);
    imgid_free(&imgam);
    imgid_free(&imgph);
    imgid_free(&imgre);
    imgid_free(&imgim);
    return ret;
}

errno_t mk_amph_from_reim_IMGID(IMGID *imgre, IMGID *imgim, IMGID *imgam, IMGID *imgph)
{
    DEBUG_TRACE_FSTART();

    IMGID imgC       = imgid_make_from_name("Ctmp");
    imgC.mdt->shared = 0;

    FUNC_CHECK_RETURN(mk_complex_from_reim_IMGID(imgre, imgim, &imgC));

    FUNC_CHECK_RETURN(mk_amph_from_complex_IMGID(&imgC, imgam, imgph));

    FUNC_CHECK_RETURN(delete_image_IMGID(&imgC, DELETE_IMAGE_ERRMODE_WARNING));
    imgid_free(&imgC);

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}

errno_t mk_amph_from_reim(const char *re_name,
                          const char *im_name,
                          const char *am_out_name,
                          const char *ph_out_name,
                          int         sharedmem)
{
    IMGID imgre       = imgid_make_from_name(re_name);
    IMGID imgim       = imgid_make_from_name(im_name);
    IMGID imgam       = imgid_make_from_name(am_out_name);
    IMGID imgph       = imgid_make_from_name(ph_out_name);
    imgam.mdt->shared = sharedmem;
    imgph.mdt->shared = sharedmem;

    errno_t ret = mk_amph_from_reim_IMGID(&imgre, &imgim, &imgam, &imgph);
    imgid_free(&imgre);
    imgid_free(&imgim);
    imgid_free(&imgam);
    imgid_free(&imgph);
    return ret;
}
