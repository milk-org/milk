// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    image_complex.c
 * @brief   complex number conversion
 */

#include <math.h>

#include "CommandLineInterface/CLIcore.h"
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

    IMGID imgC  = mkIMGID_from_name("Ctmp");
    imgC.shared = 0;

    FUNC_CHECK_RETURN(mk_complex_from_amph_IMGID(imgam, imgph, &imgC));

    FUNC_CHECK_RETURN(mk_reim_from_complex_IMGID(&imgC, imgre, imgim));

    FUNC_CHECK_RETURN(delete_image_IMGID(&imgC, DELETE_IMAGE_ERRMODE_WARNING));

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}

errno_t mk_reim_from_amph(const char *am_name,
                          const char *ph_name,
                          const char *re_out_name,
                          const char *im_out_name,
                          int         sharedmem)
{
    IMGID imgam  = mkIMGID_from_name(am_name);
    IMGID imgph  = mkIMGID_from_name(ph_name);
    IMGID imgre  = mkIMGID_from_name(re_out_name);
    IMGID imgim  = mkIMGID_from_name(im_out_name);
    imgre.shared = sharedmem;
    imgim.shared = sharedmem;

    return mk_reim_from_amph_IMGID(&imgam, &imgph, &imgre, &imgim);
}

errno_t mk_amph_from_reim_IMGID(IMGID *imgre, IMGID *imgim, IMGID *imgam, IMGID *imgph)
{
    DEBUG_TRACE_FSTART();

    IMGID imgC  = mkIMGID_from_name("Ctmp");
    imgC.shared = 0;

    FUNC_CHECK_RETURN(mk_complex_from_reim_IMGID(imgre, imgim, &imgC));

    FUNC_CHECK_RETURN(mk_amph_from_complex_IMGID(&imgC, imgam, imgph));

    FUNC_CHECK_RETURN(delete_image_IMGID(&imgC, DELETE_IMAGE_ERRMODE_WARNING));

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}

errno_t mk_amph_from_reim(const char *re_name,
                          const char *im_name,
                          const char *am_out_name,
                          const char *ph_out_name,
                          int         sharedmem)
{
    IMGID imgre  = mkIMGID_from_name(re_name);
    IMGID imgim  = mkIMGID_from_name(im_name);
    IMGID imgam  = mkIMGID_from_name(am_out_name);
    IMGID imgph  = mkIMGID_from_name(ph_out_name);
    imgam.shared = sharedmem;
    imgph.shared = sharedmem;

    return mk_amph_from_reim_IMGID(&imgre, &imgim, &imgam, &imgph);
}
