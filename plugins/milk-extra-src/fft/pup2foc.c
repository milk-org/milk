// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file pup2foc.c
 * @brief Pup2foc module
 */

/** @file pupfft.c
 */

#include "milk_runnable.h"

#include "COREMOD_memory/COREMOD_memory.h"

#include "dofft.h"
#include "permut.h"


/* ================================================================
 * 2.  LOCAL PARAMETER VARIABLES
 * ============================================================= */

static char inamp[FUNCTION_PARAMETER_STRMAXLEN]  = "";
static char inpha[FUNCTION_PARAMETER_STRMAXLEN]  = "";
static char outamp[FUNCTION_PARAMETER_STRMAXLEN] = "";
static char outpha[FUNCTION_PARAMETER_STRMAXLEN] = "";


/* ================================================================
 * 3.  UNIFIED PARAMETER TABLE (X-Macro)
 * ============================================================= */

#define FPS_PARAMS(X)                                                                \
    X(".inamp", inamp, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "input WF ampl")  \
    X(".inpha", inpha, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "input WF phase") \
    X(".outa", outamp, FPTYPE_STRING, 1, FPFLAG_DEFAULT_INPUT, "output WF ampl")     \
    X(".outp", outpha, FPTYPE_STRING, 1, FPFLAG_DEFAULT_INPUT, "output WF phase")


/* ================================================================
 * 5.  BINDINGS, FARG, AND CLI DATA
 * ============================================================= */

MILK_PARAMS_DEFINE(pup2foc, FPS_PARAMS);

/* inv = 0 for direct fft and 1 for inverse fft */
/* direct = focal plane -> pupil plane  equ. fft2d(..,..,..,1) */
/* inverse = pupil plane -> focal plane equ. fft2d(..,..,..,0) */
/* options :  -reim  takes real/imaginary input and creates real/imaginary output
               -inv  for inverse fft (inv=1) */
errno_t pup2foc_fft(const char *__restrict ID_name_ampl,
                    const char *__restrict ID_name_pha,
                    const char *__restrict ID_name_ampl_out,
                    const char *__restrict ID_name_pha_out,
                    const char *__restrict options)
{
    int reim;
    int inv;

    char Ctmpname[STRINGMAXLEN_IMGNAME];
    char C1tmpname[STRINGMAXLEN_IMGNAME];

    reim = 0;
    inv  = 0;

    if (strstr(options, "-reim") != NULL)
    {
        /*	printf("taking real / imaginary input/output\n");*/
        reim = 1;
    }

    if (strstr(options, "-inv") != NULL)
    {
        /*printf("doing the inverse Fourier transform\n");*/
        inv = 1;
    }

    WRITE_IMAGENAME(Ctmpname, "_Ctmp_%d", (int) getpid());

    if (reim == 0)
    {
        mk_complex_from_amph(ID_name_ampl, ID_name_pha, Ctmpname, 0);
    }
    else
    {
        mk_complex_from_reim(ID_name_ampl, ID_name_pha, Ctmpname, 0);
    }

    permut(Ctmpname);


    WRITE_IMAGENAME(C1tmpname, "_C1tmp_%d", (int) getpid());

    if (inv == 0)
    {
        do2dfft(Ctmpname, C1tmpname); /* equ. fft2d(..,1) */
    }
    else
    {
        do2dffti(Ctmpname, C1tmpname); /* equ. fft2d(..,0) */
    }

    delete_image_ID(Ctmpname, DELETE_IMAGE_ERRMODE_WARNING);

    if (reim == 0)
    {
        /* if this line is removed, the program crashes... why ??? */
        /*	list_image_ID(data); */
        mk_amph_from_complex(C1tmpname, ID_name_ampl_out, ID_name_pha_out, 0);
    }
    else
    {
        mk_reim_from_complex(C1tmpname, ID_name_ampl_out, ID_name_pha_out, 0);
    }

    delete_image_ID(C1tmpname, DELETE_IMAGE_ERRMODE_WARNING);

    permut(ID_name_ampl_out);
    permut(ID_name_pha_out);

    return RETURN_SUCCESS;
}


static MILK_HOT errno_t pup2foc_compute(const MILK_RUNNABLE *self)
{
    DEBUG_TRACE_FSTART();

    IMGID imgamp = imgid_make_from_name(inamp);
    resolveIMGID(&imgamp, ERRMODE_WARN, dcimg, dcnimg);

    IMGID imgpha = imgid_make_from_name(inpha);
    if (imgamp.ID == -1)
    {
        return RETURN_FAILURE;
    }
    resolveIMGID(&imgpha, ERRMODE_WARN, dcimg, dcnimg);

    if (imgpha.ID == -1)
    {
        return RETURN_FAILURE;
    }
    MILK_PROCINFO_INIT(self)

    // custom initialization
    printf(" COMPUTE Flags = %ld\n", self->rt->cmddata.cmdsettings->flags);
    if (self->rt->cmddata.cmdsettings->flags & CLICMDFLAG_PROCINFO)
    {
        // procinfo is accessible here
    }

    MILK_PROCINFO_LOOPSTART(self)
    {
        pup2foc_fft(inamp, inpha, outamp, outpha, "");
    }
    MILK_PROCINFO_END(self)

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}


/* ================================================================
 * 7.  RUNNABLE DESCRIPTOR
 * ============================================================= */

const MILK_RUNNABLE runnable_fft_pup2foc = {
    .kind    = MILK_RUNNABLE_FPS,
    .app     = { .fps_name    = "pup2foc",
                 .cmdkey      = "pup2foc",
                 .description = "pupil to focus by FFT",
                 .description_long =
                     "Propagate a wavefront from pupil plane to focal plane using FFT. Applies "
                     "the Fraunhofer diffraction integral to compute the PSF from a pupil "
                     "amplitude/phase map." },
    .srcfile = __FILE__,
    MILK_PARAMS_REF(pup2foc),
    .compute = pup2foc_compute,
};
EXPORT_FPSEXEC(milk, "fft-pup2foc", runnable_fft_pup2foc);
