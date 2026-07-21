// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file imexpand.c
 * @brief Expand/upsample images
 */

#ifdef MILK_NO_CLI
#    include "CLIcore_standalone.h"
#else
#    include "CLIcore.h"
#endif
#include "fps.h"

#include "COREMOD_memory/COREMOD_memory.h"

// Forward declarations
imageID basic_expand(const char *ID_name, const char *ID_name_out, int n1, int n2);

imageID basic_expand3D(const char *ID_name, const char *ID_name_out, int n1, int n2, int n3);

/* ---- Command 1: imexpand ---- */

static char      pe1_in[FUNCTION_PARAMETER_STRMAXLEN]  = "im1";
static char      pe1_out[FUNCTION_PARAMETER_STRMAXLEN] = "im2";
static long long pe1_fx                                = 2;
static long long pe1_fy                                = 2;

static FPS_APP_INFO FPS_app_info_1 = {
    .fps_name         = "imexpand",
    .cmdkey           = "imexpand",
    .description      = "expand 2D image",
    .description_long = "Expand (upsample) a 2D or 3D image by integer factors. Each input pixel "
                        "is replicated into a block of output pixels."
};

#define FPS_PARAMS_1(X)                                                              \
    X(".in_name", pe1_in, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "input image") \
    X(".out_name", pe1_out, FPTYPE_STRING, 1, FPFLAG_DEFAULT_INPUT, "output image")  \
    X(".factx", &pe1_fx, FPTYPE_INT64, 1, FPFLAG_DEFAULT_INPUT, "x expand factor")   \
    X(".facty", &pe1_fy, FPTYPE_INT64, 1, FPFLAG_DEFAULT_INPUT, "y expand factor")

static FPS_CLI_BINDING my_bindings_1[] = { FPS_PARAMS_1(FPS_X_BINDING) };
static const int       nb_bindings_1   = sizeof(my_bindings_1) / sizeof(FPS_CLI_BINDING);
static CLICMDARGDEF    farg_1[]        = { FPS_PARAMS_1(FPS_X_FARG) };
static CLICMDDATA      CLIcmddata_1    = {
    "",   "",   __FILE__, sizeof(farg_1) / sizeof(CLICMDARGDEF), farg_1, CLICMDFLAG_FPS,
    NULL, NULL, NULL
};
FPS_CMDSETTINGS_INIT(1, CLIcmddata_1, FPS_app_info_1)


static MILK_HOT errno_t compute_function_1()
{
    basic_expand(pe1_in, pe1_out, (int) pe1_fx, (int) pe1_fy);
    return RETURN_SUCCESS;
}

static errno_t CLIfunction_1(void)
{
    return safe_fps_generic_CLIfunction(&FPS_app_info_1, farg_1, &CLIcmddata_1, my_bindings_1,
                                        nb_bindings_1, compute_function_1);
}

/* ---- Command 2: imexpand3D ---- */

static char      pe2_in[FUNCTION_PARAMETER_STRMAXLEN]  = "im1";
static char      pe2_out[FUNCTION_PARAMETER_STRMAXLEN] = "im2";
static long long pe2_fx                                = 2;
static long long pe2_fy                                = 2;
static long long pe2_fz                                = 2;

static FPS_APP_INFO FPS_app_info_2 = {
    .fps_name         = "imexpand3D",
    .cmdkey           = "imexpand3D",
    .description      = "expand 3D image",
    .description_long = "Expand (upsample) a 2D or 3D image by integer factors. Each input pixel "
                        "is replicated into a block of output pixels."
};

#define FPS_PARAMS_2(X)                                                              \
    X(".in_name", pe2_in, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "input image") \
    X(".out_name", pe2_out, FPTYPE_STRING, 1, FPFLAG_DEFAULT_INPUT, "output image")  \
    X(".factx", &pe2_fx, FPTYPE_INT64, 1, FPFLAG_DEFAULT_INPUT, "x expand factor")   \
    X(".facty", &pe2_fy, FPTYPE_INT64, 1, FPFLAG_DEFAULT_INPUT, "y expand factor")   \
    X(".factz", &pe2_fz, FPTYPE_INT64, 1, FPFLAG_DEFAULT_INPUT, "z expand factor")

static FPS_CLI_BINDING my_bindings_2[] = { FPS_PARAMS_2(FPS_X_BINDING) };
static const int       nb_bindings_2   = sizeof(my_bindings_2) / sizeof(FPS_CLI_BINDING);
static CLICMDARGDEF    farg_2[]        = { FPS_PARAMS_2(FPS_X_FARG) };
static CLICMDDATA      CLIcmddata_2    = {
    "",   "",   __FILE__, sizeof(farg_2) / sizeof(CLICMDARGDEF), farg_2, CLICMDFLAG_FPS,
    NULL, NULL, NULL
};
FPS_CMDSETTINGS_INIT(2, CLIcmddata_2, FPS_app_info_2)


static MILK_HOT errno_t compute_function_2()
{
    basic_expand3D(pe2_in, pe2_out, (int) pe2_fx, (int) pe2_fy, (int) pe2_fz);
    return RETURN_SUCCESS;
}

static errno_t CLIfunction_2(void)
{
    return safe_fps_generic_CLIfunction(&FPS_app_info_2, farg_2, &CLIcmddata_2, my_bindings_2,
                                        nb_bindings_2, compute_function_2);
}

errno_t CLIADDCMD_image_basic__imexpand()
{
    safe_fps_fill_farg_examples(farg_1, my_bindings_1, nb_bindings_1);
    {
        int cmdi                 = RegisterCLIcmd(CLIcmddata_1, CLIfunction_1);
        CLIcmddata_1.cmdsettings = &data.cmd[cmdi].cmdsettings;
    }

    safe_fps_fill_farg_examples(farg_2, my_bindings_2, nb_bindings_2);
    {
        int cmdi                 = RegisterCLIcmd(CLIcmddata_2, CLIfunction_2);
        CLIcmddata_2.cmdsettings = &data.cmd[cmdi].cmdsettings;
    }

    return RETURN_SUCCESS;
}

/**
 * Expand image by factor n1 along x and
 * n2 along y (nearest-neighbour).
 */
imageID basic_expand(const char *ID_name, const char *ID_name_out, int n1, int n2)
{
    DEBUG_TRACE_FSTART();

    IMGID imgin = imgid_make_from_name(ID_name);
    resolveIMGID(&imgin, ERRMODE_WARN, dcimg, dcnimg);
    if (imgin.ID == -1)
    {
        return RETURN_FAILURE;
    }

    long naxes[2];
    naxes[0] = imgin.md->size[0];
    naxes[1] = imgin.md->size[1];
    long naxes_out[2];
    naxes_out[0] = naxes[0] * n1;
    naxes_out[1] = naxes[1] * n2;

    IMGID imgout       = imgid_make_from_name_2D(ID_name_out, naxes_out[0], naxes_out[1]);
    imgout.mdt->shared = 0;
    imgout.im          = (IMAGE *) calloc(1, sizeof(IMAGE));
    imgid_mkimage(&imgout);

    for (long jj = 0; jj < naxes[1]; jj++)
    {
        for (long ii = 0; ii < naxes[0]; ii++)
        {
            for (int i = 0; i < n1; i++)
            {
                for (int j = 0; j < n2; j++)
                {
                    imgout.im->array.F[(jj * n2 + j) * naxes_out[0] + ii * n1 + i] =
                        imgin.im->array.F[jj * naxes[0] + ii];
                }
            }
        }
    }

    DEBUG_TRACE_FEXIT();
    return imgout.ID;
}

/**
 * Expand 3D image by factors n1, n2, n3
 * along x, y, z (nearest-neighbour).
 */
imageID basic_expand3D(const char *ID_name, const char *ID_name_out, int n1, int n2, int n3)
{
    IMGID imgin = imgid_make_from_name(ID_name);
    resolveIMGID(&imgin, ERRMODE_WARN, dcimg, dcnimg);
    if (imgin.ID == -1)
    {
        return RETURN_FAILURE;
    }

    long naxes[3];
    naxes[0] = imgin.md->size[0];
    naxes[1] = (imgin.md->naxis > 1) ? imgin.md->size[1] : 1;
    naxes[2] = (imgin.md->naxis == 3) ? imgin.md->size[2] : 1;

    long naxes_out[3];
    naxes_out[0] = naxes[0] * n1;
    naxes_out[1] = naxes[1] * n2;
    naxes_out[2] = naxes[2] * n3;

    printf(" %ld %ld %ld"
           " -> %ld %ld %ld\n",
           naxes[0], naxes[1], naxes[2], naxes_out[0], naxes_out[1], naxes_out[2]);

    IMGID imgout = imgid_make_from_name_3D(ID_name_out, naxes_out[0], naxes_out[1], naxes_out[2]);
    imgout.mdt->shared = 0;
    imgout.im          = (IMAGE *) calloc(1, sizeof(IMAGE));
    imgid_mkimage(&imgout);

    list_image_ID();

    for (long kk = 0; kk < naxes[2]; kk++)
    {
        for (long jj = 0; jj < naxes[1]; jj++)
        {
            for (long ii = 0; ii < naxes[0]; ii++)
            {
                for (int i = 0; i < n1; i++)
                {
                    for (int j = 0; j < n2; j++)
                    {
                        for (int k = 0; k < n3; k++)
                        {
                            imgout.im->array.F[(kk * n3 + k) * naxes_out[0] * naxes_out[1] +
                                               (jj * n2 + j) * naxes_out[0] + ii * n1 + i] =
                                imgin.im->array.F[kk * naxes[0] * naxes[1] + jj * naxes[0] + ii];
                        }
                    }
                }
            }
        }
    }

    return imgout.ID;
}
