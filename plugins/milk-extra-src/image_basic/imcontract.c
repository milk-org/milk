/**
 * @file imcontract.c
 * @brief Image binning
 */

#ifdef MILK_NO_CLI
#include "CLIcore_standalone.h"
#else
#include "CLIcore.h"
#endif
#include "fps.h"

#include "COREMOD_memory/COREMOD_memory.h"

// Forward declarations
imageID basic_contract(
    const char *ID_name,
    const char *ID_name_out,
    int n1, int n2);

imageID basic_contract3D(
    const char *ID_name,
    const char *ID_name_out,
    int n1, int n2, int n3);

/* ---- Command 1: imcontract ---- */

static char pc1_in[FUNCTION_PARAMETER_STRMAXLEN]
    = "im1";
static char pc1_out[FUNCTION_PARAMETER_STRMAXLEN]
    = "outim";
static long long pc1_bx = 4;
static long long pc1_by = 4;

static FPS_APP_INFO FPS_app_info_1 = {
    .fps_name    = "imcontract",
    .cmdkey      = "imcontract",
    .description = "image binning"
};

#define FPS_PARAMS_1(X) \
    X(".in_name", pc1_in, \
      FPTYPE_STREAMNAME, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "input image") \
    X(".out_name", pc1_out, \
      FPTYPE_STRING, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "output image") \
    X(".binx", &pc1_bx, \
      FPTYPE_INT64, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "x bin factor") \
    X(".biny", &pc1_by, \
      FPTYPE_INT64, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "y bin factor")

static FPS_CLI_BINDING my_bindings_1[] = {
    FPS_PARAMS_1(FPS_X_BINDING)
};
static const int nb_bindings_1 =
    sizeof(my_bindings_1) /
    sizeof(FPS_CLI_BINDING);
static CLICMDARGDEF farg_1[] = {
    FPS_PARAMS_1(FPS_X_FARG)
};
static CLICMDDATA CLIcmddata_1 = {
    "", "", __FILE__,
    sizeof(farg_1) / sizeof(CLICMDARGDEF),
    farg_1, CLICMDFLAG_FPS,
    NULL, NULL, NULL
};
static CMDSETTINGS cms_1 = {0};

static __attribute__((constructor))
void init_cms_1(void)
{
    strncpy(CLIcmddata_1.key,
            FPS_app_info_1.cmdkey,
            sizeof(CLIcmddata_1.key) - 1);
    strncpy(CLIcmddata_1.description,
            FPS_app_info_1.description,
            sizeof(CLIcmddata_1.description)
            - 1);
    if (CLIcmddata_1.cmdsettings == NULL) {
        CLIcmddata_1.cmdsettings = &cms_1;
    }
}

static errno_t compute_function_1()
{
    basic_contract(pc1_in, pc1_out,
                   (int) pc1_bx,
                   (int) pc1_by);
    return RETURN_SUCCESS;
}

static errno_t CLIfunction_1(void)
{
    return safe_fps_generic_CLIfunction(
        &FPS_app_info_1, farg_1,
        &CLIcmddata_1,
        my_bindings_1, nb_bindings_1,
        compute_function_1);
}

/* ---- Command 2: imcontract3D ---- */

static char pc2_in[FUNCTION_PARAMETER_STRMAXLEN]
    = "im1";
static char pc2_out[FUNCTION_PARAMETER_STRMAXLEN]
    = "outim";
static long long pc2_bx = 4;
static long long pc2_by = 4;
static long long pc2_bz = 1;

static FPS_APP_INFO FPS_app_info_2 = {
    .fps_name    = "imcontract3D",
    .cmdkey      = "imcontract3D",
    .description = "image binning (3D)"
};

#define FPS_PARAMS_2(X) \
    X(".in_name", pc2_in, \
      FPTYPE_STREAMNAME, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "input image") \
    X(".out_name", pc2_out, \
      FPTYPE_STRING, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "output image") \
    X(".binx", &pc2_bx, \
      FPTYPE_INT64, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "x bin factor") \
    X(".biny", &pc2_by, \
      FPTYPE_INT64, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "y bin factor") \
    X(".binz", &pc2_bz, \
      FPTYPE_INT64, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "z bin factor")

static FPS_CLI_BINDING my_bindings_2[] = {
    FPS_PARAMS_2(FPS_X_BINDING)
};
static const int nb_bindings_2 =
    sizeof(my_bindings_2) /
    sizeof(FPS_CLI_BINDING);
static CLICMDARGDEF farg_2[] = {
    FPS_PARAMS_2(FPS_X_FARG)
};
static CLICMDDATA CLIcmddata_2 = {
    "", "", __FILE__,
    sizeof(farg_2) / sizeof(CLICMDARGDEF),
    farg_2, CLICMDFLAG_FPS,
    NULL, NULL, NULL
};
static CMDSETTINGS cms_2 = {0};

static __attribute__((constructor))
void init_cms_2(void)
{
    strncpy(CLIcmddata_2.key,
            FPS_app_info_2.cmdkey,
            sizeof(CLIcmddata_2.key) - 1);
    strncpy(CLIcmddata_2.description,
            FPS_app_info_2.description,
            sizeof(CLIcmddata_2.description)
            - 1);
    if (CLIcmddata_2.cmdsettings == NULL) {
        CLIcmddata_2.cmdsettings = &cms_2;
    }
}

static errno_t compute_function_2()
{
    basic_contract3D(pc2_in, pc2_out,
                     (int) pc2_bx,
                     (int) pc2_by,
                     (int) pc2_bz);
    return RETURN_SUCCESS;
}

static errno_t CLIfunction_2(void)
{
    return safe_fps_generic_CLIfunction(
        &FPS_app_info_2, farg_2,
        &CLIcmddata_2,
        my_bindings_2, nb_bindings_2,
        compute_function_2);
}

errno_t
CLIADDCMD_image_basic__imcontract()
{
    safe_fps_fill_farg_examples(
        farg_1, my_bindings_1,
        nb_bindings_1);
    {
        int cmdi = RegisterCLIcmd(
            CLIcmddata_1, CLIfunction_1);
        CLIcmddata_1.cmdsettings =
            &data.cmd[cmdi].cmdsettings;
    }

    safe_fps_fill_farg_examples(
        farg_2, my_bindings_2,
        nb_bindings_2);
    {
        int cmdi = RegisterCLIcmd(
            CLIcmddata_2, CLIfunction_2);
        CLIcmddata_2.cmdsettings =
            &data.cmd[cmdi].cmdsettings;
    }

    return RETURN_SUCCESS;
}

imageID
basic_contract(const char *ID_name, const char *ID_name_out, int n1, int n2)
{
    imageID  ID;
    imageID  ID_out; /* ID for the output image */
    uint32_t naxes[2], naxes_out[2];
    int      i, j;

    ID       = image_ID(ID_name, dcimg, dcnimg);
    naxes[0] = dcimg[ID].md[0].size[0];
    naxes[1] = dcimg[ID].md[0].size[1];

    naxes_out[0] = naxes[0] / n1;
    naxes_out[1] = naxes[1] / n2;

    //  printf("%ld %ld  ->  %ld %ld\n",naxes[0],naxes[1],naxes_out[0],naxes_out[1]);
    create_2Dimage_ID(ID_name_out, naxes_out[0], naxes_out[1], &ID_out);

    for(uint32_t jj = 0; jj < naxes_out[1]; jj++)
        for(uint32_t ii = 0; ii < naxes_out[0]; ii++)
            for(i = 0; i < n1; i++)
                for(j = 0; j < n2; j++)
                {
                    dcimg[ID_out].array.F[jj * naxes_out[0] + ii] +=
                        dcimg[ID]
                        .array.F[(jj * n2 + j) * naxes[0] + ii * n1 + i];
                }

    return (ID_out);
}

imageID basic_contract3D(
    const char *ID_name, const char *ID_name_out, int n1, int n2, int n3)
{
    DEBUG_TRACE_FSTART();

    imageID   ID;
    imageID   ID_out; /* ID for the output image */
    uint32_t  naxes[3];
    uint32_t *naxes_out;
    uint8_t   datatype;

    ID       = image_ID(ID_name, dcimg, dcnimg);
    datatype = dcimg[ID].md[0].datatype;
    naxes[0] = dcimg[ID].md[0].size[0];
    naxes[1] = dcimg[ID].md[0].size[1];
    naxes[2] = dcimg[ID].md[0].size[2];

    naxes_out = (uint32_t *) malloc(sizeof(uint32_t) * 3);
    if(naxes_out == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }
    naxes_out[0] = naxes[0] / n1;
    naxes_out[1] = naxes[1] / n2;
    naxes_out[2] = naxes[2] / n3;

    if(naxes_out[2] == 1)
    {
        create_2Dimage_ID(ID_name_out, naxes_out[0], naxes_out[1], NULL);
    }
    else
    {
        printf("(%ld x %ld x %ld)  ->  (%ld x %ld x %ld)\n",
               (long) naxes[0],
               (long) naxes[1],
               (long) naxes[2],
               (long) naxes_out[0],
               (long) naxes_out[1],
               (long) naxes_out[2]);
        create_image_ID(ID_name_out, 3, naxes_out, datatype, 0, 0, 0, NULL);
    }

    ID_out = image_ID(ID_name_out, dcimg, dcnimg);

    switch(datatype)
    {
        case _DATATYPE_FLOAT:
            for(uint32_t jj = 0; jj < naxes_out[1]; jj++)
                for(uint32_t ii = 0; ii < naxes_out[0]; ii++)
                    for(uint32_t kk = 0; kk < naxes_out[2]; kk++)
                        for(int i = 0; i < n1; i++)
                            for(int j = 0; j < n2; j++)
                                for(int k = 0; k < n3; k++)
                                {
                                    dcimg[ID_out]
                                    .array.F[kk * naxes_out[0] * naxes_out[1] +
                                                jj * naxes_out[0] + ii] +=
                                                 dcimg[ID]
                                                 .array
                                                 .F[(kk * n3 + k) * naxes[0] * naxes[1] +
                                                                  (jj * n2 + j) * naxes[0] + ii * n1 +
                                                                  i];
                                }
            break;
        case _DATATYPE_DOUBLE:
            for(uint32_t jj = 0; jj < naxes_out[1]; jj++)
                for(uint32_t ii = 0; ii < naxes_out[0]; ii++)
                    for(uint32_t kk = 0; kk < naxes_out[2]; kk++)
                        for(int i = 0; i < n1; i++)
                            for(int j = 0; j < n2; j++)
                                for(int k = 0; k < n3; k++)
                                {
                                    dcimg[ID_out]
                                    .array.D[kk * naxes_out[0] * naxes_out[1] +
                                                jj * naxes_out[0] + ii] +=
                                                 dcimg[ID]
                                                 .array
                                                 .D[(kk * n3 + k) * naxes[0] * naxes[1] +
                                                                  (jj * n2 + j) * naxes[0] + ii * n1 +
                                                                  i];
                                }
            break;
        case _DATATYPE_COMPLEX_FLOAT:
            for(uint32_t jj = 0; jj < naxes_out[1]; jj++)
                for(uint32_t ii = 0; ii < naxes_out[0]; ii++)
                    for(uint32_t kk = 0; kk < naxes_out[2]; kk++)
                        for(int i = 0; i < n1; i++)
                            for(int j = 0; j < n2; j++)
                                for(int k = 0; k < n3; k++)
                                {
                                    dcimg[ID_out]
                                    .array
                                    .CF[kk * naxes_out[0] * naxes_out[1] +
                                           jj * naxes_out[0] + ii]
                                    .re += dcimg[ID]
                                           .array
                                           .CF[(kk * n3 + k) * naxes[0] *
                                                             naxes[1] +
                                                             (jj * n2 + j) * naxes[0] +
                                                             ii * n1 + i]
                                           .re;
                                    dcimg[ID_out]
                                    .array
                                    .CF[kk * naxes_out[0] * naxes_out[1] +
                                           jj * naxes_out[0] + ii]
                                    .im += dcimg[ID]
                                           .array
                                           .CF[(kk * n3 + k) * naxes[0] *
                                                             naxes[1] +
                                                             (jj * n2 + j) * naxes[0] +
                                                             ii * n1 + i]
                                           .im;
                                }
            break;
        case _DATATYPE_COMPLEX_DOUBLE:
            for(uint32_t jj = 0; jj < naxes_out[1]; jj++)
                for(uint32_t ii = 0; ii < naxes_out[0]; ii++)
                    for(uint32_t kk = 0; kk < naxes_out[2]; kk++)
                        for(int i = 0; i < n1; i++)
                            for(int j = 0; j < n2; j++)
                                for(int k = 0; k < n3; k++)
                                {
                                    dcimg[ID_out]
                                    .array
                                    .CD[kk * naxes_out[0] * naxes_out[1] +
                                           jj * naxes_out[0] + ii]
                                    .re += dcimg[ID]
                                           .array
                                           .CD[(kk * n3 + k) * naxes[0] *
                                                             naxes[1] +
                                                             (jj * n2 + j) * naxes[0] +
                                                             ii * n1 + i]
                                           .re;
                                    dcimg[ID_out]
                                    .array
                                    .CD[kk * naxes_out[0] * naxes_out[1] +
                                           jj * naxes_out[0] + ii]
                                    .im += dcimg[ID]
                                           .array
                                           .CD[(kk * n3 + k) * naxes[0] *
                                                             naxes[1] +
                                                             (jj * n2 + j) * naxes[0] +
                                                             ii * n1 + i]
                                           .im;
                                }
            break;
    }

    free(naxes_out);

    DEBUG_TRACE_FEXIT();

    return (ID_out);
}
