/**
 * @file image_add.c
 * @brief Add 2D/3D images with offsets
 */

#ifdef MILK_NO_CLI
#    include "CLIcore_standalone.h"
#else
#    include "CLIcore.h"
#endif
#include "fps.h"

#include "COREMOD_memory/COREMOD_memory.h"

// Forward declarations
imageID basic_add(const char *__restrict ID_name1,
                  const char *__restrict ID_name2,
                  const char *__restrict ID_name_out,
                  long off1,
                  long off2);

imageID basic_add3D(const char *__restrict ID_name1,
                    const char *__restrict ID_name2,
                    const char *__restrict ID_name_out,
                    long off1,
                    long off2,
                    long off3);

/* ---- Command 1: addim ---- */

static char      p1_in1[FUNCTION_PARAMETER_STRMAXLEN] = "im1";
static char      p1_in2[FUNCTION_PARAMETER_STRMAXLEN] = "im2";
static char      p1_out[FUNCTION_PARAMETER_STRMAXLEN] = "outim";
static long long p1_ox                                = 23;
static long long p1_oy                                = 201;

static FPS_APP_INFO FPS_app_info_1 = {
    .fps_name         = "addim",
    .cmdkey           = "addim",
    .description      = "add two 2D images with offset",
    .description_long = "Add two 3D image cubes with a configurable spatial offset. Supports "
                        "sub-pixel alignment via interpolation."
};

#define FPS_PARAMS_1(X)                                                            \
    X(".in1", p1_in1, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "input image 1") \
    X(".in2", p1_in2, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "input image 2") \
    X(".out", p1_out, FPTYPE_STRING, 1, FPFLAG_DEFAULT_INPUT, "output image")      \
    X(".offx", &p1_ox, FPTYPE_INT64, 1, FPFLAG_DEFAULT_INPUT, "x offset")          \
    X(".offy", &p1_oy, FPTYPE_INT64, 1, FPFLAG_DEFAULT_INPUT, "y offset")

static FPS_CLI_BINDING my_bindings_1[] = { FPS_PARAMS_1(FPS_X_BINDING) };
static const int       nb_bindings_1   = sizeof(my_bindings_1) / sizeof(FPS_CLI_BINDING);
static CLICMDARGDEF    farg_1[]        = { FPS_PARAMS_1(FPS_X_FARG) };
static CLICMDDATA      CLIcmddata_1    = {
    "",   "",   __FILE__, sizeof(farg_1) / sizeof(CLICMDARGDEF), farg_1, CLICMDFLAG_FPS,
    NULL, NULL, NULL
};
static CMDSETTINGS cms_1 = { 0 };

static __attribute__((constructor)) void init_cms_1(void)
{
    strncpy(CLIcmddata_1.key, FPS_app_info_1.cmdkey, sizeof(CLIcmddata_1.key) - 1);
    strncpy(CLIcmddata_1.description, FPS_app_info_1.description,
            sizeof(CLIcmddata_1.description) - 1);
    if (CLIcmddata_1.cmdsettings == NULL)
    {
        CLIcmddata_1.cmdsettings = &cms_1;
    }
}

static MILK_HOT errno_t compute_function_1()
{
    basic_add(p1_in1, p1_in2, p1_out, (long) p1_ox, (long) p1_oy);
    return RETURN_SUCCESS;
}

static errno_t CLIfunction_1(void)
{
    return safe_fps_generic_CLIfunction(&FPS_app_info_1, farg_1, &CLIcmddata_1, my_bindings_1,
                                        nb_bindings_1, compute_function_1);
}

/* ---- Command 2: addim3D ---- */

static char      p2_in1[FUNCTION_PARAMETER_STRMAXLEN] = "im1";
static char      p2_in2[FUNCTION_PARAMETER_STRMAXLEN] = "im2";
static char      p2_out[FUNCTION_PARAMETER_STRMAXLEN] = "outim";
static long long p2_ox                                = 23;
static long long p2_oy                                = 201;
static long long p2_oz                                = 0;

static FPS_APP_INFO FPS_app_info_2 = {
    .fps_name         = "addim3D",
    .cmdkey           = "addim3D",
    .description      = "add two 3D images with offset",
    .description_long = "Add two 3D image cubes with a configurable spatial offset. Supports "
                        "sub-pixel alignment via interpolation."
};

#define FPS_PARAMS_2(X)                                                            \
    X(".in1", p2_in1, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "input image 1") \
    X(".in2", p2_in2, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "input image 2") \
    X(".out", p2_out, FPTYPE_STRING, 1, FPFLAG_DEFAULT_INPUT, "output image")      \
    X(".offx", &p2_ox, FPTYPE_INT64, 1, FPFLAG_DEFAULT_INPUT, "x offset")          \
    X(".offy", &p2_oy, FPTYPE_INT64, 1, FPFLAG_DEFAULT_INPUT, "y offset")          \
    X(".offz", &p2_oz, FPTYPE_INT64, 1, FPFLAG_DEFAULT_INPUT, "z offset")

static FPS_CLI_BINDING my_bindings_2[] = { FPS_PARAMS_2(FPS_X_BINDING) };
static const int       nb_bindings_2   = sizeof(my_bindings_2) / sizeof(FPS_CLI_BINDING);
static CLICMDARGDEF    farg_2[]        = { FPS_PARAMS_2(FPS_X_FARG) };
static CLICMDDATA      CLIcmddata_2    = {
    "",   "",   __FILE__, sizeof(farg_2) / sizeof(CLICMDARGDEF), farg_2, CLICMDFLAG_FPS,
    NULL, NULL, NULL
};
static CMDSETTINGS cms_2 = { 0 };

static __attribute__((constructor)) void init_cms_2(void)
{
    strncpy(CLIcmddata_2.key, FPS_app_info_2.cmdkey, sizeof(CLIcmddata_2.key) - 1);
    strncpy(CLIcmddata_2.description, FPS_app_info_2.description,
            sizeof(CLIcmddata_2.description) - 1);
    if (CLIcmddata_2.cmdsettings == NULL)
    {
        CLIcmddata_2.cmdsettings = &cms_2;
    }
}

static MILK_HOT errno_t compute_function_2()
{
    basic_add3D(p2_in1, p2_in2, p2_out, (long) p2_ox, (long) p2_oy, (long) p2_oz);
    return RETURN_SUCCESS;
}

static errno_t CLIfunction_2(void)
{
    return safe_fps_generic_CLIfunction(&FPS_app_info_2, farg_2, &CLIcmddata_2, my_bindings_2,
                                        nb_bindings_2, compute_function_2);
}

errno_t CLIADDCMD_image_basic__image_add()
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

imageID basic_add(const char *__restrict ID_name1,
                  const char *__restrict ID_name2,
                  const char *__restrict ID_name_out,
                  long off1,
                  long off2)
{
    imageID ID1, ID2; /* ID for the 2 images added */
    imageID ID_out;   /* ID for the output image */
    long    ii, jj;
    long    naxes1[2], naxes2[2], naxes[2];
    long    xmin, ymin, xmax, ymax; /* extrema in the ID1 coordinates */
    uint8_t datatype1, datatype2, datatype;
    int     datatypeOK;

    ID1       = image_ID(ID_name1, dcimg, dcnimg);
    ID2       = image_ID(ID_name2, dcimg, dcnimg);
    naxes1[0] = dcimg[ID1].md[0].size[0];
    naxes1[1] = dcimg[ID1].md[0].size[1];
    naxes2[0] = dcimg[ID2].md[0].size[0];
    naxes2[1] = dcimg[ID2].md[0].size[1];

    datatype1 = dcimg[ID1].md[0].datatype;
    datatype2 = dcimg[ID2].md[0].datatype;

    datatypeOK = 0;

    if ((datatype1 == _DATATYPE_FLOAT) && (datatype2 == _DATATYPE_FLOAT))
    {
        datatype   = _DATATYPE_FLOAT;
        datatypeOK = 1;
    }
    if ((datatype1 == _DATATYPE_DOUBLE) && (datatype2 == _DATATYPE_DOUBLE))
    {
        datatype   = _DATATYPE_DOUBLE;
        datatypeOK = 1;
    }

    if (datatypeOK == 0)
    {
        printf("ERROR in basic_add: data type combination not supported\n");
        exit(EXIT_FAILURE);
    }

    /*  if(dcquiet==0)*/
    /* printf("add called with %s ( %ld x %ld ) %s ( %ld x %ld ) and offset ( %ld x %ld )\n",ID_name1,naxes1[0],naxes1[1],ID_name2,naxes2[0],naxes2[1],off1,off2);*/
    xmin = 0;
    if (off1 < 0)
    {
        xmin = off1;
    }
    ymin = 0;
    if (off2 < 0)
    {
        ymin = off2;
    }
    xmax = naxes1[0];
    if ((naxes2[0] + off1) > naxes1[0])
    {
        xmax = (naxes2[0] + off1);
    }
    ymax = naxes1[1];
    if ((naxes2[1] + off2) > naxes1[1])
    {
        ymax = (naxes2[1] + off2);
    }

    if (datatype == _DATATYPE_FLOAT)
    {
        create_2Dimage_ID(ID_name_out, (xmax - xmin), (ymax - ymin), &ID_out);
        naxes[0] = dcimg[ID_out].md[0].size[0];
        naxes[1] = dcimg[ID_out].md[0].size[1];

        for (jj = 0; jj < naxes[1]; jj++)
        {
            for (ii = 0; ii < naxes[0]; ii++)
            {
                {
                    dcimg[ID_out].array.F[jj * naxes[0] + ii] = 0;
                    /* if pixel is in ID1 */
                    if (((ii + xmin) >= 0) && ((ii + xmin) < naxes1[0]))
                    {
                        if (((jj + ymin) >= 0) && ((jj + ymin) < naxes1[1]))
                        {
                            dcimg[ID_out].array.F[jj * naxes[0] + ii] +=
                                dcimg[ID1].array.F[(jj + ymin) * naxes1[0] + (ii + xmin)];
                        }
                    }
                    /* if pixel is in ID2 */
                    if (((ii + xmin - off1) >= 0) && ((ii + xmin - off1) < naxes2[0]))
                    {
                        if (((jj + ymin - off2) >= 0) && ((jj + ymin - off2) < naxes2[1]))
                        {
                            dcimg[ID_out].array.F[jj * naxes[0] + ii] +=
                                dcimg[ID2]
                                    .array.F[(jj + ymin - off2) * naxes2[0] + (ii + xmin - off1)];
                        }
                    }
                }
            }
        }
    }

    if (datatype == _DATATYPE_DOUBLE)
    {
        create_2Dimage_ID_double(ID_name_out, (xmax - xmin), (ymax - ymin), &ID_out);
        naxes[0] = dcimg[ID_out].md[0].size[0];
        naxes[1] = dcimg[ID_out].md[0].size[1];

        for (jj = 0; jj < naxes[1]; jj++)
        {
            for (ii = 0; ii < naxes[0]; ii++)
            {
                {
                    dcimg[ID_out].array.D[jj * naxes[0] + ii] = 0;
                    /* if pixel is in ID1 */
                    if (((ii + xmin) >= 0) && ((ii + xmin) < naxes1[0]))
                    {
                        if (((jj + ymin) >= 0) && ((jj + ymin) < naxes1[1]))
                        {
                            dcimg[ID_out].array.D[jj * naxes[0] + ii] +=
                                dcimg[ID1].array.D[(jj + ymin) * naxes1[0] + (ii + xmin)];
                        }
                    }
                    /* if pixel is in ID2 */
                    if (((ii + xmin - off1) >= 0) && ((ii + xmin - off1) < naxes2[0]))
                    {
                        if (((jj + ymin - off2) >= 0) && ((jj + ymin - off2) < naxes2[1]))
                        {
                            dcimg[ID_out].array.D[jj * naxes[0] + ii] +=
                                dcimg[ID2]
                                    .array.D[(jj + ymin - off2) * naxes2[0] + (ii + xmin - off1)];
                        }
                    }
                }
            }
        }
    }

    return (ID_out);
}

imageID basic_add3D(const char *__restrict ID_name1,
                    const char *__restrict ID_name2,
                    const char *__restrict ID_name_out,
                    long off1,
                    long off2,
                    long off3)
{
    imageID  ID1, ID2; /* ID for the 2 images added */
    imageID  ID_out;   /* ID for the output image */
    uint32_t naxes1[3], naxes2[3], naxes[3];
    long     xmin, ymin, zmin, xmax, ymax, zmax; /* extrema in the ID1 coordinates */
    uint8_t  datatype1, datatype2, datatype;
    int      datatypeOK;

    ID1       = image_ID(ID_name1, dcimg, dcnimg);
    ID2       = image_ID(ID_name2, dcimg, dcnimg);
    naxes1[0] = dcimg[ID1].md[0].size[0];
    naxes1[1] = dcimg[ID1].md[0].size[1];
    naxes1[2] = dcimg[ID1].md[0].size[2];

    naxes2[0] = dcimg[ID2].md[0].size[0];
    naxes2[1] = dcimg[ID2].md[0].size[1];
    naxes2[2] = dcimg[ID2].md[0].size[2];

    datatype1 = dcimg[ID1].md[0].datatype;
    datatype2 = dcimg[ID2].md[0].datatype;

    datatypeOK = 0;

    if ((datatype1 == _DATATYPE_FLOAT) && (datatype2 == _DATATYPE_FLOAT))
    {
        datatype   = _DATATYPE_FLOAT;
        datatypeOK = 1;
    }
    if ((datatype1 == _DATATYPE_DOUBLE) && (datatype2 == _DATATYPE_DOUBLE))
    {
        datatype   = _DATATYPE_DOUBLE;
        datatypeOK = 1;
    }

    if (datatypeOK == 0)
    {
        printf("ERROR in basic_add: data type combination not supported\n");
        exit(0);
    }

    /*  if(dcquiet==0)*/
    /* printf("add called with %s ( %ld x %ld ) %s ( %ld x %ld ) and offset ( %ld x %ld )\n",ID_name1,naxes1[0],naxes1[1],ID_name2,naxes2[0],naxes2[1],off1,off2);*/
    xmin = 0;
    if (off1 < 0)
    {
        xmin = off1;
    }

    ymin = 0;
    if (off2 < 0)
    {
        ymin = off2;
    }

    zmin = 0;
    if (off3 < 0)
    {
        zmin = off3;
    }

    xmax = naxes1[0];
    if ((naxes2[0] + off1) > naxes1[0])
    {
        xmax = (naxes2[0] + off1);
    }

    ymax = naxes1[1];
    if ((naxes2[1] + off2) > naxes1[1])
    {
        ymax = (naxes2[1] + off2);
    }

    zmax = naxes1[2];
    if ((naxes2[2] + off3) > naxes1[2])
    {
        zmax = (naxes2[2] + off3);
    }

    if (datatype == _DATATYPE_FLOAT)
    {
        create_3Dimage_ID(ID_name_out, (xmax - xmin), (ymax - ymin), (zmax - zmin), &ID_out);
        naxes[0] = dcimg[ID_out].md[0].size[0];
        naxes[1] = dcimg[ID_out].md[0].size[1];
        naxes[2] = dcimg[ID_out].md[0].size[2];

        for (uint32_t kk = 0; kk < naxes[2]; kk++)
        {
            for (uint32_t jj = 0; jj < naxes[1]; jj++)
            {
                for (uint32_t ii = 0; ii < naxes[0]; ii++)
                {
                    {
                        dcimg[ID_out].array.F[kk * naxes[1] * naxes[0] + jj * naxes[0] + ii] = 0;
                        /* if pixel is in ID1 */

                        if (((ii + xmin) >= 0) && ((ii + xmin) < naxes1[0]))
                        {
                            if (((jj + ymin) >= 0) && ((jj + ymin) < naxes1[1]))
                            {
                                if (((kk + zmin) >= 0) && ((kk + zmin) < naxes1[2]))
                                {
                                    dcimg[ID_out]
                                        .array.F[kk * naxes[1] * naxes[0] + jj * naxes[0] + ii] +=
                                        dcimg[ID1].array.F[(kk + zmin) * naxes1[1] * naxes1[0] +
                                                           (jj + ymin) * naxes1[0] + (ii + xmin)];
                                }
                            }
                        }
                        /* if pixel is in ID2 */
                        if (((ii + xmin - off1) >= 0) && ((ii + xmin - off1) < naxes2[0]))
                        {
                            if (((jj + ymin - off2) >= 0) && ((jj + ymin - off2) < naxes2[1]))
                            {
                                if (((kk + zmin - off3) >= 0) && ((kk + zmin - off3) < naxes2[2]))
                                {
                                    dcimg[ID_out]
                                        .array.F[kk * naxes[1] * naxes[0] + jj * naxes[0] + ii] +=
                                        dcimg[ID2]
                                            .array
                                            .F[(kk + zmin - off3) * naxes2[1] * naxes2[0] +
                                               (jj + ymin - off2) * naxes2[0] + (ii + xmin - off1)];
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    if (datatype == _DATATYPE_DOUBLE)
    {
        create_3Dimage_ID_double(ID_name_out, (xmax - xmin), (ymax - ymin), (zmax - zmin), &ID_out);
        naxes[0] = dcimg[ID_out].md[0].size[0];
        naxes[1] = dcimg[ID_out].md[0].size[1];
        naxes[2] = dcimg[ID_out].md[0].size[2];

        for (uint32_t kk = 0; kk < naxes[2]; kk++)
        {
            for (uint32_t jj = 0; jj < naxes[1]; jj++)
            {
                for (uint32_t ii = 0; ii < naxes[0]; ii++)
                {
                    {
                        dcimg[ID_out].array.D[kk * naxes[1] * naxes[0] + jj * naxes[0] + ii] = 0;
                        /* if pixel is in ID1 */
                        if (((ii + xmin) >= 0) && ((ii + xmin) < naxes1[0]))
                        {
                            if (((jj + ymin) >= 0) && ((jj + ymin) < naxes1[1]))
                            {
                                if (((kk + zmin) >= 0) && ((kk + zmin) < naxes1[2]))
                                {
                                    dcimg[ID_out]
                                        .array.D[kk * naxes[1] * naxes[0] + jj * naxes[0] + ii] +=
                                        dcimg[ID1].array.D[(kk + zmin) * naxes1[1] * naxes1[0] +
                                                           (jj + ymin) * naxes1[0] + (ii + xmin)];
                                }
                            }
                        }
                        /* if pixel is in ID2 */
                        if (((ii + xmin - off1) >= 0) && ((ii + xmin - off1) < naxes2[0]))
                        {
                            if (((jj + ymin - off2) >= 0) && ((jj + ymin - off2) < naxes2[1]))
                            {
                                if (((kk + zmin - off3) >= 0) && ((kk + zmin - off3) < naxes2[2]))
                                {
                                    dcimg[ID_out]
                                        .array.D[kk * naxes[1] * naxes[0] + jj * naxes[0] + ii] +=
                                        dcimg[ID2]
                                            .array
                                            .D[(kk + zmin - off3) * naxes2[1] * naxes2[0] +
                                               (jj + ymin - off2) * naxes2[0] + (ii + xmin - off1)];
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    return (ID_out);
}
