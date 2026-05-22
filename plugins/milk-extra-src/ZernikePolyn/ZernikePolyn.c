/**
 * @file ZernikePolyn.c
 * @brief ==================================================================
 */


/* ================================================================== */
/* ================================================================== */
/*            MODULE INFO                                             */
/* ================================================================== */
/* ================================================================== */

// module default short name
// all CLI calls to this module functions will be <shortname>.<funcname>
// if set to "", then calls use <funcname>
#define MODULE_SHORTNAME_DEFAULT "zern"

// Module short description
#define MODULE_DESCRIPTION "Create and fit Zernike polynomials"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <fitsio.h> /* required by every program that uses CFITSIO  */

#include "CLIcore.h"

#include "COREMOD_arith/COREMOD_arith.h"
#include "COREMOD_iofits/COREMOD_iofits.h"
#include "COREMOD_memory/COREMOD_memory.h"

#include "image_gen/image_gen.h"


#include "zernike.h"
#include "zernike_value.h"

#include "ZernikePolyn/ZernikePolyn.h"

#include "mkzercube.h"

#define SWAP(x, y) \
    tmp = (x);     \
    x   = (y);     \
    y   = tmp;

#define PI 3.14159265358979323846264338328

//extern DATA data;

//ZERNIKE Zernike;

/* ================================================================== */
/* ================================================================== */
/*            INITIALIZE LIBRARY                                      */
/* ================================================================== */
/* ================================================================== */

// Module initialization macro in CLIcore.h
// macro argument defines module name for bindings
//
INIT_MODULE_LIB(ZernikePolyn)

/* ================================================================== */
/* ================================================================== */
/*            COMMAND LINE INTERFACE (CLI) FUNCTIONS                  */
/* ================================================================== */
/* ================================================================== */

// Forward declarations
imageID mk_zer(const char *ID_name, long SIZE, long zer_nb, float rpix);

long ZERNIKEPOLYN_rmPiston(const char *ID_name, const char *IDmask_name);

/* ============================================
 * Command: mkzer
 * ============================================ */

static char    mkz_out[FUNCTION_PARAMETER_STRMAXLEN] = "z43";
static int64_t mkz_size                              = 512;
static int64_t mkz_index                             = 43;
static double  mkz_rpix                              = 100.0;

static FPS_APP_INFO FPS_app_info_mkz = {
    .fps_name         = "mkzer",
    .cmdkey           = "mkzer",
    .description      = "create Zernike polynomial",
    .description_long = "Compute and manipulate Zernike polynomials for wavefront analysis. "
                        "Supports piston removal and modal decomposition."
};

#define FPS_PARAMS_MKZ(X)                                                              \
    X(".out_name", mkz_out, FPTYPE_STRING, 1, FPFLAG_DEFAULT_INPUT, "output image")    \
    X(".size", &mkz_size, FPTYPE_INT64, 1, FPFLAG_DEFAULT_INPUT, "image size")         \
    X(".zerindex", &mkz_index, FPTYPE_INT64, 1, FPFLAG_DEFAULT_INPUT, "Zernike index") \
    X(".rpix", &mkz_rpix, FPTYPE_FLOAT64, 1, FPFLAG_DEFAULT_INPUT, "radius in pixels")

#include "fps.h"

static FPS_CLI_BINDING mkz_bindings[]  = { FPS_PARAMS_MKZ(FPS_X_BINDING) };
static const int       mkz_nb_bindings = sizeof(mkz_bindings) / sizeof(FPS_CLI_BINDING);
static CLICMDARGDEF    farg[]          = { FPS_PARAMS_MKZ(FPS_X_FARG) };
static CLICMDDATA      CLIcmddata      = { "", "", CLICMD_FIELDS_DEFAULTS };
static CMDSETTINGS     mkz_cms         = { 0 };

static __attribute__((constructor)) void init_mkz_cms(void)
{
    strncpy(CLIcmddata.key, FPS_app_info_mkz.cmdkey, sizeof(CLIcmddata.key) - 1);
    strncpy(CLIcmddata.description, FPS_app_info_mkz.description,
            sizeof(CLIcmddata.description) - 1);
    CLIcmddata.nbarg         = sizeof(farg) / sizeof(CLICMDARGDEF);
    CLIcmddata.funcfpscliarg = farg;
    CLIcmddata.flags         = CLICMDFLAG_FPS;
    if (CLIcmddata.cmdsettings == NULL)
    {
        CLIcmddata.cmdsettings = &mkz_cms;
    }
}

static errno_t mkz_compute(void)
{
    mk_zer(mkz_out, (long) mkz_size, (long) mkz_index, (float) mkz_rpix);
    return RETURN_SUCCESS;
}

static errno_t CLIfunction(void)
{
    return safe_fps_generic_CLIfunction(&FPS_app_info_mkz, farg, &CLIcmddata, mkz_bindings,
                                        mkz_nb_bindings, mkz_compute);
}

/* ============================================
 * Command: rmcpiston
 * ============================================ */

static char rmp_in[FUNCTION_PARAMETER_STRMAXLEN]   = "wfc";
static char rmp_mask[FUNCTION_PARAMETER_STRMAXLEN] = "mask";

static FPS_APP_INFO FPS_app_info_rmp = {
    .fps_name         = "rmcpiston",
    .cmdkey           = "rmcpiston",
    .description      = "remove piston term from WF cube",
    .description_long = "Compute and manipulate Zernike polynomials for wavefront analysis. "
                        "Supports piston removal and modal decomposition."
};

#define FPS_PARAMS_RMP(X)                                                        \
    X(".in_name", rmp_in, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "WF cube") \
    X(".mask_name", rmp_mask, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "aperture mask")

static FPS_CLI_BINDING rmp_bindings[]  = { FPS_PARAMS_RMP(FPS_X_BINDING) };
static const int       rmp_nb_bindings = sizeof(rmp_bindings) / sizeof(FPS_CLI_BINDING);
static CLICMDARGDEF    rmp_farg[]      = { FPS_PARAMS_RMP(FPS_X_FARG) };
static CLICMDDATA      rmp_CLIcmddata  = { "", "", CLICMD_FIELDS_DEFAULTS };
static CMDSETTINGS     rmp_cms         = { 0 };

static __attribute__((constructor)) void init_rmp_cms(void)
{
    strncpy(rmp_CLIcmddata.key, FPS_app_info_rmp.cmdkey, sizeof(rmp_CLIcmddata.key) - 1);
    strncpy(rmp_CLIcmddata.description, FPS_app_info_rmp.description,
            sizeof(rmp_CLIcmddata.description) - 1);
    rmp_CLIcmddata.nbarg         = sizeof(rmp_farg) / sizeof(CLICMDARGDEF);
    rmp_CLIcmddata.funcfpscliarg = rmp_farg;
    rmp_CLIcmddata.flags         = CLICMDFLAG_FPS;
    if (rmp_CLIcmddata.cmdsettings == NULL)
    {
        rmp_CLIcmddata.cmdsettings = &rmp_cms;
    }
}

static errno_t rmp_compute(void)
{
    ZERNIKEPOLYN_rmPiston(rmp_in, rmp_mask);
    return RETURN_SUCCESS;
}

static errno_t rmp_CLIfunction(void)
{
    return safe_fps_generic_CLIfunction(&FPS_app_info_rmp, rmp_farg, &rmp_CLIcmddata, rmp_bindings,
                                        rmp_nb_bindings, rmp_compute);
}

static errno_t init_module_CLI()
{
    /* mkzer */
    {
        safe_fps_fill_farg_examples(farg, mkz_bindings, mkz_nb_bindings);
        int cmdi               = RegisterCLIcmd(CLIcmddata, CLIfunction);
        CLIcmddata.cmdsettings = &data.cmd[cmdi].cmdsettings;
    }

    /* rmcpiston */
    {
        safe_fps_fill_farg_examples(rmp_farg, rmp_bindings, rmp_nb_bindings);
        int cmdi                   = RegisterCLIcmd(rmp_CLIcmddata, rmp_CLIfunction);
        rmp_CLIcmddata.cmdsettings = &data.cmd[cmdi].cmdsettings;
    }

    CLIADDCMD_ZernikePolyn__mkzercube();

    // add atexit functions here

    return RETURN_SUCCESS;
}


imageID mk_zer(const char *ID_name, long SIZE, long zer_nb, float rpix)
{
    long    ii, jj;
    double  r, theta;
    imageID ID;
    long    naxes[2];
    long    n, m;
    double  coeffextend1 = -1.0;
    double  coeffextend2 = 0.3;
    double  coeffextend3 = 4.0;
    double  ss           = 0.0;
    double  xoffset      = 0.0;
    double  yoffset      = 0.0;
    double  x, y;

    ID = variable_ID("ZEXTENDc1");
    if (ID != -1)
    {
        coeffextend1 = dcvar[ID].value.f;
        printf("ZEXTENDc1 = %f\n", coeffextend1);
    }

    ID = variable_ID("ZEXTENDc2");
    if (ID != -1)
    {
        coeffextend2 = dcvar[ID].value.f;
        printf("ZEXTENDc2 = %f\n", coeffextend2);
    }

    ID = variable_ID("Zxoffset");
    if (ID != -1)
    {
        xoffset = dcvar[ID].value.f;
        printf("Zxoffset = %f\n", xoffset);
    }
    ID = variable_ID("Zyoffset");
    if (ID != -1)
    {
        yoffset = dcvar[ID].value.f;
        printf("Zyoffset = %f\n", yoffset);
    }

    naxes[0] = SIZE;
    naxes[1] = SIZE;

    zernike_init();


    n = Zernike_n(zer_nb);
    m = Zernike_m(zer_nb);
    printf("Z = %ld    :  n = %ld, m = %ld\n", zer_nb, n, m);
    create_2Dimage_ID(ID_name, SIZE, SIZE, &ID);

    /* let's compute the polar coordinates */
    ss = 0.0;
    for (ii = 0; ii < SIZE; ii++)
    {
        for (jj = 0; jj < SIZE; jj++)
        {
            x = 1.0 * (ii - SIZE / 2) - xoffset;
            y = 1.0 * (jj - SIZE / 2) - yoffset;

            r     = sqrt(x * x + y * y) / rpix;
            theta = atan2(y, x);
            if (r < 1.0)
            {
                dcimg[ID].array.F[jj * naxes[0] + ii] = Zernike_value(zer_nb, r, theta);
                //printf("%f\n", Zernike_value(zer_nb,r,theta));
                ss += dcimg[ID].array.F[jj * naxes[0] + ii] * dcimg[ID].array.F[jj * naxes[0] + ii];
            }
            else if (coeffextend1 > 0)
            {
                r = 1.0 + (r - 1.0) / (1.0 + coeffextend1 * (r - 1.0));
                dcimg[ID].array.F[jj * naxes[0] + ii] = Zernike_value(zer_nb, 1.0, theta);
                dcimg[ID].array.F[jj * naxes[0] + ii] *=
                    exp(-pow((r - 1.0) / (rpix * coeffextend2), coeffextend3));
                //	dcimg[ID].array.F[jj*naxes[0]+ii] = r;
                //printf("%f %f\n", Zernike_value(zer_nb, 1.0, theta), exp(-pow((r-1.0)/(rpix*coeffextend2), coeffextend3)));
            }
        }
    }

    if (zer_nb > 0)
    {
        double coeff_norm;

        make_disk("disk_tmp", SIZE, SIZE, SIZE / 2, SIZE / 2, rpix);
        coeff_norm = sqrt(arith_image_sumsquare("disk_tmp") / ss);
        //	printf("coeff = %f\n", coeff_norm);
        arith_image_cstmult_inplace(ID_name, coeff_norm);
        delete_image_ID("disk_tmp", DELETE_IMAGE_ERRMODE_WARNING);
    }

    if (zer_nb == 0)
    {
        for (ii = 0; ii < SIZE; ii++)
        {
            for (jj = 0; jj < SIZE; jj++)
            {
                r = sqrt((ii - SIZE / 2) * (ii - SIZE / 2) + (jj - SIZE / 2) * (jj - SIZE / 2)) /
                    rpix;
                if (r > 1.0)
                {
                    if (coeffextend1 < 0)
                    {
                        dcimg[ID].array.F[jj * naxes[0] + ii] = 0.0f;
                    }
                    else
                    {
                        dcimg[ID].array.F[jj * naxes[0] + ii] = 1.0f;
                    }
                }
            }
        }
    }

    return ID;
}

// continue Zernike exp. beyond nominal radius, using the same polynomial expression
imageID mk_zer_unbounded(const char *ID_name, long SIZE, long zer_nb, float rpix)
{
    long    ii, jj;
    double  r, theta;
    imageID ID;
    long    naxes[2];
    long    n, m;

    naxes[0] = SIZE;
    naxes[1] = SIZE;


    zernike_init();

    n = Zernike_n(zer_nb);
    m = Zernike_m(zer_nb);
    printf("Z = %ld    :  n = %ld, m = %ld\n", zer_nb, n, m);
    create_2Dimage_ID(ID_name, SIZE, SIZE, &ID);

    /* let's compute the polar coordinates */
    for (ii = 0; ii < SIZE; ii++)
    {
        for (jj = 0; jj < SIZE; jj++)
        {
            r = sqrt((ii - SIZE / 2) * (ii - SIZE / 2) + (jj - SIZE / 2) * (jj - SIZE / 2)) / rpix;
            theta = atan2((jj - SIZE / 2), (ii - SIZE / 2));
            //	  if(r<1.0)
            dcimg[ID].array.F[jj * naxes[0] + ii] = Zernike_value(zer_nb, r, theta);
        }
    }

    if (zer_nb > 0)
    {
        double coeff_norm;

        make_disk("disk_tmp", SIZE, SIZE, SIZE / 2, SIZE / 2, rpix);
        coeff_norm = sqrt(arith_image_sumsquare("disk_tmp") / arith_image_sumsquare(ID_name));
        arith_image_cstmult_inplace(ID_name, coeff_norm);
        delete_image_ID("disk_tmp", DELETE_IMAGE_ERRMODE_WARNING);
    }

    if (zer_nb == 0)
    {
        for (ii = 0; ii < SIZE; ii++)
        {
            for (jj = 0; jj < SIZE; jj++)
            {
                //r = sqrt((ii-SIZE/2)*(ii-SIZE/2)+(jj-SIZE/2)*(jj-SIZE/2))/rpix;
                //    if(r<1.0)
                dcimg[ID].array.F[jj * naxes[0] + ii] = 1.0f;
            }
        }
    }

    return ID;
}

// continue Zernike exp. beyond nominal radius, using the r=1 for r>1
imageID mk_zer_unbounded1(const char *ID_name, long SIZE, long zer_nb, float rpix)
{
    long    ii, jj;
    double  r, theta;
    imageID ID;
    long    naxes[2];
    double  coeff_norm;
    long    n, m;

    naxes[0] = SIZE;
    naxes[1] = SIZE;

    zernike_init();

    n = Zernike_n(zer_nb);
    m = Zernike_m(zer_nb);
    printf("Z = %ld    :  n = %ld, m = %ld\n", zer_nb, n, m);
    create_2Dimage_ID(ID_name, SIZE, SIZE, &ID);

    /* let's compute the polar coordinates */
    for (ii = 0; ii < SIZE; ii++)
    {
        for (jj = 0; jj < SIZE; jj++)
        {
            r = sqrt((ii - SIZE / 2) * (ii - SIZE / 2) + (jj - SIZE / 2) * (jj - SIZE / 2)) / rpix;
            theta = atan2((jj - SIZE / 2), (ii - SIZE / 2));
            if (r > 1.0)
            {
                r = 1.0;
            }
            dcimg[ID].array.F[jj * naxes[0] + ii] = Zernike_value(zer_nb, r, theta);
        }
    }

    if (zer_nb > 0)
    {
        make_disk("disk_tmp", SIZE, SIZE, SIZE / 2, SIZE / 2, rpix);
        coeff_norm = sqrt(arith_image_sumsquare("disk_tmp") / arith_image_sumsquare(ID_name));
        arith_image_cstmult_inplace(ID_name, coeff_norm);
        delete_image_ID("disk_tmp", DELETE_IMAGE_ERRMODE_WARNING);
    }

    if (zer_nb == 0)
    {
        for (ii = 0; ii < SIZE; ii++)
        {
            for (jj = 0; jj < SIZE; jj++)
            {
                dcimg[ID].array.F[jj * naxes[0] + ii] = 1.0f;
            }
        }
    }

    return ID;
}

errno_t mk_zer_series(const char *ID_name, long SIZE, long zer_nb, float rpix)
{
    long    ii, jj;
    double *r;
    double *theta;
    imageID ID;
    long    naxes[2];
    double  tmp;
    char    fname[200];
    long    j;

    j        = 0;
    naxes[0] = SIZE;
    naxes[1] = SIZE;

    zernike_init();

    create_2Dimage_ID("ztmp", SIZE, SIZE, &ID);

    r = (double *) malloc(SIZE * SIZE * sizeof(double));
    if (r == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    theta = (double *) malloc(SIZE * SIZE * sizeof(double));
    if (theta == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    if ((r == NULL) || (theta == NULL))
    {
        printf("error in memory allocation !!!\n");
    }

    /* let's compute the polar coordinates */
    for (ii = 0; ii < SIZE; ii++)
    {
        for (jj = 0; jj < SIZE; jj++)
        {
            r[jj * naxes[0] + ii] = sqrt((0.5 + ii - SIZE / 2) * (0.5 + ii - SIZE / 2) +
                                         (0.5 + jj - SIZE / 2) * (0.5 + jj - SIZE / 2)) /
                                    rpix;
            theta[jj * naxes[0] + ii] = atan2((jj - SIZE / 2), (ii - SIZE / 2));
        }
    }

    /* let's make the Zernikes */
    for (ii = 0; ii < SIZE; ii++)
    {
        for (jj = 0; jj < SIZE; jj++)
        {
            tmp = r[jj * naxes[0] + ii];
            if (tmp < 1.0)
            {
                dcimg[ID].array.F[jj * SIZE + ii] = 1.0f;
            }
            else
            {
                dcimg[ID].array.F[jj * SIZE + ii] = 0.0f;
            }
        }
    }
    snprintf(fname, sizeof(fname), "%s%ld", ID_name, j);
    save_fl_fits("ztmp", fname);

    for (j = 1; j < zer_nb; j++)
    {
        /*	printf("%ld/%ld\n",j,zer_nb);*/
        fflush(stdout);

        for (ii = 0; ii < SIZE; ii++)
        {
            for (jj = 0; jj < SIZE; jj++)
            {
                tmp = r[jj * naxes[0] + ii];
                if (tmp < 1.0)
                {
                    dcimg[ID].array.F[jj * SIZE + ii] =
                        Zernike_value(j, tmp, theta[jj * naxes[0] + ii]);
                }
                else
                {
                    dcimg[ID].array.F[jj * SIZE + ii] = 0.0f;
                }
            }
        }

        snprintf(fname, sizeof(fname), "%s%04ld", ID_name, j);
        save_fl_fits("ztmp", fname);
    }

    delete_image_ID("ztmp", DELETE_IMAGE_ERRMODE_WARNING);

    free(r);
    free(theta);

    return RETURN_SUCCESS;
}

imageID mk_zer_seriescube(const char *ID_namec, long SIZE, long zer_nb, float rpix)
{
    long    ii, jj;
    double *r;
    double *theta;
    imageID ID;
    long    naxes[2];
    double  tmp;
    long    j;

    j        = 0;
    naxes[0] = SIZE;
    naxes[1] = SIZE;

    zernike_init();

    create_3Dimage_ID(ID_namec, SIZE, SIZE, zer_nb, &ID);
    //    ID = image_ID("ztmp", dcimg, dcnimg);

    r = (double *) malloc(SIZE * SIZE * sizeof(double));
    if (r == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    theta = (double *) malloc(SIZE * SIZE * sizeof(double));
    if (theta == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    /* let's compute the polar coordinates */
    for (ii = 0; ii < SIZE; ii++)
    {
        for (jj = 0; jj < SIZE; jj++)
        {
            r[jj * naxes[0] + ii] = sqrt((0.5 + ii - SIZE / 2) * (0.5 + ii - SIZE / 2) +
                                         (0.5 + jj - SIZE / 2) * (0.5 + jj - SIZE / 2)) /
                                    rpix;
            theta[jj * naxes[0] + ii] = atan2((jj - SIZE / 2), (ii - SIZE / 2));
        }
    }

    /* let's make the Zernikes */
    for (ii = 0; ii < SIZE; ii++)
    {
        for (jj = 0; jj < SIZE; jj++)
        {
            tmp = r[jj * naxes[0] + ii];
            if (tmp < 1.0)
            {
                dcimg[ID].array.F[jj * SIZE + ii] = 1.0f;
            }
            else
            {
                dcimg[ID].array.F[jj * SIZE + ii] = 0.0f;
            }
        }
    }
    for (j = 1; j < zer_nb; j++)
    {
        /*	printf("%ld/%ld\n",j,zer_nb);*/
        //        fflush(stdout);

        for (ii = 0; ii < SIZE; ii++)
        {
            for (jj = 0; jj < SIZE; jj++)
            {
                tmp = r[jj * naxes[0] + ii];
                if (tmp < 1.0)
                {
                    dcimg[ID].array.F[j * SIZE * SIZE + jj * SIZE + ii] =
                        Zernike_value(j, tmp, theta[jj * naxes[0] + ii]);
                }
                else
                {
                    dcimg[ID].array.F[j * SIZE * SIZE + jj * SIZE + ii] = 0.0;
                }
            }
        }
    }

    free(r);
    free(theta);

    return ID;
}
