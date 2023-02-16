

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

#include "CommandLineInterface/CLIcore.h"

#include "COREMOD_arith/COREMOD_arith.h"
#include "COREMOD_iofits/COREMOD_iofits.h"
#include "COREMOD_memory/COREMOD_memory.h"

#include "image_gen/image_gen.h"


#include "zernike.h"
#include "zernike_value.h"

#include "ZernikePolyn/ZernikePolyn.h"

#include "mkzercube.h"

#define SWAP(x, y)                                                             \
    tmp = (x);                                                                 \
    x   = (y);                                                                 \
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

// CLI commands
//
// function CLI_checkarg used to check arguments
// 1: float
// 2: long
// 3: string
// 4: existing image
//

errno_t mk_zer_cli()
{
    if(CLI_checkarg(1, CLIARG_STR_NOT_IMG) + CLI_checkarg(2, CLIARG_INT64) +
            CLI_checkarg(3, CLIARG_INT64) + CLI_checkarg(4, CLIARG_FLOAT64) ==
            0)
    {
        mk_zer(data.cmdargtoken[1].val.string,
               data.cmdargtoken[2].val.numl,
               data.cmdargtoken[3].val.numl,
               data.cmdargtoken[4].val.numf);

        return CLICMD_SUCCESS;
    }
    else
    {
        return CLICMD_INVALID_ARG;
    }
}

errno_t ZERNIKEPOLYN_rmPiston_cli()
{
    if(CLI_checkarg(1, CLIARG_IMG) + CLI_checkarg(2, CLIARG_IMG) == 0)
    {
        ZERNIKEPOLYN_rmPiston(data.cmdargtoken[1].val.string,
                              data.cmdargtoken[2].val.string);

        return CLICMD_SUCCESS;
    }
    else
    {
        return CLICMD_INVALID_ARG;
    }
}

static errno_t init_module_CLI()
{

    RegisterCLIcommand(
        "mkzer",
        __FILE__,
        mk_zer_cli,
        "create Zernike polynomial",
        "<output image> <size> <zern index> <rpix>",
        "mkzer z43 512 43 100.0",
        "mk_zer(const char *ID_name, long SIZE, long zer_nb, float rpix)");

    RegisterCLIcommand("rmcpiston",
                       __FILE__,
                       ZERNIKEPOLYN_rmPiston_cli,
                       "remove piston term from WF cube",
                       "<WF cube> <aperture mask>",
                       "rmcpiston wfc mask",
                       "long ZERNIKEPOLYN_rmPiston(const char *ID_name, const "
                       "char *IDmask_name);");


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
    if(ID != -1)
    {
        coeffextend1 = data.variable[ID].value.f;
        printf("ZEXTENDc1 = %f\n", coeffextend1);
    }

    ID = variable_ID("ZEXTENDc2");
    if(ID != -1)
    {
        coeffextend2 = data.variable[ID].value.f;
        printf("ZEXTENDc2 = %f\n", coeffextend2);
    }

    ID = variable_ID("Zxoffset");
    if(ID != -1)
    {
        xoffset = data.variable[ID].value.f;
        printf("Zxoffset = %f\n", xoffset);
    }
    ID = variable_ID("Zyoffset");
    if(ID != -1)
    {
        yoffset = data.variable[ID].value.f;
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
    for(ii = 0; ii < SIZE; ii++)
        for(jj = 0; jj < SIZE; jj++)
        {
            x = 1.0 * (ii - SIZE / 2) - xoffset;
            y = 1.0 * (jj - SIZE / 2) - yoffset;

            r     = sqrt(x * x + y * y) / rpix;
            theta = atan2(y, x);
            if(r < 1.0)
            {
                data.image[ID].array.F[jj * naxes[0] + ii] =
                    Zernike_value(zer_nb, r, theta);
                //printf("%f\n", Zernike_value(zer_nb,r,theta));
                ss += data.image[ID].array.F[jj * naxes[0] + ii] *
                      data.image[ID].array.F[jj * naxes[0] + ii];
            }
            else if(coeffextend1 > 0)
            {
                r = 1.0 + (r - 1.0) / (1.0 + coeffextend1 * (r - 1.0));
                data.image[ID].array.F[jj * naxes[0] + ii] =
                    Zernike_value(zer_nb, 1.0, theta);
                data.image[ID].array.F[jj * naxes[0] + ii] *=
                    exp(-pow((r - 1.0) / (rpix * coeffextend2), coeffextend3));
                //	data.image[ID].array.F[jj*naxes[0]+ii] = r;
                //printf("%f %f\n", Zernike_value(zer_nb, 1.0, theta), exp(-pow((r-1.0)/(rpix*coeffextend2), coeffextend3)));
            }
        }

    if(zer_nb > 0)
    {
        double coeff_norm;

        make_disk("disk_tmp", SIZE, SIZE, SIZE / 2, SIZE / 2, rpix);
        coeff_norm = sqrt(arith_image_sumsquare("disk_tmp") / ss);
        //	printf("coeff = %f\n", coeff_norm);
        arith_image_cstmult_inplace(ID_name, coeff_norm);
        delete_image_ID("disk_tmp", DELETE_IMAGE_ERRMODE_WARNING);
    }

    if(zer_nb == 0)
    {
        for(ii = 0; ii < SIZE; ii++)
            for(jj = 0; jj < SIZE; jj++)
            {
                r = sqrt((ii - SIZE / 2) * (ii - SIZE / 2) +
                         (jj - SIZE / 2) * (jj - SIZE / 2)) /
                    rpix;
                if(r > 1.0)
                {
                    if(coeffextend1 < 0)
                    {
                        data.image[ID].array.F[jj * naxes[0] + ii] = 0.0;
                    }
                    else
                    {
                        data.image[ID].array.F[jj * naxes[0] + ii] = 1.0;
                    }
                }
            }
    }

    return ID;
}

// continue Zernike exp. beyond nominal radius, using the same polynomial expression
imageID
mk_zer_unbounded(const char *ID_name, long SIZE, long zer_nb, float rpix)
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
    for(ii = 0; ii < SIZE; ii++)
        for(jj = 0; jj < SIZE; jj++)
        {
            r = sqrt((ii - SIZE / 2) * (ii - SIZE / 2) +
                     (jj - SIZE / 2) * (jj - SIZE / 2)) /
                rpix;
            theta = atan2((jj - SIZE / 2), (ii - SIZE / 2));
            //	  if(r<1.0)
            data.image[ID].array.F[jj * naxes[0] + ii] =
                Zernike_value(zer_nb, r, theta);
        }

    if(zer_nb > 0)
    {
        double coeff_norm;

        make_disk("disk_tmp", SIZE, SIZE, SIZE / 2, SIZE / 2, rpix);
        coeff_norm = sqrt(arith_image_sumsquare("disk_tmp") /
                          arith_image_sumsquare(ID_name));
        arith_image_cstmult_inplace(ID_name, coeff_norm);
        delete_image_ID("disk_tmp", DELETE_IMAGE_ERRMODE_WARNING);
    }

    if(zer_nb == 0)
    {
        for(ii = 0; ii < SIZE; ii++)
            for(jj = 0; jj < SIZE; jj++)
            {
                //r = sqrt((ii-SIZE/2)*(ii-SIZE/2)+(jj-SIZE/2)*(jj-SIZE/2))/rpix;
                //    if(r<1.0)
                data.image[ID].array.F[jj * naxes[0] + ii] = 1.0;
            }
    }

    return ID;
}

// continue Zernike exp. beyond nominal radius, using the r=1 for r>1
imageID
mk_zer_unbounded1(const char *ID_name, long SIZE, long zer_nb, float rpix)
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
    for(ii = 0; ii < SIZE; ii++)
        for(jj = 0; jj < SIZE; jj++)
        {
            r = sqrt((ii - SIZE / 2) * (ii - SIZE / 2) +
                     (jj - SIZE / 2) * (jj - SIZE / 2)) /
                rpix;
            theta = atan2((jj - SIZE / 2), (ii - SIZE / 2));
            if(r > 1.0)
            {
                r = 1.0;
            }
            data.image[ID].array.F[jj * naxes[0] + ii] =
                Zernike_value(zer_nb, r, theta);
        }

    if(zer_nb > 0)
    {
        make_disk("disk_tmp", SIZE, SIZE, SIZE / 2, SIZE / 2, rpix);
        coeff_norm = sqrt(arith_image_sumsquare("disk_tmp") /
                          arith_image_sumsquare(ID_name));
        arith_image_cstmult_inplace(ID_name, coeff_norm);
        delete_image_ID("disk_tmp", DELETE_IMAGE_ERRMODE_WARNING);
    }

    if(zer_nb == 0)
    {
        for(ii = 0; ii < SIZE; ii++)
            for(jj = 0; jj < SIZE; jj++)
            {
                data.image[ID].array.F[jj * naxes[0] + ii] = 1.0;
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
    if(r == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    theta = (double *) malloc(SIZE * SIZE * sizeof(double));
    if(theta == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    if((r == NULL) || (theta == NULL))
    {
        printf("error in memory allocation !!!\n");
    }

    /* let's compute the polar coordinates */
    for(ii = 0; ii < SIZE; ii++)
        for(jj = 0; jj < SIZE; jj++)
        {
            r[jj * naxes[0] + ii] =
                sqrt((0.5 + ii - SIZE / 2) * (0.5 + ii - SIZE / 2) +
                     (0.5 + jj - SIZE / 2) * (0.5 + jj - SIZE / 2)) /
                rpix;
            theta[jj * naxes[0] + ii] = atan2((jj - SIZE / 2), (ii - SIZE / 2));
        }

    /* let's make the Zernikes */
    for(ii = 0; ii < SIZE; ii++)
        for(jj = 0; jj < SIZE; jj++)
        {
            tmp = r[jj * naxes[0] + ii];
            if(tmp < 1.0)
            {
                data.image[ID].array.F[jj * SIZE + ii] = 1.0;
            }
            else
            {
                data.image[ID].array.F[jj * SIZE + ii] = 0.0;
            }
        }
    sprintf(fname, "%s%ld", ID_name, j);
    save_fl_fits("ztmp", fname);

    for(j = 1; j < zer_nb; j++)
    {
        /*	printf("%ld/%ld\n",j,zer_nb);*/
        fflush(stdout);

        for(ii = 0; ii < SIZE; ii++)
            for(jj = 0; jj < SIZE; jj++)
            {
                tmp = r[jj * naxes[0] + ii];
                if(tmp < 1.0)
                {
                    data.image[ID].array.F[jj * SIZE + ii] =
                        Zernike_value(j, tmp, theta[jj * naxes[0] + ii]);
                }
                else
                {
                    data.image[ID].array.F[jj * SIZE + ii] = 0.0;
                }
            }

        sprintf(fname, "%s%04ld", ID_name, j);
        save_fl_fits("ztmp", fname);
    }

    delete_image_ID("ztmp", DELETE_IMAGE_ERRMODE_WARNING);

    free(r);
    free(theta);

    return RETURN_SUCCESS;
}

imageID
mk_zer_seriescube(const char *ID_namec, long SIZE, long zer_nb, float rpix)
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
    //    ID = image_ID("ztmp");

    r = (double *) malloc(SIZE * SIZE * sizeof(double));
    if(r == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    theta = (double *) malloc(SIZE * SIZE * sizeof(double));
    if(theta == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    /* let's compute the polar coordinates */
    for(ii = 0; ii < SIZE; ii++)
        for(jj = 0; jj < SIZE; jj++)
        {
            r[jj * naxes[0] + ii] =
                sqrt((0.5 + ii - SIZE / 2) * (0.5 + ii - SIZE / 2) +
                     (0.5 + jj - SIZE / 2) * (0.5 + jj - SIZE / 2)) /
                rpix;
            theta[jj * naxes[0] + ii] = atan2((jj - SIZE / 2), (ii - SIZE / 2));
        }

    /* let's make the Zernikes */
    for(ii = 0; ii < SIZE; ii++)
        for(jj = 0; jj < SIZE; jj++)
        {
            tmp = r[jj * naxes[0] + ii];
            if(tmp < 1.0)
            {
                data.image[ID].array.F[jj * SIZE + ii] = 1.0;
            }
            else
            {
                data.image[ID].array.F[jj * SIZE + ii] = 0.0;
            }
        }
    for(j = 1; j < zer_nb; j++)
    {
        /*	printf("%ld/%ld\n",j,zer_nb);*/
        //        fflush(stdout);

        for(ii = 0; ii < SIZE; ii++)
            for(jj = 0; jj < SIZE; jj++)
            {
                tmp = r[jj * naxes[0] + ii];
                if(tmp < 1.0)
                {
                    data.image[ID].array.F[j * SIZE * SIZE + jj * SIZE + ii] =
                        Zernike_value(j, tmp, theta[jj * naxes[0] + ii]);
                }
                else
                {
                    data.image[ID].array.F[j * SIZE * SIZE + jj * SIZE + ii] =
                        0.0;
                }
            }
    }

    free(r);
    free(theta);

    return ID;
}

double get_zer(const char *ID_name, long zer_nb, double radius)
{
    double  value;
    long    SIZE;
    imageID ID;
    char    fname[200];
    char    fname1[200];

    ID   = image_ID(ID_name);
    SIZE = data.image[ID].md[0].size[0];
    make_disk("disktmp", SIZE, SIZE, 0.5 * SIZE, 0.5 * SIZE, radius);

    sprintf(fname, "/RAID0/tmp/Zernike/Z_%ld", zer_nb);
    sprintf(fname1, "Z_%ld", zer_nb);

    if((ID = image_ID(fname1)) == -1)
    {
        if(file_exists(fname) == 1)
        {
            imageID IDtmp;
            load_fits(fname, fname1, 1, &IDtmp);
        }
        else
        {
            mk_zer(fname1, SIZE, zer_nb, radius);
        }
    }

    arith_image_mult(fname1, ID_name, "mult_tmp");
    value = arith_image_total("mult_tmp") / arith_image_total("disktmp");
    /* printf("value is %e\n",value);*/
    delete_image_ID("disktmp", DELETE_IMAGE_ERRMODE_WARNING);
    /*  delete_image_ID("zernike_tmp");*/
    delete_image_ID("mult_tmp", DELETE_IMAGE_ERRMODE_WARNING);

    return (value);
}

double
get_zer_crop(const char *ID_name, long zer_nb, double radius, double radius1)
{
    double  value;
    long    SIZE;
    imageID ID;
    char    fname[200];
    char    fname1[200];

    ID   = image_ID(ID_name);
    SIZE = data.image[ID].md[0].size[0];
    make_disk("disktmp", SIZE, SIZE, 0.5 * SIZE, 0.5 * SIZE, radius1);

    sprintf(fname, "/RAID0/tmp/Zernike/Z_%ld", zer_nb);
    sprintf(fname1, "Z_%ld", zer_nb);

    if((ID = image_ID(fname1)) == -1)
    {
        if(file_exists(fname) == 1)
        {
            imageID IDtmp;
            load_fits(fname, fname1, 1, &IDtmp);
        }
        else
        {
            mk_zer(fname1, SIZE, zer_nb, radius);
        }
    }

    arith_image_mult(fname1, ID_name, "mult_tmp");
    arith_image_mult("mult_tmp", "disktmp", "mult_tmp1");
    value = arith_image_total("mult_tmp1") / arith_image_total("disktmp");
    /* printf("value is %e\n",value);*/
    delete_image_ID("disktmp", DELETE_IMAGE_ERRMODE_WARNING);
    /*  delete_image_ID("zernike_tmp");*/
    delete_image_ID("mult_tmp", DELETE_IMAGE_ERRMODE_WARNING);
    delete_image_ID("mult_tmp1", DELETE_IMAGE_ERRMODE_WARNING);

    return (value);
}

int get_zerns(const char *ID_name, long max_zer, double radius)
{
    for(long i = 0; i < max_zer; i++)
    {
        printf("%ld %e\n", i, get_zer(ID_name, i, radius));
    }

    return (0);
}

int get_zern_array(const char *ID_name,
                   long        max_zer,
                   double      radius,
                   double     *array)
{
    for(long i = 0; i < max_zer; i++)
    {
        double tmp;

        tmp = get_zer(ID_name, i, radius);
        /*     printf("%ld %e\n",i,tmp);*/
        array[i] = tmp;
    }

    return (0);
}

int remove_zerns(const char *ID_name,
                 const char *ID_name_out,
                 int         max_zer,
                 double      radius)
{
    imageID ID;
    long    SIZE;

    copy_image_ID(ID_name, ID_name_out, 0);
    ID   = image_ID(ID_name);
    SIZE = data.image[ID].md[0].size[0];
    for(int i = 0; i < max_zer; i++)
    {
        double coeff;

        mk_zer("zer_tmp", SIZE, i, radius);
        coeff = -1.0 * get_zer(ID_name, i, radius);
        arith_image_cstmult_inplace("zer_tmp", coeff);
        arith_image_add(ID_name_out, "zer_tmp", "tmp");
        delete_image_ID(ID_name_out, DELETE_IMAGE_ERRMODE_WARNING);
        copy_image_ID("tmp", ID_name_out, 0);
        delete_image_ID("tmp", DELETE_IMAGE_ERRMODE_WARNING);
        delete_image_ID("zer_tmp", DELETE_IMAGE_ERRMODE_WARNING);
    }
    return (0);
}

long ZERNIKEPOLYN_rmPiston(const char *ID_name, const char *IDmask_name)
{
    imageID ID, IDmask;
    long    xsize, ysize, zsize, xysize;
    long    ii, kk;

    ID     = image_ID(ID_name);
    xsize  = data.image[ID].md[0].size[0];
    ysize  = data.image[ID].md[0].size[1];
    zsize  = data.image[ID].md[0].size[2];
    xysize = xsize * ysize;

    IDmask = image_ID(IDmask_name);

    for(kk = 0; kk < zsize; kk++)
    {
        double tot1, tot2, ave;

        tot1 = 0.0;
        tot2 = 0.0;
        for(ii = 0; ii < xysize; ii++)
        {
            tot1 += data.image[ID].array.F[kk * xysize + ii] *
                    data.image[IDmask].array.F[ii];
            tot2 += data.image[IDmask].array.F[ii];
        }
        ave = tot1 / tot2;
        for(ii = 0; ii < xysize; ii++)
        {
            data.image[ID].array.F[kk * xysize + ii] -= ave;
        }
    }

    return (ID);
}

int remove_TTF(const char *ID_name, const char *ID_name_out, double radius)
{
    int     i;
    double  coeff;
    imageID ID;
    long    SIZE;

    //  printf("-- %s  --- %s --\n",ID_name,ID_name_out);
    copy_image_ID(ID_name, ID_name_out, 0);
    ID   = image_ID(ID_name);
    SIZE = data.image[ID].md[0].size[0];
    make_disk("disktmpttf", SIZE, SIZE, 0.5 * SIZE, 0.5 * SIZE, radius);
    //  list_image_ID();
    for(i = 0; i < 5; i++)
    {
        if((i == 0) || (i == 1) || (i == 2) || (i == 4))
        {
            mk_zer("zer_tmp", SIZE, i, radius);
            arith_image_mult("zer_tmp", ID_name, "mult_tmp");
            //coeff = arith_image_total("mult_tmp")/arith_image_total("disktmpttf");
            delete_image_ID("mult_tmp", DELETE_IMAGE_ERRMODE_WARNING);
            coeff               = -1.0 * get_zer(ID_name, i, radius);
            data.DOUBLEARRAY[i] = coeff;
            mk_zer("zer_tmpu", SIZE, i, radius);
            arith_image_cstmult_inplace("zer_tmpu", coeff);
            //	  basic_add(ID_name_out,"zer_tmpu","tmp",0,0);
            arith_image_add(ID_name_out, "zer_tmpu", "tmp");
            delete_image_ID(ID_name_out, DELETE_IMAGE_ERRMODE_WARNING);
            copy_image_ID("tmp", ID_name_out, 0);
            delete_image_ID("tmp", DELETE_IMAGE_ERRMODE_WARNING);
            delete_image_ID("zer_tmp", DELETE_IMAGE_ERRMODE_WARNING);
            delete_image_ID("zer_tmpu", DELETE_IMAGE_ERRMODE_WARNING);
        }
    }
    delete_image_ID("disktmpttf", DELETE_IMAGE_ERRMODE_WARNING);

    return (0);
}

double fit_zer(const char *ID_name,
               long        maxzer_nb,
               double      radius,
               double     *zvalue,
               double     *residual)
{
    long    SIZE;
    imageID ID, IDZ, IDdisk;
    char    fname[200];
    char    fname1[200];
    long    i;
    long    ii;
    double  tmp;
    double  disktot = 0.0;
    long    NBpass, pass;
    double  value;
    double  residualf = 0.0;

    NBpass = 10;

    copy_image_ID(ID_name, "resid", 0);

    ID   = image_ID("resid");
    SIZE = data.image[ID].md[0].size[0];
    IDdisk =
        make_disk("dtmp", SIZE, SIZE, 0.5 * SIZE, 0.5 * SIZE, 0.999 * radius);

    for(ii = 0; ii < SIZE * SIZE; ii++)
        if(data.image[IDdisk].array.F[ii] > 0.5)
        {
            disktot += 1.0;
        }

    for(i = 0; i < maxzer_nb; i++)
    {
        residual[i] = 0.0;
        zvalue[i]   = 0.0;
    }

    for(pass = 0; pass < NBpass; pass++)
    {
        for(i = 0; i < maxzer_nb; i++)
        {
            sprintf(fname, "/RAID0/tmp/Zernike/Z_%ld", i);
            sprintf(fname1, "Z_%ld", i);

            if((IDZ = image_ID(fname1)) == -1)
            {
                if(file_exists(fname) == 1)
                {
                    load_fits(fname, fname1, 1, &IDZ);
                }
                else
                {
                    IDZ = mk_zer(fname1, SIZE, i, radius);
                }
            }
            tmp = 0.0;
            for(ii = 0; ii < SIZE * SIZE; ii++)
                if(data.image[IDdisk].array.F[ii] > 0.5)
                {
                    tmp += data.image[IDZ].array.F[ii] *
                           data.image[ID].array.F[ii];
                }
            value = tmp / disktot;

            for(ii = 0; ii < SIZE * SIZE; ii++)
                if(data.image[IDdisk].array.F[ii] > 0.5)
                {
                    data.image[ID].array.F[ii] -=
                        value * data.image[IDZ].array.F[ii];
                }
            zvalue[i] += value;
            tmp = 0.0;
            for(ii = 0; ii < SIZE * SIZE; ii++)
                if(data.image[IDdisk].array.F[ii] > 0.5)
                {
                    tmp +=
                        data.image[ID].array.F[ii] * data.image[ID].array.F[ii];
                }

            residualf = sqrt(tmp / disktot);
        }
    }

    residual[maxzer_nb - 1] = residualf;
    for(i = maxzer_nb - 1; i > 0; i--)
    {
        residual[i - 1] =
            sqrt(residual[i] * residual[i] + zvalue[i] * zvalue[i]);
    }

    for(ii = 0; ii < SIZE * SIZE; ii++)
    {
        if(data.image[IDdisk].array.F[ii] < 0.5)
        {
            data.image[ID].array.F[ii] = 0.0;
        }
    }

    delete_image_ID("dtmp", DELETE_IMAGE_ERRMODE_WARNING);

    return (residualf);
}
