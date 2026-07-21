// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file ZernikePolyn_fit.c
 * @brief Zernike polynomial fitting and analysis
 *
 * Contains get_zer, get_zer_crop, get_zerns,
 * get_zern_array, rmPiston, remove_TTF, fit_zer.
 *
 * @see ZernikePolyn.c for creation functions.
 */

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <stdbool.h>
#include <unistd.h>

#ifdef MILK_NO_CLI
#    include "CLIcore_standalone.h"
#else
#    include "libmilkdata/milkdata.h"
#    include "milkDebugTools.h"
#    include "fps.h"
#    include "ImageStreamIO/ImageStreamIO.h"
#endif
#include "COREMOD_arith/COREMOD_arith.h"
#include "COREMOD_iofits/COREMOD_iofits.h"
#include "COREMOD_memory/COREMOD_memory.h"

#include "image_gen/image_gen.h"

#include "zernike.h"
#include "zernike_value.h"

#define PI 3.14159265358979323846264338328

/* Forward declarations from ZernikePolyn.c */
imageID mk_zer(const char *ID_name, long SIZE, long zer_nb, float rpix);

double get_zer(const char *ID_name, long zer_nb, double radius)
{
    double  value;
    long    SIZE;
    imageID ID;
    char    fname[200];
    char    fname1[200];

    ID   = image_ID(ID_name, dcimg, dcnimg);
    SIZE = dcimg[ID].md[0].size[0];
    make_disk("disktmp", SIZE, SIZE, 0.5 * SIZE, 0.5 * SIZE, radius);

    snprintf(fname, sizeof(fname), "/RAID0/tmp/Zernike/Z_%ld", zer_nb);
    snprintf(fname1, sizeof(fname1), "Z_%ld", zer_nb);

    if ((ID = image_ID(fname1, dcimg, dcnimg)) == -1)
    {
        if (file_exists(fname) == 1)
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

double get_zer_crop(const char *ID_name, long zer_nb, double radius, double radius1)
{
    double  value;
    long    SIZE;
    imageID ID;
    char    fname[200];
    char    fname1[200];

    ID   = image_ID(ID_name, dcimg, dcnimg);
    SIZE = dcimg[ID].md[0].size[0];
    make_disk("disktmp", SIZE, SIZE, 0.5 * SIZE, 0.5 * SIZE, radius1);

    snprintf(fname, sizeof(fname), "/RAID0/tmp/Zernike/Z_%ld", zer_nb);
    snprintf(fname1, sizeof(fname1), "Z_%ld", zer_nb);

    if ((ID = image_ID(fname1, dcimg, dcnimg)) == -1)
    {
        if (file_exists(fname) == 1)
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
    for (long i = 0; i < max_zer; i++)
    {
        printf("%ld %e\n", i, get_zer(ID_name, i, radius));
    }

    return (0);
}

int get_zern_array(const char *ID_name, long max_zer, double radius, double *array)
{
    for (long i = 0; i < max_zer; i++)
    {
        double tmp;

        tmp = get_zer(ID_name, i, radius);
        /*     printf("%ld %e\n",i,tmp);*/
        array[i] = tmp;
    }

    return (0);
}

int remove_zerns(const char *ID_name, const char *ID_name_out, int max_zer, double radius)
{
    imageID ID;
    long    SIZE;

    copy_image_ID(ID_name, ID_name_out, 0);
    ID   = image_ID(ID_name, dcimg, dcnimg);
    SIZE = dcimg[ID].md[0].size[0];
    for (int i = 0; i < max_zer; i++)
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

    ID     = image_ID(ID_name, dcimg, dcnimg);
    xsize  = dcimg[ID].md[0].size[0];
    ysize  = dcimg[ID].md[0].size[1];
    zsize  = dcimg[ID].md[0].size[2];
    xysize = xsize * ysize;

    IDmask = image_ID(IDmask_name, dcimg, dcnimg);

    for (kk = 0; kk < zsize; kk++)
    {
        double tot1, tot2, ave;

        tot1 = 0.0;
        tot2 = 0.0;
        for (ii = 0; ii < xysize; ii++)
        {
            tot1 += dcimg[ID].array.F[kk * xysize + ii] * dcimg[IDmask].array.F[ii];
            tot2 += dcimg[IDmask].array.F[ii];
        }
        ave = tot1 / tot2;
        for (ii = 0; ii < xysize; ii++)
        {
            dcimg[ID].array.F[kk * xysize + ii] -= ave;
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
    ID   = image_ID(ID_name, dcimg, dcnimg);
    SIZE = dcimg[ID].md[0].size[0];
    make_disk("disktmpttf", SIZE, SIZE, 0.5 * SIZE, 0.5 * SIZE, radius);
    //  list_image_ID();
    for (i = 0; i < 5; i++)
    {
        if ((i == 0) || (i == 1) || (i == 2) || (i == 4))
        {
            mk_zer("zer_tmp", SIZE, i, radius);
            arith_image_mult("zer_tmp", ID_name, "mult_tmp");
            //coeff = arith_image_total("mult_tmp")/arith_image_total("disktmpttf");
            delete_image_ID("mult_tmp", DELETE_IMAGE_ERRMODE_WARNING);
            coeff          = -1.0 * get_zer(ID_name, i, radius);
            dcdoublearr[i] = coeff;
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

double fit_zer(const char *ID_name, long maxzer_nb, double radius, double *zvalue, double *residual)
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

    ID     = image_ID("resid", dcimg, dcnimg);
    SIZE   = dcimg[ID].md[0].size[0];
    IDdisk = make_disk("dtmp", SIZE, SIZE, 0.5 * SIZE, 0.5 * SIZE, 0.999 * radius);

    for (ii = 0; ii < SIZE * SIZE; ii++)
    {
        if (dcimg[IDdisk].array.F[ii] > 0.5f)
        {
            disktot += 1.0;
        }
    }

    for (i = 0; i < maxzer_nb; i++)
    {
        residual[i] = 0.0;
        zvalue[i]   = 0.0;
    }

    for (pass = 0; pass < NBpass; pass++)
    {
        for (i = 0; i < maxzer_nb; i++)
        {
            snprintf(fname, sizeof(fname), "/RAID0/tmp/Zernike/Z_%ld", i);
            snprintf(fname1, sizeof(fname1), "Z_%ld", i);

            if ((IDZ = image_ID(fname1, dcimg, dcnimg)) == -1)
            {
                if (file_exists(fname) == 1)
                {
                    load_fits(fname, fname1, 1, &IDZ);
                }
                else
                {
                    IDZ = mk_zer(fname1, SIZE, i, radius);
                }
            }
            tmp = 0.0;
            for (ii = 0; ii < SIZE * SIZE; ii++)
            {
                if (dcimg[IDdisk].array.F[ii] > 0.5f)
                {
                    tmp += dcimg[IDZ].array.F[ii] * dcimg[ID].array.F[ii];
                }
            }
            value = tmp / disktot;

            for (ii = 0; ii < SIZE * SIZE; ii++)
            {
                if (dcimg[IDdisk].array.F[ii] > 0.5f)
                {
                    dcimg[ID].array.F[ii] -= value * dcimg[IDZ].array.F[ii];
                }
            }
            zvalue[i] += value;
            tmp = 0.0;
            for (ii = 0; ii < SIZE * SIZE; ii++)
            {
                if (dcimg[IDdisk].array.F[ii] > 0.5f)
                {
                    tmp += dcimg[ID].array.F[ii] * dcimg[ID].array.F[ii];
                }
            }

            residualf = sqrt(tmp / disktot);
        }
    }

    residual[maxzer_nb - 1] = residualf;
    for (i = maxzer_nb - 1; i > 0; i--)
    {
        residual[i - 1] = sqrt(residual[i] * residual[i] + zvalue[i] * zvalue[i]);
    }

    for (ii = 0; ii < SIZE * SIZE; ii++)
    {
        if (dcimg[IDdisk].array.F[ii] < 0.5f)
        {
            dcimg[ID].array.F[ii] = 0.0f;
        }
    }

    delete_image_ID("dtmp", DELETE_IMAGE_ERRMODE_WARNING);

    return (residualf);
}
