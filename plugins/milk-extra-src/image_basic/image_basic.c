/**
 * @file    image_basic.c
 * @brief   basic image functions
 *
 * Simple image routines
 *
 */

/* ================================================================== */
/* ================================================================== */
/*            MODULE INFO                                             */
/* ================================================================== */
/* ================================================================== */

// module default short name
// all CLI calls to this module functions will be <shortname>.<funcname>
// if set to "", then calls use <funcname>
#define MODULE_SHORTNAME_DEFAULT "imgbasic"

// Module short description
#define MODULE_DESCRIPTION "standard image operations"

//#include <stdint.h>
//#include <string.h>
//#include <math.h>
//#include <stdlib.h>
//#include <stdio.h>
//#include <errno.h>
//#include <unistd.h>
//#include <sched.h>

//#include <fitsio.h>  /* required by every program that uses CFITSIO  */

#include "CommandLineInterface/CLIcore.h"
//#include "COREMOD_tools/COREMOD_tools.h"
//#include "COREMOD_memory/COREMOD_memory.h"
//#include "COREMOD_iofits/COREMOD_iofits.h"
//#include "COREMOD_arith/COREMOD_arith.h"

/*
#include "fft/fft.h"
#include "image_filter/image_filter.h"
#include "image_gen/image_gen.h"
#include "info/info.h"
#include "kdtree/kdtree.h"
#include "linopt_imtools/linopt_imtools.h"
#include "statistic/statistic.h"
*/

//#include "image_basic/image_basic.h"

#include "cubecollapse.h"
#include "im3Dto2D.h"
#include "image_add.h"
#include "imcontract.h"
#include "imexpand.h"
#include "imgetcircasym.h"
#include "imgetcircsym.h"
#include "imresize.h"
#include "imrotate.h"
#include "imswapaxis2D.h"
#include "indexmap.h"
#include "loadfitsimgcube.h"
#include "streamfeed.h"
#include "streamrecord.h"

/*
#define SBUFFERSIZE 1000

#define SWAP(x,y)  temp=(x);x=(y);y=temp;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


char errmsg[SBUFFERSIZE];
*/

/* ================================================================== */
/* ================================================================== */
/*            INITIALIZE LIBRARY                                      */
/* ================================================================== */
/* ================================================================== */

// Module initialization macro in CLIcore.h
// macro argument defines module name for bindings
//
INIT_MODULE_LIB(image_basic)

/* ================================================================== */
/* ================================================================== */
/*            COMMAND LINE INTERFACE (CLI) FUNCTIONS                  */
/* ================================================================== */
/* ================================================================== */

static errno_t init_module_CLI()
{

    imswapaxis2D_addCLIcmd();
    im3Dto2D_addCLIcmd();
    image_add_addCLIcmd();
    imexpand_addCLIcmd();
    imgetcircsym_addCLIcmd();
    imgetcircasym_addCLIcmd();
    imresize_addCLIcmd();
    imcontract_addCLIcmd();
    imrotate_addCLIcmd();
    loadfitsimgcube_addCLIcmd();
    streamfeed_addCLIcmd();
    streamrecord_addCLIcmd();
    cubecollapse_addCLIcmd();

    // add atexit functions here

    return RETURN_SUCCESS;
}

/*

int basic_mincontract(
    __attribute__((unused)) const char *ID_name,
    __attribute__((unused)) uint8_t     axis,
    __attribute__((unused)) const char *out_name)
{



    return(0);
}



int basic_lmin_im(
    const char *ID_name,
    const char *out_name
)
{
    imageID IDin, IDout;
    long ii, jj;
    long naxes[2];
    float tmp;

    IDin = image_ID(ID_name);
    naxes[0] = data.image[IDin].md[0].size[0];
    naxes[1] = data.image[IDin].md[0].size[1];

    IDout = create_2Dimage_ID(out_name, naxes[0], 1);

    for(ii = 0; ii < naxes[0]; ii++)
    {
        data.image[IDout].array.F[ii] = data.image[IDin].array.F[ii];
    }

    for(jj = 1; jj < naxes[1]; jj++)
        for(ii = 0; ii < naxes[0]; ii++)
        {
            tmp = data.image[IDin].array.F[jj * naxes[0] + ii];
            if(tmp < data.image[IDout].array.F[ii])
            {
                data.image[IDout].array.F[ii] = tmp;
            }
        }

    return(0);
}




int basic_lmax_im(
    const char *ID_name,
    const char *out_name
)
{
    imageID IDin, IDout;
    long ii, jj;
    long naxes[2];
    float tmp;

    IDin = image_ID(ID_name);
    naxes[0] = data.image[IDin].md[0].size[0];
    naxes[1] = data.image[IDin].md[0].size[1];

    IDout = create_2Dimage_ID(out_name, naxes[0], 1);

    for(ii = 0; ii < naxes[0]; ii++)
    {
        data.image[IDout].array.F[ii] = data.image[IDin].array.F[ii];
    }

    for(jj = 1; jj < naxes[1]; jj++)
        for(ii = 0; ii < naxes[0]; ii++)
        {
            tmp = data.image[IDin].array.F[jj * naxes[0] + ii];
            if(tmp > data.image[IDout].array.F[ii])
            {
                data.image[IDout].array.F[ii] = tmp;
            }
        }

    return(0);
}







long basic_diff(const char *ID_name1, const char *ID_name2,
                const char *ID_name_out, long off1, long off2)
{
    int ID1, ID2;
    int ID_out;
    long ii, jj;
    long naxes1[2], naxes2[2], naxes[2];
    long xmin, ymin, xmax, ymax; // extrema in the ID1 coordinates

    ID1 = image_ID(ID_name1);
    ID2 = image_ID(ID_name2);
    naxes1[0] = data.image[ID1].md[0].size[0];
    naxes1[1] = data.image[ID1].md[0].size[1];
    naxes2[0] = data.image[ID2].md[0].size[0];
    naxes2[1] = data.image[ID2].md[0].size[1];

    printf("add called with %s ( %ld x %ld ) %s ( %ld x %ld ) and offset ( %ld x %ld )\n",
           ID_name1, naxes1[0], naxes1[1], ID_name2, naxes2[0], naxes2[1], off1, off2);
    xmin = 0;
    if(off1 < 0)
    {
        xmin = off1;
    }
    ymin = 0;
    if(off2 < 0)
    {
        ymin = off2;
    }
    xmax = naxes1[0];
    if((naxes2[0] + off1) > naxes1[0])
    {
        xmax = (naxes2[0] + off1);
    }
    ymax = naxes1[1];
    if((naxes2[1] + off2) > naxes1[1])
    {
        ymax = (naxes2[1] + off2);
    }

    create_2Dimage_ID(ID_name_out, (xmax - xmin), (ymax - ymin));
    ID_out = image_ID(ID_name_out);
    naxes[0] = data.image[ID_out].md[0].size[0];
    naxes[1] = data.image[ID_out].md[0].size[1];

    for(jj = 0; jj < naxes[1]; jj++)
        for(ii = 0; ii < naxes[0]; ii++)
        {
            {
                data.image[ID_out].array.F[jj * naxes[0] + ii] = 0;
                // if pixel is in ID1
                if(((ii + xmin) >= 0) && ((ii + xmin) < naxes1[0]))
                    if(((jj + ymin) >= 0) && ((jj + ymin) < naxes1[1]))
                    {
                        data.image[ID_out].array.F[jj * naxes[0] + ii] += data.image[ID1].array.F[(jj +
                                ymin) * naxes1[0] + (ii + xmin)];
                    }
                // if pixel is in ID2
                if(((ii + xmin - off1) >= 0) && ((ii + xmin - off1) < naxes2[0]))
                    if(((jj + ymin - off2) >= 0) && ((jj + ymin - off2) < naxes2[1]))
                    {
                        data.image[ID_out].array.F[jj * naxes[0] + ii] -= data.image[ID2].array.F[(jj +
                                ymin - off2) * naxes2[0] + (ii + xmin - off1)];
                    }
            }
        }
    return(ID_out);
}



int basic_add_cst(const char *ID_name, float f1, int sign) // add a constant
{
    int ID;
    long ii, jj;
    long naxes[2];

    ID = image_ID(ID_name);
    naxes[0] = data.image[ID].md[0].size[0];
    naxes[1] = data.image[ID].md[0].size[1];

    for(jj = 0; jj < naxes[1]; jj++)
        for(ii = 0; ii < naxes[0]; ii++)
        {
            data.image[ID].array.F[jj * naxes[0] + ii] = data.image[ID].array.F[jj *
                    naxes[0] + ii] + f1 * sign;
        }

    return(0);
}



// extracts a n1xn2 subwindow of an image at offset n3,n4
imageID basic_extract(
    const char *ID_in_name,
    const char *ID_out_name,
    long n1,
    long n2,
    long n3,
    long n4
)
{
    imageID ID_in;
    imageID ID_out;
    long ii, jj;
    char name[SBUFFERSIZE];
    int n;

    ID_in = image_ID(ID_in_name);
    n = snprintf(name, SBUFFERSIZE, "%s", ID_out_name);
    if(n >= SBUFFERSIZE)
    {
        PRINT_ERROR("Attempted to write string buffer with too many characters");
    }

    create_2Dimage_ID(name, n1, n2);
    fflush(stdout);
    ID_out = image_ID(ID_out_name);
    for(ii = 0; ii < n1; ii++)
        for(jj = 0; jj < n2; jj++)
        {
            data.image[ID_out].array.F[jj * n1 + ii] = data.image[ID_in].array.F[(jj + n4) *
                    data.image[ID_in].md[0].size[0] + ii + n3];
        }

    return(ID_out);
}



int basic_trunc_circ(const char *ID_name, float f1)
{
    imageID ID;
    long ii, jj;
    long naxes[2];

    ID = image_ID(ID_name);
    naxes[0] = data.image[ID].md[0].size[0];
    naxes[1] = data.image[ID].md[0].size[1];

    for(jj = 0; jj < naxes[1]; jj++)
        for(ii = 0; ii < naxes[0]; ii++)
        {
            data.image[ID].array.F[jj * naxes[0] + ii] = fmod(data.image[ID].array.F[jj *
                    naxes[0] + ii], f1);
        }

    return(0);
}



imageID basic_zoom2(
    const char *ID_name,
    const char *ID_name_out
)
{
    imageID ID;
    imageID ID_out; // ID for the output image
    uint32_t naxes[2], naxes_out[2];
    char lstring[SBUFFERSIZE];
    int n;

    n = snprintf(lstring, SBUFFERSIZE, "%s=%s*1", ID_name_out, ID_name);
    if(n >= SBUFFERSIZE)
    {
        PRINT_ERROR("Attempted to write string buffer with too many characters");
    }


    execute_arith(lstring);
    ID = image_ID(ID_name);
    naxes[0] = data.image[ID].md[0].size[0];
    naxes[1] = data.image[ID].md[0].size[1];
    naxes_out[0] = naxes[0];
    naxes_out[1] = naxes[1];
    ID_out = image_ID(ID_name_out);

    for(uint32_t jj = 0; jj < naxes[1] / 2; jj++)
        for(uint32_t ii = 0; ii < naxes[0] / 2; ii++)
        {
            data.image[ID_out].array.F[(2 * jj)*naxes_out[0] + (2 * ii)] =
                data.image[ID].array.F[(jj + naxes[1] / 4) * naxes[0] + (ii + naxes[1] / 4)];
            data.image[ID_out].array.F[(2 * jj + 1)*naxes_out[0] + (2 * ii)] = 0.5 *
                    (data.image[ID].array.F[(jj + naxes[1] / 4) * naxes[0] +
                                            (ii + naxes[1] / 4)] + data.image[ID].array.F[(jj + naxes[1] / 4 + 1) *
                                                    naxes[0] + (ii + naxes[1] / 4)]);
            data.image[ID_out].array.F[(2 * jj)*naxes_out[0] + (2 * ii + 1)] = 0.5 *
                    (data.image[ID].array.F[(jj + naxes[1] / 4) * naxes[0] +
                                            (ii + naxes[1] / 4)] + data.image[ID].array.F[(jj + naxes[1] / 4) * naxes[0] +
                                                    (ii + naxes[1] / 4 + 1)]);
            data.image[ID_out].array.F[(2 * jj + 1)*naxes_out[0] + (2 * ii + 1)] = 0.25 *
                    (data.image[ID].array.F[(jj + naxes[1] / 4) * naxes[0] +
                                            (ii + naxes[1] / 4)] + data.image[ID].array.F[(jj + naxes[1] / 4) * naxes[0] +
                                                    (ii + naxes[1] / 4 + 1)] + data.image[ID].array.F[(jj + naxes[1] / 4 + 1) *
                                                            naxes[0] + (ii + naxes[1] / 4)] + data.image[ID].array.F[(jj + naxes[1] / 4 + 1)
                                                                    * naxes[0] + (ii + naxes[1] / 4 + 1)]);
        }

    return(ID_out);
}




long basic_average_column(
    __attribute__((unused)) const char *ID_name,
    __attribute__((unused)) const char *IDout_name
)
{
    long IDout = -1;

    // TO BE WRITTEN

    return(IDout);
}



imageID basic_padd(
    const char *ID_name,
    const char *ID_name_out,
    int n1,
    int n2
)
{
    imageID ID;
    imageID ID_out; // ID for the output image
    uint32_t naxes[2], naxes_out[2];

    ID = image_ID(ID_name);
    naxes[0] = data.image[ID].md[0].size[0];
    naxes[1] = data.image[ID].md[0].size[1];

    naxes_out[0] = naxes[0] + 2 * n1;
    naxes_out[1] = naxes[1] + 2 * n2;

    create_2Dimage_ID(ID_name_out, naxes_out[0], naxes_out[1]);
    ID_out = image_ID(ID_name_out);

    for(uint32_t jj = 0; jj < naxes[1]; jj++)
        for(uint32_t ii = 0; ii < naxes[0]; ii++)
        {
            data.image[ID_out].array.F[(jj + n2)*naxes_out[0] + ii + n1] =
                data.image[ID].array.F[jj * naxes[0] + ii];
        }

    return(ID_out);
}


// flip an image relative to the horizontal axis
imageID basic_fliph(
    const char *ID_name
)
{
    imageID ID;
    long naxes[2];
    uint32_t tmp_long;
    float temp;

    ID = image_ID(ID_name);
    naxes[0] = data.image[ID].md[0].size[0];
    naxes[1] = data.image[ID].md[0].size[1];

    temp = 0.0;
    tmp_long = (uint32_t) (naxes[1] / 2);
    for(uint32_t jj = 0; jj < tmp_long; jj++)
        for(uint32_t ii = 0; ii < naxes[0]; ii++)
        {
            temp = data.image[ID].array.F[jj * naxes[0] + ii];
            data.image[ID].array.F[jj * naxes[0] + ii] = data.image[ID].array.F[(naxes[1] -
                    jj - 1) * naxes[0] + ii];
            data.image[ID].array.F[(naxes[1] - jj - 1)*naxes[0] + ii] = temp;
        }
    return(ID);
}




// flip an image relative to the vertical axis
imageID basic_flipv(
    const char *ID_name
)
{
    imageID ID;
    uint32_t naxes[2];
    uint32_t tmp_long;
    float temp;

    ID = image_ID(ID_name);
    naxes[0] = data.image[ID].md[0].size[0];
    naxes[1] = data.image[ID].md[0].size[1];

    temp = 0.0;
    tmp_long = (uint32_t) (naxes[0] / 2);
    for(uint32_t jj = 0; jj < naxes[1]; jj++)
        for(uint32_t ii = 0; ii < tmp_long; ii++)
        {
            temp = data.image[ID].array.F[jj * naxes[0] + ii];
            data.image[ID].array.F[jj * naxes[0] + ii] = data.image[ID].array.F[jj *
                    naxes[0] + (naxes[0] - ii - 1)];
            data.image[ID].array.F[jj * naxes[0] + (naxes[0] - ii - 1)] = temp;
        }
    return(ID);
}



// flip an image horizontally and vertically
imageID basic_fliphv(
    const char *ID_name
)
{
    imageID ID;
    uint32_t naxes[2];
    uint32_t tmp_long;
    float temp;

    ID = image_ID(ID_name);
    naxes[0] = data.image[ID].md[0].size[0];
    naxes[1] = data.image[ID].md[0].size[1];

    temp = 0.0;
    tmp_long = (uint32_t) (naxes[1] / 2);
    for(uint32_t jj = 0; jj < tmp_long; jj++)
        for(uint32_t ii = 0; ii < naxes[0]; ii++)
        {
            temp = data.image[ID].array.F[jj * naxes[0] + ii];
            data.image[ID].array.F[jj * naxes[0] + ii] = data.image[ID].array.F[(naxes[1] -
                    jj - 1) * naxes[0] + (naxes[0] - ii - 1)];
            data.image[ID].array.F[(naxes[1] - jj - 1)*naxes[0] + (naxes[0] - ii - 1)] =
                temp;
        }
    return(ID);
}


// median of the images specified in options, output is ID_name
int basic_median(
    const char *ID_name,
    const char *options
)
{
    unsigned int Nb_files;
    imageID ID;
    unsigned int file_nb;
    int str_pos;
    imageID *IDn;
    char file_name[STRINGMAXLEN_FILENAME];
    uint32_t naxes[2];
    int medianpt = 0;

    unsigned long i, j;
    float *array;

    Nb_files = 0;
    i = 0;
    str_pos = 0;
    while((options[i + str_pos] != '\n') && (options[i + str_pos] != '\0'))
    {
        if(options[i + str_pos] == ' ')
        {
            Nb_files += 1;
        }
        i++;
    }

    printf("%d files\n", Nb_files);
    medianpt = (int)(0.5 * (Nb_files - 1));

    IDn = (imageID *) malloc(sizeof(imageID) * Nb_files);
    if(IDn == NULL)
    {
        C_ERRNO = errno;
        PRINT_ERROR("malloc() error");
        exit(0);
    }

    array = (float *) malloc(sizeof(float) * Nb_files);
    if(array == NULL)
    {
        C_ERRNO = errno;
        PRINT_ERROR("malloc() error");
        exit(0);
    }

    i = 1;
    j = 0;
    file_nb = 0;
    while(file_nb < Nb_files)
    {
        if((options[i + str_pos] == ' ') || (options[i + str_pos] == '\0')
                || (options[i + str_pos] == '\n'))
        {
            file_name[j] = '\0';
            IDn[file_nb] = image_ID(file_name);
            printf("%d %s \n", (int) IDn[file_nb], file_name);
            file_nb += 1;
            j = 0;
        }
        else
        {
            file_name[j] = options[i + str_pos];
            j++;
        }
        i++;
    }

    naxes[0] = data.image[IDn[0]].md[0].size[0];
    naxes[1] = data.image[IDn[0]].md[0].size[1];
    create_2Dimage_ID(ID_name, naxes[0], naxes[1]);
    ID = image_ID(ID_name);

    for(uint32_t jj = 0; jj < naxes[1]; jj++)
        for(uint32_t ii = 0; ii < naxes[0]; ii++)
        {
            for(i = 0; i < Nb_files; i++)
            {
                array[i] = data.image[IDn[i]].array.F[jj * naxes[0] + ii];
            }
            quick_sort_float(array, Nb_files);
            if((0.5 * (Nb_files - 1) - medianpt) < 0.1)
            {
                data.image[ID].array.F[jj * naxes[0] + ii] = array[medianpt];
            }
            else
            {
                data.image[ID].array.F[jj * naxes[0] + ii] = 0.5 * array[medianpt] + 0.5 *
                        array[medianpt + 1];
            }
        }

    printf("%d %d \n", Nb_files, medianpt);
    if((0.5 * (Nb_files - 1) - medianpt) > 0.1)
    {
        printf("median of an even number of number: average of the 2 closest \n");
    }

    free(IDn);
    free(array);
    return(0);
}


imageID basic_renorm_max(
    const char *ID_name
)
{
    imageID ID;
    long ii, jj;
    long naxes[2];
    float max;

    ID = image_ID(ID_name);
    naxes[0] = data.image[ID].md[0].size[0];
    naxes[1] = data.image[ID].md[0].size[1];

    max = 0;

    for(jj = 0; jj < naxes[1]; jj++)
        for(ii = 0; ii < naxes[0]; ii++)
            if(data.image[ID].array.F[jj * naxes[0] + ii] > max)
            {
                max = data.image[ID].array.F[jj * naxes[0] + ii];
            }

    if(max != 0)
    {
        for(jj = 0; jj < naxes[1]; jj++)
            for(ii = 0; ii < naxes[0]; ii++)
            {
                data.image[ID].array.F[jj * naxes[0] + ii] /= max;
            }
    }

    return(ID);
}




int basic_translate(
    const char *ID_name,
    const char *ID_out,
    float xtransl,
    float ytransl
)
{
    imageID ID;
    long naxes[2];
    long onaxes[2];
    long ii, jj;
    int n0, n1;
    float coeff;

    ID = image_ID(ID_name);
    naxes[0] = data.image[ID].md[0].size[0];
    naxes[1] = data.image[ID].md[0].size[1];
    onaxes[0] = naxes[0];
    onaxes[1] = naxes[1];
    n0 = (int)((log10(naxes[0]) / log10(2)) + 0.01);
    n1 = (int)((log10(naxes[0]) / log10(2)) + 0.01);

    if((n0 == n1) && (naxes[0] == (int) pow(2, n0))
            && (naxes[1] == (int) pow(2, n1)))
    {
        create_2Dimage_ID("zero_tmp", naxes[0], naxes[1]);
        pupfft(ID_name, "zero_tmp", "out_transl_re_tmp", "out_transl_im_tmp", "-reim");
        delete_image_ID("zero_tmp");
        mk_amph_from_reim("out_transl_re_tmp", "out_transl_im_tmp",
                          "out_transl_ampl_tmp", "out_transl_pha_tmp", 0);
        delete_image_ID("out_transl_re_tmp");
        delete_image_ID("out_transl_im_tmp");

        ID = image_ID("out_transl_pha_tmp");
        for(jj = 1; jj < naxes[1]; jj++)
            for(ii = 1; ii < naxes[0]; ii++)
            {
                data.image[ID].array.F[jj * naxes[0] + ii] -= xtransl * 2.0 * M_PI /
                        naxes[0] * (ii - naxes[0] / 2) + ytransl * 2.0 * M_PI / naxes[1] *
                        (jj - naxes[1] / 2);
            }

        coeff = 1.0 / (naxes[0] * naxes[1]);
        ID = image_ID("out_transl_ampl_tmp");
        for(jj = 0; jj < naxes[1]; jj++)
            for(ii = 0; ii < naxes[0]; ii++)
            {
                data.image[ID].array.F[jj * naxes[0] + ii] *= coeff;
            }


        mk_reim_from_amph("out_transl_ampl_tmp", "out_transl_pha_tmp", "out_re_tmp",
                          "out_im_tmp", 0);
        delete_image_ID("out_transl_ampl_tmp");
        delete_image_ID("out_transl_pha_tmp");
        pupfft("out_re_tmp", "out_im_tmp", ID_out, "tbe_tmp", "-reim -inv");
        delete_image_ID("out_re_tmp");
        delete_image_ID("out_im_tmp");
        delete_image_ID("tbe_tmp");
    }
    else
    {
        basic_add(ID_name, ID_name, "tmp1t", naxes[0], 0);
        basic_add("tmp1t", "tmp1t", "tmp2t", 0, naxes[1]);
        delete_image_ID("tmp1t");
        basic_extract("tmp2t", "tmp3t", pow(2, n0 + 1), pow(2, n1 + 1), 0, 0);
        delete_image_ID("tmp2t");
        ID = image_ID("tmp3t");
        naxes[0] = data.image[ID].md[0].size[0];
        naxes[1] = data.image[ID].md[0].size[1];
        create_2Dimage_ID("zero_tmp", naxes[0], naxes[1]);

        pupfft("tmp3t", "zero_tmp", "out_transl_re_tmp", "out_transl_im_tmp", "-reim");
        delete_image_ID("zero_tmp");
        delete_image_ID("tmp3t");
        mk_amph_from_reim("out_transl_re_tmp", "out_transl_im_tmp",
                          "out_transl_ampl_tmp", "out_transl_pha_tmp", 0);
        delete_image_ID("out_transl_re_tmp");
        delete_image_ID("out_transl_im_tmp");

        ID = image_ID("out_transl_pha_tmp");
        for(jj = 1; jj < naxes[1]; jj++)
            for(ii = 1; ii < naxes[0]; ii++)
            {
                data.image[ID].array.F[jj * naxes[0] + ii] -= xtransl * 2.0 * M_PI /
                        naxes[0] * (ii - naxes[0] / 2) + ytransl * 2.0 * M_PI / naxes[1] *
                        (jj - naxes[1] / 2);
            }
        coeff = 1.0 / (naxes[0] * naxes[1]);
        ID = image_ID("out_transl_ampl_tmp");
        for(jj = 0; jj < naxes[1]; jj++)
            for(ii = 0; ii < naxes[0]; ii++)
            {
                data.image[ID].array.F[jj * naxes[0] + ii] *= coeff;
            }

        mk_reim_from_amph("out_transl_ampl_tmp", "out_transl_pha_tmp", "out_re_tmp",
                          "out_im_tmp", 0);
        delete_image_ID("out_transl_ampl_tmp");
        delete_image_ID("out_transl_pha_tmp");
        pupfft("out_re_tmp", "out_im_tmp", "outtmp", "tbe_tmp", "-reim -inv");
        delete_image_ID("out_re_tmp");
        delete_image_ID("out_im_tmp");
        delete_image_ID("tbe_tmp");

        basic_extract("outtmp", ID_out, onaxes[0], onaxes[1], 0, 0);
        delete_image_ID("outtmp");
    }

    return(0);
}




float basic_correlation(
    const char *ID_name1,
    const char *ID_name2
)
{
    float correl;
    imageID ID1, ID2;
    uint32_t naxes1[2];
    uint32_t naxes2[2];

    ID1 = image_ID(ID_name1);
    naxes1[0] = data.image[ID1].md[0].size[0];
    naxes1[1] = data.image[ID1].md[0].size[1];
    ID2 = image_ID(ID_name2);
    naxes2[0] = data.image[ID2].md[0].size[0];
    naxes2[1] = data.image[ID2].md[0].size[1];

    if((naxes1[0] != naxes2[0]) || (naxes1[1] != naxes2[1]))
    {
        printf("correlation : file size do not match\n");
        exit(1);
    }
    correl = 0;

    for(uint32_t jj = 0; jj < naxes1[1]; jj++)
        for(uint32_t ii = 0; ii < naxes1[0]; ii++)
        {
            correl += (data.image[ID1].array.F[jj * naxes1[0] + ii] -
                       data.image[ID2].array.F[jj * naxes1[0] + ii]) * (data.image[ID1].array.F[jj *
                               naxes1[0] + ii] - data.image[ID2].array.F[jj * naxes1[0] + ii]);
        }

    return(correl);
}











int gauss_histo_image(
    const char *ID_name,
    const char *ID_out_name,
    float sigma,
    float center
)
{
    imageID ID, ID_out;
    uint32_t naxes[2];
    long k, k1;
    float x;
    long N = 100000;
    float *histo = NULL;
    float *imp = NULL;
    float *impr = NULL;
    float *imprinv = NULL;

    ID = image_ID(ID_name);
    naxes[0] = data.image[ID].md[0].size[0];
    naxes[1] = data.image[ID].md[0].size[1];

    histo = (float *) malloc(sizeof(float) * N);
    if(histo == NULL)
    {
        C_ERRNO = errno;
        PRINT_ERROR("malloc() error");
        exit(0);
    }

    imp = (float *) malloc(sizeof(float) * N);
    if(imp == NULL)
    {
        C_ERRNO = errno;
        PRINT_ERROR("malloc() error");
        exit(0);
    }

    impr = (float *) malloc(sizeof(float) * N);
    if(impr == NULL)
    {
        C_ERRNO = errno;
        PRINT_ERROR("malloc() error");
        exit(0);
    }

    imprinv = (float *) malloc(sizeof(float) * N);
    if(imprinv == NULL)
    {
        C_ERRNO = errno;
        PRINT_ERROR("malloc() error");
        exit(0);
    }

    for(uint32_t ii = 0; ii < N; ii++)
    {
        histo[ii] = 0.0;
        imp[ii] = 0.0;
    }

    for(uint32_t  jj = 0; jj < naxes[1]; jj++)
        for(uint32_t  ii = 0; ii < naxes[0]; ii++)
        {
            k = (long)(data.image[ID].array.F[jj * naxes[0] + ii] * N);
            if(k < 0)
            {
                k = 0;
            }
            if(k > N - 1)
            {
                k = N - 1;
            }
            histo[k]++;
        }
    for(k = 0; k < N; k++)
    {
        histo[k] *= 1.0 / naxes[1] / naxes[0];
    }

    imp[0] = histo[0];
    for(k = 1; k < N; k++)
    {
        imp[k] = imp[k - 1] + histo[k];
    }
    for(k = 0; k < N; k++)
    {
        imp[k] /= imp[N - 1];
    }


    printf("SIGMA = %f, CENTER = %f\n", sigma, center);

    for(uint32_t  ii = 0; ii < N; ii++)
    {
        x = 2.0 * (1.0 * ii / N - center);
        histo[ii] = exp(-(x * x) / sigma / sigma);
        impr[ii] = 0.0;
    }
    impr[0] = histo[0];
    for(k = 1; k < N; k++)
    {
        impr[k] = impr[k - 1] + histo[k];
    }
    for(k = 0; k < N; k++)
    {
        impr[k] /= impr[N - 1];
    }

    k = 0;
    for(k1 = 0; k1 < N; k1++)
    {
        x = 1.0 * k1 / N;
        while(impr[k] < x)
        {
            k++;
        }
        if(k > N - 1)
        {
            k = N - 1;
        }
        imprinv[k1] = 1.0 * k / N;
    }

    ID_out = create_2Dimage_ID(ID_out_name, naxes[0], naxes[1]);
    for(uint32_t jj = 0; jj < naxes[1]; jj++)
        for(uint32_t ii = 0; ii < naxes[0]; ii++)
        {
            k1 = (long)(data.image[ID].array.F[jj * naxes[0] + ii] * N);
            if(k1 < 0)
            {
                k1 = 0;
            }
            if(k1 > N - 1)
            {
                k1 = N - 1;
            }
            k = (long)(imp[k1] * N);
            if(k < 0)
            {
                k = 0;
            }
            if(k > N - 1)
            {
                k = N - 1;
            }
            data.image[ID_out].array.F[jj * naxes[0] + ii] = imprinv[k];
        }

    free(histo);
    free(imp);
    free(impr);
    free(imprinv);

    return(0);
}





// load all images matching strfilter + .fits
// return number of images loaded
// image name in buffer is same as file name without extension
long load_fitsimages(
    const char *strfilter
)
{
    long cnt = 0;
    char fname[STRINGMAXLEN_FILENAME];
    char fname1[STRINGMAXLEN_FILENAME];
    FILE *fp;

	EXECUTE_SYSTEM_COMMAND("ls %s.fits > flist.tmp\n", strfilter);


    if((fp = fopen("flist.tmp", "r")) == NULL)
    {
        C_ERRNO = errno;
        PRINT_ERROR("fopen() error");
        exit(0);
    }

    while(fgets(fname, STRINGMAXLEN_FILENAME, fp) != NULL)
    {
        fname[strlen(fname) - 1] = '\0';
        strncpy(fname1, fname, STRINGMAXLEN_FILENAME);
        fname1[strlen(fname) - 5] = '\0';
        load_fits(fname, fname1, 1);
        printf("[%ld] Image %s loaded -> %s\n", cnt, fname, fname1);
        fflush(stdout);
        cnt++;
    }

    fclose(fp);

    EXECUTE_SYSTEM_COMMAND("rm flist.tmp");

    printf("%ld images loaded\n", cnt);

    return(cnt);
}







// recenter cube frames such that the photocenter is on the central pixel
// images are recentered by integer number of pixels
imageID basic_cube_center(
    const char *ID_in_name,
    const char *ID_out_name
)
{
    imageID IDin, IDout;
    long xsize, ysize, ksize;
    long ii, jj, kk, ii1, jj1;
    double tot, totii, totjj;
    long index0, index1, index;
    double v;
    long *tx = NULL;
    long *ty = NULL;

    IDin = image_ID(ID_in_name);
    xsize = data.image[IDin].md[0].size[0];
    ysize = data.image[IDin].md[0].size[1];
    ksize = data.image[IDin].md[0].size[2];

    tx = (long *) malloc(sizeof(long) * ksize);
    if(tx == NULL)
    {
        C_ERRNO = errno;
        PRINT_ERROR("malloc() error");
        exit(0);
    }

    ty = (long *) malloc(sizeof(long) * ksize);
    if(ty == NULL)
    {
        C_ERRNO = errno;
        PRINT_ERROR("malloc() error");
        exit(0);
    }


    IDout = create_3Dimage_ID(ID_out_name, xsize, ysize, ksize);

    for(kk = 0; kk < ksize; kk++)
    {
        tot = 0.0;
        totii = 0.0;
        totjj = 0.0;
        index0 = kk * xsize * ysize;

        for(jj = 0; jj < ysize; jj++)
        {
            index1 = index0 + jj * xsize;
            for(ii = 0; ii < xsize; ii++)
            {
                index = index1 + ii;
                v = data.image[IDin].array.F[index];
                totii += v * ii;
                totjj += v * jj;
                tot += v;
            }
        }
        totii /= tot;
        totjj /= tot;
        tx[kk] = ((long)(totii + 0.5)) - xsize / 2;
        ty[kk] = ((long)(totjj + 0.5)) - ysize / 2;

        for(ii = 0; ii < xsize; ii++)
            for(jj = 0; jj < ysize; jj++)
            {
                ii1 = ii + tx[kk];
                jj1 = jj + ty[kk];
                if((ii1 > -1) && (ii1 < xsize) && (jj1 > -1) && (jj1 < ysize))
                {
                    data.image[IDout].array.F[index0 + jj * xsize + ii] =
                        data.image[IDin].array.F[index0 + jj1 * xsize + ii1];
                }
                else
                {
                    data.image[IDout].array.F[index0 + jj * xsize + ii] = 0.0;
                }
            }
    }

    free(tx);
    free(ty);

    return IDout;
}




//
// average frames in a cube
// excludes point which are more than alpha x sigma
// writes an rms value frame as rmsim
//
imageID cube_average(
    const char *ID_in_name,
    const char *ID_out_name,
    float alpha
)
{
    imageID IDin, IDout, IDrms;
    long xsize, ysize, ksize;
    long ii, kk;
    double *array = NULL;
    double ave, ave1, rms;
    long cnt;
    long cnt1;

    IDin = image_ID(ID_in_name);
    xsize = data.image[IDin].md[0].size[0];
    ysize = data.image[IDin].md[0].size[1];
    ksize = data.image[IDin].md[0].size[2];

    IDout = create_2Dimage_ID(ID_out_name, xsize, ysize);
    IDrms = create_2Dimage_ID("rmsim", xsize, ysize);

    array = (double *) malloc(sizeof(double) * ksize);
    if(array == NULL)
    {
        C_ERRNO = errno;
        PRINT_ERROR("malloc() error");
        exit(0);
    }

    cnt1 = 0;
    for(ii = 0; ii < xsize * ysize; ii++)
    {
        for(kk = 0; kk < ksize; kk++)
        {
            array[kk] = (double) data.image[IDin].array.F[kk * xsize * ysize + ii];
        }

        ave = 0.0;
        for(kk = 0; kk < ksize; kk++)
        {
            ave += array[kk];
        }
        ave /= ksize;

        rms = 0.0;
        for(kk = 0; kk < ksize; kk++)
        {
            rms += (array[kk] - ave) * (array[kk] - ave);
        }
        rms = sqrt(rms / ksize);

        data.image[IDrms].array.F[ii] = (float) rms;

        ave1 = 0.0;
        cnt = 0;
        for(kk = 0; kk < ksize; kk++)
        {
            if(fabs(array[kk] - ave) < alpha * rms)
            {
                ave1 += array[kk];
                cnt ++;
            }
        }
        if(cnt > 0.5)
        {
            data.image[IDout].array.F[ii] = (float)(ave1 / cnt);
        }
        else
        {
            data.image[IDout].array.F[ii] = (float) ave;
        }
        cnt1 += cnt;
    }

    free(array);

    printf("(alpha = %f) fraction of pixel values selected = %ld/%ld = %.20g\n",
           alpha, cnt1, xsize * ysize * ksize,
           (double)(1.0 * cnt1 / (xsize * ysize * ksize)));
    printf("RMS written into image rmsim\n");

    return(IDout);
}








// coadd all images matching strfilter + .fits
// return number of images added
long basic_addimagesfiles(
    const char *strfilter,
    const char *outname
)
{
    long cnt = 0;
    char fname[STRINGMAXLEN_FILENAME];
    char fname1[STRINGMAXLEN_FILENAME];
    FILE *fp;
    imageID ID;
    int init = 0; // becomes 1 when first image encountered

	EXECUTE_SYSTEM_COMMAND("ls %s.fits > flist.tmp\n", strfilter);


    if((fp = fopen("flist.tmp", "r")) == NULL)
    {
        PRINT_ERROR("fopen() error");
        exit(0);
    }
    while(fgets(fname, STRINGMAXLEN_FILENAME, fp) != NULL)
    {
        fname[strlen(fname) - 1] = '\0';
        strncpy(fname1, fname, STRINGMAXLEN_FILENAME);

        fname1[strlen(fname) - 5] = '\0';
        ID = load_fits(fname, fname1, 1);
        printf("Image %s loaded -> %s\n", fname, fname1);
        if(init == 0)
        {
            init = 1;
            copy_image_ID(data.image[ID].name, outname, 0);
        }
        else
        {
            arith_image_add_inplace(outname, data.image[ID].name);
        }
        delete_image_ID(fname1);
        printf("Image %s added\n", data.image[ID].name);
        cnt++;
    }

    fclose(fp);

    EXECUTE_SYSTEM_COMMAND("rm flist.tmp");

    printf("%ld images coadded (stored in variable imcnt) -> %s\n", cnt, outname);
    create_variable_ID("imcnt", 1.0 * cnt);

    return(cnt);
}




// coadd all images matching strfilter + .fits
// return number of images added
long basic_aveimagesfiles(
    const char *strfilter,
    const char *outname
)
{
    long cnt;

    cnt = basic_addimagesfiles(strfilter, outname);
    arith_image_cstmult_inplace(outname, 1.0 / cnt);

    return(cnt);
}



// add all images starting with prefix
// return number of images added
long basic_addimages(
    const char *prefix,
    const char *ID_out
)
{
    long i;
    int init = 0; // becomes 1 when first image encountered
    long cnt = 0;

    for(i = 0; i < data.NB_MAX_IMAGE; i++)
        if(data.image[i].used == 1)
        {
            if(strncmp(prefix, data.image[i].name, strlen(prefix)) == 0)
            {
                if(init == 0)
                {
                    init = 1;
                    copy_image_ID(data.image[i].name, ID_out, 0);
                }
                else
                {
                    arith_image_add_inplace(ID_out, data.image[i].name);
                }
                printf("Image %s added\n", data.image[i].name);
                cnt ++;
            }
        }

    return(cnt);
}


// paste all images starting with prefix
long basic_pasteimages(
    const char *prefix,
    long        NBcol,
    const char *IDout_name
)
{
    long i;
    long cnt = 0;
    long row = 0;
    long col = 0;
    long colmax = 0;
    long xsizeout = 0;
    long ysizeout = 0;
    long xsize1max = 0;
    long ysize1max = 0;
    long xsize1, ysize1;
    long iioffset, jjoffset;
    long ii, jj, ii1, jj1;
    imageID IDout;

    for(i = 0; i < data.NB_MAX_IMAGE; i++)
        if(data.image[i].used == 1)
        {
            if(strncmp(prefix, data.image[i].name, strlen(prefix)) == 0)
            {
                if(data.image[i].md[0].size[0] > xsize1max)
                {
                    xsize1max = data.image[i].md[0].size[0];
                }
                if(data.image[i].md[0].size[1] > ysize1max)
                {
                    ysize1max = data.image[i].md[0].size[1];
                }

                if(col == NBcol)
                {
                    col = 0;
                    row ++;
                }
                if(col > colmax)
                {
                    colmax = col;
                }

                printf("Image %s[%ld] will be pasted at [%ld %ld]\n", data.image[i].name, cnt,
                       row, col);
                col ++;
            }
        }
    xsizeout = (colmax + 1) * xsize1max;
    ysizeout = (row + 1) * ysize1max;
    IDout = create_2Dimage_ID(IDout_name, xsizeout, ysizeout);


    col = 0;
    row = 0;
    for(i = 0; i < data.NB_MAX_IMAGE; i++)
        if(data.image[i].used == 1)
        {
            if(strncmp(prefix, data.image[i].name, strlen(prefix)) == 0)
            {
                if(col == NBcol)
                {
                    col = 0;
                    row ++;
                }

                iioffset = col * xsize1max;
                jjoffset = row * ysize1max;

                xsize1 = data.image[i].md[0].size[0];
                ysize1 = data.image[i].md[0].size[1];
                for(ii = 0; ii < xsize1; ii++)
                    for(jj = 0; jj < ysize1; jj++)
                    {
                        ii1 = ii + iioffset;
                        jj1 = jj + jjoffset;
                        data.image[IDout].array.F[jj1 * xsizeout + ii1] = data.image[i].array.F[jj *
                                xsize1 + ii];
                    }

                printf("Image %s[%ld] pasted at [%ld %ld]\n", data.image[i].name, cnt, row,
                       col);
                col ++;
            }
        }

    return(cnt);
}



// average all images starting with prefix
// return number of images added
long basic_averageimages(
    const char *prefix,
    const char *ID_out
)
{
    long cnt;

    cnt = basic_addimages(prefix, ID_out);
    arith_image_cstmult_inplace(ID_out, (float)(1.0 / cnt));

    return(cnt);
}




*/
