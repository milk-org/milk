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

#ifdef MILK_NO_CLI
#    include "CLIcore_standalone.h"
#else
#    include "CLIcore.h"
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

#    include "cubecollapse.h"
#    include "im3Dto2D.h"
#    include "image_add.h"
#    include "imcontract.h"
#    include "imexpand.h"
#    include "imgetcircasym.h"
#    include "imgetcircsym.h"
#    include "imresize.h"
#    include "imrotate.h"
#    include "imswapaxis2D.h"
#    include "indexmap.h"
#    include "loadfitsimgcube.h"
#    include "streamfeed.h"
#    include "streamrecord.h"

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
    CLIADDCMD_image_basic__imswapaxis2D();
    CLIADDCMD_image_basic__im3Dto2D();
    CLIADDCMD_image_basic__image_add();
    CLIADDCMD_image_basic__imexpand();
    CLIADDCMD_image_basic__imgetcircsym();
    CLIADDCMD_image_basic__imgetcircasym();
    CLIADDCMD_image_basic__imresize();
    CLIADDCMD_image_basic__imcontract();
    CLIADDCMD_image_basic__imrotate();
    CLIADDCMD_image_basic__loadfitsimgcube();
    CLIADDCMD_image_basic__streamfeed();
    CLIADDCMD_image_basic__streamrecord();
    CLIADDCMD_image_basic__cubecollapse();
    CLIADDCMD_image_basic__indexmap();

    // add atexit functions here

    return RETURN_SUCCESS;
}
#endif /* MILK_NO_CLI */

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

    IDin = image_ID(ID_name, dcimg, dcnimg);
    naxes[0] = dcimg[IDin].md[0].size[0];
    naxes[1] = dcimg[IDin].md[0].size[1];

    IDout = create_2Dimage_ID(out_name, naxes[0], 1);

    for(ii = 0; ii < naxes[0]; ii++)
    {
        dcimg[IDout].array.F[ii] = dcimg[IDin].array.F[ii];
    }

    for(jj = 1; jj < naxes[1]; jj++)
        for(ii = 0; ii < naxes[0]; ii++)
        {
            tmp = dcimg[IDin].array.F[jj * naxes[0] + ii];
            if(tmp < dcimg[IDout].array.F[ii])
            {
                dcimg[IDout].array.F[ii] = tmp;
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

    IDin = image_ID(ID_name, dcimg, dcnimg);
    naxes[0] = dcimg[IDin].md[0].size[0];
    naxes[1] = dcimg[IDin].md[0].size[1];

    IDout = create_2Dimage_ID(out_name, naxes[0], 1);

    for(ii = 0; ii < naxes[0]; ii++)
    {
        dcimg[IDout].array.F[ii] = dcimg[IDin].array.F[ii];
    }

    for(jj = 1; jj < naxes[1]; jj++)
        for(ii = 0; ii < naxes[0]; ii++)
        {
            tmp = dcimg[IDin].array.F[jj * naxes[0] + ii];
            if(tmp > dcimg[IDout].array.F[ii])
            {
                dcimg[IDout].array.F[ii] = tmp;
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

    ID1 = image_ID(ID_name1, dcimg, dcnimg);
    ID2 = image_ID(ID_name2, dcimg, dcnimg);
    naxes1[0] = dcimg[ID1].md[0].size[0];
    naxes1[1] = dcimg[ID1].md[0].size[1];
    naxes2[0] = dcimg[ID2].md[0].size[0];
    naxes2[1] = dcimg[ID2].md[0].size[1];

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
    ID_out = image_ID(ID_name_out, dcimg, dcnimg);
    naxes[0] = dcimg[ID_out].md[0].size[0];
    naxes[1] = dcimg[ID_out].md[0].size[1];

    for(jj = 0; jj < naxes[1]; jj++)
        for(ii = 0; ii < naxes[0]; ii++)
        {
            {
                dcimg[ID_out].array.F[jj * naxes[0] + ii] = 0;
                // if pixel is in ID1
                if(((ii + xmin) >= 0) && ((ii + xmin) < naxes1[0]))
                    if(((jj + ymin) >= 0) && ((jj + ymin) < naxes1[1]))
                    {
                        dcimg[ID_out].array.F[jj * naxes[0] + ii] += dcimg[ID1].array.F[(jj +
                                ymin) * naxes1[0] + (ii + xmin)];
                    }
                // if pixel is in ID2
                if(((ii + xmin - off1) >= 0) && ((ii + xmin - off1) < naxes2[0]))
                    if(((jj + ymin - off2) >= 0) && ((jj + ymin - off2) < naxes2[1]))
                    {
                        dcimg[ID_out].array.F[jj * naxes[0] + ii] -= dcimg[ID2].array.F[(jj +
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

    ID = image_ID(ID_name, dcimg, dcnimg);
    naxes[0] = dcimg[ID].md[0].size[0];
    naxes[1] = dcimg[ID].md[0].size[1];

    for(jj = 0; jj < naxes[1]; jj++)
        for(ii = 0; ii < naxes[0]; ii++)
        {
            dcimg[ID].array.F[jj * naxes[0] + ii] = dcimg[ID].array.F[jj *
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

    ID_in = image_ID(ID_in_name, dcimg, dcnimg);
    n = snprintf(name, SBUFFERSIZE, "%s", ID_out_name);
    if(n >= SBUFFERSIZE)
    {
        PRINT_ERROR("Attempted to write string buffer with too many characters");
    }

    create_2Dimage_ID(name, n1, n2);
    fflush(stdout);
    ID_out = image_ID(ID_out_name, dcimg, dcnimg);
    for(ii = 0; ii < n1; ii++)
        for(jj = 0; jj < n2; jj++)
        {
            dcimg[ID_out].array.F[jj * n1 + ii] = dcimg[ID_in].array.F[(jj + n4) *
                    dcimg[ID_in].md[0].size[0] + ii + n3];
        }

    return(ID_out);
}



int basic_trunc_circ(const char *ID_name, float f1)
{
    imageID ID;
    long ii, jj;
    long naxes[2];

    ID = image_ID(ID_name, dcimg, dcnimg);
    naxes[0] = dcimg[ID].md[0].size[0];
    naxes[1] = dcimg[ID].md[0].size[1];

    for(jj = 0; jj < naxes[1]; jj++)
        for(ii = 0; ii < naxes[0]; ii++)
        {
            dcimg[ID].array.F[jj * naxes[0] + ii] = fmod(dcimg[ID].array.F[jj *
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
    ID = image_ID(ID_name, dcimg, dcnimg);
    naxes[0] = dcimg[ID].md[0].size[0];
    naxes[1] = dcimg[ID].md[0].size[1];
    naxes_out[0] = naxes[0];
    naxes_out[1] = naxes[1];
    ID_out = image_ID(ID_name_out, dcimg, dcnimg);

    for(uint32_t jj = 0; jj < naxes[1] / 2; jj++)
        for(uint32_t ii = 0; ii < naxes[0] / 2; ii++)
        {
            dcimg[ID_out].array.F[(2 * jj)*naxes_out[0] + (2 * ii)] =
                dcimg[ID].array.F[(jj + naxes[1] / 4) * naxes[0] + (ii + naxes[1] / 4)];
            dcimg[ID_out].array.F[(2 * jj + 1)*naxes_out[0] + (2 * ii)] = 0.5f *
                    (dcimg[ID].array.F[(jj + naxes[1] / 4) * naxes[0] +
                                            (ii + naxes[1] / 4)] + dcimg[ID].array.F[(jj + naxes[1] / 4 + 1) *
                                                    naxes[0] + (ii + naxes[1] / 4)]);
            dcimg[ID_out].array.F[(2 * jj)*naxes_out[0] + (2 * ii + 1)] = 0.5f *
                    (dcimg[ID].array.F[(jj + naxes[1] / 4) * naxes[0] +
                                            (ii + naxes[1] / 4)] + dcimg[ID].array.F[(jj + naxes[1] / 4) * naxes[0] +
                                                    (ii + naxes[1] / 4 + 1)]);
            dcimg[ID_out].array.F[(2 * jj + 1)*naxes_out[0] + (2 * ii + 1)] = 0.25f *
                    (dcimg[ID].array.F[(jj + naxes[1] / 4) * naxes[0] +
                                            (ii + naxes[1] / 4)] + dcimg[ID].array.F[(jj + naxes[1] / 4) * naxes[0] +
                                                    (ii + naxes[1] / 4 + 1)] + dcimg[ID].array.F[(jj + naxes[1] / 4 + 1) *
                                                            naxes[0] + (ii + naxes[1] / 4)] + dcimg[ID].array.F[(jj + naxes[1] / 4 + 1)
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

    ID = image_ID(ID_name, dcimg, dcnimg);
    naxes[0] = dcimg[ID].md[0].size[0];
    naxes[1] = dcimg[ID].md[0].size[1];

    naxes_out[0] = naxes[0] + 2 * n1;
    naxes_out[1] = naxes[1] + 2 * n2;

    create_2Dimage_ID(ID_name_out, naxes_out[0], naxes_out[1]);
    ID_out = image_ID(ID_name_out, dcimg, dcnimg);

    for(uint32_t jj = 0; jj < naxes[1]; jj++)
        for(uint32_t ii = 0; ii < naxes[0]; ii++)
        {
            dcimg[ID_out].array.F[(jj + n2)*naxes_out[0] + ii + n1] =
                dcimg[ID].array.F[jj * naxes[0] + ii];
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

    ID = image_ID(ID_name, dcimg, dcnimg);
    naxes[0] = dcimg[ID].md[0].size[0];
    naxes[1] = dcimg[ID].md[0].size[1];

    temp = 0.0;
    tmp_long = (uint32_t) (naxes[1] / 2);
    for(uint32_t jj = 0; jj < tmp_long; jj++)
        for(uint32_t ii = 0; ii < naxes[0]; ii++)
        {
            temp = dcimg[ID].array.F[jj * naxes[0] + ii];
            dcimg[ID].array.F[jj * naxes[0] + ii] = dcimg[ID].array.F[(naxes[1] -
                    jj - 1) * naxes[0] + ii];
            dcimg[ID].array.F[(naxes[1] - jj - 1)*naxes[0] + ii] = temp;
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

    ID = image_ID(ID_name, dcimg, dcnimg);
    naxes[0] = dcimg[ID].md[0].size[0];
    naxes[1] = dcimg[ID].md[0].size[1];

    temp = 0.0;
    tmp_long = (uint32_t) (naxes[0] / 2);
    for(uint32_t jj = 0; jj < naxes[1]; jj++)
        for(uint32_t ii = 0; ii < tmp_long; ii++)
        {
            temp = dcimg[ID].array.F[jj * naxes[0] + ii];
            dcimg[ID].array.F[jj * naxes[0] + ii] = dcimg[ID].array.F[jj *
                    naxes[0] + (naxes[0] - ii - 1)];
            dcimg[ID].array.F[jj * naxes[0] + (naxes[0] - ii - 1)] = temp;
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

    ID = image_ID(ID_name, dcimg, dcnimg);
    naxes[0] = dcimg[ID].md[0].size[0];
    naxes[1] = dcimg[ID].md[0].size[1];

    temp = 0.0;
    tmp_long = (uint32_t) (naxes[1] / 2);
    for(uint32_t jj = 0; jj < tmp_long; jj++)
        for(uint32_t ii = 0; ii < naxes[0]; ii++)
        {
            temp = dcimg[ID].array.F[jj * naxes[0] + ii];
            dcimg[ID].array.F[jj * naxes[0] + ii] = dcimg[ID].array.F[(naxes[1] -
                    jj - 1) * naxes[0] + (naxes[0] - ii - 1)];
            dcimg[ID].array.F[(naxes[1] - jj - 1)*naxes[0] + (naxes[0] - ii - 1)] =
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
            IDn[file_nb] = image_ID(file_name, dcimg, dcnimg);
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

    naxes[0] = dcimg[IDn[0]].md[0].size[0];
    naxes[1] = dcimg[IDn[0]].md[0].size[1];
    create_2Dimage_ID(ID_name, naxes[0], naxes[1]);
    ID = image_ID(ID_name, dcimg, dcnimg);

    for(uint32_t jj = 0; jj < naxes[1]; jj++)
        for(uint32_t ii = 0; ii < naxes[0]; ii++)
        {
            for(i = 0; i < Nb_files; i++)
            {
                array[i] = dcimg[IDn[i]].array.F[jj * naxes[0] + ii];
            }
            quick_sort_float(array, Nb_files);
            if((0.5 * (Nb_files - 1) - medianpt) < 0.1)
            {
                dcimg[ID].array.F[jj * naxes[0] + ii] = array[medianpt];
            }
            else
            {
                dcimg[ID].array.F[jj * naxes[0] + ii] = 0.5f * array[medianpt] + 0.5f *
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

    ID = image_ID(ID_name, dcimg, dcnimg);
    naxes[0] = dcimg[ID].md[0].size[0];
    naxes[1] = dcimg[ID].md[0].size[1];

    max = 0;

    for(jj = 0; jj < naxes[1]; jj++)
        for(ii = 0; ii < naxes[0]; ii++)
            if(dcimg[ID].array.F[jj * naxes[0] + ii] > max)
            {
                max = dcimg[ID].array.F[jj * naxes[0] + ii];
            }

    if(max != 0)
    {
        for(jj = 0; jj < naxes[1]; jj++)
            for(ii = 0; ii < naxes[0]; ii++)
            {
                dcimg[ID].array.F[jj * naxes[0] + ii] /= max;
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

    ID = image_ID(ID_name, dcimg, dcnimg);
    naxes[0] = dcimg[ID].md[0].size[0];
    naxes[1] = dcimg[ID].md[0].size[1];
    onaxes[0] = naxes[0];
    onaxes[1] = naxes[1];
    n0 = (int)((log10(naxes[0]) / log10(2)) + 0.01);
    n1 = (int)((log10(naxes[0]) / log10(2)) + 0.01);

    if((n0 == n1) && (naxes[0] == (1 << n0))
            && (naxes[1] == (1 << n1)))
    {
        create_2Dimage_ID("zero_tmp", naxes[0], naxes[1]);
        pupfft(ID_name, "zero_tmp", "out_transl_re_tmp", "out_transl_im_tmp", "-reim");
        delete_image_ID("zero_tmp");
        mk_amph_from_reim("out_transl_re_tmp", "out_transl_im_tmp",
                          "out_transl_ampl_tmp", "out_transl_pha_tmp", 0);
        delete_image_ID("out_transl_re_tmp");
        delete_image_ID("out_transl_im_tmp");

        ID = image_ID("out_transl_pha_tmp", dcimg, dcnimg);
        for(jj = 1; jj < naxes[1]; jj++)
            for(ii = 1; ii < naxes[0]; ii++)
            {
                dcimg[ID].array.F[jj * naxes[0] + ii] -= xtransl * 2.0f * M_PI /
                        naxes[0] * (ii - naxes[0] / 2) + ytransl * 2.0 * M_PI / naxes[1] *
                        (jj - naxes[1] / 2);
            }

        coeff = 1.0 / (naxes[0] * naxes[1]);
        ID = image_ID("out_transl_ampl_tmp", dcimg, dcnimg);
        for(jj = 0; jj < naxes[1]; jj++)
            for(ii = 0; ii < naxes[0]; ii++)
            {
                dcimg[ID].array.F[jj * naxes[0] + ii] *= coeff;
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
        basic_extract("tmp2t", "tmp3t", (1 << (n0 + 1)), (1 << (n1 + 1)), 0, 0);
        delete_image_ID("tmp2t");
        ID = image_ID("tmp3t", dcimg, dcnimg);
        naxes[0] = dcimg[ID].md[0].size[0];
        naxes[1] = dcimg[ID].md[0].size[1];
        create_2Dimage_ID("zero_tmp", naxes[0], naxes[1]);

        pupfft("tmp3t", "zero_tmp", "out_transl_re_tmp", "out_transl_im_tmp", "-reim");
        delete_image_ID("zero_tmp");
        delete_image_ID("tmp3t");
        mk_amph_from_reim("out_transl_re_tmp", "out_transl_im_tmp",
                          "out_transl_ampl_tmp", "out_transl_pha_tmp", 0);
        delete_image_ID("out_transl_re_tmp");
        delete_image_ID("out_transl_im_tmp");

        ID = image_ID("out_transl_pha_tmp", dcimg, dcnimg);
        for(jj = 1; jj < naxes[1]; jj++)
            for(ii = 1; ii < naxes[0]; ii++)
            {
                dcimg[ID].array.F[jj * naxes[0] + ii] -= xtransl * 2.0f * M_PI /
                        naxes[0] * (ii - naxes[0] / 2) + ytransl * 2.0 * M_PI / naxes[1] *
                        (jj - naxes[1] / 2);
            }
        coeff = 1.0 / (naxes[0] * naxes[1]);
        ID = image_ID("out_transl_ampl_tmp", dcimg, dcnimg);
        for(jj = 0; jj < naxes[1]; jj++)
            for(ii = 0; ii < naxes[0]; ii++)
            {
                dcimg[ID].array.F[jj * naxes[0] + ii] *= coeff;
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

    ID1 = image_ID(ID_name1, dcimg, dcnimg);
    naxes1[0] = dcimg[ID1].md[0].size[0];
    naxes1[1] = dcimg[ID1].md[0].size[1];
    ID2 = image_ID(ID_name2, dcimg, dcnimg);
    naxes2[0] = dcimg[ID2].md[0].size[0];
    naxes2[1] = dcimg[ID2].md[0].size[1];

    if((naxes1[0] != naxes2[0]) || (naxes1[1] != naxes2[1]))
    {
        printf("correlation : file size do not match\n");
        exit(1);
    }
    correl = 0;

    for(uint32_t jj = 0; jj < naxes1[1]; jj++)
        for(uint32_t ii = 0; ii < naxes1[0]; ii++)
        {
            correl += (dcimg[ID1].array.F[jj * naxes1[0] + ii] -
                       dcimg[ID2].array.F[jj * naxes1[0] + ii]) * (dcimg[ID1].array.F[jj *
                               naxes1[0] + ii] - dcimg[ID2].array.F[jj * naxes1[0] + ii]);
        }

    return(correl);
}











*/
