/**
 * @file    image_filter.c
 * @brief   Image filtering
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
#define MODULE_SHORTNAME_DEFAULT "imgfilt"

// Module short description
#define MODULE_DESCRIPTION "Image filtering"

#include "CommandLineInterface/CLIcore.h"

#include "fconvolve.h"
#include "gaussfilter.h"

/* ================================================================== */
/* ================================================================== */
/*            INITIALIZE LIBRARY                                      */
/* ================================================================== */
/* ================================================================== */

// Module initialization macro in CLIcore.h
// macro argument defines module name for bindings
//
INIT_MODULE_LIB(image_filter)

/* ================================================================== */
/* ================================================================== */
/*            COMMAND LINE INTERFACE (CLI) FUNCTIONS                  */
/* ================================================================== */
/* ================================================================== */

//long fconvolve(const char *ID_in, const char *ID_ke, const char *ID_out);

static errno_t init_module_CLI()
{
    gaussfilter_addCLIcmd();
    fconvolve_addCLIcmd();

    // add atexit functions here

    return RETURN_SUCCESS;
}

//int f_filter(const char *ID_name, const char *ID_out, float f1, float f2)
//{
//    printf("%s %s %f %f\n", ID_name, ID_out, f1, f2);
/*  char lstring[1000];
    int ID;
    long naxes[2];
    float a=200.0;

    ID=image_ID(ID_name);
    naxes[0]=data.image[ID].size[0];
    naxes[1]=data.image[ID].size[1];

    sprintf(lstring,"zer_tmp=%s*0",ID_name);
    execute_arith(lstring);
    pupfft(ID_name,"zer_tmp","ffamp","ffpha","");
    delete_image_ID("zer_tmp");

    make_dist("ffdist",naxes[0],naxes[1],1.0*naxes[0]/2,1.0*naxes[1]/2);
    sprintf(lstring,"ffd=(ffdist-%f)*(%f-ffdist)",f1,f2);
    execute_arith(lstring);
    sprintf(lstring,"ffd=ffd/(ffd+%f)",a);
    execute_arith(lstring);
    execute_arith("ffd=ffd*ffd");

    make_disk("ffd1",naxes[0],naxes[1],naxes[0]/2,naxes[1]/2,f1);
    make_disk("ffd2",naxes[0],naxes[1],naxes[0]/2,naxes[1]/2,f2);
    execute_arith("ffd=ffd*(ffd2-ffd1)");
    delete_image_ID("ffd1");
    delete_image_ID("ffd2");

    execute_arith("ffamp=ffamp*ffd");
    delete_image_ID("ffd");
    pupfft("ffamp","ffpha","fftbea","fftbep","-inv");
    delete_image_ID("ffamp");
    delete_image_ID("ffpha");
    ampl_pha_2_re_im("fftbea","fftbep",ID_out,"fftbe");

    delete_image_ID("fftbe");
    delete_image_ID("fftbea");
    delete_image_ID("fftbep");
    */
//   return(0);
//}

/*
int film_scanner_vsripes_remove(const char *IDname, const char *IDout, long l1,
                                long l2)
{
    long ID;
    long naxes[2];
    long ii, jj;
    float *smarray;
    float value;

    ID = image_ID(IDname);
    naxes[0] = data.image[ID].md[0].size[0];
    naxes[1] = data.image[ID].md[0].size[1];
    printf("%s\n", IDout);

    // fill up linarray
    smarray = (float *) malloc(sizeof(float) * (l2 - l1));

    for(ii = 0; ii < naxes[0]; ii++)
    {
        for(jj = l1; jj < l2; jj++)
        {
            smarray[jj - l1] = data.image[ID].array.F[jj * naxes[0] + ii];
        }
        quick_sort_float(smarray, l2 - l1);
        value = smarray[(long)(0.5 * (l2 - l1))];
        for(jj = 0; jj < naxes[1]; jj++)
        {
            data.image[ID].array.F[jj * naxes[0] + ii] -= value;
        }
    }

    free(smarray);

    return(0);
}
*/
