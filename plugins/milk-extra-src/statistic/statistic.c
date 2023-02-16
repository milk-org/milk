/**
 * @file    statistic.c
 * @brief   statistical tools module
 *
 * Random numbers, photon noise
 *
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
#define MODULE_SHORTNAME_DEFAULT "stat"

// Module short description
#define MODULE_DESCRIPTION "Statistics functions and tools"

/* =============================================================================================== */
/* =============================================================================================== */
/*                                        HEADER FILES                                             */
/* =============================================================================================== */
/* =============================================================================================== */

#include "CommandLineInterface/CLIcore.h"
#include <gsl/gsl_randist.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "COREMOD_memory/COREMOD_memory.h"
#include "statistic/statistic.h"

/* =============================================================================================== */
/* =============================================================================================== */
/*                                  GLOBAL DATA DECLARATION                                        */
/* =============================================================================================== */
/* =============================================================================================== */

typedef struct
{
    int active; // 1 if active, 0 otherwise

    int    NBpt; // number of points
    float *sum;  // sum
    float *ssum; // sum of squares

    int   leaf; // 1 if leaf, 0 if non-leaf
    long  parent_index;
    long  NBchildren;
    long *children_index;
} BIRCHCF;

/* ================================================================== */
/* ================================================================== */
/*            INITIALIZE LIBRARY                                      */
/* ================================================================== */
/* ================================================================== */

// Module initialization macro in CLIcore.h
// macro argument defines module name for bindings
//
INIT_MODULE_LIB(statistic)

/* ================================================================== */
/* ================================================================== */
/*            COMMAND LINE INTERFACE (CLI) FUNCTIONS                  */
/* ================================================================== */
/* ================================================================== */

/** @name CLI bindings */

errno_t statistic_putphnoise_cli()
{

    if(CLI_checkarg(1, CLIARG_IMG) + CLI_checkarg(2, CLIARG_STR_NOT_IMG) == 0)
    {
        put_poisson_noise(data.cmdargtoken[1].val.string,
                          data.cmdargtoken[2].val.string);

        return CLICMD_SUCCESS;
    }
    else
    {
        return CLICMD_INVALID_ARG;
    }
}

errno_t statistic_putgaussnoise_cli()
{

    if(CLI_checkarg(1, CLIARG_IMG) + CLI_checkarg(2, CLIARG_STR_NOT_IMG) +
            CLI_checkarg(3, CLIARG_FLOAT64) ==
            0)
    {
        put_gauss_noise(data.cmdargtoken[1].val.string,
                        data.cmdargtoken[2].val.string,
                        data.cmdargtoken[3].val.numf);

        return CLICMD_SUCCESS;
    }
    else
    {
        return CLICMD_INVALID_ARG;
    }
}

/* =============================================================================================== */
/* =============================================================================================== */
/*                                    MODULE INITIALIZATION                                        */
/* =============================================================================================== */
/* =============================================================================================== */
/** @name Module initialization */

static errno_t init_module_CLI()
{
    RegisterCLIcommand("putphnoise",
                       __FILE__,
                       statistic_putphnoise_cli,
                       "add photon noise to image",
                       "input output",
                       "putphnoise im0 im1",
                       "int put_poisson_noise(const char *ID_in_name, const "
                       "char *ID_out_name)");

    RegisterCLIcommand("putgaussnoise",
                       __FILE__,
                       statistic_putgaussnoise_cli,
                       "add gaussian noise to image",
                       "input output amplitude",
                       "putgaussnoise im0 im1 0.2",
                       "long put_gauss_noise(const char *ID_in_name, const "
                       "char *ID_out_name, doule ampl)");

    // add atexit functions here

    return RETURN_SUCCESS;
}

/* =============================================================================================== */
/* =============================================================================================== */
/*                                    FUNCTIONS SOURCE CODE                                        */
/* =============================================================================================== */
/* =============================================================================================== */
/** @name STATISTIC functions */

double ran1()
{
    double value;

    value = data.INVRANDMAX * rand();
    // gsl_rng_uniform (data.rndgen);// data.INVRANDMAX*rand();

    return (value);
}

double gauss()
{
    // use first option if using ranlxs generator
    // return(gsl_ran_ugaussian (data.rndgen));

    // for speed (4.1x faster than default), but not that random (some fringes appear in image)
    // return(gsl_ran_gaussian_ziggurat (data.rndgen,1.0));

    // default
    return (gsl_ran_gaussian(data.rndgen, 1.0));
}

double gauss_trc()
{
    double value;

    value = gauss();
    while(fabs(value) > 1.0)
    {
        value = gauss();
    }
    return (value);
}

long poisson(double mu)
{
    return (gsl_ran_poisson(data.rndgen, (double) mu));
}

double cfits_gammaln(double xx)
{
    /* ln of the Gamma function */
    int    j;
    double cof[6];
    double stp;
    double ser;
    double tmp, x, y;
    double result;

    cof[0] = 76.18009172947146;
    cof[1] = -86.50532032941677;
    cof[2] = 24.01409824083091;
    cof[3] = -1.231739572450155;
    cof[4] = 0.001208650973866179;
    cof[5] = 0.000005395239384953;
    stp    = 2.5066282746310005;
    ser    = 1.000000000190015;

    x   = xx;
    y   = x;
    tmp = x + 5.5;
    tmp = (x + 0.5) * log(tmp) - tmp;
    for(j = 0; j < 6; j++)
    {
        y   = y + 1;
        ser = ser + cof[j] / y;
    }
    result = tmp + log(stp * ser / x);
    return (result);
}

double fast_poisson(double mu)
{
    /* a fast, but approximate, poisson distribution generator */
    double em;

    em = 0;
    em = (double)((long long)(mu + gauss() * sqrt(mu)));
    if(em < 0.0)
    {
        em = 0.0;
    }

    return (em);
}

// better_poisson seems to give a very weird value every once in a while
// probability this happens is ~1e-8 to 1e-9
double better_poisson(double mu)
{
    /* a better poisson distribution generator... see num. rec. section 7.3. */
    double inv_randmax;
    double em;

    inv_randmax = 1.0 / RAND_MAX;

    em = 0;
    if(mu < 100)
    {
        em = (double) poisson(mu);
    }
    else
    {
        double logmu;
        double sq, g, y, t;

        sq    = sqrt(2 * mu);
        logmu = log(mu);
        g     = mu * logmu - cfits_gammaln(mu + 1);

        y  = tan(PI * (inv_randmax * rand()));
        em = sq * y + mu;
        while(em < 0)
        {
            y  = tan(PI * (inv_randmax * rand()));
            em = sq * y + mu;
        }
        em = (int) em;
        t  = 0.9 * (1 + y * y) * exp(em * logmu - cfits_gammaln(em + 1) - g);

        while((inv_randmax * rand()) > t)
        {
            y  = tan(PI * (inv_randmax * rand()));
            em = sq * y + mu;
            while(em < 0)
            {
                y  = tan(PI * (inv_randmax * rand()));
                em = sq * y + mu;
            }
            em = (long) em;
            t = 0.9 * (1 + y * y) * exp(em * logmu - cfits_gammaln(em + 1) - g);
        }
    }

    return (1.0 * em);
}

long put_poisson_noise(const char *ID_in_name, const char *ID_out_name)
{
    long ID_in;
    long ID_out;
    long ii;
    long nelements;
    long naxis;
    long i;

    ID_in     = image_ID(ID_in_name);
    naxis     = data.image[ID_in].md[0].naxis;
    nelements = 1;
    for(i = 0; i < naxis; i++)
    {
        nelements *= data.image[ID_in].md[0].size[i];
    }

    copy_image_ID(ID_in_name, ID_out_name, 0);

    ID_out = image_ID(ID_out_name);
    //  srand(time(NULL));

    for(ii = 0; ii < nelements; ii++)
    {
        data.image[ID_out].array.F[ii] = poisson(data.image[ID_in].array.F[ii]);
    }

    return (ID_out);
}

long put_gauss_noise(const char *ID_in_name,
                     const char *ID_out_name,
                     double      ampl)
{
    long ID_in;
    long ID_out;
    long ii;
    long nelements;
    long naxis;
    long i;

    ID_in     = image_ID(ID_in_name);
    naxis     = data.image[ID_in].md[0].naxis;
    nelements = 1;
    for(i = 0; i < naxis; i++)
    {
        nelements *= data.image[ID_in].md[0].size[i];
    }

    copy_image_ID(ID_in_name, ID_out_name, 0);

    ID_out = image_ID(ID_out_name);
    //  srand(time(NULL));

    for(ii = 0; ii < nelements; ii++)
    {
        data.image[ID_out].array.F[ii] =
            data.image[ID_in].array.F[ii] + ampl * gauss();
    }

    return (ID_out);
}

/**
 * ## Purpose
 *
 * Apply BIRCH clustering to images
 *
 * ## Overview
 *
 * Images input is 3D array, one image per slice\n
 * Euclidian distance adopted\n
 *
 * B is the number of branches
 *
 * epsilon is the maximum distance (Euclidian)
 *
 *
 * ## Details
 *
 */

long statistic_BIRCH_clustering(__attribute__((unused)) const char *IDin_name,
                                __attribute__((unused)) int         B,
                                __attribute__((unused)) double      epsilon,
                                __attribute__((unused)) const char *IDout_name)
{
    //long IDin;
    //long xsize, ysize;
    //long zsize;

    // node definition:

    // leaf or not ?
    //
    // pointers to children - if leaf, these point to single samples
    // pointer to parent
    // level
    //
    // CF_N (1 if sample)
    // CF_S
    // CF_SS
    //

    /*
    typedef struct
    {
    int active;  // 1 if active, 0 otherwise

    int NBpt;    // number of points
    float *sum;  // sum
    float *ssum; // sum of squares

    long level;   // 0 is root, and so on
    int leaf;    // 1 if leaf, 0 if non-leaf
    long parent_index;
    long NBchildren;
    long *children_index;
    } BIRCHCF;
    */

    /*
    IDin = image_ID(IDin_name);
    xsize = data.image[IDin].md[0].size[0];
    ysize = data.image[IDin].md[0].size[1];
    zsize = data.image[IDin].md[0].size[2];

    long xysize = xsize*ysize;

    BIRCHCF *BirchCFarray;

    long NBnodeMax = zsize;
    BirchCFarray = (BIRCHCF*) malloc(sizeof(BIRCHCF)*NBnodeMax);

    // initialize
    long node;
    for(node=0; node<NBnodeMax; node++)
    {
    	BirchCFarray[node].active = 0;
    	BirchCFarray[node].level = 0;

    	BirchCFarray[node].NBpt = 0;
    	BirchCFarray[node].sum = NULL;
    	BirchCFarray[node].ssum = NULL;


    	BirchCFarray[node].leaf = 0;
    	BirchCFarray[node].parent_index = 0;
    	BirchCFarray[node].NBchildren = 0;
    	BirchCFarray[node].children_index = NULL;
    }

    node = 0;



    long NBnode = 1; // number of nodes



    // initialize to single node
    node = 0;
    BirchCFarray[node].N = 1;

    NBnode = 1;



    long k;
    for(k=0; k<NBCFmax; k++) // Insert sample into tree
    {
    	BirchCFarray[k].active = 0;
    	BirchCFarray[k].NBpt = 0;

    	BirchCFarray[k].sum = (float*) malloc(sizeof(float)*xysize);
    	BirchCFarray[k].ssum = (float*) malloc(sizeof(float)*xysize);

    	BirchCFarray[k].leaf = 1;
    	BirchCFarray[k].parent_index = -1;

    	BirchCFarray[k].NBchildren = 0;
    	BirchCFarray[k].children_index = (long*) malloc(sizeof(long)*B);

    	long kk;
    	for(kk=0;kk<B;kk++)
    		BirchCFarray[k].children_index[kk] = 0;
    }


    // first slice
    k = 0;
    BirchCFarray[k].active = 1;
    BirchCFarray[k].NBpt = 1;
    memcpy(BirchCFarray[k].sum, data.image[IDin].array.F, sizeof(float)*xysize);

    long ii;
    for(ii=0;ii<xysize;ii++)
    	BirchCFarray[k].ssum[ii] = data.image[IDin].array.F[ii]*data.image[IDin].array.F[ii];


    //
    // Scan through array
    // kin is input array index
    //
    long kin;
    for(kin=1;kin<zsize;kin++)
    {
    	k = 0; // root

    	while(BirchCFarray[k].leaf == 0) // if non-leaf, find path
    	{
    		double distmin = 0.0;
    		double dist;
    		long kkmin; // path


    		long kk = 0;
    		for(ii=0;ii<xysize;ii++)
    		{
    			double tmpv;
    			tmpv = BirchCFarray[BirchCFarray[k].children_index[kk]].sum[ii]/BirchCFarray[BirchCFarray[k].children_index[kk]].NBpt - data.image[IDin].array.F[kin*xysize+ii];
    			distmin += tmpv*tmpv;
    		}

    		for(kk=1;kk<BirchCFarray[k].NBchildren;kk++)
    		{
    			double dist = 0.0;
    			for(ii=0;ii<xysize;ii++)
    			{
    				double tmpv;
    				tmpv = BirchCFarray[BirchCFarray[k].children_index[kk]].sum[ii]/BirchCFarray[BirchCFarray[k].children_index[kk]].NBpt - data.image[IDin].array.F[kin*xysize+ii];
    				dist += tmpv*tmpv;
    			}
    			if(dist<distmin)
    			{
    				distmin = dist;
    				kkmin = kk;
    			}
    		}
    		k = kkmin;
    	}



    	// leaf node children point to input entries
    	if(BirchCFarray[k].leaf == 1) // If leaf node, add to leaf node
    	{
    		// Measure distance to existing

    		if(BirchCFarray[k].NBpt == B-1) // split leaf node
    		{

    			// identify maximum distance pair
    			double maxdist = 0.0;
    			long kk1max, kk2max;
    			long kk1, kk2;

    			for(kk1=0;kk1<BirchCFarray[k].NBpt;kk1++)
    				for(kk2=kk1+1; kk2<BirchCFarray[k].NBpt;kk2++)
    				{
    					double dist = 0.0;
    					for(ii=0;ii<xysize;ii++)
    					{
    						double tmpv;
    						tmpv = BirchCFarray[BirchCFarray[k].children_index[kk1]].sum[ii] - BirchCFarray[BirchCFarray[k].children_index[kk2]].sum[ii];
    						dist += tmpv*tmpv;
    					}
    					if(dist>maxdist)
    					{
    						kk1max = kk1;
    						kk2max = kk2;
    						maxdist = dist;
    					}
    				}

    			// create two new leaf nodes
    			long k1next, k2next;
    			long ksearch = 0;
    			while(BirchCFarray[ksearch].active==1)
    				ksearch ++;
    			k1next = ksearch;
    			BirchCFarray[k1next].active = 1;
    			BirchCFarray[k1next].NBpt = 0;

    			while(BirchCFarray[ksearch].active==1)
    				ksearch ++;
    			k2next = ksearch;
    			BirchCFarray[k2next].active = 1;
    			BirchCFarray[k2next].NBpt = 0;

    			// populate new leaf nodes


    			// edit source node

    		}
    		else
    		{
    			long kk = BirchCFarray[k].NBpt;
    			BirchCFarray[k].children_index[kk] = kin;
    			BirchCFarray[k].NBpt ++;

    			for(ii=0;ii<xysize;ii++)
    			{
    				BirchCFarray[k].sum[ii] += data.image[IDin].array.F[kin*xysize+ii];
    				BirchCFarray[k].ssum[ii] += data.image[IDin].array.F[kin*xysize+ii]*data.image[IDin].array.F[kin*xysize+ii];
    			}
    		}
    	}
    	else
    	{
    		printf("ERROR: BIRCH scan ends up in npn-leaf\n");
    		exit(0);
    	}




    }



    for(k=0; k<NBCFmax; k++)
    {
    	free(BirchCFarray[k].sum);
    	free(BirchCFarray[k].ssum);
    	free(BirchCFarray[k].children_index);
    }

    free(BirchCFarray);
    */

    return 0;
}
