/**
 * @file    stream_monproc.c
 * @brief   monitor stream
 */

#define NCURSES_WIDECHAR 1

#include <math.h>

#include "CommandLineInterface/CLIcore.h"




static char *inimname;

// time binning flag
// binary digits encode binning
// 2 [bin 10] : bin2
// 6 [bin 110] : bin2 and bin4
//  [bin 110000] : bin16 and bin32
static uint64_t *tbinflag;
static long     fpi_tbinflag = -1;


static CLICMDARGDEF farg[] =
{
    {
        CLIARG_IMG,
        ".in_name",
        "input image",
        "im1",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &inimname,
        NULL
    },
    {
        CLIARG_UINT64,
        ".tbinflag",
        "time binning flag, bin digit",
        "48",
        CLIARG_HIDDEN_DEFAULT,
        (void **) &tbinflag,
        &fpi_tbinflag
    }
};


// Optional custom configuration setup
// Runs once at conf startup
//
// To use this function, set :
// CLIcmddata.FPS_customCONFsetup = customCONFsetup
// when registering function
// (see end of this file)
//
static errno_t customCONFsetup()
{

    return RETURN_SUCCESS;
}

// Optional custom configuration checks
// Runs at every configuration check loop iteration
//
// To use this function, set :
// CLIcmddata.FPS_customCONFcheck = customCONFcheck
// when registering function
// (see end of this file)
//
static errno_t customCONFcheck()
{
    if(data.fpsptr != NULL)
    {

    }

    return RETURN_SUCCESS;
}



static CLICMDDATA CLIcmddata =
{
    "streammon",
    "stream monitor",
    CLICMD_FIELDS_DEFAULTS
};




// detailed help
static errno_t help_function()
{
    return RETURN_SUCCESS;
}


static errno_t stream_monitor_process(
    IMGID *inimg
)
{
    DEBUG_TRACE_FSTART();
    // custom stream process function code

    // resolve image
    // This function call has low overhead, as it will acknowledge existing image
    resolveIMGID(inimg, ERRMODE_ABORT);

    uint32_t xsize  = inimg->size[0];
    uint32_t ysize  = inimg->size[1];
    uint64_t xysize = xsize * ysize;


    // Create output image if needed
    //imcreateIMGID(outimg);

    //outimg->md->write = 1;

    //for(uint64_t ii = 0; ii < xysize; ii++)
    //{
    //    outimg->im->array.F[ii] = sqrt(inimg->im->array.F[ii]);
    //}

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}





static errno_t compute_function()
{
    DEBUG_TRACE_FSTART();


    // Check if image is in memory
    // First, create an IMGIG with the image name
    IMGID inimg = mkIMGID_from_name(inimname);
    // Then resolve it (connect it to an image in memory if possible)
    resolveIMGID(&inimg, ERRMODE_ABORT);
    uint32_t xsize  = inimg.size[0];
    uint32_t ysize  = inimg.size[1];
    uint64_t xysize = xsize * ysize;


    printf(" COMPUTE Flags = %ld\n", CLIcmddata.cmdsettings->flags);
    INSERT_STD_PROCINFO_COMPUTEFUNC_INIT

    // custom initialization
    printf(" COMPUTE Flags = %ld\n", CLIcmddata.cmdsettings->flags);
    if(CLIcmddata.cmdsettings->flags & CLICMDFLAG_PROCINFO)
    {
        // procinfo is accessible here
    }

    // If custom initialization with access to procinfo is not required
    // then replace
    // INSERT_STD_PROCINFO_COMPUTEFUNC_INIT
    // INSERT_STD_PROCINFO_COMPUTEFUNC_LOOPSTART
    // With :
    // INSERT_STD_PROCINFO_COMPUTEFUNC_START



    printf("tbinflag = %lu\n", *tbinflag);


    // We assume a 64-bit integer for this loop.
    // Usually sizeof(int) * 8 is safer for portability.
    int numbin = 0;
    int tbinarray[100];
    int bincounter[100];
    for (int i = 0; i < sizeof(uint64_t) * 8; i++) {

        // Check if the bit at position 'i' is 1
        if ((*tbinflag >> i) & 1) {
            // 1U << i calculates 2 to the power of i
            // We use 1U (unsigned) to ensure it handles the 31st bit correctly
            //printf("%u ", (1U << i));
            tbinarray[numbin] = (int) (1U << i);
            bincounter[numbin] = 0;
            numbin++;
        }
    }
    printf("\n");

    // internal array for accumulation
    double** arraysum = (double**)malloc(numbin * sizeof(double*));
    double** arraysumsq = (double**)malloc(numbin * sizeof(double*));

    IMGID* imgoutbin = (IMGID*) malloc(sizeof(IMGID)*numbin);
    IMGID* imgoutbinrms = (IMGID*) malloc(sizeof(IMGID)*numbin);

    for(int tbin=0; tbin<numbin; tbin++)
    {
        printf("Allocating arrays for time binning factor %d\n", tbinarray[tbin]);
        arraysum[tbin] = (double*)malloc(xysize * sizeof(double));
        arraysumsq[tbin] = (double*)malloc(xysize * sizeof(double));


        // Creating img
        {
            char imoutsanme[STRINGMAXLEN_IMGNAME];
            WRITE_IMAGENAME(imoutsanme, "%s.tbin%d", inimg.name, tbinarray[tbin]);
            imgoutbin[tbin] = mkIMGID_from_name(imoutsanme);
            imgoutbin[tbin].shared = 1;
            imgoutbin[tbin].NBkw = inimg.NBkw;
            imgoutbin[tbin].CBsize = inimg.CBsize;
            imgoutbin[tbin] = stream_connect_create_2D(imoutsanme, xsize, ysize, _DATATYPE_FLOAT);
            imcreateIMGID(&imgoutbin[tbin]);
        }
        {
            char imoutsanme[STRINGMAXLEN_IMGNAME];
            WRITE_IMAGENAME(imoutsanme, "%s.tbin%d.rms", inimg.name, tbinarray[tbin]);
            imgoutbinrms[tbin] = mkIMGID_from_name(imoutsanme);
            imgoutbinrms[tbin].shared = 1;
            imgoutbinrms[tbin].NBkw = inimg.NBkw;
            imgoutbinrms[tbin].CBsize = inimg.CBsize;
            imgoutbinrms[tbin] = stream_connect_create_2D(imoutsanme, xsize, ysize, _DATATYPE_FLOAT);
            imcreateIMGID(&imgoutbin[tbin]);
        }
    }

    uint64_t loopcnt = 0;
    INSERT_STD_PROCINFO_COMPUTEFUNC_LOOPSTART
    {
        printf("[%8lu]  ", loopcnt);
        for(int binindex=0; binindex<numbin; binindex++)
        {
            printf("%d %3d/%3d     ", binindex, bincounter[binindex], tbinarray[binindex]);
        }
        printf("\n");


        printf("    Adding input frame to bin index 0\n");
        for (uint64_t pixi=0; pixi<xysize; pixi++)
        {
            double pval = inimg.im->array.F[pixi];
            arraysum[0][pixi] += pval;
            arraysumsq[0][pixi] += pval*pval;
        }

        bincounter[0] ++;

        for(int binindex=0; binindex<numbin; binindex++)
        {
            if (bincounter[binindex] == tbinarray[binindex])
            {
                if(binindex+1 < numbin)
                {
                    printf("    Adding bin index %d to bin index %d\n", binindex, binindex+1);

                    for (uint64_t pixi=0; pixi<xysize; pixi++)
                    {
                        arraysum[binindex+1][pixi] += arraysum[binindex][pixi];
                        arraysumsq[binindex+1][pixi] += arraysumsq[binindex][pixi];
                    }
                    bincounter[binindex+1] += bincounter[binindex];
                }


                printf("    >>>>>>>> UPDATING BIN FRAME %d\n", binindex);

                imgoutbin[binindex].md->write = 1;
                for (uint64_t pixi=0; pixi<xysize; pixi++)
                {
                    imgoutbin[binindex].im->array.F[pixi] = arraysum[binindex][pixi]/bincounter[binindex];
                }
                processinfo_update_output_stream(processinfo, imgoutbin[binindex].ID);


                imgoutbinrms[binindex].md->write = 1;
                for (uint64_t pixi=0; pixi<xysize; pixi++)
                {
                    double v1 = arraysumsq[binindex][pixi] / bincounter[binindex];
                    double v2 = arraysum[binindex][pixi] / bincounter[binindex];

                    imgoutbinrms[binindex].im->array.F[pixi] = sqrt(v1 - v2*v2);
                }
                processinfo_update_output_stream(processinfo, imgoutbinrms[binindex].ID);

                for (uint64_t pixi=0; pixi<xysize; pixi++)
                {
                    arraysum[binindex][pixi] = 0.0;
                    arraysumsq[binindex][pixi] = 0.0;
                }
                bincounter[binindex] = 0;
            }
        }


        stream_monitor_process(&inimg);

        // stream is updated here, and not in the function called above, so that
        // the above function can be chained with others
        //processinfo_update_output_stream(processinfo, outimg.ID);

        loopcnt++;

    }
    INSERT_STD_PROCINFO_COMPUTEFUNC_END

    for(int tbin=0; tbin<numbin; tbin++)
    {
        free(arraysum[tbin]);
        free(arraysumsq[tbin]);
    }
    free(arraysum);
    free(arraysumsq);

    free(imgoutbin);
    free(imgoutbinrms);

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}



INSERT_STD_FPSCLIfunctions



// Register function in CLI
errno_t
CLIADDCMD_info__stream_monproc()
{
    CLIcmddata.FPS_customCONFsetup = customCONFsetup;
    CLIcmddata.FPS_customCONFcheck = customCONFcheck;

    INSERT_STD_CLIREGISTERFUNC

    return RETURN_SUCCESS;
}
