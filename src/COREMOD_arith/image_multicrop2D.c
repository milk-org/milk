/**
 * @file    image_multicrop2D.c
 * @brief   crop 2D function, multiple windows
 *
 */

#include "CommandLineInterface/CLIcore.h"

static char *mcropinsname;
static long fpi_mcropinsname;

static char *outsname;
static long fpi_outsname;


static uint32_t *outxsize;
static long fpi_outxsize;

static uint32_t *outysize;
static long fpi_outysize;


/*
#define CROPWINDOWVARS(WINDEX) \
static int64_t *w#WINDEXactive; \
static long     fpi_w#WINDEXactive; \
static uint32_t *w#WINDEXcropxstart; \
long fpi_w#WINDEXcropxstart; \
static uint32_t *w#WINDEXcropxsize; \
long fpi_w#WINDEXcropxsize; \
static uint32_t *w#WINDEXcropystart; \
long fpi_w#WINDEXcropystart; \
static uint32_t *w#WINDEXcropysize; \
long fpi_w#WINDEXcropysize;

CROPWINDOWVARS(00)
*/


// Window 00

#define MAXNB_CROPWINDOW 8

static int64_t *wactive[MAXNB_CROPWINDOW];
static long     fpi_wactive[MAXNB_CROPWINDOW];

// addmode = 0 if replacing pixels, 1 if adding
static int64_t *waddmode[MAXNB_CROPWINDOW];
static long     fpi_waddmode[MAXNB_CROPWINDOW];

static uint32_t *wcropxstart[MAXNB_CROPWINDOW];
static long fpi_wcropxstart[MAXNB_CROPWINDOW];

static uint32_t *wcropxsize[MAXNB_CROPWINDOW];
static long fpi_wcropxsize[MAXNB_CROPWINDOW];

static uint32_t *wcropystart[MAXNB_CROPWINDOW];
static long fpi_wcropystart[MAXNB_CROPWINDOW];

static uint32_t *wcropysize[MAXNB_CROPWINDOW];
static long fpi_wcropysize[MAXNB_CROPWINDOW];


// output position

static uint32_t *wcropxpos[MAXNB_CROPWINDOW];
static long fpi_wcropxpos[MAXNB_CROPWINDOW];

static uint32_t *wcropypos[MAXNB_CROPWINDOW];
static long fpi_wcropypos[MAXNB_CROPWINDOW];





#define STR_EXPAND(tok) #tok
#define STR(tok) STR_EXPAND(tok)


#define CROPWINDOWONOFF(wn) \
    { \
        CLIARG_ONOFF,\
        ".w"#wn".active",\
        "crop window active flag", \
        "0",\
        CLIARG_HIDDEN_DEFAULT,\
        (void **) &wactive[wn],\
        &fpi_wactive[wn]\
    }

#define CROPWINDOWADDMODE(wn) \
    { \
        CLIARG_ONOFF,\
        ".w"#wn".addmode",\
        "1 if adding, 0 if replacing", \
        "0",\
        CLIARG_HIDDEN_DEFAULT,\
        (void **) &waddmode[wn],\
        &fpi_waddmode[wn]\
    }

#define CROPWINDOWXSTART(wn) \
    {\
        CLIARG_UINT32,\
        ".w"#wn".cropxstart",\
        "crop x coord start",\
        "30",\
        CLIARG_HIDDEN_DEFAULT,\
        (void **) &wcropxstart[wn],\
        &fpi_wcropxstart[wn]\
    }

#define CROPWINDOWXSIZE(wn) \
    {\
        CLIARG_UINT32,\
        ".w"#wn".cropxsize",\
        "crop x coord size",\
        "30",\
        CLIARG_HIDDEN_DEFAULT,\
        (void **) &wcropxsize[wn],\
        &fpi_wcropxsize[wn]\
    }

#define CROPWINDOWYSTART(wn) \
    {\
        CLIARG_UINT32,\
        ".w"#wn".cropystart",\
        "crop y coord start",\
        "30",\
        CLIARG_HIDDEN_DEFAULT,\
        (void **) &wcropystart[wn],\
        &fpi_wcropystart[wn]\
    }

#define CROPWINDOWYSIZE(wn) \
    {\
        CLIARG_UINT32,\
        ".w"#wn".cropysize",\
        "crop y coord size",\
        "30",\
        CLIARG_HIDDEN_DEFAULT,\
        (void **) &wcropysize[wn],\
        &fpi_wcropysize[wn]\
    }

#define CROPWINDOWXPOS(wn) \
    {\
        CLIARG_UINT32,\
        ".w"#wn".cropxpos",\
        "crop x placement in output",\
        "30",\
        CLIARG_HIDDEN_DEFAULT,\
        (void **) &wcropxpos[wn],\
        &fpi_wcropxpos[wn]\
    }

#define CROPWINDOWYPOS(wn) \
    {\
        CLIARG_UINT32,\
        ".w"#wn".cropypos",\
        "crop y placement in output",\
        "30",\
        CLIARG_HIDDEN_DEFAULT,\
        (void **) &wcropypos[wn],\
        &fpi_wcropypos[wn]\
    }



static CLICMDARGDEF farg[] =
{
    {
        CLIARG_IMG,
        ".insname",
        "input stream name",
        "inim",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &mcropinsname,
        &fpi_mcropinsname
    },
    {
        CLIARG_STR,
        ".outsname",
        "output stream name",
        "outim",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &outsname,
        &fpi_outsname
    },
    {
        CLIARG_UINT32,
        ".outxsize",
        "output x size",
        "200",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &outxsize,
        &fpi_outxsize
    },
    {
        CLIARG_UINT32,
        ".outysize",
        "output y size",
        "200",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &outysize,
        &fpi_outysize
    },
    CROPWINDOWONOFF(00),
    CROPWINDOWADDMODE(00),
    CROPWINDOWXSTART(00),
    CROPWINDOWXSIZE(00),
    CROPWINDOWYSTART(00),
    CROPWINDOWYSIZE(00),
    CROPWINDOWXPOS(00),
    CROPWINDOWYPOS(00),

    CROPWINDOWONOFF(01),
    CROPWINDOWADDMODE(01),
    CROPWINDOWXSTART(01),
    CROPWINDOWXSIZE(01),
    CROPWINDOWYSTART(01),
    CROPWINDOWYSIZE(01),
    CROPWINDOWXPOS(01),
    CROPWINDOWYPOS(01),

    CROPWINDOWONOFF(02),
    CROPWINDOWADDMODE(02),
    CROPWINDOWXSTART(02),
    CROPWINDOWXSIZE(02),
    CROPWINDOWYSTART(02),
    CROPWINDOWYSIZE(02),
    CROPWINDOWXPOS(02),
    CROPWINDOWYPOS(02),

    CROPWINDOWONOFF(03),
    CROPWINDOWADDMODE(03),
    CROPWINDOWXSTART(03),
    CROPWINDOWXSIZE(03),
    CROPWINDOWYSTART(03),
    CROPWINDOWYSIZE(03),
    CROPWINDOWXPOS(03),
    CROPWINDOWYPOS(03),

    CROPWINDOWONOFF(04),
    CROPWINDOWADDMODE(04),
    CROPWINDOWXSTART(04),
    CROPWINDOWXSIZE(04),
    CROPWINDOWYSTART(04),
    CROPWINDOWYSIZE(04),
    CROPWINDOWXPOS(04),
    CROPWINDOWYPOS(04),

    CROPWINDOWONOFF(05),
    CROPWINDOWADDMODE(05),
    CROPWINDOWXSTART(05),
    CROPWINDOWXSIZE(05),
    CROPWINDOWYSTART(05),
    CROPWINDOWYSIZE(05),
    CROPWINDOWXPOS(05),
    CROPWINDOWYPOS(05),

    CROPWINDOWONOFF(06),
    CROPWINDOWADDMODE(06),
    CROPWINDOWXSTART(06),
    CROPWINDOWXSIZE(06),
    CROPWINDOWYSTART(06),
    CROPWINDOWYSIZE(06),
    CROPWINDOWXPOS(06),
    CROPWINDOWYPOS(06),

    CROPWINDOWONOFF(07),
    CROPWINDOWADDMODE(07),
    CROPWINDOWXSTART(07),
    CROPWINDOWXSIZE(07),
    CROPWINDOWYSTART(07),
    CROPWINDOWYSIZE(07),
    CROPWINDOWXPOS(07),
    CROPWINDOWYPOS(07)
};



// Optional custom configuration setup.
// Runs once at conf startup
//
static errno_t customCONFsetup()
{
    if(data.fpsptr != NULL)
    {
        data.fpsptr->parray[fpi_mcropinsname].fpflag |=
            FPFLAG_STREAM_RUN_REQUIRED | FPFLAG_CHECKSTREAM;
    }

    return RETURN_SUCCESS;
}

// Optional custom configuration checks.
// Runs at every configuration check loop iteration
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
    "multicrop2D", "crop 2D image, multiple crops", CLICMD_FIELDS_DEFAULTS
};

// detailed help
static errno_t help_function()
{
    return RETURN_SUCCESS;
}







static errno_t compute_function()
{
    DEBUG_TRACE_FSTART();

    // CONNECT TO INPUT STREAM
    IMGID imgin = mkIMGID_from_name(mcropinsname);
    resolveIMGID(&imgin, ERRMODE_ABORT);

    // CONNNECT TO OR CREATE OUTPUT STREAM
    IMGID imgout = stream_connect_create_2D(outsname, *outxsize, *outysize, imgin.md->datatype);


    INSERT_STD_PROCINFO_COMPUTEFUNC_INIT;



    INSERT_STD_PROCINFO_COMPUTEFUNC_LOOPSTART
    {

        // zero output array
        switch (imgin.md->datatype)
        {
        case _DATATYPE_FLOAT:
            for(uint64_t ii=0; ii < *outxsize * *outysize; ii++)
            {
                imgout.im->array.F[ii] = 0.0;
            }
            break;

        case _DATATYPE_DOUBLE:
            for(uint64_t ii=0; ii < *outxsize * *outysize; ii++)
            {
                imgout.im->array.D[ii] = 0.0;
            }
            break;

        case _DATATYPE_UINT8:
            for(uint64_t ii=0; ii < *outxsize * *outysize; ii++)
            {
                imgout.im->array.UI8[ii] = 0;
            }
            break;

        case _DATATYPE_UINT16:
            for(uint64_t ii=0; ii < *outxsize * *outysize; ii++)
            {
                imgout.im->array.UI16[ii] = 0;
            }
            break;

        case _DATATYPE_UINT32:
            for(uint64_t ii=0; ii < *outxsize * *outysize; ii++)
            {
                imgout.im->array.UI32[ii] = 0;
            }
            break;

        case _DATATYPE_UINT64:
            for(uint64_t ii=0; ii < *outxsize * *outysize; ii++)
            {
                imgout.im->array.UI64[ii] = 0;
            }
            break;

        case _DATATYPE_INT8:
            for(uint64_t ii=0; ii < *outxsize * *outysize; ii++)
            {
                imgout.im->array.SI8[ii] = 0;
            }
            break;

        case _DATATYPE_INT16:
            for(uint64_t ii=0; ii < *outxsize * *outysize; ii++)
            {
                imgout.im->array.SI16[ii] = 0;
            }
            break;

        case _DATATYPE_INT32:
            for(uint64_t ii=0; ii < *outxsize * *outysize; ii++)
            {
                imgout.im->array.SI32[ii] = 0;
            }
            break;

        case _DATATYPE_INT64:
            for(uint64_t ii=0; ii < *outxsize * *outysize; ii++)
            {
                imgout.im->array.SI64[ii] = 0;
            }
            break;

        }



        for(int cropwindow=0; cropwindow < MAXNB_CROPWINDOW ; cropwindow++)
        {
            if ( *wactive[cropwindow] == 1)
            {
                uint32_t iimax = *wcropxsize[cropwindow];
                printf("%d  iimax %5d\n", __LINE__, iimax);
                if ( iimax +  *wcropxpos[cropwindow] > (*outxsize))
                {
                    iimax = (*outxsize) - *wcropxpos[cropwindow];
                }
                printf("%d  iimax %5d\n", __LINE__, iimax);

                if ( iimax + *wcropxstart[cropwindow] > imgin.md->size[0])
                {
                    iimax = imgin.md->size[0] - *wcropxstart[cropwindow];
                }
                printf("%d  iimax %5d\n", __LINE__, iimax);


                uint32_t jjmax = *wcropysize[cropwindow];
                if ( jjmax +  *wcropypos[cropwindow] > (*outysize))
                {
                    jjmax = (*outysize) - *wcropypos[cropwindow];
                }

                if ( jjmax + *wcropystart[cropwindow] > imgin.md->size[1])
                {
                    jjmax = imgin.md->size[1] - *wcropystart[cropwindow];
                }

                printf("iimax %5d jjmax %5d\n", iimax, jjmax);



                for(uint32_t jj = 0; jj < jjmax; jj++)
                {
                    uint64_t indjj = jj + (*wcropystart[cropwindow]);
                    indjj *=  imgin.md->size[0];


                    if ( *waddmode[cropwindow] == 0 )
                    {
                        switch (imgin.md->datatype)
                        {
                        case _DATATYPE_FLOAT:
                            memcpy( &imgout.im->array.F[ (*wcropypos[cropwindow] + jj) * (*outxsize) + *wcropxpos[cropwindow] ],
                                    &imgin.im->array.F[ indjj + (*wcropxstart[cropwindow]) ],
                                    iimax * SIZEOF_DATATYPE_FLOAT);
                            break;

                        case _DATATYPE_DOUBLE:
                            memcpy( &imgout.im->array.D[ (*wcropypos[cropwindow] + jj) * (*outxsize) + *wcropxpos[cropwindow] ],
                                    &imgin.im->array.D[ indjj + (*wcropxstart[cropwindow]) ],
                                    iimax * SIZEOF_DATATYPE_DOUBLE);
                            break;

                        case _DATATYPE_UINT8:
                            memcpy( &imgout.im->array.UI8[ (*wcropypos[cropwindow] + jj) * (*outxsize) + *wcropxpos[cropwindow] ],
                                    &imgin.im->array.UI8[ indjj + (*wcropxstart[cropwindow]) ],
                                    iimax * SIZEOF_DATATYPE_UINT8);
                            break;

                        case _DATATYPE_UINT16:
                            memcpy( &imgout.im->array.UI16[ (*wcropypos[cropwindow] + jj) * (*outxsize) + *wcropxpos[cropwindow] ],
                                    &imgin.im->array.UI16[ indjj + (*wcropxstart[cropwindow]) ],
                                    iimax * SIZEOF_DATATYPE_UINT16);
                            break;

                        case _DATATYPE_UINT32:
                            memcpy( &imgout.im->array.UI32[ (*wcropypos[cropwindow] + jj) * (*outxsize) + *wcropxpos[cropwindow] ],
                                    &imgin.im->array.UI32[ indjj + (*wcropxstart[cropwindow]) ],
                                    iimax * SIZEOF_DATATYPE_UINT32);
                            break;

                        case _DATATYPE_UINT64:
                            memcpy( &imgout.im->array.UI64[ (*wcropypos[cropwindow] + jj) * (*outxsize) + *wcropxpos[cropwindow] ],
                                    &imgin.im->array.UI64[ indjj + (*wcropxstart[cropwindow]) ],
                                    iimax * SIZEOF_DATATYPE_UINT64);
                            break;

                        case _DATATYPE_INT8:
                            memcpy( &imgout.im->array.SI8[ (*wcropypos[cropwindow] + jj) * (*outxsize) + *wcropxpos[cropwindow] ],
                                    &imgin.im->array.SI8[ indjj + (*wcropxstart[cropwindow]) ],
                                    iimax * SIZEOF_DATATYPE_INT8);
                            break;

                        case _DATATYPE_INT16:
                            memcpy( &imgout.im->array.SI16[ (*wcropypos[cropwindow] + jj) * (*outxsize) + *wcropxpos[cropwindow] ],
                                    &imgin.im->array.SI16[ indjj + (*wcropxstart[cropwindow]) ],
                                    iimax * SIZEOF_DATATYPE_INT16);
                            break;

                        case _DATATYPE_INT32:
                            memcpy( &imgout.im->array.SI32[ (*wcropypos[cropwindow] + jj) * (*outxsize) + *wcropxpos[cropwindow] ],
                                    &imgin.im->array.SI32[ indjj + (*wcropxstart[cropwindow]) ],
                                    iimax * SIZEOF_DATATYPE_INT32);
                            break;

                        case _DATATYPE_INT64:
                            memcpy( &imgout.im->array.SI64[ (*wcropypos[cropwindow] + jj) * (*outxsize) + *wcropxpos[cropwindow] ],
                                    &imgin.im->array.SI64[ indjj + (*wcropxstart[cropwindow]) ],
                                    iimax * SIZEOF_DATATYPE_INT64);
                            break;

                        }
                    }
                    else  // add pixels
                    {
                        switch (imgin.md->datatype)
                        {

                        case _DATATYPE_FLOAT:
                            for(int ii=0; ii< iimax; ii++)
                            {
                                imgout.im->array.F[ (*wcropypos[cropwindow] + jj) * (*outxsize) + (*wcropxpos[cropwindow] + ii)] +=
                                    imgin.im->array.F[ indjj + (*wcropxstart[cropwindow] + ii)];
                            }
                            break;

                        case _DATATYPE_DOUBLE:
                            for(int ii=0; ii< iimax; ii++)
                            {
                                imgout.im->array.D[ (*wcropypos[cropwindow] + jj)  * (*outxsize) + (*wcropxpos[cropwindow] + ii)] +=
                                    imgin.im->array.D[ indjj + (*wcropxstart[cropwindow] + ii)];
                            }
                            break;

                        case _DATATYPE_UINT8:
                            for(int ii=0; ii< iimax; ii++)
                            {
                                imgout.im->array.UI8[ (*wcropypos[cropwindow] + jj)  * (*outxsize) + (*wcropxpos[cropwindow] + ii)] +=
                                    imgin.im->array.UI8[ indjj + (*wcropxstart[cropwindow] + ii)];
                            }
                            break;

                        case _DATATYPE_UINT16:
                            for(int ii=0; ii< iimax; ii++)
                            {
                                imgout.im->array.UI16[ (*wcropypos[cropwindow] + jj)  * (*outxsize) + (*wcropxpos[cropwindow] + ii)] +=
                                    imgin.im->array.UI16[ indjj + (*wcropxstart[cropwindow] + ii)];
                            }
                            break;

                        case _DATATYPE_UINT32:
                            for(int ii=0; ii< iimax; ii++)
                            {
                                imgout.im->array.UI32[ (*wcropypos[cropwindow] + jj)  * (*outxsize) + (*wcropxpos[cropwindow] + ii)] +=
                                    imgin.im->array.UI32[ indjj + (*wcropxstart[cropwindow] + ii)];
                            }
                            break;

                        case _DATATYPE_UINT64:
                            for(int ii=0; ii< iimax; ii++)
                            {
                                imgout.im->array.UI64[ (*wcropypos[cropwindow] + jj)  * (*outxsize) + (*wcropxpos[cropwindow] + ii)] +=
                                    imgin.im->array.UI64[ indjj + (*wcropxstart[cropwindow] + ii)];
                            }
                            break;

                        case _DATATYPE_INT8:
                            for(int ii=0; ii< iimax; ii++)
                            {
                                imgout.im->array.SI8[ (*wcropypos[cropwindow] + jj)  * (*outxsize) + (*wcropxpos[cropwindow] + ii)] +=
                                    imgin.im->array.SI8[ indjj + (*wcropxstart[cropwindow] + ii)];
                            }
                            break;

                        case _DATATYPE_INT16:
                            for(int ii=0; ii< iimax; ii++)
                            {
                                imgout.im->array.SI16[ (*wcropypos[cropwindow] + jj)  * (*outxsize) + (*wcropxpos[cropwindow] + ii)] +=
                                    imgin.im->array.SI16[ indjj + (*wcropxstart[cropwindow] + ii)];
                            }
                            break;

                        case _DATATYPE_INT32:
                            for(int ii=0; ii< iimax; ii++)
                            {
                                imgout.im->array.SI32[ (*wcropypos[cropwindow] + jj)  * (*outxsize) + (*wcropxpos[cropwindow] + ii)] +=
                                    imgin.im->array.SI32[ indjj + (*wcropxstart[cropwindow] + ii)];
                            }
                            break;

                        case _DATATYPE_INT64:
                            for(int ii=0; ii< iimax; ii++)
                            {
                                imgout.im->array.SI64[ (*wcropypos[cropwindow] + jj)  * (*outxsize)+ (*wcropxpos[cropwindow]  + ii)] +=
                                    imgin.im->array.SI64[ indjj + (*wcropxstart[cropwindow] + ii)];
                            }
                            break;
                        }

                    }
                }
            }
        }
        processinfo_update_output_stream(processinfo, imgout.ID);

    }
    INSERT_STD_PROCINFO_COMPUTEFUNC_END

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}





INSERT_STD_FPSCLIfunctions




// Register function in CLI
errno_t
CLIADDCMD_COREMODE_arith__multicrop2D()
{
    CLIcmddata.FPS_customCONFsetup = customCONFsetup;
    CLIcmddata.FPS_customCONFcheck = customCONFcheck;
    INSERT_STD_CLIREGISTERFUNC

    return RETURN_SUCCESS;
}
