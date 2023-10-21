/** @file pupfft.c
 */

#include "CommandLineInterface/CLIcore.h"

#include "COREMOD_memory/COREMOD_memory.h"

#include "dofft.h"
#include "permut.h"



static char *inamp;
static char *inpha;

static char *outamp;
static char *outpha;




static CLICMDARGDEF farg[] =
{
    {
        CLIARG_IMG,
        ".inamp",
        "input WF ampl",
        "ima",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &inamp,
        NULL
    },
    {
        CLIARG_IMG,
        ".inpha",
        "input WF phase",
        "imp",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &inpha,
        NULL
    },
    {
        CLIARG_STR,
        ".outa",
        "output WF ampl",
        "outa",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &outamp,
        NULL
    },
    {
        CLIARG_STR,
        ".outp",
        "output WF phase",
        "outp",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &outpha,
        NULL
    }
};


static errno_t customCONFsetup()
{
    return RETURN_SUCCESS;
}


static errno_t customCONFcheck()
{
    return RETURN_SUCCESS;
}


static CLICMDDATA CLIcmddata =
{
    "pup2foc",
    "pupil to focus by FFT",
    CLICMD_FIELDS_DEFAULTS
};



// detailed help
static errno_t help_function()
{
    return RETURN_SUCCESS;
}




/* inv = 0 for direct fft and 1 for inverse fft */
/* direct = focal plane -> pupil plane  equ. fft2d(..,..,..,1) */
/* inverse = pupil plane -> focal plane equ. fft2d(..,..,..,0) */
/* options :  -reim  takes real/imaginary input and creates real/imaginary output
               -inv  for inverse fft (inv=1) */
errno_t pup2foc_fft(
    const char * __restrict ID_name_ampl,
    const char * __restrict ID_name_pha,
    const char * __restrict ID_name_ampl_out,
    const char * __restrict ID_name_pha_out,
    const char * __restrict options
)
{
    int reim;
    int inv;

    char Ctmpname[STRINGMAXLEN_IMGNAME];
    char C1tmpname[STRINGMAXLEN_IMGNAME];

    reim = 0;
    inv  = 0;

    if(strstr(options, "-reim") != NULL)
    {
        /*	printf("taking real / imaginary input/output\n");*/
        reim = 1;
    }

    if(strstr(options, "-inv") != NULL)
    {
        /*printf("doing the inverse Fourier transform\n");*/
        inv = 1;
    }

    WRITE_IMAGENAME(Ctmpname, "_Ctmp_%d", (int) getpid());

    if(reim == 0)
    {
        mk_complex_from_amph(ID_name_ampl, ID_name_pha, Ctmpname, 0);
    }
    else
    {
        mk_complex_from_reim(ID_name_ampl, ID_name_pha, Ctmpname, 0);
    }

    permut(Ctmpname);



    WRITE_IMAGENAME(C1tmpname, "_C1tmp_%d", (int) getpid());

    if(inv == 0)
    {
        do2dfft(Ctmpname, C1tmpname); /* equ. fft2d(..,1) */
    }
    else
    {
        do2dffti(Ctmpname, C1tmpname); /* equ. fft2d(..,0) */
    }

    delete_image_ID(Ctmpname, DELETE_IMAGE_ERRMODE_WARNING);

    if(reim == 0)
    {
        /* if this line is removed, the program crashes... why ??? */
        /*	list_image_ID(data); */
        mk_amph_from_complex(C1tmpname, ID_name_ampl_out, ID_name_pha_out, 0);
    }
    else
    {
        mk_reim_from_complex(C1tmpname, ID_name_ampl_out, ID_name_pha_out, 0);
    }

    delete_image_ID(C1tmpname, DELETE_IMAGE_ERRMODE_WARNING);

    permut(ID_name_ampl_out);
    permut(ID_name_pha_out);

    return RETURN_SUCCESS;
}





static errno_t compute_function()
{
    DEBUG_TRACE_FSTART();

    IMGID imgamp = mkIMGID_from_name(inamp);
    resolveIMGID(&imgamp, ERRMODE_ABORT);

    IMGID imgpha = mkIMGID_from_name(inpha);
    resolveIMGID(&imgpha, ERRMODE_ABORT);

//    printf(" COMPUTE Flags = %ld\n", CLIcmddata.cmdsettings->flags);
    INSERT_STD_PROCINFO_COMPUTEFUNC_INIT

    // custom initialization
    printf(" COMPUTE Flags = %ld\n", CLIcmddata.cmdsettings->flags);
    if(CLIcmddata.cmdsettings->flags & CLICMDFLAG_PROCINFO)
    {
        // procinfo is accessible here
    }

    INSERT_STD_PROCINFO_COMPUTEFUNC_LOOPSTART
    {
        pup2foc_fft(inamp, inpha, outamp, outpha, "");
    }
    INSERT_STD_PROCINFO_COMPUTEFUNC_END

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}



INSERT_STD_FPSCLIfunctions



// Register function in CLI
errno_t
CLIADDCMD_milk_fft__pup2foc()
{
    CLIcmddata.FPS_customCONFsetup = customCONFsetup;
    CLIcmddata.FPS_customCONFcheck = customCONFcheck;

    INSERT_STD_CLIREGISTERFUNC

    return RETURN_SUCCESS;
}
