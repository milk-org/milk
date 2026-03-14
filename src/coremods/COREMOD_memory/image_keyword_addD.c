/**
 * @file image_keyword_addD.c
 * @brief Image keyword addd module
 */

#ifdef MILK_NO_CLI
#include "CLIcore_standalone.h"
#else
#include "CLIcore.h"
#endif

static char   *inimname;
static char   *kwname;
static double *kwval;
static char   *comment;

static CLICMDARGDEF farg[] = {{
        CLIARG_IMG,
        ".in_name",
        "input image",
        "im1",
        (FPFLAG_DEFAULT_INPUT | FPFLAG_CLI_INPUT),
        (void **) &inimname,
        NULL
    },
    {
        CLIARG_STR,
        ".kwname",
        "keyword name",
        "KW1",
        (FPFLAG_DEFAULT_INPUT | FPFLAG_CLI_INPUT),
        (void **) &kwname,
        NULL
    },
    {
        CLIARG_FLOAT64,
        ".kwval",
        "keyword value",
        "1.234",
        (FPFLAG_DEFAULT_INPUT | FPFLAG_CLI_INPUT),
        (void **) &kwval,
        NULL
    },
    {
        CLIARG_STR,
        ".comment",
        "comment",
        "keyword comment",
        (FPFLAG_DEFAULT_INPUT | FPFLAG_CLI_INPUT),
        (void **) &comment,
        NULL
    }
};

static CLICMDDATA CLIcmddata =
{
    "imkwaddD", "add float type image keyword", CLICMD_FIELDS_NOFPS
};

errno_t image_keyword_addD(IMGID img, char *kwname, double kwval, char *comment)
{
    resolveIMGID(&img, ERRMODE_ABORT, dcimg, dcnimg);

    int NBkw = img.md->NBkw; // max nb kw

    int kw = 0;
    while((img.im->kw[kw].type != 'N') && (kw < NBkw))
    {
        kw++;
    }
    int kw0 = kw;

    if(kw0 == NBkw)
    {
        printf("WARNING: no available keyword entry -> ignored\n");
        //abort();
    }
    else
    {
        strcpy(img.im->kw[kw].name, kwname);
        img.im->kw[kw].type       = 'D';
        img.im->kw[kw].value.numf = kwval;
        strcpy(img.im->kw[kw].comment, comment);
    }

    return RETURN_SUCCESS;
}

static MILK_HOT errno_t compute_function()
{
    image_keyword_addD(imgid_make_from_name(inimname), kwname, *kwval, comment);
    return RETURN_SUCCESS;
}

INSERT_STD_CLIfunction

errno_t
CLIADDCMD_COREMOD_memory__image_keyword_addD()
{
    INSERT_STD_CLIREGISTERFUNC

    return RETURN_SUCCESS;
}
