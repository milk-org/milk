/**
 * @file image_keyword_addS.c
 * @brief Image keyword adds module
 */

#ifdef MILK_NO_CLI
#include "CLIcore_standalone.h"
#include "COREMOD_memory/COREMOD_memory.h"
#else
#include "CLIcore.h"
#endif

static char *inimname;
static char *kwname;
static char *kwval;
static char *comment;

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
        CLIARG_STR,
        ".kwval",
        "keyword value",
        "blue",
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
    "imkwaddS", "add string type image keyword", CLICMD_FIELDS_NOFPS
};

errno_t image_keyword_addS(
    IMGID img,
    char *kwname,
    char *kwval,
    char *comment
)
{
    resolveIMGID(&img, ERRMODE_WARN, dcimg, dcnimg);

    int NBkw = img.md->NBkw; // max nb kw
    if (img.ID == -1) {
        return RETURN_FAILURE;
    }

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
        //printf("writing kw %3d / %3d  \"%s\"  %d %d %d\n", kw, NBkw, kwname, strlen(kwname), strlen(kwval), strlen(comment));

        strncpy(img.im->kw[kw].name, kwname, KEYWORD_MAX_STRING);
        img.im->kw[kw].type = 'S';
        strncpy(img.im->kw[kw].value.valstr, kwval, KEYWORD_MAX_STRING);
        strncpy(img.im->kw[kw].comment, comment, KEYWORD_MAX_COMMENT);
    }

    return RETURN_SUCCESS;
}

static MILK_HOT errno_t __attribute__((unused)) compute_function()
{
    image_keyword_addS(imgid_make_from_name(inimname), kwname, kwval, comment);
    return RETURN_SUCCESS;
}

INSERT_STD_CLIfunction

errno_t
CLIADDCMD_COREMOD_memory__image_keyword_addS()
{
    INSERT_STD_CLIREGISTERFUNC

    return RETURN_SUCCESS;
}
