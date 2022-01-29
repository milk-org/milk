#include "CommandLineInterface/CLIcore.h"

static char *inimname;

static CLICMDARGDEF farg[] = {{CLIARG_IMG,
                               ".in_name",
                               "input image",
                               "im1",
                               CLIARG_VISIBLE_DEFAULT,
                               (void **) &inimname,
                               NULL}};

static CLICMDDATA CLIcmddata = {
    "imkwlist", "list image keywords", CLICMD_FIELDS_NOFPS};

errno_t image_keywords_list(IMGID img)
{
    resolveIMGID(&img, ERRMODE_ABORT);

    int NBkw  = img.md->NBkw;
    int kwcnt = 0;
    for (int kw = 0; kw < NBkw; kw++)
    {
        char tmpkwvalstr[81];
        switch (img.im->kw[kw].type)
        {
        case 'L':
            printf("[L] %-8s= %20ld / %s\n",
                   img.im->kw[kw].name,
                   img.im->kw[kw].value.numl,
                   img.im->kw[kw].comment);
            kwcnt++;
            break;

        case 'D':
            printf("[D] %-8s= %20g / %s\n",
                   img.im->kw[kw].name,
                   img.im->kw[kw].value.numf,
                   img.im->kw[kw].comment);
            kwcnt++;
            break;

        case 'S':
            sprintf(tmpkwvalstr, "'%s'", img.im->kw[kw].value.valstr);
            printf("[S] %-8s= %-20s / %s\n",
                   img.im->kw[kw].name,
                   tmpkwvalstr,
                   img.im->kw[kw].comment);
            kwcnt++;
            break;

        default:
            break;
        }
    }

    printf("%d / %d keyword(s)\n", kwcnt, NBkw);

    return RETURN_SUCCESS;
}

static errno_t compute_function()
{
    image_keywords_list(mkIMGID_from_name(inimname));
    return RETURN_SUCCESS;
}

INSERT_STD_CLIfunction

    errno_t
    CLIADDCMD_COREMOD_memory__image_keyword_list()
{
    INSERT_STD_CLIREGISTERFUNC

    return RETURN_SUCCESS;
}
