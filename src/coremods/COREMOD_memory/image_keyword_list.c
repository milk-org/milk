// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file image_keyword_list.c
 * @brief Image keyword list module
 */

#ifdef MILK_NO_CLI
#    include "CLIcore_standalone.h"
#    include "COREMOD_memory/COREMOD_memory.h"
#else
#    include "CLIcore.h"
#endif

static char *inimname;

static CLICMDARGDEF farg[] = { { CLIARG_IMG, ".in_name", "input image", "im1",
                                 (FPFLAG_DEFAULT_INPUT | FPFLAG_CLI_INPUT), (void **) &inimname,
                                 NULL } };

static CLICMDDATA CLIcmddata = { "imkwlist", "list image keywords", CLICMD_FIELDS_NOFPS };

errno_t image_keywords_list(IMGID img)
{
    resolveIMGID(&img, ERRMODE_WARN, dcimg, dcnimg);

    int NBkw = img.md->NBkw;
    if (img.ID == -1)
    {
        return RETURN_FAILURE;
    }
    int kwcnt = 0;
    for (int kw = 0; kw < NBkw; kw++)
    {
        char tmpkwvalstr[81];
        switch (img.im->kw[kw].type)
        {
        case 'L':
            printf("[L] %-8s= %20ld / %s\n", img.im->kw[kw].name, img.im->kw[kw].value.numl,
                   img.im->kw[kw].comment);
            kwcnt++;
            break;

        case 'D':
            printf("[D] %-8s= %20g / %s\n", img.im->kw[kw].name, img.im->kw[kw].value.numf,
                   img.im->kw[kw].comment);
            kwcnt++;
            break;

        case 'S':
            snprintf(tmpkwvalstr, sizeof(tmpkwvalstr), "'%s'", img.im->kw[kw].value.valstr);
            printf("[S] %-8s= %-20s / %s\n", img.im->kw[kw].name, tmpkwvalstr,
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

static MILK_HOT errno_t __attribute__((unused)) compute_function()
{
    image_keywords_list(imgid_make_from_name(inimname));
    return RETURN_SUCCESS;
}

INSERT_STD_CLIfunction

    errno_t
    CLIADDCMD_COREMOD_memory__image_keyword_list()
{
    INSERT_STD_CLIREGISTERFUNC return RETURN_SUCCESS;
}
