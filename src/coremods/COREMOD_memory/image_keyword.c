/**
 * @file    image_keyword.c
 * @brief   image keyword read/write
 *
 * Uses FPS V2 framework.
 */

#ifdef MILK_NO_CLI
#include "CLIcore_standalone.h"
#else
#include "CLIcore.h"
#endif
#include "fps.h"

#include "image_ID.h"

/* forward decls */
long image_write_keyword_L(
    const char *IDname, const char *kname,
    long value, const char *comment);

imageID image_list_keywords(
    const char *restrict IDname);


/* ================================================================
 *  CMD 1: imlistkw (1 arg)
 * ============================================================= */

static FPS_APP_INFO FPS_app_info_listkw = {
    .fps_name    = "imlistkw",
    .cmdkey      = "imlistkw",
    .description = "list image keywords"
};

static char p_listkw_imname[
    FUNCTION_PARAMETER_STRMAXLEN] = "im1";

#define FPS_PARAMS_listkw(X) \
    X(".imname", p_listkw_imname, \
      FPTYPE_STREAMNAME, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "image name")

static FPS_CLI_BINDING bindings_listkw[] = {
    FPS_PARAMS_listkw(FPS_X_BINDING)
};

static const int nb_bindings_listkw =
    sizeof(bindings_listkw) /
    sizeof(FPS_CLI_BINDING);

static CLICMDARGDEF farg_listkw[] = {
    FPS_PARAMS_listkw(FPS_X_FARG)
};

static CLICMDDATA CLIcmddata_listkw = {
    "",
    "",
    CLICMD_FIELDS_NOPARAM
};

static CMDSETTINGS cms_listkw = {0};

static __attribute__((constructor))
void init_cms_listkw(void)
{
    strncpy(CLIcmddata_listkw.key,
            FPS_app_info_listkw.cmdkey,
            sizeof(CLIcmddata_listkw.key)
            - 1);
    strncpy(
        CLIcmddata_listkw.description,
        FPS_app_info_listkw.description,
        sizeof(
            CLIcmddata_listkw.description
        ) - 1);
    if (CLIcmddata_listkw.cmdsettings
        == NULL) {
        CLIcmddata_listkw.cmdsettings =
            &cms_listkw;
    }
}

static errno_t compute_listkw()
{
    image_list_keywords(p_listkw_imname);
    return RETURN_SUCCESS;
}


/* ================================================================
 *  CMD 2: imwritekwL (4 args, primary)
 * ============================================================= */

static FPS_APP_INFO FPS_app_info = {
    .fps_name    = "imwritekwL",
    .cmdkey      = "imwritekwL",
    .description =
        "write long type keyword"
};

static char p_imname[
    FUNCTION_PARAMETER_STRMAXLEN] = "im1";
static char p_kname[
    FUNCTION_PARAMETER_STRMAXLEN] = "kw2";
static long long p_kwval = 34;
static char p_comment[
    FUNCTION_PARAMETER_STRMAXLEN]
    = "my_keyword_comment";

#define FPS_PARAMS(X) \
    X(".imname", p_imname, \
      FPTYPE_STREAMNAME, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "image name") \
    X(".kname", p_kname, \
      FPTYPE_STRING, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "keyword name") \
    X(".kwval", &p_kwval, \
      FPTYPE_INT64, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "keyword value") \
    X(".comment", p_comment, \
      FPTYPE_STRING, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "comment")

static FPS_CLI_BINDING my_bindings[] = {
    FPS_PARAMS(FPS_X_BINDING)
};

static const int nb_bindings =
    sizeof(my_bindings) /
    sizeof(FPS_CLI_BINDING);

static CLICMDARGDEF farg[] = {
    FPS_PARAMS(FPS_X_FARG)
};

static CLICMDDATA CLIcmddata = {
    "",
    "",
    CLICMD_FIELDS_DEFAULTS
};

static CMDSETTINGS cms_writkw = {0};

static __attribute__((constructor))
void init_cms_writkw(void)
{
    strncpy(CLIcmddata.key,
            FPS_app_info.cmdkey,
            sizeof(CLIcmddata.key) - 1);
    strncpy(CLIcmddata.description,
            FPS_app_info.description,
            sizeof(CLIcmddata.description)
            - 1);
    if (CLIcmddata.cmdsettings == NULL) {
        CLIcmddata.cmdsettings =
            &cms_writkw;
    }
}

static MILK_HOT errno_t compute_function()
{
    DEBUG_TRACE_FSTART();
    INSERT_STD_PROCINFO_COMPUTEFUNC_START
    image_write_keyword_L(
        p_imname, p_kname,
        p_kwval, p_comment);
    INSERT_STD_PROCINFO_COMPUTEFUNC_END
    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}


/* ================================================================
 *  REGISTRATION
 * ============================================================= */

#if !defined(FPS_STANDALONE) && !defined(MILK_NO_CLI)

static errno_t CLIfunction_listkw(void)
{
    return safe_fps_generic_CLIfunction(
        &FPS_app_info_listkw,
        farg_listkw, &CLIcmddata_listkw,
        bindings_listkw, nb_bindings_listkw,
        compute_listkw);
}

static errno_t CLIfunction(void)
{
    return safe_fps_generic_CLIfunction(
        &FPS_app_info, farg, &CLIcmddata,
        my_bindings, nb_bindings,
        compute_function);
}

errno_t
CLIADDCMD_COREMOD_memory__image_keyword()
{
    {
        safe_fps_fill_farg_examples(
            farg_listkw, bindings_listkw,
            nb_bindings_listkw);

        int cmdi = RegisterCLIcmd(
            CLIcmddata_listkw,
            CLIfunction_listkw);
        CLIcmddata_listkw.cmdsettings =
            &data.cmd[cmdi].cmdsettings;
    }

    {
        safe_fps_fill_farg_examples(
            farg, my_bindings, nb_bindings);

        int cmdi = RegisterCLIcmd(
            CLIcmddata, CLIfunction);
        CLIcmddata.cmdsettings =
            &data.cmd[cmdi].cmdsettings;
    }

    return RETURN_SUCCESS;
}
#endif


/* ================================================================
 *  COMPUTATION CODE
 * ============================================================= */

long image_write_keyword_L(
    const char *IDname,
    const char *kname,
    long        value,
    const char *comment)
{
    imageID ID;
    long    kw, NBkw, kw0;

    ID   = image_ID(IDname, dcimg, dcnimg);
    NBkw = dcimg[ID].md[0].NBkw;

    kw = 0;
    while((dcimg[ID].kw[kw].type != 'N')
          && (kw < NBkw))
    {
        kw++;
    }
    kw0 = kw;

    if(kw0 == NBkw)
    {
        printf("ERROR: no available"
               " keyword entry\n");
        exit(0);
    }
    else
    {
        strcpy(dcimg[ID].kw[kw].name,
               kname);
        dcimg[ID].kw[kw].type       = 'L';
        dcimg[ID].kw[kw].value.numl = value;
        strcpy(dcimg[ID].kw[kw].comment,
               comment);
    }

    return kw0;
}

long image_write_keyword_D(
    const char *IDname,
    const char *kname,
    double      value,
    const char *comment)
{
    imageID ID;
    long    kw;
    long    NBkw;
    long    kw0;

    ID   = image_ID(IDname, dcimg, dcnimg);
    NBkw = dcimg[ID].md[0].NBkw;

    kw = 0;
    while((dcimg[ID].kw[kw].type != 'N')
          && (kw < NBkw))
    {
        kw++;
    }
    kw0 = kw;

    if(kw0 == NBkw)
    {
        printf("ERROR: no available"
               " keyword entry\n");
        exit(0);
    }
    else
    {
        strcpy(dcimg[ID].kw[kw].name,
               kname);
        dcimg[ID].kw[kw].type       = 'D';
        dcimg[ID].kw[kw].value.numf = value;
        strcpy(dcimg[ID].kw[kw].comment,
               comment);
    }

    return kw0;
}

long image_write_keyword_S(
    const char *IDname,
    const char *kname,
    const char *value,
    const char *comment)
{
    imageID ID;
    long    kw;
    long    NBkw;
    long    kw0;

    ID   = image_ID(IDname, dcimg, dcnimg);
    NBkw = dcimg[ID].md[0].NBkw;

    kw = 0;
    while((dcimg[ID].kw[kw].type != 'N')
          && (kw < NBkw))
    {
        kw++;
    }
    kw0 = kw;

    if(kw0 == NBkw)
    {
        printf("ERROR: no available"
               " keyword entry\n");
        exit(0);
    }
    else
    {
        strcpy(dcimg[ID].kw[kw].name,
               kname);
        dcimg[ID].kw[kw].type = 'D';
        strcpy(dcimg[ID].kw[kw].value.valstr,
               value);
        strcpy(dcimg[ID].kw[kw].comment,
               comment);
    }

    return kw0;
}

imageID image_list_keywords(
    const char *restrict IDname)
{
    imageID ID;
    long    kw;

    ID = image_ID(IDname, dcimg, dcnimg);

    int kwcnt = 0;
    for(kw = 0;
        kw < dcimg[ID].md->NBkw;
        kw++)
    {

        switch (dcimg[ID].kw[kw].type)
        {
        case 'L' :
            printf(
                "%18s  %20ld %s\n",
                dcimg[ID].kw[kw].name,
                dcimg[ID].kw[kw].value.numl,
                dcimg[ID].kw[kw].comment);
            kwcnt ++;
            break;

        case 'D' :
            printf(
                "%18s  %20lf %s\n",
                dcimg[ID].kw[kw].name,
                dcimg[ID].kw[kw].value.numf,
                dcimg[ID].kw[kw].comment);
            kwcnt ++;
            break;

        case 'S' :
            printf(
                "%18s  %20s %s\n",
                dcimg[ID].kw[kw].name,
                dcimg[ID].kw[kw].value.valstr,
                dcimg[ID].kw[kw].comment);
            kwcnt ++;
            break;
        }
    }

    printf("%d / %d keywords set\n",
           kwcnt, dcimg[ID].md->NBkw);

    return ID;
}

long image_read_keyword_D(
    const char *IDname,
    const char *kname,
    double *val)
{
    variableID ID;
    long       kw;
    long       kw0;

    ID  = image_ID(IDname, dcimg, dcnimg);
    kw0 = -1;
    for(kw = 0;
        kw < dcimg[ID].md[0].NBkw;
        kw++)
    {
        if((dcimg[ID].kw[kw].type == 'D')
           && (strncmp(
                   kname,
                   dcimg[ID].kw[kw].name,
                   strlen(kname)) == 0))
        {
            kw0  = kw;
            *val =
                dcimg[ID].kw[kw].value.numf;
        }
    }

    return kw0;
}

long image_read_keyword_L(
    const char *IDname,
    const char *kname,
    long *val)
{
    variableID ID;
    long       kw;
    long       kw0;

    ID  = image_ID(IDname, dcimg, dcnimg);
    kw0 = -1;
    for(kw = 0;
        kw < dcimg[ID].md[0].NBkw;
        kw++)
    {
        if((dcimg[ID].kw[kw].type == 'L')
           && (strncmp(
                   kname,
                   dcimg[ID].kw[kw].name,
                   strlen(kname)) == 0))
        {
            kw0  = kw;
            *val =
                dcimg[ID].kw[kw].value.numl;
        }
    }

    return kw0;
}
