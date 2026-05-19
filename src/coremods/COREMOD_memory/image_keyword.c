/**
 * @file    image_keyword.c
 * @brief   Image keyword read/write API
 *
 * FITS-like keyword metadata attached to images.
 * Each image has a fixed-size keyword array; entries
 * are typed as Long ('L'), Double ('D'), or
 * String ('S'). Unused slots have type 'N'.
 *
 * CLI commands:
 *  - imlistkw   — list all keywords of an image
 *  - imwritekwL — write a long-type keyword
 */

#ifdef MILK_NO_CLI
#include "CLIcore_standalone.h"
#else
#include "CLIcore.h"
#endif
#include "fps.h"
#include "COREMOD_memory/COREMOD_memory.h"

#include "image_ID.h"

/* forward decls */
long image_write_keyword_L(
    const char *IDname,
    const char *kname,
    long       value,
    const char *comment);

/**
 * @brief Print all keywords attached to an image.
 */
imageID image_list_keywords(
    const char *restrict IDname);


/* ================================================================
 *  CMD 1: imlistkw (1 arg)
 * ============================================================= */

static FPS_APP_INFO FPS_app_info_listkw =
{
    .fps_name    = "imlistkw",
    .cmdkey      = "imlistkw",
    .description = "list image keywords",
    .description_long =
    "List, read, or modify FITS-style keywords attached to a shared memory image stream. Keywords are stored in the stream metadata header."
};

static char p_listkw_imname[FUNCTION_PARAMETER_STRMAXLEN] = "im1";

#define FPS_PARAMS_listkw(X) \
    X(".imname", p_listkw_imname, \
      FPTYPE_STREAMNAME, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "image name")

static FPS_CLI_BINDING bindings_listkw[] =
{
    FPS_PARAMS_listkw(FPS_X_BINDING)
};

static const int __attribute__((unused)) nb_bindings_listkw =
    sizeof(bindings_listkw) /
    sizeof(FPS_CLI_BINDING);

static CLICMDARGDEF __attribute__((unused)) farg_listkw[] =
{
    FPS_PARAMS_listkw(FPS_X_FARG)
};

static CLICMDDATA CLIcmddata_listkw =
{
    "",
    "",
    CLICMD_FIELDS_NOPARAM
};

FPS_CMDSETTINGS_INIT(listkw, CLIcmddata_listkw, FPS_app_info_listkw)

static errno_t __attribute__((unused)) compute_listkw()
{
    image_list_keywords(p_listkw_imname);
    return RETURN_SUCCESS;
}


/* ================================================================
 *  CMD 2: imwritekwL (4 args, primary)
 * ============================================================= */

static FPS_APP_INFO FPS_app_info =
{
    .fps_name    = "imwritekwL",
    .cmdkey      = "imwritekwL",
    .description =
    "write long type keyword",
    .description_long =
    "List, read, or modify FITS-style keywords attached to a shared memory image stream. Keywords are stored in the stream metadata header."
};

static char p_imname[FUNCTION_PARAMETER_STRMAXLEN] = "im1";
static char p_kname[FUNCTION_PARAMETER_STRMAXLEN] = "kw2";
static long long p_kwval = 34;
static char p_comment[FUNCTION_PARAMETER_STRMAXLEN] = "my_keyword_comment";

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

static FPS_CLI_BINDING my_bindings[] =
{
    FPS_PARAMS(FPS_X_BINDING)
};

static const int __attribute__((unused)) nb_bindings =
    sizeof(my_bindings) /
    sizeof(FPS_CLI_BINDING);

static CLICMDARGDEF farg[] =
{
    FPS_PARAMS(FPS_X_FARG)
};

static CLICMDDATA CLIcmddata =
{
    "",
    "",
    CLICMD_FIELDS_DEFAULTS
};

FPS_CMDSETTINGS_INIT(writkw, CLIcmddata, FPS_app_info)

static MILK_HOT errno_t __attribute__((unused)) compute_function()
{
    DEBUG_TRACE_FSTART();
    INSERT_STD_PROCINFO_COMPUTEFUNC_START
    image_write_keyword_L(p_imname, p_kname, p_kwval, p_comment);
    INSERT_STD_PROCINFO_COMPUTEFUNC_END DEBUG_TRACE_FEXIT();
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
               bindings_listkw, nb_bindings_listkw, compute_listkw);
}

static errno_t CLIfunction(void)
{
    return safe_fps_generic_CLIfunction(
               &FPS_app_info, farg, &CLIcmddata, my_bindings, nb_bindings, compute_function);
}

errno_t
CLIADDCMD_COREMOD_memory__image_keyword()
{
    {
        safe_fps_fill_farg_examples(farg_listkw, bindings_listkw, nb_bindings_listkw);

        int cmdi = RegisterCLIcmd(CLIcmddata_listkw, CLIfunction_listkw);
        CLIcmddata_listkw.cmdsettings = &data.cmd[cmdi].cmdsettings;
    }

    {
        safe_fps_fill_farg_examples(farg, my_bindings, nb_bindings);

        int cmdi = RegisterCLIcmd(CLIcmddata, CLIfunction);
        CLIcmddata.cmdsettings = &data.cmd[cmdi].cmdsettings;
    }

    return RETURN_SUCCESS;
}
#endif


/* ================================================================
 *  COMPUTATION CODE
 * ============================================================= */

/**
 * @brief Internal helper to write a keyword
 */
static long _image_write_keyword(
    IMGID      img,
    const char *kname,
    char       type,
    long       numl,
    double     numf,
    const char *valstr,
    const char *comment)
{
    resolveIMGID(&img, ERRMODE_WARN, dcimg, dcnimg);
    long ID = img.ID;
    if(ID == -1)
    {
        return RETURN_FAILURE;
    }
    long NBkw = dcimg[ID].md[0].NBkw;

    long kw = 0;
    while((dcimg[ID].kw[kw].type != 'N') && (kw < NBkw))
    {
        kw++;
    }
    long kw0 = kw;

    if(kw0 == NBkw)
    {
        PRINT_ERROR("no available keyword slot");
        return -1;
    }
    else
    {
        snprintf(dcimg[ID].kw[kw].name, KEYWORD_MAX_STRING, "%s", kname);
        dcimg[ID].kw[kw].type = type;
        if(type == 'L')
        {
            dcimg[ID].kw[kw].value.numl = numl;
        }
        else if(type == 'D')
        {
            dcimg[ID].kw[kw].value.numf = numf;
        }
        else if(type == 'S')
        {
            snprintf(dcimg[ID].kw[kw].value.valstr, KEYWORD_MAX_STRING, "%s", valstr);
        }
        snprintf(dcimg[ID].kw[kw].comment, KEYWORD_MAX_COMMENT, "%s", comment);
    }

    return kw0;
}

/**
 * @brief Write a long-type keyword to an image
 *
 * Finds the first empty keyword slot (type 'N')
 * and writes name, value, and comment. Returns
 * the slot index, or exits if no slots available.
 *
 * @param IDname   Image name
 * @param kname    Keyword name
 * @param value    Long integer value
 * @param comment  Keyword comment string
 * @return Keyword slot index
 */
long image_write_keyword_L(
    const char *IDname,
    const char *kname,
    long       value,
    const char *comment)
{
    IMGID img = imgid_make_from_name(IDname);
    return _image_write_keyword(img, kname, 'L', value, 0.0, NULL, comment);
}

/**
 * @brief Write a double-type keyword to an image
 *
 * @param IDname   Image name
 * @param kname    Keyword name
 * @param value    Double-precision value
 * @param comment  Keyword comment string
 * @return Keyword slot index
 */
long image_write_keyword_D(
    const char *IDname,
    const char *kname,
    double     value,
    const char *comment)
{
    IMGID img = imgid_make_from_name(IDname);
    return _image_write_keyword(img, kname, 'D', 0, value, NULL, comment);
}

/**
 * @brief Write a string-type keyword to an image
 *
 * @param IDname   Image name
 * @param kname    Keyword name
 * @param value    String value
 * @param comment  Keyword comment string
 * @return Keyword slot index
 */
long image_write_keyword_S(
    const char *IDname,
    const char *kname,
    const char *value,
    const char *comment)
{
    IMGID img = imgid_make_from_name(IDname);
    return _image_write_keyword(img, kname, 'S', 0, 0.0, value, comment);
}

/**
 * @brief Legacy wrappers taking IMGID
 */
errno_t image_keyword_addL(
    IMGID      img,
    const char *kwname,
    long       kwval,
    const char *comment)
{
    _image_write_keyword(img, kwname, 'L', kwval, 0.0, NULL, comment);
    return RETURN_SUCCESS;
}

errno_t image_keyword_addD(
    IMGID      img,
    const char *kwname,
    double     kwval,
    const char *comment)
{
    _image_write_keyword(img, kwname, 'D', 0, kwval, NULL, comment);
    return RETURN_SUCCESS;
}

errno_t image_keyword_addS(
    IMGID      img,
    const char *kwname,
    const char *kwval,
    const char *comment)
{
    _image_write_keyword(img, kwname, 'S', 0, 0.0, kwval, comment);
    return RETURN_SUCCESS;
}

/**
 * @brief List all keywords of an image
 *
 * Prints each keyword's name, typed value, and
 * comment to stdout.
 *
 * @param IDname  Image name
 * @return Image ID
 */
imageID image_list_keywords(
    const char *restrict IDname)
{
    IMGID img = imgid_make_from_name(IDname);
    resolveIMGID(&img, ERRMODE_WARN, dcimg, dcnimg);
    long ID = img.ID;
    if(img.ID == -1)
    {
        return RETURN_FAILURE;
    }


    int kwcnt = 0;
    for(long kw = 0;
            kw < dcimg[ID].md->NBkw;
            kw++)
    {

        switch(dcimg[ID].kw[kw].type)
        {
        case 'L' :
            printf(
                "%18s  %20ld %s\n",
                dcimg[ID].kw[kw].name, dcimg[ID].kw[kw].value.numl, dcimg[ID].kw[kw].comment);
            kwcnt ++;
            break;

        case 'D' :
            printf(
                "%18s  %20lf %s\n",
                dcimg[ID].kw[kw].name, dcimg[ID].kw[kw].value.numf, dcimg[ID].kw[kw].comment);
            kwcnt ++;
            break;

        case 'S' :
            printf(
                "%18s  %20s %s\n",
                dcimg[ID].kw[kw].name, dcimg[ID].kw[kw].value.valstr, dcimg[ID].kw[kw].comment);
            kwcnt ++;
            break;
        }
    }

    printf("%d / %d keywords set\n", kwcnt, dcimg[ID].md->NBkw);

    return ID;
}

/**
 * @brief Read a double-type keyword value
 *
 * Scans keyword array for a matching 'D'-type entry.
 *
 * @param IDname  Image name
 * @param kname   Keyword name to find
 * @param val     Output: keyword value
 * @return Keyword slot index, or -1 if not found
 */
long image_read_keyword_D(
    const char *IDname,
    const char *kname,
    double     *val)
{
    IMGID img = imgid_make_from_name(IDname);
    resolveIMGID(&img, ERRMODE_NULL, dcimg, dcnimg);
    if(img.ID == -1)
    {
        return -1;
    }
    long ID = img.ID;

    long       kw0;

    kw0 = -1;
    for(long kw = 0;
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
            *val = dcimg[ID].kw[kw].value.numf;
        }
    }

    return kw0;
}

/**
 * @brief Read a long-type keyword value
 *
 * @param IDname  Image name
 * @param kname   Keyword name to find
 * @param val     Output: keyword value
 * @return Keyword slot index, or -1 if not found
 */
long image_read_keyword_L(
    const char *IDname,
    const char *kname,
    long       *val)
{
    IMGID img = imgid_make_from_name(IDname);
    resolveIMGID(&img, ERRMODE_NULL, dcimg, dcnimg);
    if(img.ID == -1)
    {
        return -1;
    }
    long ID = img.ID;

    long       kw0;

    kw0 = -1;
    for(long kw = 0;
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
            *val = dcimg[ID].kw[kw].value.numl;
        }
    }

    return kw0;
}
