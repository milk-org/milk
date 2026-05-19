/**
 * @file milk-fps-set.c
 * @brief Milk fps set module
 */

#include <getopt.h>

#include "fps_disconnect.h"
#include "fps_GetParamIndex.h"

/* Helper to check if string starts with prefix */
static int starts_with(
    const char *pre,
    const char *str)
{
    size_t lenpre = strlen(pre);
    size_t lenstr = strlen(str);
    return lenstr < lenpre ? 0 : strncmp(pre, str, lenpre) == 0;
}

/**
 * @brief Map FPS type code to a short name string.
 *
 * Returns "INT64", "FLOAT32", etc.
 */
static const char *get_type_name(uint32_t type)
{
    if(type & FPTYPE_INT32)
    {
        return "INT32";
    }
    if(type & FPTYPE_UINT32)
    {
        return "UINT32";
    }
    if(type & FPTYPE_INT64)
    {
        return "INT64";
    }
    if(type & FPTYPE_UINT64)
    {
        return "UINT64";
    }
    if(type & FPTYPE_FLOAT32)
    {
        return "FLOAT32";
    }
    if(type & FPTYPE_FLOAT64)
    {
        return "FLOAT64";
    }
    if(type & FPTYPE_STRING)
    {
        return "STRING";
    }
    if(type & FPTYPE_FILENAME)
    {
        return "FILENAME";
    }
    if(type & FPTYPE_FITSFILENAME)
    {
        return "FITSFILENAME";
    }
    if(type & FPTYPE_EXECFILENAME)
    {
        return "EXECFILENAME";
    }
    if(type & FPTYPE_DIRNAME)
    {
        return "DIRNAME";
    }
    if(type & FPTYPE_STREAMNAME)
    {
        return "STREAMNAME";
    }
    if(type & FPTYPE_FPSNAME)
    {
        return "FPSNAME";
    }
    if(type & FPTYPE_ONOFF)
    {
        return "ONOFF";
    }
    if(type & FPTYPE_TIMESPEC)
    {
        return "TIMESPEC";
    }
    if(type & FPTYPE_PID)
    {
        return "PID";
    }
    return "UNKNOWN";
}

#define FSET_DESC \
    "set a parameter value in a Function Parameter Structure (FPS)"
#define FSET_DESC_LONG \
    "Write a new value to a named FPS parameter in shared memory.\n" \
    "The parameter is specified as <FPSname>.<paramkey>.\n" \
    "Supported types: INT32, UINT32, INT64, UINT64, FLOAT32, FLOAT64,\n" \
    "STRING, FILENAME, STREAMNAME, ONOFF, TIMESPEC, PID.\n" \
    "TIMESPEC parameters accept a float value in seconds (e.g. 0.001)."

/**
 * @brief Print help message for milk-fps-set.
 */
static void print_help(
    const char *progname,
    int        mh_color)
{
    milk_help_banner(progname, FSET_DESC, mh_color);
    milk_help_section("Usage", mh_color);
    printf("  %s%s%s %s<FPSname>.<param>%s %s<value>%s\n\n",
           mh_color ? MH_CMD : "", progname, mh_color ? MH_RST : "",
           mh_color ? MH_ARG : "", mh_color ? MH_RST : "",
           mh_color ? MH_ARG : "", mh_color ? MH_RST : "");
    milk_help_section("Description", mh_color);
    printf("  %s\n\n", FSET_DESC_LONG);
    milk_help_section("Options", mh_color);
    printf("  %s%-25s%s %s\n",
           mh_color ? MH_OPT : "", "-h, --help",
           mh_color ? MH_RST : "", "Show this help and exit");
    printf("  %s%-25s%s %s\n",
           mh_color ? MH_OPT : "", "-h1, --help-oneline",
           mh_color ? MH_RST : "", "One-line description and exit");
    printf("  %s%-25s%s %s\n",
           mh_color ? MH_OPT : "", "-h2, --help-description",
           mh_color ? MH_RST : "", "Verbose description and exit");
    printf("  %s%-25s%s %s\n\n",
           mh_color ? MH_OPT : "", "-hm, --help-mono",
           mh_color ? MH_RST : "", "Full help, no ANSI color");
    milk_help_section("Examples", mh_color);
    printf("  %s$ milk-fps-set%s %smyfps00.procinfo.enabled%s %sTRUE%s\n",
           mh_color ? MH_CMD : "", mh_color ? MH_RST : "",
           mh_color ? MH_ARG : "", mh_color ? MH_RST : "",
           mh_color ? MH_DFLT : "", mh_color ? MH_RST : "");
    printf("  %s$ milk-fps-set%s %smyfps00.loopfreq%s %s100.0%s\n\n",
           mh_color ? MH_CMD : "", mh_color ? MH_RST : "",
           mh_color ? MH_ARG : "", mh_color ? MH_RST : "",
           mh_color ? MH_DFLT : "", mh_color ? MH_RST : "");
    const char *see_also[] =
    {
        "milk-fps-info:inspect FPS directory contents",
        "milk-fps-list:list active FPS instances",
        "milk-fps-track:monitor FPS parameters"
    };
    milk_help_see_also(see_also, 3, mh_color);
}

/**
 * @brief Scan FPS entries for tab completion.
 *
 * Builds a list of matching parameter keywords
 * for readline tab completion.
 */
void do_completion_scan(const char *word)
{
    char *dot = strchr(word, '.');

    if(dot == NULL)
    {
        // No dot, scan for FPS names
        FPS *fpsarray = (FPS *) calloc(NB_FPS_MAX, sizeof(FPS));
        KEYWORD_TREE_NODE *keywnode = (KEYWORD_TREE_NODE *) calloc(NB_KEYWNODE_MAX,
                                      sizeof(KEYWORD_TREE_NODE));
        int NBkwn = 0;
        int NBfps = 0;
        long NBpindex = 0;

        functionparameter_scan_fps(0, "_ALL", fpsarray, keywnode, &NBkwn, &NBfps, &NBpindex, 0);

        for(int ii = 0; ii < NBfps; ii++)
        {
            if(starts_with(word, fpsarray[ii].md->name))
            {
                printf("%s.\n", fpsarray[ii].md->name);
            }
            fps_disconnect(&fpsarray[ii]);
        }
        free(fpsarray);
        free(keywnode);
    }
    else
    {
        // Dot found, extract FPS name
        char fpsname[128];
        int len = dot - word;
        if(len >= 128)
        {
            len = 127;
        }
        strncpy(fpsname, word, len);
        fpsname[len] = '\0';

        char *param_prefix = dot + 1;

        FPS fps;
        long NBparam = fps_connect(fpsname, &fps, FPSCONNECT_SIMPLE);
        if(NBparam != -1)
        {
            fps.NBparam = NBparam;
            for(int ii = 0; ii < fps.NBparam; ii++)
            {
                if(!(fps.parray[ii].fpflag & FPFLAG_ACTIVE))
                {
                    continue;
                }

                const char *kw = fps.parray[ii].keywordfull;
                // kw is typically "FPSname.keyword"

                // We want to match param_prefix against the part AFTER FPSname.

                // Verify kw starts with fpsname
                if(strncmp(kw, fpsname, strlen(fpsname)) == 0 && kw[strlen(fpsname)] == '.')
                {
                    const char *suffix = kw + strlen(fpsname) + 1;
                    if(starts_with(param_prefix, suffix))
                    {
                        printf("%s\n", kw);
                    }
                }
                else
                {
                    // Fallback if kw format is unexpected (e.g. just .keyword)
                    // If kw starts with '.', treat it as suffix directly
                    if(kw[0] == '.')
                    {
                        if(starts_with(param_prefix, kw + 1))
                        {
                            printf("%s%s\n", fpsname, kw);
                        }
                    }
                }
            }
            fps_disconnect(&fps);
        }
    }
}

int main(
    int argc,
    char *argv[])
{
    int action = milk_help_init(argc, argv, FSET_DESC, FSET_DESC_LONG);
    if(action == MH_ACTION_H1 || action == MH_ACTION_H2)
    {
        return 0;
    }
    int mh_color = (action == MH_ACTION_HELP);
    if(action == MH_ACTION_HELP || action == MH_ACTION_MONO)
    {
        print_help(argv[0], mh_color);
        return 0;
    }

    /* Check for bash completion mode */
    if(getenv("COMP_LINE") != NULL)
    {
        if(argc >= 3)
        {
            do_completion_scan(argv[2]);
            return 0;
        }
    }

    int opt;
    int option_index = 0;
    static struct option long_options[] =
    {
        {"help",    no_argument,       0, 'h'},
        {"help-oneline", no_argument, 0, '1'},
        {0, 0, 0, 0}
    };

    while((opt = getopt_long(argc, argv, "h1",
                             long_options, &option_index)) != -1)
    {
        switch(opt)
        {
        case 'h':
        case '1':
            break; /* handled above */
        default: printf("\n\033[1;31mERROR\033[0m invalid option\n\n");
            print_help(argv[0], 1);
            return 1;
        }
    }

    if(optind + 2 > argc)
    {
        if(optind + 1 == argc)
        {
            printf("\n\033[1;31mERROR\033[0m missing value\n\n");
        }
        else
        {
            printf("\n\033[1;31mERROR\033[0m missing arguments\n\n");
        }
        print_help(argv[0], 1);
        return 1;
    }

        {
    char *fullkey = argv[optind];
        char *value_str = argv[optind + 1];

        char *dot = strchr(fullkey, '.');
        if(dot == NULL)
        {
            PRINT_ERROR("Error: Invalid format '%s'. Expected <FPSname>.<parameter>", fullkey);
            return 1;
        }

        char fpsname[128];
        int len = dot - fullkey;
        if(len >= 128)
        {
            len = 127;
        }
        strncpy(fpsname, fullkey, len);
        fpsname[len] = '\0';

        char *keyword = dot; // Includes the dot

        FPS fps;
        long NBparam = fps_connect(fpsname, &fps, FPSCONNECT_SIMPLE);
        if(NBparam == -1)
        {
            PRINT_ERROR("Error: Could not connect to FPS '%s'", fpsname);
            return 1;
        }
        fps.NBparam = NBparam;

        long pindex = functionparameter_GetParamIndex(&fps, keyword);
        if(pindex == -1)
        {
            pindex = functionparameter_GetParamIndex(&fps, dot + 1);
            if(pindex == -1)
            {
                PRINT_ERROR("Error: Parameter '%s' not found in FPS '%s'", keyword, fpsname);
                fps_disconnect(&fps);
                return 1;
            }
        }

        int type = fps.parray[pindex].type;
        int vOK = 1;
        if(functionparameter_SetParamValue_fromString(&fps, pindex, value_str) == RETURN_SUCCESS)
        {
            printf("Parameter '%s' set to '%s'\n", fullkey, value_str);
        }
        else
        {
            vOK = 0;
            PRINT_ERROR("Error: Failed to set parameter '%s'. Type mismatch or invalid format.", fullkey);
            PRINT_ERROR("       Parameter Type: %s", get_type_name(type));
            PRINT_ERROR("       Input Value:    '%s'", value_str);
        }

        fps_disconnect(&fps);
        return vOK ? 0 : 1;

    }
}
