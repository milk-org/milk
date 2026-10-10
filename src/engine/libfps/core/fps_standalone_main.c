// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    fps_standalone_main.c
 * @brief   Command-line entrypoint (fpsexec_main_function)
 *
 * Compiled once into milkfpsStandalone with -DMILK_NO_CLI (see CMakeLists.txt).
 */

#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "CLIcore_standalone.h"
#include "fps.h"

/**
 * @brief Fill a human-readable type string for a binding's FPS type.
 */
static void fp_type_to_typestring(uint64_t fptype, char ts[20])
{
    switch (fptype)
    {
    case FPTYPE_INT32:
        strncpy(ts, "INT32", 19);
        break;
    case FPTYPE_UINT32:
        strncpy(ts, "UINT32", 19);
        break;
    case FPTYPE_INT64:
        strncpy(ts, "INT64", 19);
        break;
    case FPTYPE_UINT64:
        strncpy(ts, "UINT64", 19);
        break;
    case FPTYPE_FLOAT32:
        strncpy(ts, "FLOAT32", 19);
        break;
    case FPTYPE_FLOAT64:
        strncpy(ts, "FLOAT64", 19);
        break;
    case FPTYPE_ONOFF:
        strncpy(ts, "ONOFF", 19);
        break;
    case FPTYPE_STREAMNAME:
        strncpy(ts, "STREAMNAME", 19);
        break;
    case FPTYPE_FILENAME:
        strncpy(ts, "FILENAME", 19);
        break;
    case FPTYPE_FITSFILENAME:
        strncpy(ts, "FITSFILE", 19);
        break;
    case FPTYPE_EXECFILENAME:
        strncpy(ts, "EXECFILE", 19);
        break;
    case FPTYPE_DIRNAME:
        strncpy(ts, "DIRNAME", 19);
        break;
    case FPTYPE_FPSNAME:
        strncpy(ts, "FPSNAME", 19);
        break;
    case FPTYPE_PROCESS:
        strncpy(ts, "PROCESS", 19);
        break;
    case FPTYPE_STRING:
    case FPTYPE_STRING_NOT_STREAM:
        strncpy(ts, "STRING", 19);
        break;
    case FPTYPE_PID:
        strncpy(ts, "PID", 19);
        break;
    case FPTYPE_TIMESPEC:
        strncpy(ts, "TIMESPEC", 19);
        break;
    default:
        break;
    }
    ts[19] = '\0';
}

/* ---- helper: fill val_str for V2 params ---- */
/**
 * @brief Fill a human-readable default-value string for a binding.
 */
static void fp_type_and_ptr_to_valuestring(uint64_t fptype, void *valueptr, char vs[64])
{
    switch (fptype)
    {
    case FPTYPE_INT32:
        snprintf(vs, 64, "%d", *(int32_t *) valueptr);
        break;
    case FPTYPE_UINT32:
        snprintf(vs, 64, "%u", *(uint32_t *) valueptr);
        break;
    case FPTYPE_INT64:
        snprintf(vs, 64, "%ld", *(int64_t *) valueptr);
        break;
    case FPTYPE_UINT64:
        snprintf(vs, 64, "%lu", *(uint64_t *) valueptr);
        break;
    case FPTYPE_FLOAT32:
        snprintf(vs, 64, "%f", *(float *) valueptr);
        break;
    case FPTYPE_FLOAT64:
        snprintf(vs, 64, "%f", *(double *) valueptr);
        break;
    case FPTYPE_ONOFF:
        snprintf(vs, 64, "%s", (*(int32_t *) valueptr) ? "ON" : "OFF");
        break;
    case FPTYPE_PID:
        snprintf(vs, 64, "%d", (int) *(pid_t *) valueptr);
        break;
    case FPTYPE_TIMESPEC:
        snprintf(vs, 64, "%ld.%09ld", ((struct timespec *) valueptr)->tv_sec,
                 ((struct timespec *) valueptr)->tv_nsec);
        break;
    /* all string-like types (covers FPTYPE_IS_STRING members) */
    case FPTYPE_STRING:
    case FPTYPE_STRING_NOT_STREAM:
    case FPTYPE_STREAMNAME:
    case FPTYPE_FILENAME:
    case FPTYPE_FITSFILENAME:
    case FPTYPE_EXECFILENAME:
    case FPTYPE_DIRNAME:
    case FPTYPE_FPSNAME:
    case FPTYPE_PROCESS:
        strncpy(vs, (char *) valueptr, 63);
        vs[63] = '\0';
        break;
    default:
        break;
    }
}

/**
 * @brief Column widths for the "-h" parameters table.
 */
struct HELPER_PRETTYPRINT
{
    int col_kw_w;
    int col_tp_w;
    int col_df_w;
    int show_help_color;
    int CLIargcnt;
};

/**
 * @brief Measure column widths for 6-arg binding.
 */
static void X_HELP_MEASURE_V2(FPS_CLI_BINDING *binding, struct HELPER_PRETTYPRINT *pp)
{
    const char *_kp =
        (binding->fpskeyword[0] == '.') ? &binding->fpskeyword[1] : binding->fpskeyword;
    int _kl = (int) strlen(_kp);
    if (_kl > pp->col_kw_w)
    {
        pp->col_kw_w = _kl;
    }
    char _ts[20] = "???";
    fp_type_to_typestring(binding->type, _ts);
    int _tl = (int) strlen(_ts);
    if (_tl > pp->col_tp_w)
    {
        pp->col_tp_w = _tl;
    }
    char _vs[64] = "";
    fp_type_and_ptr_to_valuestring(binding->type, binding->ptr, _vs);
    int _vl = (int) strlen(_vs);
    if (_vl > pp->col_df_w)
    {
        pp->col_df_w = _vl;
    }
}

static void X_HELP_MEASURE_V2_LOOP(FPS_CLI_BINDING           *bindings,
                                   int                        nb_bindings,
                                   struct HELPER_PRETTYPRINT *pp)
{
    for (int k = 0; k < nb_bindings; ++k)
    {
        X_HELP_MEASURE_V2(&bindings[k], pp);
    }
}

/**
 * @brief Help-print for the 6-arg binding format.
 *
 * Uses col_kw_w, col_tp_w, col_df_w (set by
 * X_HELP_MEASURE_V2) for dynamic column widths.
 */
static void X_HELP_PRINT_V2(FPS_CLI_BINDING *binding, struct HELPER_PRETTYPRINT *pp)
{
    char        cli_idx_str[8];
    char        val_str[64]  = "";
    char        type_str[20] = "???";
    const char *kw           = binding->fpskeyword;
    const char *disp_kw      = (kw[0] == '.') ? &kw[1] : kw;
    if (binding->is_primary)
    {
        snprintf(cli_idx_str, sizeof(cli_idx_str), "%3d", pp->CLIargcnt);
    }
    else
    {
        strncpy(cli_idx_str, " - ", sizeof(cli_idx_str) - 1);
    }
    fp_type_to_typestring(binding->type, type_str);
    fp_type_and_ptr_to_valuestring(binding->type, binding->ptr, val_str);
    const char *_trig_tag = "";
    if ((binding->fpflag) & FPFLAG_TRIGGER_STREAM)
    {
        _trig_tag = pp->show_help_color ? " \033[48;5;23m\033[38;2;80;220;220m [TRIGGER] \033[0m"
                                        : " [TRIGGER]";
    }
    if (pp->show_help_color)
    {
        const char *_clr = binding->is_primary ? COLORPRIMARY : COLORARGnotCLI;
        printf("  %s %s%-*s%s"
               " %-*s %-*s %s%s\n",
               cli_idx_str, _clr, pp->col_kw_w, disp_kw, COLORRESET, pp->col_tp_w, type_str,
               pp->col_df_w, val_str, binding->descr, _trig_tag);
    }
    else
    {
        printf("  %s %-*s %-*s"
               " %-*s %s%s\n",
               cli_idx_str, pp->col_kw_w, disp_kw, pp->col_tp_w, type_str, pp->col_df_w, val_str,
               binding->descr, _trig_tag);
    }
    if (binding->is_primary)
    {
        pp->CLIargcnt++;
    }
}

static void X_HELP_PRINT_V2_LOOP(FPS_CLI_BINDING            bindings[],
                                 int                        nb_bindings,
                                 struct HELPER_PRETTYPRINT *pp)
{
    for (int k = 0; k < nb_bindings; ++k)
    {
        X_HELP_PRINT_V2(&bindings[k], pp);
    }
}


/* ----------------------------------------------------------------
 * Helpers factoring out the "connect, sync CLI args, apply -loops/
 * -loopd override, disconnect" pattern repeated across fpsinit,
 * exec, runstart and their -tmux dispatch counterparts below.
 * ---------------------------------------------------------------- */

/** @brief Sync CLI args into an already-connected FPS, then apply -loops/-loopd. */
static void fps_apply_cli_sync_and_loop_overrides(FPS             *fps,
                                                  CLICMDARGDEF    *farg,
                                                  FPS_CLI_BINDING *bindings,
                                                  int              nb_b,
                                                  int              use_loop,
                                                  double           loop_delay)
{
    fps_process_cli_and_sync(fps, farg, bindings, nb_b);
    if (use_loop == 1)
    {
        fps_loop_override_trigger(fps, bindings, nb_b);
    }
    else if (use_loop == 2)
    {
        fps_loop_override_delay(fps, loop_delay);
    }
}

/** @brief Connect to fps_name, sync CLI args + apply loop overrides, disconnect. No-op if connect fails. */
static void fps_sync_and_override_by_name(const char      *fps_name,
                                          CLICMDARGDEF    *farg,
                                          FPS_CLI_BINDING *bindings,
                                          int              nb_b,
                                          int              use_loop,
                                          double           loop_delay)
{
    FPS fps;
    if (fps_connect(fps_name, &fps, FPSCONNECT_SIMPLE) != -1)
    {
        fps_apply_cli_sync_and_loop_overrides(&fps, farg, bindings, nb_b, use_loop, loop_delay);
        fps_disconnect(&fps);
    }
}

/** @brief Append " -procinfo"/" -loops"/" -loopd SEC" to buf, mirroring the parsed CLI flags. */
static void fps_append_loop_flags(char  *buf,
                                  size_t bufsize,
                                  int    use_procinfo,
                                  int    use_loop,
                                  double loop_delay)
{
    if (use_procinfo)
    {
        strncat(buf, " -procinfo", bufsize - strlen(buf) - 1);
    }
    if (use_loop == 1)
    {
        strncat(buf, " -loops", bufsize - strlen(buf) - 1);
    }
    else if (use_loop == 2)
    {
        char ld[64];
        snprintf(ld, sizeof(ld), " -loopd %.6f", loop_delay);
        strncat(buf, ld, bufsize - strlen(buf) - 1);
    }
}

/** @brief Append " fpsname:command" (custom name) or " command" (default name) to buf. */
static void fps_append_dispatch_target(char       *buf,
                                       size_t      bufsize,
                                       const char *fps_name,
                                       const char *default_name,
                                       const char *command)
{
    size_t l = strlen(buf);
    if (strcmp(fps_name, default_name) != 0)
    {
        snprintf(buf + l, bufsize - l, " %s:%s", fps_name, command);
    }
    else
    {
        snprintf(buf + l, bufsize - l, " %s", command);
    }
}

/** @brief For 'exec': create the FPS if missing, reuse it otherwise (printing NEW/REUSE). */
static void fps_exec_autoinit(const char      *fps_name,
                              FPS_APP_INFO    *app_info,
                              FPS_CLI_BINDING *bindings,
                              int              nb_b,
                              int              use_procinfo)
{
    if (fps_name[0] == '_')
    {
        return; // local (non-shared-memory) FPS names are never auto-initialized
    }
    FPS fc_;
    if (fps_connect(fps_name, &fc_, FPSCONNECT_SIMPLE) == -1)
    {
        printf("FPS " COLORCOMMAND "%s" COLORRESET " exec -> \033[33mNEW\033[0m\n", fps_name);
        fps_generic_init(fps_name, app_info, bindings, nb_b, use_procinfo);
    }
    else
    {
        printf("FPS " COLORCOMMAND "%s" COLORRESET " exec -> " COLORCOMMAND "REUSE" COLORRESET "\n",
               fps_name);
        fps_disconnect(&fc_);
    }
}

int fpsexec_main_function(int              argc,
                          char            *argv[],
                          FPS_APP_INFO    *APP_INFO,
                          FPS_CLI_BINDING  bindings[],
                          int              nb_bindings,
                          CLICMDARGDEF     farg[],
                          fps_compute_fn   COMPUTE_FN,
                          fps_confcheck_fn CONFCHECK_FN)
{
    /* Pre-getopt scan: -h1 and -h2 must be handled before anything else */
    /* so -h1 is never split into -h + 1. */
    for (int _i = 1; _i < argc; _i++)
    {
        if (strcmp(argv[_i], "-h1") == 0 || strcmp(argv[_i], "--help-oneline") == 0)
        {
            printf("%s\n", APP_INFO->description);
            return 0;
        }
        if (strcmp(argv[_i], "-h2") == 0 || strcmp(argv[_i], "--help-description") == 0)
        {
            const char *_dl = APP_INFO->description_long;
            printf("%s\n", _dl ? _dl : APP_INFO->description);
            return 0;
        }
    }
    milk_data_init();
    extern void milkfps_set_image_array(IMAGE * imarray, long nb_max);
    milkfps_set_image_array(milk_data.image, milk_data.NB_MAX_IMAGE);
    fps_cli_set_standalone_args(argc, argv);
    char fps_name[STRINGMAXLEN_FPS_NAME] = "";
    strncpy(fps_name, APP_INFO->fps_name, STRINGMAXLEN_FPS_NAME - 1);
    char arg_fps_name[STRINGMAXLEN_FPS_NAME] = "";
    int  use_tmux                            = 0;
    int  use_procinfo                        = 0;
    if (getenv("MILK_FPSPROCINFO") != NULL)
    {
        use_procinfo = 1;
    }
    int    use_loop    = 0;
    double loop_delay  = -1.0;
    int    show_help   = 0;
    int    mh_color    = 1;
    char  *command     = NULL;
    char  *keywords    = NULL;
    char  *description = NULL;
    char  *colon_pos   = NULL;
    //FPS_CLI_BINDING bindings[] = bindings;
    //int             nb_bindings   = sizeof(bindings) / sizeof(FPS_CLI_BINDING);
    //CLICMDARGDEF    farg[]        = farg;
    for (int ii = 1; ii < argc; ii++)
    {
        if (strcmp(argv[ii], "-h") == 0 || strcmp(argv[ii], "--help") == 0)
        {
            show_help = 1;
        }
        else if (strcmp(argv[ii], "-hm") == 0 || strcmp(argv[ii], "--help-mono") == 0)
        {
            show_help = 1;
            mh_color  = 0;
        }
        else if (strcmp(argv[ii], "-tmux") == 0)
        {
            use_tmux = 1;
        }
        else if (strcmp(argv[ii], "-procinfo") == 0 || strcmp(argv[ii], "--procinfo") == 0)
        {
            use_procinfo = 1;
        }
        else if (strcmp(argv[ii], "-loops") == 0 || strcmp(argv[ii], "--loops") == 0)
        {
            use_loop     = 1;
            use_procinfo = 1;
        }
        else if (strcmp(argv[ii], "-loopd") == 0 || strcmp(argv[ii], "--loopd") == 0)
        {
            use_loop     = 2;
            use_procinfo = 1;
            if (ii + 1 < argc)
            {
                char c0 = argv[ii + 1][0];
                if ((c0 >= '0' && c0 <= '9') || c0 == '.')
                {
                    loop_delay = atof(argv[++ii]);
                }
                else
                {
                    loop_delay = 0.0;
                }
            }
            else
            {
                loop_delay = 0.0;
            }
        }
        else if ((strcmp(argv[ii], "-k") == 0 || strcmp(argv[ii], "--keywords") == 0) &&
                 ii + 1 < argc)
        {
            keywords = argv[++ii];
        }
        else if ((strcmp(argv[ii], "-d") == 0 || strcmp(argv[ii], "--description") == 0) &&
                 ii + 1 < argc)
        {
            description = argv[++ii];
        }
        else if ((strcmp(argv[ii], "-n") == 0 || strcmp(argv[ii], "--name") == 0) && ii + 1 < argc)
        {
            strncpy(arg_fps_name, argv[++ii], STRINGMAXLEN_FPS_NAME - 1);
        }
        else if (command == NULL)
        {
            command = argv[ii];
            if ((colon_pos = strchr(command, ':')) != NULL)
            {
                *colon_pos = '\0';
                strncpy(arg_fps_name, command, STRINGMAXLEN_FPS_NAME - 1); // copies to colon_pos
                *colon_pos =
                    ':'; // must restore the colon otherwise argc/argv mismatch are reparse in fps_process_cli_and_sync
                command = colon_pos + 1;
            }
        }
    }
    if (command == NULL)
    {
        command = (char *) "run"; /* cast for C++ compilation compatibility */
    }
    else
    {
        if (strcmp(command, "fpsinit") != 0 && strcmp(command, "fps") != 0 &&
            strcmp(command, "fpslist") != 0 && strcmp(command, "confstart") != 0 &&
            strcmp(command, "confstep") != 0 && strcmp(command, "confstop") != 0 &&
            strcmp(command, "runstart") != 0 && strcmp(command, "runstop") != 0 &&
            strcmp(command, "exec") != 0 && strcmp(command, "set") != 0 &&
            strcmp(command, "run") != 0)
        {
            fprintf(stderr,
                    MH_ERR "Error:" MH_RST " '%s' is not a valid command. Run with -h for help.\n",
                    command);
            return 1;
        }
    }
    if (strlen(arg_fps_name) > 0)
    {
        strncpy(fps_name, arg_fps_name, STRINGMAXLEN_FPS_NAME - 1);
    }
    if (show_help || (argc < 2))
    {
        /* TTY auto-detect for -h */
        if (show_help && mh_color && !isatty(STDOUT_FILENO))
        {
            mh_color = 0;
        }
        /* ---- BANNER ---- */
        milk_help_banner(argv[0], APP_INFO->description, mh_color);
        /* ---- USAGE ---- */
        milk_help_section("Usage", mh_color);
        printf("  %s %s %s%s\n\n", argv[0], MH(MH_OPT, "[options]"), MH(MH_OPT, "[fpsname:]"),
               MH(MH_CMD, "<command>"));
        /* ---- DESCRIPTION ---- */
        {
            const char *_dl   = APP_INFO->description_long;
            const char *_desc = _dl ? _dl : APP_INFO->description;
            milk_help_section("Description", mh_color);
            printf("  %s\n\n", _desc);
        }
        /* ---- FPS NOTE ---- */
        if (mh_color)
        {
            printf("  " MH_NOTE "This is a fpsexec command. Command parameters map"
                   " to FPS parameters." MH_RST "\n  Run " MH_CMD "milk-fpsexec-help" MH_RST
                   " for detailed FPS framework info.\n\n");
        }
        else
        {
            printf("  NOTE: This is a fpsexec command. Command parameters map to FPS parameters.\n"
                   "  Run milk-fpsexec-help for detailed FPS framework info.\n\n");
        }
        /* ---- OPTIONS ---- */
        milk_help_section("Options", mh_color);
        printf("  " MH_PAD_FMT "    Show this help\n", MH_PAD(MH_OPT, "-h, --help", 15));
        printf("  " MH_PAD_FMT "    One-line description\n", MH_PAD(MH_OPT, "-h1", 15));
        printf("  " MH_PAD_FMT "    Verbose description\n", MH_PAD(MH_OPT, "-h2", 15));
        printf("  " MH_PAD_FMT "    Monochrome help\n", MH_PAD(MH_OPT, "-hm", 15));
        printf("  " MH_PAD_FMT "    Run inside tmux session\n", MH_PAD(MH_OPT, "-tmux", 15));
        printf("  " MH_PAD_FMT "    Enable processinfo\n", MH_PAD(MH_OPT, "-procinfo", 15));
        printf("  " MH_PAD_FMT "    Infinite loop, semaphore trigger\n",
               MH_PAD(MH_OPT, "-loops", 15));
        printf("  %s %s       Infinite loop, delay trigger\n", MH(MH_OPT, "-loopd"),
               MH(MH_ARG, "SEC"));
        printf("  %s %s        Set FPS instance name\n\n", MH(MH_OPT, "-n"), MH(MH_ARG, "NAME"));
        /* ---- COMMANDS ---- */
        milk_help_section("Commands", mh_color);
        printf("  " MH_PAD_FMT "          Create the FPS\n", MH_PAD(MH_CMD, "fpsinit", 15));
        printf("  " MH_PAD_FMT "          Print FPS content\n", MH_PAD(MH_CMD, "fps", 15));
        printf("  " MH_PAD_FMT "          List matching FPS instances\n",
               MH_PAD(MH_CMD, "fpslist", 15));
        printf("  " MH_PAD_FMT "          Configuration loop\n", MH_PAD(MH_CMD, "confstart", 15));
        printf("  " MH_PAD_FMT "          Single config step\n", MH_PAD(MH_CMD, "confstep", 15));
        printf("  " MH_PAD_FMT "          Stop config loop\n", MH_PAD(MH_CMD, "confstop", 15));
        printf("  " MH_PAD_FMT "          Main processing loop\n", MH_PAD(MH_CMD, "runstart", 15));
        printf("  " MH_PAD_FMT "          Stop processing loop\n", MH_PAD(MH_CMD, "runstop", 15));
        printf("  %s %s      Set positional args (. to skip)\n", MH(MH_CMD, "set"),
               MH(MH_OPT, "[args]"));
        printf("  %s %s     Auto-init + set"
               " args + run\n\n",
               MH(MH_CMD, "exec"), MH(MH_OPT, "[args]"));
        /* ---- PARAMETERS ---- */
        struct HELPER_PRETTYPRINT pp = {
            .col_kw_w = 7, .col_tp_w = 4, .col_df_w = 7, .show_help_color = mh_color, .CLIargcnt = 0
        };
        //PARAMS_MACRO(X_HELP_MEASURE_V2)

        X_HELP_MEASURE_V2_LOOP(bindings, nb_bindings, &pp);

        milk_help_section("Parameters", mh_color);
        printf("  %-3s %-*s %-*s %-*s %s\n", "Idx", pp.col_kw_w, "Keyword", pp.col_tp_w, "Type",
               pp.col_df_w, "Default", "Description");
        printf("  %-3s %-*s %-*s %-*s %s\n", "---", pp.col_kw_w, "-------", pp.col_tp_w, "----",
               pp.col_df_w, "-------", "-----------");

        //PARAMS_MACRO(X_HELP_PRINT_V2)
        X_HELP_PRINT_V2_LOOP(bindings, nb_bindings, &pp);
        printf("\n");
        /* ---- SEE ALSO ---- */
        {
            const char *_sa[] = { "milk-fpsexec-help:print detailed FPS framework info",
                                  "milk-fpsCTRL:launch the FPS dashboard TUI",
                                  "milk-fpsexec-list:list installed fpsexec commands" };
            milk_help_see_also(_sa, 3, mh_color);
        }
        printf("\n");
        return 0;
    }
    if (command == NULL)
    {
        fprintf(stderr, "Error: Missing command argument.\n");
        return 1;
    }
    if (strcmp(command, "exec") != 0)
    {
        printf("FPS " COLORCOMMAND "%s" COLORRESET " %s\n", fps_name, command);
    }
    if (strcmp(command, "fps") == 0)
    {
        FPS fps;
        if (fps_connect(fps_name, &fps, FPSCONNECT_SIMPLE) == -1)
        {
            fprintf(stderr, "Error: cannot connect to FPS '%s'.\n", fps_name);
            return 1;
        }
        function_parameter_print_info(&fps, 0, 0);
        fps_disconnect(&fps);
        return 0;
    }
    else if (strcmp(command, "fpslist") == 0)
    {
        FPS *fpsarray = (FPS *) calloc(NB_FPS_MAX, sizeof(FPS));
        if (fpsarray == NULL)
        {
            return 1;
        }
        for (int ii = 0; ii < NB_FPS_MAX; ii++)
        {
            fpsarray[ii].SMfd = -1;
        }
        KEYWORD_TREE_NODE *keywnode =
            (KEYWORD_TREE_NODE *) calloc(NB_KEYWNODE_MAX, sizeof(KEYWORD_TREE_NODE));
        if (keywnode == NULL)
        {
            free(fpsarray);
            return 1;
        }
        int  NBkwn = 0, NBfps = 0;
        long NBpindex = 0;
        functionparameter_scan_fps(0, (char *) "_ALL", /* cast for C++ compilation compatibily */
                                   fpsarray, keywnode, &NBkwn, &NBfps, &NBpindex, 0);
        if (NBfps > 0)
        {
            char *eb = strrchr(argv[0], '/');
            if (eb)
            {
                eb++;
            }
            else
            {
                eb = argv[0];
            }
            int found = 0;
            for (int ii = 0; ii < NBfps; ii++)
            {
                char *fb = strrchr(fpsarray[ii].md->execfullpath, '/');
                if (fb)
                {
                    fb++;
                }
                else
                {
                    fb = fpsarray[ii].md->execfullpath;
                }
                if (strcmp(eb, fb) == 0)
                {
                    if (!found)
                    {
                        printf("%-30s %-10s %s\n", "FPS Name", "Status", "Description");
                        printf("----------------------------------------\n");
                        found = 1;
                    }
                    char ss[32] = "UNKNOWN";
                    if (fpsarray[ii].md->status & FUNCTION_PARAMETER_STRUCT_STATUS_CONF)
                    {
                        strncpy(ss, "CONF", sizeof(ss) - 1);
                    }
                    else if (fpsarray[ii].md->status & FUNCTION_PARAMETER_STRUCT_STATUS_RUN)
                    {
                        strncpy(ss, "RUN", sizeof(ss) - 1);
                    }
                    printf("%-30s %-10s %s\n", fpsarray[ii].md->name, ss,
                           fpsarray[ii].md->description);
                }
                fps_disconnect(&fpsarray[ii]);
            }
            if (!found)
            {
                printf("No matching FPS for '%s'.\n", eb);
            }
        }
        else
        {
            printf("No FPS found.\n");
        }
        free(keywnode);
        free(fpsarray);
        return 0;
    }
    if (use_tmux)
    {
        char path[PATH_MAX];
        if (functionparameter_FPS_get_executable_path(path, sizeof(path)) == NULL)
        {
            if (realpath(argv[0], path) == NULL)
            {
                strncpy(path, argv[0], 1023);
            }
        }
        functionparameter_FPS_tmux_standalone_setup(fps_name);

        int is_exec = (strcmp(command, "exec") == 0);
        if (is_exec || strcmp(command, "runstart") == 0 || strcmp(command, "run") == 0)
        {
            /* exec always dispatches as "runstart" (auto-init, then run);
             * "run" isn't a recognized tmux window, so it falls through below. */
            const char *dispatch_cmd = is_exec ? "runstart" : command;
            if (is_exec)
            {
                fps_exec_autoinit(fps_name, APP_INFO, bindings, nb_bindings, use_procinfo);
            }
            /* Persist the loop config in FPS shared memory now (works even if the
             * tmux child doesn't re-parse -loops/-loopd), and also forward the
             * flags so the child re-applies them for consistency. */
            fps_sync_and_override_by_name(fps_name, farg, bindings, nb_bindings, use_loop,
                                          loop_delay);

            char dispatch_arg[512] = "";
            fps_append_loop_flags(dispatch_arg, sizeof(dispatch_arg), use_procinfo, use_loop,
                                  loop_delay);
            fps_append_dispatch_target(dispatch_arg, sizeof(dispatch_arg), fps_name,
                                       APP_INFO->fps_name, dispatch_cmd);
            if (functionparameter_FPS_tmux_send_dispatch(fps_name, dispatch_cmd, path,
                                                         dispatch_arg) == 0)
            {
                return 0;
            }
        }
        else
        {
            char name_arg[256] = "";
            fps_append_dispatch_target(name_arg, sizeof(name_arg), fps_name, APP_INFO->fps_name,
                                       command);
            if (functionparameter_FPS_tmux_send_dispatch(fps_name, command, path, name_arg) == 0)
            {
                return 0;
            }
        }
    }
    if (strcmp(command, "fpsinit") == 0)
    {
        int rc_ = fps_generic_init(fps_name, APP_INFO, bindings, nb_bindings, use_procinfo);
        if (use_loop == 1 && !fps_check_has_trigger_binding(bindings, nb_bindings))
        {
            fprintf(stderr, "\033[1;33mWARNING"
                            "\033[0m [-loops] No"
                            " trigger stream"
                            " binding found \xe2\x80\x94"
                            " semaphore trigger"
                            " will not be"
                            " configured.\n"
                            "  Flag a stream"
                            " parameter with"
                            " FPFLAG_TRIGGER"
                            "_STREAM.\n");
        }
        /* Sync CLI args into FPS before applying loop overrides. */
        if (use_loop && rc_ == 0)
        {
            fps_sync_and_override_by_name(fps_name, farg, bindings, nb_bindings, use_loop,
                                          loop_delay);
        }
        return rc_;
    }
    else if (strcmp(command, "confstart") == 0)
    {
        return fps_generic_conf_cb(fps_name, 1, CONFCHECK_FN);
    }
    else if (strcmp(command, "confstep") == 0)
    {
        return fps_generic_conf_cb(fps_name, 0, CONFCHECK_FN);
    }
    else if (strcmp(command, "confstop") == 0)
    {
        return fps_generic_confstop(fps_name);
    }
    else if (strcmp(command, "set") == 0)
    {
        FPS fps;
        if (fps_connect(fps_name, &fps, FPSCONNECT_SIMPLE) == -1)
        {
            fprintf(stderr,
                    "Error: FPS '%s' not found."
                    " Run fpsinit first.\n",
                    fps_name);
            return 1;
        }
        fps_process_cli_and_sync(&fps, farg, bindings, nb_bindings);
        fps_disconnect(&fps);
        printf("FPS " COLORCOMMAND "%s" COLORRESET " set done\n", fps_name);
        return 0;
    }
    else if (strcmp(command, "exec") == 0 || strcmp(command, "runstart") == 0 ||
             strcmp(command, "run") == 0)
    {
        if (strcmp(command, "exec") == 0)
        {
            fps_exec_autoinit(fps_name, APP_INFO, bindings, nb_bindings, use_procinfo);
        }
        fps_sync_and_override_by_name(fps_name, farg, bindings, nb_bindings, use_loop, loop_delay);
        return fps_generic_run(fps_name, APP_INFO, farg, bindings, nb_bindings, COMPUTE_FN);
    }
    else if (strcmp(command, "runstop") == 0)
    {
        return fps_generic_runstop(fps_name);
    }
    fprintf(stderr, "Invalid command: %s\n", command);
    return 1;
}
