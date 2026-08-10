// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    milk_help.h
 * @brief   Unified help message color palette
 *          and helper macros
 *
 * Provides a consistent look-and-feel for `-h`,
 * `-h1`, and `-hm` help output across all milk
 * and cacao executables.
 *
 * Engine tier -- no CLI dependency.
 *
 * Usage:
 * @code
 * #include "milk_help.h"
 *
 * int main(int argc, char *argv[])
 * {
 *     int action = milk_help_init(argc, argv,
 *         "create a shared-memory image stream");
 *     if (action == MH_ACTION_H1)
 *         return 0;
 *     int mh_color = (action == MH_ACTION_HELP);
 *     if (action == MH_ACTION_HELP ||
 *         action == MH_ACTION_MONO)
 *     {
 *         print_help(argv[0], mh_color);
 *         return 0;
 *     }
 *     // ... normal execution ...
 * }
 * @endcode
 */

#ifndef MILK_HELP_H
#define MILK_HELP_H

#include <stdio.h>
#include <string.h>
#include <unistd.h>

/* -------------------------------------------
 * ANSI Color Palette
 *
 * Semantic names map to a curated palette that
 * is readable on both dark and light terminals.
 * ------------------------------------------- */

/** @brief Reset all attributes */
#define MH_RST "\033[0m"
/** @brief Bold white */
#define MH_BOLD "\033[1m"
/** @brief Dim gray */
#define MH_DIM "\033[2m"

/** @brief Cyan bold -- program name, primary params */
#define MH_TITLE "\033[1;36m"
/** @brief Blue bold -- section headers */
#define MH_HDR "\033[1;34m"
/** @brief Green bold -- commands, executables */
#define MH_CMD "\033[1;32m"
/** @brief Yellow -- option flags */
#define MH_OPT "\033[33m"
/** @brief Magenta bold -- required arguments */
#define MH_ARG "\033[1;35m"
/** @brief Cyan -- default values */
#define MH_DFLT "\033[36m"
/** @brief Yellow bold -- notes, see-also intro */
#define MH_NOTE "\033[1;33m"
/** @brief Red bold -- error messages */
#define MH_ERR "\033[1;31m"

/* -------------------------------------------
 * Color Dispatch Macro
 *
 * MH(color, text) emits color+text+reset when
 * mh_color is true, or just text when false.
 *
 * Requires a local `int mh_color` in scope.
 * ------------------------------------------- */

/**
 * @brief Conditionally wrap text in color.
 *
 * @param c  One of MH_TITLE, MH_HDR, etc.
 * @param t  String literal to colorize.
 *
 * Expands to colored string when mh_color is
 * true, plain text otherwise.
 */
#define MH(c, t) (mh_color ? (c t MH_RST) : (t))

/**
 * @brief Format spec matching the MH_PAD() argument pack.
 */
#define MH_PAD_FMT "%s%-*s%s"

/**
 * @brief Colorized, width-padded text for use with MH_PAD_FMT.
 *
 * Unlike MH(), the field width is applied to the plain text
 * only, so ANSI escape codes are never counted toward it and
 * printf() alignment is preserved with color on or off.
 *
 * @param c  One of MH_TITLE, MH_HDR, etc.
 * @param t  String literal to colorize.
 * @param w  Field width (as passed to %-*s).
 *
 * Requires a local `int mh_color` in scope.
 */
#define MH_PAD(c, t, w) (mh_color ? (c) : ""), (w), (t), (mh_color ? MH_RST : "")

/* -------------------------------------------
 * Return values from milk_help_init()
 * ------------------------------------------- */

/** @brief No help flag found -- continue */
#define MH_ACTION_NONE 0
/** @brief -h1 was printed, caller should return 0 */
#define MH_ACTION_H1 1
/** @brief -h requested, use color (tty detected) */
#define MH_ACTION_HELP 2
/** @brief -hm requested, or -h but not a tty */
#define MH_ACTION_MONO 3
/** @brief -h2 was printed, caller should return 0 */
#define MH_ACTION_H2 4

/* -------------------------------------------
 * Initialization
 *
 * Scans argv for -h1, -h, -hm BEFORE getopt
 * runs, so -h1 is never split into -h + 1.
 * ------------------------------------------- */

/**
 * @brief Scan argv for help flags.
 *
 * @param argc             Argument count.
 * @param argv             Argument vector.
 * @param description      One-line description for -h1.
 * @param description_long Verbose description for -h2
 *                         (NULL falls back to description).
 * @return MH_ACTION_NONE, MH_ACTION_H1, MH_ACTION_H2,
 *         MH_ACTION_HELP, or MH_ACTION_MONO.
 *
 * -h1 and -h2 print immediately and return.
 * The caller should then `return 0`.
 */
static inline int milk_help_init(int         argc,
                                 char       *argv[],
                                 const char *description,
                                 const char *description_long)
{
    int want_help = 0;
    int want_mono = 0;

    for (int ii = 1; ii < argc; ii++)
    {
        if (strcmp(argv[ii], "-h1") == 0 || strcmp(argv[ii], "--help-oneline") == 0)
        {
            printf("%s\n", description);
            return MH_ACTION_H1;
        }

        if (strcmp(argv[ii], "-h2") == 0 || strcmp(argv[ii], "--help-description") == 0)
        {
            const char *desc = (description_long != NULL) ? description_long : description;
            printf("%s\n", desc);
            return MH_ACTION_H2;
        }

        if (strcmp(argv[ii], "-hm") == 0 || strcmp(argv[ii], "--help-mono") == 0)
        {
            want_help = 1;
            want_mono = 1;
        }

        if (strcmp(argv[ii], "-h") == 0 || strcmp(argv[ii], "--help") == 0)
        {
            want_help = 1;
        }
    }

    if (!want_help)
    {
        return MH_ACTION_NONE;
    }

    if (want_mono)
    {
        return MH_ACTION_MONO;
    }

    /* Auto-detect: suppress color if not a tty */
    if (isatty(STDOUT_FILENO))
    {
        return MH_ACTION_HELP;
    }

    return MH_ACTION_MONO;
}

/* -------------------------------------------
 * Section Printing Helpers
 * ------------------------------------------- */

/**
 * @brief Print the help banner line.
 *
 * Format: "<progname> -- <description>"
 *
 * @param progname    Executable name (argv[0] basename).
 * @param description One-line description.
 * @param color       Non-zero to emit ANSI color.
 */
static inline void milk_help_banner(const char *progname, const char *description, int color)
{
    /* Extract basename */
    const char *base = strrchr(progname, '/');
    if (base)
    {
        base++;
    }
    else
    {
        base = progname;
    }

    if (color)
    {
        printf("\n" MH_TITLE "%s" MH_RST " -- %s\n\n", base, description);
    }
    else
    {
        printf("\n%s -- %s\n\n", base, description);
    }
}

/**
 * @brief Print a section header.
 *
 * @param title  Section name (e.g., "Usage").
 * @param color  Non-zero to emit ANSI color.
 */
static inline void milk_help_section(const char *title, int color)
{
    if (color)
    {
        printf(MH_HDR "%s:" MH_RST "\n", title);
    }
    else
    {
        printf("%s:\n", title);
    }
}

/**
 * @brief Print a "See also" footer.
 *
 * @param cmds   Array of command name strings.
 * @param ncmds  Number of entries in cmds.
 * @param color  Non-zero to emit ANSI color.
 */
static inline void milk_help_see_also(const char *cmds[], int ncmds, int color)
{
    if (color)
    {
        printf(MH_HDR "See Also:" MH_RST "\n");
    }
    else
    {
        printf("See Also:\n");
    }

    for (int ii = 0; ii < ncmds; ii++)
    {
        const char *cmd   = cmds[ii];
        const char *colon = strchr(cmd, ':');
        if (colon != NULL)
        {
            int cmd_len = colon - cmd;
            if (color)
            {
                printf("  " MH_CMD "%-24.*s" MH_RST " -- %s\n", cmd_len, cmd, colon + 1);
            }
            else
            {
                printf("  %-24.*s -- %s\n", cmd_len, cmd, colon + 1);
            }
        }
        else
        {
            if (color)
            {
                printf("  " MH_CMD "%s" MH_RST "\n", cmd);
            }
            else
            {
                printf("  %s\n", cmd);
            }
        }
    }
    printf("\n");
}

#endif /* MILK_HELP_H */
