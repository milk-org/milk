#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <libgen.h>
#include <malloc.h>

#include <fitsio.h>
#ifdef USE_READLINE
#include <readline/readline.h>
#endif
#include <regex.h>

#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <sys/ioctl.h>
#include "CLIcore.h"
#include "ImageStreamIO/ImageStreamIO_config.h" // For IMAGESTRUCT_VERSION

#include "fps.h"
#include "fps_connect.h"


#define C_RST    "\033[0m"
#define C_TITLE  "\033[1;36m"
#define C_HDR    "\033[1;35m"
#define C_CMD    "\033[32m"
#define C_BOLD   "\033[1m"
#define C_NOTE   "\033[33m"
#define C_ERR    "\033[1;31m"

// Compatibility aliases
#ifndef COLORRESET
#define COLORRESET     C_RST
#endif

#ifndef COLORCMD
#define COLORCMD       C_CMD
#endif

#ifndef COLORINFO
#define COLORINFO      "\033[32m" // green
#endif

#ifndef COLORARGCLI
#define COLORARGCLI    "\033[36m" // argument part of CLI call: cyan
#endif

#ifndef COLORARGnotCLI
#define COLORARGnotCLI "\033[35m" // argument not part of CLI call: yellow
#endif


int help_format_mode = 0;

/**
 * @brief Print system and build information.
 *
 * Displays PID, package version, compiler info,
 * precision settings, memory usage, number of
 * images, loaded modules, environment directories,
 * and malloc statistics. Used by the 'soloinfo'
 * and 'dinfo' CLI commands.
 */
errno_t printInfo()
{
    float f1;
    printf("\n");
    printf("  PID = %d\n", CLIPID);

    printf("--------------- GENERAL ----------------------\n");
    printf("%s  %s\n", dcpkgname, dcpkgver);
    printf("IMAGESTRUCT_VERSION %s\n", IMAGESTRUCT_VERSION);
    printf("%s BUILT   %s %s\n", __FILE__, __DATE__, __TIME__);
    printf("\n");
    printf("--------------- SETTINGS ---------------------\n");
    printf("procinfo status = %d\n", dcprocinfo);

    if(dcprecision == 0)
    {
        printf("Default precision upon startup : float\n");
    }
    if(dcprecision == 1)
    {
        printf("Default precision upon startup : double\n");
    }
    printf("sizeof(struct timespec)        = %4ld bit\n",
           sizeof(struct timespec) * 8);
    printf("sizeof(pid_t)                  = %4ld bit\n", sizeof(pid_t) * 8);
    printf("sizeof(short int)              = %4ld bit\n",
           sizeof(short int) * 8);
    printf("sizeof(int)                    = %4ld bit\n", sizeof(int) * 8);
    printf("sizeof(long)                   = %4ld bit\n", sizeof(long) * 8);
    printf("sizeof(long long)              = %4ld bit\n",
           sizeof(long long) * 8);
    printf("sizeof(int_fast8_t)            = %4ld bit\n",
           sizeof(int_fast8_t) * 8);
    printf("sizeof(int_fast16_t)           = %4ld bit\n",
           sizeof(int_fast16_t) * 8);
    printf("sizeof(int_fast32_t)           = %4ld bit\n",
           sizeof(int_fast32_t) * 8);
    printf("sizeof(int_fast64_t)           = %4ld bit\n",
           sizeof(int_fast64_t) * 8);
    printf("sizeof(uint_fast8_t)           = %4ld bit\n",
           sizeof(uint_fast8_t) * 8);
    printf("sizeof(uint_fast16_t)          = %4ld bit\n",
           sizeof(uint_fast16_t) * 8);
    printf("sizeof(uint_fast32_t)          = %4ld bit\n",
           sizeof(uint_fast32_t) * 8);
    printf("sizeof(uint_fast64_t)          = %4ld bit\n",
           sizeof(uint_fast64_t) * 8);
    printf("sizeof(IMAGE_KEYWORD)          = %4ld bit\n",
           sizeof(IMAGE_KEYWORD) * 8);

    size_t offsetval  = 0;
    size_t offsetval0 = 0;

    printf(
        "sizeof(IMAGE_METADATA)         = %4ld bit  = %4zu byte "
        "------------------\n",
        sizeof(IMAGE_METADATA) * 8,
        sizeof(IMAGE_METADATA));

    offsetval = offsetof(IMAGE_METADATA, version);

    offsetval0 = offsetval;
    offsetval  = offsetof(IMAGE_METADATA, name);
    printf(
        "   version                     offset = %4zu bit  = %4zu byte     "
        "[%4zu byte]\n",
        8 * offsetval0,
        offsetval0,
        offsetval - offsetval0);

    offsetval0 = offsetval;
    offsetval  = offsetof(IMAGE_METADATA, naxis);
    printf(
        "   name                        offset = %4zu bit  = %4zu byte     "
        "[%4zu byte]\n",
        8 * offsetval0,
        offsetval0,
        offsetval - offsetval0);

    offsetval0 = offsetval;
    offsetval  = offsetof(IMAGE_METADATA, size);
    printf(
        "   naxis                       offset = %4zu bit  = %4zu byte     "
        "[%4zu byte]\n",
        8 * offsetval0,
        offsetval0,
        offsetval - offsetval0);

    offsetval0 = offsetval;
    offsetval  = offsetof(IMAGE_METADATA, nelement);
    printf(
        "   size                        offset = %4zu bit  = %4zu byte     "
        "[%4zu byte]\n",
        8 * offsetval0,
        offsetval0,
        offsetval - offsetval0);

    offsetval0 = offsetval;
    offsetval  = offsetof(IMAGE_METADATA, datatype);
    printf(
        "   nelement                    offset = %4zu bit  = %4zu byte     "
        "[%4zu byte]\n",
        8 * offsetval0,
        offsetval0,
        offsetval - offsetval0);

    offsetval0 = offsetval;
    offsetval  = offsetof(IMAGE_METADATA, imagetype);
    printf(
        "   datatype                    offset = %4zu bit  = %4zu byte     "
        "[%4zu byte]\n",
        8 * offsetval0,
        offsetval0,
        offsetval - offsetval0);

    offsetval0 = offsetval;
    offsetval  = offsetof(IMAGE_METADATA, creationtime);
    printf(
        "   imagetype                   offset = %4zu bit  = %4zu byte     "
        "[%4zu byte]\n",
        8 * offsetval0,
        offsetval0,
        offsetval - offsetval0);

    offsetval0 = offsetval;
    offsetval  = offsetof(IMAGE_METADATA, lastaccesstime);
    printf(
        "   creationtime                offset = %4zu bit  = %4zu byte     "
        "[%4zu byte]\n",
        8 * offsetval0,
        offsetval0,
        offsetval - offsetval0);

    offsetval0 = offsetval;
    offsetval  = offsetof(IMAGE_METADATA, atime);
    printf(
        "   lastaccesstime              offset = %4zu bit  = %4zu byte     "
        "[%4zu byte]\n",
        8 * offsetval0,
        offsetval0,
        offsetval - offsetval0);

    offsetval0 = offsetval;
    offsetval  = offsetof(IMAGE_METADATA, writetime);
    printf(
        "   atime                       offset = %4zu bit  = %4zu byte     "
        "[%4zu byte]\n",
        8 * offsetval0,
        offsetval0,
        offsetval - offsetval0);

    offsetval0 = offsetval;
    offsetval  = offsetof(IMAGE_METADATA, location);
    printf(
        "   writetime                   offset = %4zu bit  = %4zu byte     "
        "[%4zu byte]\n",
        8 * offsetval0,
        offsetval0,
        offsetval - offsetval0);

    offsetval0 = offsetval;
    offsetval  = offsetof(IMAGE_METADATA, location);
    printf(
        "   shared                      offset = %4zu bit  = %4zu byte     "
        "[%4zu byte]\n",
        8 * offsetval0,
        offsetval0,
        offsetval - offsetval0);

    offsetval0 = offsetval;
    offsetval  = offsetof(IMAGE_METADATA, status);
    printf(
        "   location                    offset = %4zu bit  = %4zu byte     "
        "[%4zu byte]\n",
        8 * offsetval0,
        offsetval0,
        offsetval - offsetval0);

    offsetval0 = offsetval;
    offsetval  = offsetof(IMAGE_METADATA, flag);
    printf(
        "   status                      offset = %4zu bit  = %4zu byte     "
        "[%4zu byte]\n",
        8 * offsetval0,
        offsetval0,
        offsetval - offsetval0);

    offsetval0 = offsetval;
    offsetval  = offsetof(IMAGE_METADATA, sem);
    printf(
        "   flag                        offset = %4zu bit  = %4zu byte     "
        "[%4zu byte]\n",
        8 * offsetval0,
        offsetval0,
        offsetval - offsetval0);

    offsetval0 = offsetval;
    offsetval  = offsetof(IMAGE_METADATA, sem);
    printf(
        "   logflag                     offset = %4zu bit  = %4zu byte     "
        "[%4zu byte]\n",
        8 * offsetval0,
        offsetval0,
        offsetval - offsetval0);

    offsetval0 = offsetval;
    offsetval  = offsetof(IMAGE_METADATA, cnt0);
    printf(
        "   sem                         offset = %4zu bit  = %4zu byte     "
        "[%4zu byte]\n",
        8 * offsetval0,
        offsetval0,
        offsetval - offsetval0);

    offsetval0 = offsetval;
    offsetval  = offsetof(IMAGE_METADATA, cnt1);
    printf(
        "   cnt0                        offset = %4zu bit  = %4zu byte     "
        "[%4zu byte]\n",
        8 * offsetval0,
        offsetval0,
        offsetval - offsetval0);

    offsetval0 = offsetval;
    offsetval  = offsetof(IMAGE_METADATA, cnt2);
    printf(
        "   cnt1                        offset = %4zu bit  = %4zu byte     "
        "[%4zu byte]\n",
        8 * offsetval0,
        offsetval0,
        offsetval - offsetval0);

    offsetval0 = offsetval;
    offsetval  = offsetof(IMAGE_METADATA, write);
    printf(
        "   cnt2                        offset = %4zu bit  = %4zu byte     "
        "[%4zu byte]\n",
        8 * offsetval0,
        offsetval0,
        offsetval - offsetval0);

    offsetval0 = offsetval;
    offsetval  = offsetof(IMAGE_METADATA, NBkw);
    printf(
        "   write                       offset = %4zu bit  = %4zu byte     "
        "[%4zu byte]\n",
        8 * offsetval0,
        offsetval,
        offsetval - offsetval0);

    offsetval0 = offsetval;
    offsetval  = offsetof(IMAGE_METADATA, cudaMemHandle);
    printf(
        "   NBkw                        offset = %4zu bit  = %4zu byte     "
        "[%4zu byte]\n",
        8 * offsetval0,
        offsetval0,
        offsetval - offsetval0);

    offsetval0 = offsetval;
    printf("   cudaMemHandle               offset = %4zu bit  = %4zu byte\n",
           8 * offsetval0,
           offsetval0);

    printf(
        "sizeof(IMAGE)                  offset = %4zu bit  = %4zu byte "
        "------------------\n",
        sizeof(IMAGE) * 8,
        sizeof(IMAGE));
    printf("   name                        offset = %4zu bit  = %4zu byte\n",
           8 * offsetof(IMAGE, name),
           offsetof(IMAGE, name));

    printf("   used                        offset = %4zu bit  = %4zu byte\n",
           8 * offsetof(IMAGE, used),
           offsetof(IMAGE, used));

    printf("   shmfd                       offset = %4zu bit  = %4zu byte\n",
           8 * offsetof(IMAGE, shmfd),
           offsetof(IMAGE, shmfd));

    printf("   memsize                     offset = %4zu bit  = %4zu byte\n",
           8 * offsetof(IMAGE, memsize),
           offsetof(IMAGE, memsize));

    printf("   semlog                      offset = %4zu bit  = %4zu byte\n",
           8 * offsetof(IMAGE, semlog),
           offsetof(IMAGE, semlog));

    printf("   md                          offset = %4zu bit  = %4zu byte\n",
           8 * offsetof(IMAGE, md),
           offsetof(IMAGE, md));

    printf("   atimearray                  offset = %4zu bit  = %4zu byte\n",
           8 * offsetof(IMAGE, atimearray),
           offsetof(IMAGE, atimearray));

    printf("   writetimearray              offset = %4zu bit  = %4zu byte\n",
           8 * offsetof(IMAGE, writetimearray),
           offsetof(IMAGE, writetimearray));

    printf("   flagarray                   offset = %4zu bit  = %4zu byte\n",
           8 * offsetof(IMAGE, flagarray),
           offsetof(IMAGE, flagarray));

    printf("   cntarray                    offset = %4zu bit  = %4zu byte\n",
           8 * offsetof(IMAGE, cntarray),
           offsetof(IMAGE, cntarray));

    printf("   array                       offset = %4zu bit  = %4zu byte\n",
           8 * offsetof(IMAGE, array),
           offsetof(IMAGE, array));

    printf("   semptr                      offset = %4zu bit  = %4zu byte\n",
           8 * offsetof(IMAGE, semptr),
           offsetof(IMAGE, semptr));

    printf("   kw                          offset = %4zu bit  = %4zu byte\n",
           8 * offsetof(IMAGE, kw),
           offsetof(IMAGE, kw));

    printf(
        "sizeof(IMAGE_KEYWORD)          offset = %4zu bit  = %4zu byte "
        "------------------\n",
        sizeof(IMAGE_KEYWORD) * 8,
        sizeof(IMAGE_KEYWORD));

    printf("   name                        offset = %4zu bit  = %4zu byte\n",
           8 * offsetof(IMAGE_KEYWORD, name),
           offsetof(IMAGE_KEYWORD, name));

    printf("   type                        offset = %4zu bit  = %4zu byte\n",
           8 * offsetof(IMAGE_KEYWORD, type),
           offsetof(IMAGE_KEYWORD, type));

    printf("   value                       offset = %4zu bit  = %4zu byte\n",
           8 * offsetof(IMAGE_KEYWORD, value),
           offsetof(IMAGE_KEYWORD, value));

    printf("   comment                     offset = %4zu bit  = %4zu byte\n",
           8 * offsetof(IMAGE_KEYWORD, comment),
           offsetof(IMAGE_KEYWORD, comment));

    printf("\n");
    printf("--------------- LIBRARIES --------------------\n");
#ifdef USE_READLINE
    printf("READLINE : version %x\n", RL_READLINE_VERSION);
#else
    printf("READLINE : disabled\n");
#endif
#ifdef _OPENMP
    printf("OPENMP   : Compiled by an OpenMP-compliant implementation.\n");
#endif
    printf("CFITSIO  : version %f\n", fits_get_version(&f1));
    printf("\n");

    printf("--------------- DIRECTORIES ------------------\n");
    printf("CONFIGDIR = %s\n", dcconfigdir);
    printf("SOURCEDIR = %s\n", dcsourcedir);
    printf("\n");

    printf("--------------- MALLOC INFO ------------------\n");
    malloc_stats();

    printf("\n");

    return RETURN_SUCCESS;
}


/**
 * @brief List all registered CLI commands.
 *
 * Prints a table of every command including its
 * index, keyword, module, description, and
 * example. Column widths adapt to the terminal
 * width.
 */
errno_t list_commands()
{
    int cols = 120; // default
#ifdef TIOCGWINSZ
    struct winsize ws;
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) >= 0) {
        if (ws.ws_col > 0) {
            cols = ws.ws_col;
        }
    }
#endif

    int  cmdinfoslen = 50;
    
    // adjust cmdinfoslen if terminal is very small
    int base_length = 4 + 3 + 24 + 2 + 16 + 2; // 51
    if (cols - base_length - 2 < cmdinfoslen) {
        cmdinfoslen = cols - base_length - 2;
        if (cmdinfoslen < 5) cmdinfoslen = 5;
    }

    char cmdinfoshort[cmdinfoslen];

    printf("================================================ LIST OF COMMANDS ================================================\n");
    for(unsigned int i = 0; i < data.NBcmd; i++)
    {
        strncpy(cmdinfoshort, data.cmd[i].info, cmdinfoslen - 1);
        cmdinfoshort[cmdinfoslen - 1] = '\0';
        
        int example_max_len = cols - (base_length + cmdinfoslen - 1 + 2);
        char example_short[256];
        if (example_max_len < 0) example_max_len = 0;
        if (example_max_len > 255) example_max_len = 255;
        
        strncpy(example_short, data.cmd[i].example, example_max_len);
        example_short[example_max_len] = '\0';

        printf("%4u > " COLORCMD " %-24s " COLORRESET " %-16s " COLORINFO " %-*s " COLORRESET " %s\n",
               i,
               data.cmd[i].key,
               data.cmd[i].module,
               cmdinfoslen - 1,
               cmdinfoshort,
               example_short);
    }

    return RETURN_SUCCESS;
}


/**
 * @brief List commands for a specific module.
 *
 * Shows module metadata (name, version, package)
 * followed by all commands belonging to @modulename.
 * Used by the 'mload?' and 'm?' CLI commands.
 *
 * @param modulename  Module name to filter by
 */
errno_t list_commands_module(
    const char *__restrict modulename
)
{
    int cols = 120; // default
#ifdef TIOCGWINSZ
    struct winsize ws;
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) >= 0) {
        if (ws.ws_col > 0) {
            cols = ws.ws_col;
        }
    }
#endif

    int  mOK         = 0;
    int  cmdinfoslen = 38;
    
    // adjust cmdinfoslen if terminal is very small
    int base_length = 3 + 24 + 2; // 29
    if (cols - base_length < cmdinfoslen) {
        cmdinfoslen = cols - base_length;
        if (cmdinfoslen < 5) cmdinfoslen = 5;
    }

    char cmdinfoshort[cmdinfoslen];

    int moduleindex = -1;
    for(int m = 0; m < data.NBmodule; m++)
    {
        if(strcmp(modulename, data.module[m].name) == 0)
        {
            moduleindex = m;
        }
    }
    if(moduleindex == -1)
    {
        printf("---- MODULE %s DOES NOT EXIST / NOT LOADED ---------\n",
               modulename);
    }
    else
    {
        printf("   name         %s\n", data.module[moduleindex].name);
        printf("   type         %d\n", data.module[moduleindex].type);
        printf("   short name   %s\n", data.module[moduleindex].shortname);
        printf("   package      %s\n", data.module[moduleindex].package);
        printf("   loadname     %s\n", data.module[moduleindex].loadname);
        printf("   sofilename   %s\n", data.module[moduleindex].sofilename);
        printf("   version      %d %d %d\n",
               data.module[moduleindex].versionmajor,
               data.module[moduleindex].versionminor,
               data.module[moduleindex].versionpatch);
        printf("   date         %s %s\n",
               data.module[moduleindex].datestring,
               data.module[moduleindex].timestring);
        printf("   info         %s\n", data.module[moduleindex].info);

        for(unsigned int i = 0; i < data.NBcmd; i++)
        {
            int cmdstrlen = 200;
            char cmpstring[cmdstrlen];
            snprintf(cmpstring, cmdstrlen, "%s", data.cmd[i].module);

            if(strcmp(modulename, cmpstring) == 0)
            {
                if(mOK == 0)
                {
                    printf("---- MODULE %s COMMANDS ---------\n", modulename);
                }

                strncpy(cmdinfoshort, data.cmd[i].info, cmdinfoslen - 1);
                cmdinfoshort[cmdinfoslen - 1] = '\0';
                
                printf(COLORCMD "   %-24s" COLORRESET COLORINFO
                       "  %-*s\n" COLORRESET,
                       data.cmd[i].key,
                       cmdinfoslen - 1,
                       cmdinfoshort);
                mOK = 1;
            }
        }

        if(mOK == 0)
        {
            if(strlen(modulename) > 0)
            {
                printf(
                    "---- MODULE %s DOES NOT HAVE COMMANDS "
                    "---------\n",
                    modulename);
            }
        }
    }

    return RETURN_SUCCESS;
}


/** @brief Construct command line (CLI) arguments help string
 *
 */
int CLIhelp_make_argstring(
    CLICMDARGDEF fpscliarg[],
    int nbarg,
    char *outargstring)
{
    char tmpstr[STRINGMAXLEN_CMD_SYNTAX];
    tmpstr[0] = '\0';

    int CLIargcnt = 0;
    for(int arg = 0; arg < nbarg; arg++)
    {
        if(fpscliarg[arg].fpflag & FPFLAG_PRIMARY_CLI_INPUT)
        {
            char typestring[100] = "?";

            switch(fpscliarg[arg].type)
            {
            case CLIARG_FLOAT32:
                snprintf(typestring,
                         sizeof(typestring),
                         "FLOAT32");
                break;

            case CLIARG_FLOAT64:
                snprintf(typestring,
                         sizeof(typestring),
                         "FLOAT64");
                break;

            case CLIARG_INT32:
                snprintf(typestring,
                         sizeof(typestring),
                         "INT32");
                break;

            case CLIARG_UINT32:
                snprintf(typestring,
                         sizeof(typestring),
                         "UINT32");
                break;

            case CLIARG_INT64:
                snprintf(typestring,
                         sizeof(typestring),
                         "INT64");
                break;

            case CLIARG_UINT64:
                snprintf(typestring,
                         sizeof(typestring),
                         "UINT64");
                break;

            case CLIARG_STR_NOT_IMG:
                snprintf(typestring,
                         sizeof(typestring),
                         "STRING");
                break;

            case CLIARG_IMG:
                snprintf(typestring,
                         sizeof(typestring),
                         "STREAMNAME");
                break;

            case CLIARG_STR:
                snprintf(typestring,
                         sizeof(typestring),
                         "STRING");
                break;

            case CLIARG_ONOFF:
                snprintf(typestring,
                         sizeof(typestring),
                         "ONOFF");
                break;

            case CLIARG_FILENAME:
                snprintf(typestring,
                         sizeof(typestring),
                         "FILENAME");
                break;

            case CLIARG_FITSFILENAME:
                snprintf(typestring,
                         sizeof(typestring),
                         "FITSFILENAME");
                break;

            case CLIARG_FPSNAME:
                snprintf(typestring,
                         sizeof(typestring),
                         "FPSNAME");
                break;
            }

            char tmpstr1[STRINGMAXLEN_CMD_SYNTAX];
            if(CLIargcnt == 0)
            {
                snprintf(tmpstr1,
                         STRINGMAXLEN_CMD_SYNTAX,
                         "<%s [%s]>",
                         fpscliarg[arg].descr,
                         typestring);
            }
            else
            {
                snprintf(tmpstr1,
                         STRINGMAXLEN_CMD_SYNTAX - 1,
                         " <%s [%s]>",
                         fpscliarg[arg].descr,
                         typestring);
            }

            // max number of chars we can write
            int n = STRINGMAXLEN_CMD_SYNTAX - strlen(tmpstr);
            if(n > (int) strlen(tmpstr1))
            {
                strcat(tmpstr, tmpstr1);
            }
            CLIargcnt++;
        }
    }
    strncpy(outargstring, tmpstr, STRINGMAXLEN_CMD_SYNTAX - 1);

    return strlen(outargstring);
}


/** @brief Assemble command line (CLI) example command string
 *
 */
int CLIhelp_make_cmdexamplestring(
    CLICMDARGDEF fpscliarg[],
    int nbarg,
    char *shortname,
    char *outcmdexstring)
{
    char tmpstr[STRINGMAXLEN_CMD_EXAMPLE];

    snprintf(tmpstr, STRINGMAXLEN_CMD_EXAMPLE, "%s", shortname);

    for(int arg = 0; arg < nbarg; arg++)
    {
        if(fpscliarg[arg].fpflag & FPFLAG_PRIMARY_CLI_INPUT)
        {
            char tmpstr1[STRINGMAXLEN_CMD_EXAMPLE];
            snprintf(tmpstr1,
                     STRINGMAXLEN_CMD_EXAMPLE - 1,
                     " %s",
                     fpscliarg[arg].example);

            // max number of chars we can write
            int n = STRINGMAXLEN_CMD_EXAMPLE - strlen(tmpstr);
            if(n > (int) strlen(tmpstr1))
            {
                strcat(tmpstr, tmpstr1);
            }
        }
    }
    strncpy(outcmdexstring, tmpstr, STRINGMAXLEN_CMD_EXAMPLE - 1);

    return strlen(outcmdexstring);
}

/**
 * @brief Print a 64-bit flag with ON/OFF indicator.
 *
 * Tests whether @testflag is set in @flags and
 * prints a colored ON or OFF label followed by
 * @flagdescription.
 *
 * @param flags           Combined flag word
 * @param testflag        Bit(s) to test
 * @param flagdescription Human-readable label
 * @return 1 if flag is set, 0 if not
 */
