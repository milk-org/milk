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


errno_t list_commands_module(const char *__restrict modulename)
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
int CLIhelp_make_argstring(CLICMDARGDEF fpscliarg[],
                           int          nbarg,
                           char        *outargstring)
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
                strcpy(typestring, "FLOAT32");
                break;

            case CLIARG_FLOAT64:
                strcpy(typestring, "FLOAT64");
                break;

            case CLIARG_INT32:
                strcpy(typestring, "INT32");
                break;

            case CLIARG_UINT32:
                strcpy(typestring, "UINT32");
                break;

            case CLIARG_INT64:
                strcpy(typestring, "INT64");
                break;

            case CLIARG_UINT64:
                strcpy(typestring, "UINT64");
                break;

            case CLIARG_STR_NOT_IMG:
                strcpy(typestring, "STRING");
                break;

            case CLIARG_IMG:
                strcpy(typestring, "STREAMNAME");
                break;

            case CLIARG_STR:
                strcpy(typestring, "STRING");
                break;

            case CLIARG_ONOFF:
                strcpy(typestring, "ONOFF");
                break;

            case CLIARG_FILENAME:
                strcpy(typestring, "FILENAME");
                break;

            case CLIARG_FITSFILENAME:
                strcpy(typestring,
                       "FITSFILENAME");
                break;

            case CLIARG_FPSNAME:
                strcpy(typestring, "FPSNAME");
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
int CLIhelp_make_cmdexamplestring(CLICMDARGDEF fpscliarg[],
                                  int          nbarg,
                                  char        *shortname,
                                  char        *outcmdexstring)
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

static int checkFlag64(uint64_t flags, uint64_t testflag, char *flagdescription)
{
    int rval = 0;

    // printf("--------- flags: %ld\n", flags);

    if(flags & testflag)
    {
        rval = 1;
        printf("    [%c[%d;%dm ON%c[%dm]  %s\n",
               (char) 27,
               1,
               32,
               (char) 27,
               0,
               flagdescription);
    }
    else
    {
        rval = 0;
        printf("    [%c[%d;%dmOFF%c[%dm]  %s\n",
               (char) 27,
               1,
               31,
               (char) 27,
               0,
               flagdescription);
    }
    return rval;
}

/**
 * @brief command help\n
 *
 * @param[in] cmdkey Commmand name
 */

errno_t help_command(
    const char *__restrict cmdkey
)
{
    int cOK = 0;

    for(unsigned int cmdi = 0; cmdi < data.NBcmd; cmdi++)
    {
        if(!strcmp(cmdkey, data.cmd[cmdi].key))
        {
            printf("\n");
            printf(COLORCMD "%s" COLORRESET
                   " in %s [%s]\n"
                   "\t" COLORINFO
                   "%s\n" COLORRESET,
                   data.cmd[cmdi].key,
                   data.cmd[cmdi].module,
                   data.module[
                       data.cmd[cmdi]
                       .moduleindex]
                   .shortname,
                   data.cmd[cmdi].info);

            printf("\t\033[33mexample>\033[0m"
                   " \033[1m%s\033[0m\n",
                   data.cmd[cmdi].example);
            printf("\t\033[2msrc: %s\033[0m"
                   "\n",
                   data.cmd[cmdi].srcfile);

            int FPSsupport = checkFlag64(data.cmd[cmdi].cmdsettings.flags,
                                         CLICMDFLAG_FPS,
                                         "FPS support");
            if(FPSsupport == 1)
            {
                if(checkFlag64(data.cmd[cmdi].cmdsettings.flags,
                               CLICMDFLAG_PROCINFO,
                               "processinfo support (..procinfo 0/1)") == 1)
                {
                    printf("        loopcntMax         : %ld\n",
                           data.cmd[cmdi].cmdsettings.procinfo_loopcntMax);
                    printf("      Triggering:\n");

                    printf("        triggermode        : %d ",
                           data.cmd[cmdi].cmdsettings.triggermode);
                    switch(data.cmd[cmdi].cmdsettings.triggermode)
                    {
                    case PROCESSINFO_TRIGGERMODE_IMMEDIATE:
                        printf("IMMEDIATE");
                        break;
                    case PROCESSINFO_TRIGGERMODE_CNT0:
                        printf("CNT0");
                        break;
                    case PROCESSINFO_TRIGGERMODE_CNT1:
                        printf("CNT1");
                        break;
                    case PROCESSINFO_TRIGGERMODE_SEMAPHORE:
                        printf("SEMAPHORE");
                        break;
                    case PROCESSINFO_TRIGGERMODE_DELAY:
                        printf("DELAY");
                        break;
                    case PROCESSINFO_TRIGGERMODE_CNT2:
                        printf("CNT2");
                        break;
                    default:
                        printf("unknown");
                        break;
                    }
                    printf("\n");

                    printf("        triggerstreamname  : %s\n",
                           data.cmd[cmdi].cmdsettings.triggerstreamname);

                    printf("        semindexrequested  : %d\n",
                           data.cmd[cmdi].cmdsettings.semindexrequested);

                    printf(
                        "        triggerdelay       : "
                        "%lld.%09ld\n",
                        (long long) data.cmd[cmdi]
                        .cmdsettings.triggerdelay.tv_sec,
                        data.cmd[cmdi].cmdsettings.triggerdelay.tv_nsec);

                    printf(
                        "        triggertimeout     : "
                        "%lld.%09ld\n",
                        (long long) data.cmd[cmdi]
                        .cmdsettings.triggertimeout.tv_sec,
                        data.cmd[cmdi].cmdsettings.triggertimeout.tv_nsec);

                    printf("      Resources:\n");
                    printf("        RT_priority        : %d\n",

                           data.cmd[cmdi].cmdsettings.RT_priority);

                    printf("        CPUmask            : ");

                    int nproc = sysconf(_SC_NPROCESSORS_ONLN);
                    for(int cpu = 0; cpu < nproc; cpu++)
                    {
                        printf(" %d",
                               CPU_ISSET(cpu,
                                         &data.cmd[cmdi].cmdsettings.CPUmask));
                    }
                    printf("\n");
                    printf("        MeasureTiming      : %d\n",
                           data.cmd[cmdi].cmdsettings.procinfo_MeasureTiming);
                }
            }

            printf("\n");
            printf("  CLI call arguments:\n");
            //printf("  CLI#       tagname             Value         description\n");

            int CLIargcnt = 0;
            for(int argi = 0; argi < data.cmd[cmdi].nbparam; argi++)
            {
                char valuestring[STRINGMAXLEN_CLICMDARG] = "???";

                switch(data.cmd[cmdi].argdata[argi].type)
                {
                case CLIARG_FLOAT32:
                    SNPRINTF_CHECK(valuestring,
                                   STRINGMAXLEN_CLICMDARG,
                                   "[float32]  %f",
                                   data.cmd[cmdi].argdata[argi].val.f32);
                    break;

                case CLIARG_FLOAT64:
                    SNPRINTF_CHECK(valuestring,
                                   STRINGMAXLEN_CLICMDARG,
                                   "[float64]  %lf",
                                   data.cmd[cmdi].argdata[argi].val.f64);
                    break;

                case CLIARG_ONOFF:
                    SNPRINTF_CHECK(valuestring,
                                   STRINGMAXLEN_CLICMDARG,
                                   "[ ONOFF ]  %ld",
                                   data.cmd[cmdi].argdata[argi].val.ui64);
                    break;

                case CLIARG_INT32:
                    SNPRINTF_CHECK(valuestring,
                                   STRINGMAXLEN_CLICMDARG,
                                   "[ int32 ]  %d",
                                   data.cmd[cmdi].argdata[argi].val.i32);
                    break;

                case CLIARG_UINT32:
                    SNPRINTF_CHECK(valuestring,
                                   STRINGMAXLEN_CLICMDARG,
                                   "[uint32 ]  %u",
                                   data.cmd[cmdi].argdata[argi].val.ui32);
                    break;

                case CLIARG_INT64:
                    SNPRINTF_CHECK(valuestring,
                                   STRINGMAXLEN_CLICMDARG,
                                   "[ int64 ]  %ld",
                                   data.cmd[cmdi].argdata[argi].val.i64);
                    break;

                case CLIARG_UINT64:
                    SNPRINTF_CHECK(valuestring,
                                   STRINGMAXLEN_CLICMDARG,
                                   "[uint64 ]  %lu",
                                   data.cmd[cmdi].argdata[argi].val.ui64);
                    break;

                case CLIARG_STR_NOT_IMG:
                    SNPRINTF_CHECK(valuestring,
                                   STRINGMAXLEN_CLICMDARG,
                                   "[ STRnI ]  %s",
                                   data.cmd[cmdi].argdata[argi].val.s);
                    break;

                case CLIARG_IMG:
                    SNPRINTF_CHECK(valuestring,
                                   STRINGMAXLEN_CLICMDARG,
                                   "[ STREAM]  %s",
                                   data.cmd[cmdi].argdata[argi].val.s);
                    break;

                case FPTYPE_FITSFILENAME:
                case FPTYPE_FILENAME:
                case FPTYPE_DIRNAME:
                case FPTYPE_EXECFILENAME:
                    SNPRINTF_CHECK(valuestring,
                                   STRINGMAXLEN_CLICMDARG,
                                   "[ FILE  ]  %s",
                                   data.cmd[cmdi].argdata[argi].val.s);
                    break;

                case FPTYPE_STRING:
                    SNPRINTF_CHECK(valuestring,
                                   STRINGMAXLEN_CLICMDARG,
                                   "[STRING ]  %s",
                                   data.cmd[cmdi].argdata[argi].val.s);
                    break;

                case FPTYPE_PID:
                    SNPRINTF_CHECK(valuestring,
                                   STRINGMAXLEN_CLICMDARG,
                                   "[  PID  ]  %ld",
                                   data.cmd[cmdi].argdata[argi].val.i64);
                    break;

                case FPTYPE_TIMESPEC:
                    SNPRINTF_CHECK(valuestring,
                                   STRINGMAXLEN_CLICMDARG,
                                   "[ TIME  ]  %s",
                                   data.cmd[cmdi].argdata[argi].val.s);
                    break;

                case FPTYPE_PROCESS:
                    SNPRINTF_CHECK(valuestring,
                                   STRINGMAXLEN_CLICMDARG,
                                   "[PROCESS]  %s",
                                   data.cmd[cmdi].argdata[argi].val.s);
                    break;

                case FPTYPE_FPSNAME:
                    SNPRINTF_CHECK(valuestring,
                                   STRINGMAXLEN_CLICMDARG,
                                   "[FPSNAME]  %s",
                                   data.cmd[cmdi].argdata[argi].val.s);
                    break;
                }

                if(data.cmd[cmdi].argdata[argi].fpflag & FPFLAG_PRIMARY_CLI_INPUT)
                {
                    printf("%6d   " COLORARGCLI " %-16s" COLORRESET " %-24s %s\n",
                           CLIargcnt,
                           data.cmd[cmdi].argdata[argi].fpstag,
                           valuestring,
                           data.cmd[cmdi].argdata[argi].descr);
                    CLIargcnt++;
                }
                else
                {
                    printf("[hidden] " COLORARGnotCLI " %-16s" COLORRESET " %-24s %s\n",
                           data.cmd[cmdi].argdata[argi].fpstag,
                           valuestring,
                           data.cmd[cmdi].argdata[argi].descr);
                }
            }

            printf("\n");

            cOK = 1;
        }
    }

    int foundsubstring  = 0;
    int foundregexmatch = 0;
    if(cOK == 0)
    {
        printf("Command \"%s\" does not exist. Partial matches:\n", cmdkey);

        regex_t regex;
        int     reti;
        /* Compile regular expression */
        reti = regcomp(&regex, cmdkey, REG_EXTENDED);
        if(reti)
        {
            fprintf(stderr, "Could not compile regex : \"%s\"\n", cmdkey);
            exit(1);
        }
        int        maxGroups = 8;
        regmatch_t groupArray[maxGroups];

        for(unsigned int cmdi = 0; cmdi < data.NBcmd; cmdi++)
        {

            int matchsubstring = 0;
            // look for substring match

            if(strstr(data.cmd[cmdi].key, cmdkey) != NULL)
            {
                foundsubstring = 1;
                matchsubstring = 1;
                printf(COLORCMD "%s" COLORRESET " in %s [%s]\n\t" COLORINFO
                       "%s\n" COLORRESET,
                       data.cmd[cmdi].key,
                       data.cmd[cmdi].module,
                       data.module[data.cmd[cmdi].moduleindex].shortname,
                       data.cmd[cmdi].info);
            }

            // Regular expression search
            if(matchsubstring == 0)
            {
                // Regular expression search
                reti = regexec(&regex,
                               data.cmd[cmdi].key,
                               maxGroups,
                               groupArray,
                               0);
                if(!reti)
                {
                    foundregexmatch = 1;
                    printf(COLORCMD "%s" COLORRESET " in %s [%s]\n\t" COLORINFO
                           "%s\n" COLORRESET,
                           data.cmd[cmdi].key,
                           data.cmd[cmdi].module,
                           data.module[data.cmd[cmdi].moduleindex].shortname,
                           data.cmd[cmdi].info);

                    char        *cursor = data.cmd[cmdi].key;
                    unsigned int offset = 0;
                    for(int g = 0; g < maxGroups; g++)
                    {
                        if(groupArray[g].rm_so == (regoff_t)((size_t) -1))
                        {
                            break; // No more groups
                        }

                        if(g == 0)
                        {
                            offset = groupArray[g].rm_eo;
                        }

                        char cursorCopy[strlen(cursor) + 1];
                        strcpy(cursorCopy, cursor);
                        cursorCopy[groupArray[g].rm_eo] = 0;
                        /*printf("\t    Match Group %u: [%2u-%2u]: %s\n",
                               g, groupArray[g].rm_so, groupArray[g].rm_eo,
                               cursorCopy + groupArray[g].rm_so);*/
                    }
                    cursor += offset;
                }
                else if(reti == REG_NOMATCH)
                {
                    //puts("No match");
                }
                else
                {
                    char msgbuf[100];
                    regerror(reti, &regex, msgbuf, sizeof(msgbuf));
                    fprintf(stderr, "Regex match failed: %s\n", msgbuf);
                    exit(1);
                }
            }
        }

        regfree(&regex);

        if(foundsubstring == 0)
        {
            if(foundregexmatch == 0)
            {
                printf("\tNo substring or regex match to \"%s\"\n", cmdkey);
            }
        }
    }

    return RETURN_SUCCESS;
}


/**
 * @brief search for string in command info
 *
 */

errno_t command_info_search(const char *restrict searchstring)
{
    int foundsubstring  = 0;
    int foundregexmatch = 0;
    int colorcodecmd    = 31; // red
    int colorcodeinfo   = 32; // green

    regex_t regex;
    /* Compile regular expression */
    int reti = regcomp(&regex, searchstring, REG_EXTENDED);
    if(reti)
    {
        fprintf(stderr, "Could not compile regex : \"%s\"\n", searchstring);
        exit(1);
    }
    int        maxGroups = 8;
    regmatch_t groupArray[maxGroups];

    for(unsigned int cmdi = 0; cmdi < data.NBcmd; cmdi++)
    {

        int matchsubstring = 0;
        // look for substring match

        if(strstr(data.cmd[cmdi].info, searchstring) != NULL)
        {
            foundsubstring = 1;
            matchsubstring = 1;
            printf("%c[%d;%dm%s%c[%dm in %s [%s]\n\t%c[%d;%dm%s%c[%dm\n",
                   (char) 27,
                   1,
                   colorcodecmd,
                   data.cmd[cmdi].key,
                   (char) 27,
                   0,
                   data.cmd[cmdi].module,
                   data.module[data.cmd[cmdi].moduleindex].shortname,
                   (char) 27,
                   1,
                   colorcodeinfo,
                   data.cmd[cmdi].info,
                   (char) 27,
                   0);
        }

        // Regular expression search
        if(matchsubstring == 0)
        {
            // Regular expression search
            reti =
                regexec(&regex, data.cmd[cmdi].info, maxGroups, groupArray, 0);
            if(!reti)
            {
                foundregexmatch = 1;

                printf(
                    "%c[%d;%dm%s%c[%dm in %s "
                    "[%s]\n\t%c[%d;%dm%s%c[%dm\n",
                    (char) 27,
                    1,
                    colorcodecmd,
                    data.cmd[cmdi].key,
                    (char) 27,
                    0,
                    data.cmd[cmdi].module,
                    data.module[data.cmd[cmdi].moduleindex].shortname,
                    (char) 27,
                    1,
                    colorcodeinfo,
                    data.cmd[cmdi].info,
                    (char) 27,
                    0);

                char        *cursor = data.cmd[cmdi].info;
                unsigned int offset = 0;
                for(int g = 0; g < maxGroups; g++)
                {
                    if(groupArray[g].rm_so == (regoff_t)((size_t) -1))
                    {
                        break; // No more groups
                    }

                    if(g == 0)
                    {
                        offset = groupArray[g].rm_eo;
                    }

                    char cursorCopy[strlen(cursor) + 1];
                    strcpy(cursorCopy, cursor);
                    cursorCopy[groupArray[g].rm_eo] = 0;
                    /*printf("\t    Match Group %u: [%2u-%2u]: %s\n",
                           g, groupArray[g].rm_so, groupArray[g].rm_eo,
                           cursorCopy + groupArray[g].rm_so);*/
                }
                cursor += offset;
            }
            else if(reti == REG_NOMATCH)
            {
                //puts("No match");
            }
            else
            {
                char msgbuf[100];
                regerror(reti, &regex, msgbuf, sizeof(msgbuf));
                fprintf(stderr, "Regex match failed: %s\n", msgbuf);
                exit(1);
            }
        }
    }

    regfree(&regex);

    if(foundsubstring == 0)
    {
        if(foundregexmatch == 0)
        {
            printf("\tNo substring or regex match to \"%s\"\n", searchstring);
        }
    }

    return RETURN_SUCCESS;
}


void print_milk_framework_help(void)
{
    printf("\n");
    printf(C_TITLE "========================================================\n" C_RST);
    printf(C_TITLE "                    milk OVERVIEW\n" C_RST);
    printf(C_TITLE "========================================================\n" C_RST);
    printf("\n");
    printf("The milk framework is built around three core pillars for\n");
    printf("high-performance, real-time data processing:\n");
    printf("\n");

    printf(C_HDR "1. ImageStreamIO (Streams)\n" C_RST);
    printf("Fast, low-latency shared-memory data streams designed to\n");
    printf("pass images and multi-dimensional arrays between distinct\n");
    printf("processes with zero-copy overhead.\n");
    printf("  " C_NOTE "Detailed guide:" C_RST " Run " C_CMD "milk-stream-help\n" C_RST);
    printf("\n");

    printf(C_HDR "2. Function Parameter Structure (FPS)\n" C_RST);
    printf("A shared memory architecture providing a unified namespace\n");
    printf("to manage configurations, parameters, and telemetry for\n");
    printf("applications seamlessly across the CLI and API.\n");
    printf("  " C_NOTE "Detailed guide:" C_RST " Run " C_CMD "milk-fps-help\n" C_RST);
    printf("\n");

    printf(C_HDR "3. Processinfo (procinfo API)\n" C_RST);
    printf("Advanced real-time execution management, CPU affinity,\n");
    printf("scheduling policies, and stream-based process triggering.\n");
    printf("  " C_NOTE "Detailed guide:" C_RST " Run " C_CMD "milk-procinfo-help\n" C_RST);
    printf("\n");

    printf(C_TITLE "--------------------------------------------------------\n" C_RST);
    printf(C_HDR "General Usage\n" C_RST);
    printf("To enter the interactive milk shell, simply type:\n");
    printf("  $ " C_CMD "milk-cli\n" C_RST);
    printf("\n");
    printf("From within the milk shell, you can list\n");
    printf("available commands to see all capabilities:\n");
    printf("For CLI specific help, run " C_CMD "milk-cli-help\n" C_RST);
    printf("\n");
}

/* ------------------------------------------------------------------ */
/* Topic: cmdopts — command-line options                               */
/* ------------------------------------------------------------------ */

/**
 * @brief Print help for milk-cli command-line flags.
 */
void help_topic_cmdopts(void)
{
    printf("\n");
    printf(C_TITLE "========================================================\n"
           C_RST);
    printf(C_TITLE "          COMMAND LINE OPTIONS  (cmdopts)\n" C_RST);
    printf(C_TITLE "========================================================\n"
           C_RST);
    printf("\n");
    printf(C_CMD "  -h, --help         " C_RST "print help index and exit\n");
    printf(C_CMD "  -v, --version      " C_RST "print version and exit\n");
    printf(C_CMD "  -i, --info         " C_RST
           "print version, settings, info\n");
    printf(C_CMD "  --verbose          " C_RST "be verbose\n");
    printf(C_CMD "  -d <level>         " C_RST "set debug level at startup\n");
    printf(C_CMD "  -o, --overwrite    " C_RST
           "overwrite existing FITS files "
           C_NOTE "(USE WITH CAUTION)\n" C_RST);
    printf(C_CMD "  -e, --errorexit    " C_RST "exit on error\n");
    printf(C_CMD "  -Z, --idle         " C_RST
           "only run when X is idle\n");
    printf(C_CMD "  -A, --autocomplete " C_RST
           "enable inline autocomplete "
           C_NOTE "(ON by default)\n" C_RST);
    printf(C_CMD "  --no-autocomplete  " C_RST
           "disable inline autocomplete\n");
    printf(C_CMD "  --no-history-suggest " C_RST
           "disable history suggestions\n");
    printf(C_CMD "  --no-arg-hints     " C_RST
           "disable argument hint line\n");
    printf(C_CMD "  --no-fuzzy         " C_RST
           "disable fuzzy/substring matching\n");
    printf(C_CMD "  -f, --fifoflag     " C_RST
           "enable default fifo input\n");
    printf(C_CMD "  -F <fifoname>      " C_RST "specify custom fifo name\n");
    printf(C_CMD "  -s <file>          " C_RST "execute startup script\n");
    printf(C_CMD "  -n <name>          " C_RST "specify process name\n");
    printf(C_CMD "  -p <priority>      " C_RST
           "set RT priority (0-99)\n");
    printf("\n");
}


/* ------------------------------------------------------------------ */
/* Topic: syntax — shell syntax and interaction                        */
/* ------------------------------------------------------------------ */

/**
 * @brief Print help for CLI syntax and interactive features.
 */
void help_topic_syntax(void)
{
    printf("\n");
    printf(C_TITLE "========================================================\n"
           C_RST);
    printf(C_TITLE "           SYNTAX & INTERACTION  (syntax)\n" C_RST);
    printf(C_TITLE "========================================================\n"
           C_RST);
    printf("\n");
    printf(C_HDR "Syntax Rules:\n" C_RST);
    printf("  Spaces separate arguments. Use "
           C_BOLD "#" C_RST " for comments.\n");
    printf("  Example: "
           C_CMD "command arg1 arg2 # comment\n" C_RST);
    printf("\n");
    printf(C_HDR "Tab Completion & UX Features:\n" C_RST);
    printf("  1st arg: Match commands, then images, "
           "then files.\n");
    printf("  Subsequent: Match images, then files.\n");
    printf("  " C_NOTE "History:" C_RST
           " Commands saved across sessions (Up/Down).\n");
    printf("  " C_NOTE "Autocorrection:" C_RST
           " Mistyped commands suggest closest match.\n");
    printf("  " C_NOTE "Fuzzy finding:" C_RST
           " " C_CMD "fhelp" C_RST " filters interactively.\n");
    printf("  " C_NOTE "Bash completion:" C_RST
           " Source scripts/milk-completion.sh.\n");
    printf("\n");
    printf(C_HDR "Piping Commands:\n" C_RST);
    printf("  Commands can be piped via stdin:\n");
    printf("  "
           C_CMD "echo -e \"a=1\\nb=2\\nc=a+b\" | milk-cli\n"
           C_RST);
    printf("  Use " C_BOLD "\\n" C_RST
           " to separate multiple commands.\n");
    printf("\n");
    printf(C_HDR "Shell Pass-through:\n" C_RST);
    printf("  Prefix OS commands with "
           C_CMD "!" C_RST
           " in interactive mode:\n");
    printf("  " C_CMD "!ls -la\n" C_RST);
    printf("  In script files the "
           C_CMD "!" C_RST " prefix is not required.\n");
    printf("\n");
}


/* ------------------------------------------------------------------ */
/* Topic: commands — built-in CLI commands                             */
/* ------------------------------------------------------------------ */

/**
 * @brief Print help for the most important built-in CLI commands.
 */
void help_topic_commands(void)
{
    printf("\n");
    printf(C_TITLE "========================================================\n"
           C_RST);
    printf(C_TITLE "           IMPORTANT COMMANDS  (commands)\n" C_RST);
    printf(C_TITLE "========================================================\n"
           C_RST);
    printf("\n");
    printf(C_HDR "Help & Discovery:\n" C_RST);
    printf(C_CMD "  help              " C_RST
           "Show help topic index\n");
    printf(C_CMD "  help-<topic>      " C_RST
           "Show help for a specific topic\n");
    printf(C_CMD "  cmd? [cmd]        " C_RST
           "Help for a specific command\n");
    printf(C_CMD "  m? [module]       " C_RST
           "List commands in a module\n");
    printf(C_CMD "  h? [string]       " C_RST
           "Search command descriptions\n");
    printf(C_CMD "  fhelp             " C_RST
           "Interactive fuzzy command search\n");
    printf(C_CMD "  fhist             " C_RST
           "Interactive fuzzy history search\n");
    printf("\n");
    printf(C_HDR "System Info:\n" C_RST);
    printf(C_CMD "  ci                " C_RST
           "System info and memory usage\n");
    printf(C_CMD "  mem.listim        " C_RST
           "List images in memory\n");
    printf("\n");
    printf(C_HDR "File I/O:\n" C_RST);
    printf(C_CMD "  iofits.loadfits   " C_RST
           "Load FITS file "
           C_NOTE "(requires CFITSIO)\n" C_RST);
    printf(C_CMD "  iofits.savefits   " C_RST
           "Save FITS file "
           C_NOTE "(requires CFITSIO)\n" C_RST);
    printf("\n");
    printf(C_HDR "Session Control:\n" C_RST);
    printf(C_CMD "  quit / exit       " C_RST "Exit the milk shell\n");
    printf(C_CMD "  !<syscommand>     " C_RST "Execute OS shell command\n");
    printf(C_CMD "  logon / logoff    " C_RST
           "Enable/disable session log\n");
    printf("\n");
}


/* ------------------------------------------------------------------ */
/* Topic: variables — variables, arrays, arithmetic, FPS access       */
/* ------------------------------------------------------------------ */

/**
 * @brief Print help for variables, arrays, and arithmetic.
 */
void help_topic_variables(void)
{
    printf("\n");
    printf(C_TITLE "========================================================\n"
           C_RST);
    printf(C_TITLE "       VARIABLES, ARRAYS & ARITHMETIC  (variables)\n"
           C_RST);
    printf(C_TITLE "========================================================\n"
           C_RST);
    printf("\n");
    printf(C_HDR "Basic Variables:\n" C_RST);
    printf("  " C_CMD "x=42" C_RST "              Set variable\n");
    printf("  " C_CMD "echo $x" C_RST "           Print variable\n");
    printf("  " C_CMD "echo ${x}" C_RST "         Braced form\n");
    printf("  " C_CMD "unset x" C_RST "           Remove variable\n");
    printf("  " C_CMD "vars" C_RST "              List all variables\n");
    printf("\n");
    printf(C_HDR "String Operations:\n" C_RST);
    printf("  " C_CMD "${#var}" C_RST "           String length\n");
    printf("  " C_CMD "${var:2:3}" C_RST "        Substring (offset:len)\n");
    printf("  " C_CMD "${var%%pat}" C_RST "        Strip longest suffix\n");
    printf("  " C_CMD "${var##pat}" C_RST "       Strip longest prefix\n");
    printf("  " C_CMD "${var/p/r}" C_RST "        Replace first match\n");
    printf("  " C_CMD "${var//p/r}" C_RST "       Replace all matches\n");
    printf("  " C_CMD "${var^^}" C_RST "          Uppercase all\n");
    printf("  " C_CMD "${var,,}" C_RST "          Lowercase all\n");
    printf("\n");
    printf(C_HDR "Parameter Defaults:\n" C_RST);
    printf("  " C_CMD "${v:-def}" C_RST "         default if unset\n");
    printf("  " C_CMD "${v:=def}" C_RST "         assign if unset\n");
    printf("  " C_CMD "${v:+alt}" C_RST "         alt value if set\n");
    printf("  " C_CMD "${v:?err}" C_RST "         error if unset\n");
    printf("\n");
    printf(C_HDR "Arrays:\n" C_RST);
    printf("  " C_CMD "arr=(a b c)" C_RST "     Create array\n");
    printf("  " C_CMD "${arr[0]}" C_RST "        Access element\n");
    printf("  " C_CMD "${arr[@]}" C_RST "        All elements\n");
    printf("  " C_CMD "${#arr[@]}" C_RST "       Array length\n");
    printf("  " C_CMD "declare -A m" C_RST "     Associative array\n");
    printf("  " C_CMD "${m[key]}" C_RST "        Associative lookup\n");
    printf("\n");
    printf(C_HDR "Arithmetic:\n" C_RST);
    printf("  " C_CMD "y=$(( x + 5 ))" C_RST
           "  Integer +, -, *, /, %%\n");
    printf("  " C_CMD "(( expr ))" C_RST
           "        Arithmetic conditional\n");
    printf("\n");
    printf(C_HDR "FPS Parameter Access:\n" C_RST);
    printf("  " C_CMD "@fpsname.param" C_RST
           "    Read FPS parameter\n");
    printf("  " C_CMD "fpsset fps p v" C_RST
           "    Write FPS parameter\n");
    printf("\n");
    printf(C_HDR "Milk Stream Attributes:\n" C_RST);
    printf("  " C_CMD "${s.xsize}" C_RST "         Stream width\n");
    printf("  " C_CMD "${s.ysize}" C_RST "         Stream height\n");
    printf("  " C_CMD "${s.type}" C_RST "          Stream datatype\n");
    printf("  " C_CMD "${s.cnt0}" C_RST "          Stream frame counter\n");
    printf("\n");
}


/* ------------------------------------------------------------------ */
/* Topic: flowcontrol — if, loops, case, functions                    */
/* ------------------------------------------------------------------ */

/**
 * @brief Print help for flow control constructs.
 */
void help_topic_flowcontrol(void)
{
    printf("\n");
    printf(C_TITLE "========================================================\n"
           C_RST);
    printf(C_TITLE "              FLOW CONTROL  (flowcontrol)\n" C_RST);
    printf(C_TITLE "========================================================\n"
           C_RST);
    printf("\n");
    printf(C_HDR "Conditionals:\n" C_RST);
    printf("  " C_CMD "if [ $x -gt 5 ]; then" C_RST "\n");
    printf("  " C_CMD "    echo big" C_RST "\n");
    printf("  " C_CMD "elif [ $x -gt 2 ]; then"
           C_RST "  ← cascading branch\n");
    printf("  " C_CMD "else" C_RST "\n");
    printf("  " C_CMD "    echo small" C_RST "\n");
    printf("  " C_CMD "fi" C_RST "\n");
    printf("  Tests: "
           C_NOTE "-eq -ne -gt -ge -lt -le"
           C_RST " (numeric), "
           C_NOTE "= !=" C_RST " (string)\n");
    printf("  File tests: "
           C_NOTE "-f" C_RST " (file), "
           C_NOTE "-d" C_RST " (dir), "
           C_NOTE "-e" C_RST " (exists)\n");
    printf("  Negate: " C_CMD "[ ! expr ]" C_RST " logical NOT\n");
    printf("  Extended: "
           C_CMD "[[ $s =~ ^[0-9]+$ ]]"
           C_RST " regex\n");
    printf("\n");
    printf(C_HDR "Loops:\n" C_RST);
    printf("  " C_CMD "while [ $n -lt 10 ]; do"
           C_RST " ... "
           C_CMD "done" C_RST "\n");
    printf("  " C_CMD "for x in a b c; do"
           C_RST " ... "
           C_CMD "done" C_RST "\n");
    printf("  " C_CMD "for ((i=0; i<10; i++)); do"
           C_RST " ... "
           C_CMD "done" C_RST "  ← C-style\n");
    printf("  " C_CMD "break" C_RST " exits loop, "
           C_CMD "continue" C_RST " next iter\n");
    printf("  " C_CMD "break 2" C_RST
           "  / " C_CMD "continue 2" C_RST
           "  (nested)\n");
    printf("\n");
    printf(C_HDR "Case Statement:\n" C_RST);
    printf("  " C_CMD "case $var in" C_RST "\n");
    printf("  " C_CMD "  yes) echo ok ;;" C_RST "\n");
    printf("  " C_CMD "  a|b) echo ab ;;"
           C_RST "  ← alternation\n");
    printf("  " C_CMD "  *) echo default ;;" C_RST "\n");
    printf("  " C_CMD "esac" C_RST "\n");
    printf("\n");
    printf(C_HDR "Functions:\n" C_RST);
    printf("  " C_CMD "function myfunc {" C_RST
           " ... " C_CMD "}" C_RST "\n");
    printf("  " C_CMD "myfunc arg1 arg2"
           C_RST "  call with "
           C_NOTE "$1..$9" C_RST " in body\n");
    printf("  " C_CMD "return [val]" C_RST
           "      exit function, set $?\n");
    printf("  " C_CMD "local VAR=val" C_RST
           "     declare local variable\n");
    printf("\n");
    printf(C_HDR "Logical Operators:\n" C_RST);
    printf("  " C_CMD "cmd1 && cmd2" C_RST
           "  run cmd2 if cmd1 succeeds\n");
    printf("  " C_CMD "cmd1 || cmd2" C_RST
           "  run cmd2 if cmd1 fails\n");
    printf("\n");
    printf(C_HDR "Select Menu:\n" C_RST);
    printf("  " C_CMD "select x in a b c; do" C_RST "\n");
    printf("  " C_CMD "  echo $x" C_RST "\n");
    printf("  " C_CMD "done" C_RST "  interactive numbered menu\n");
    printf("\n");
    printf(C_HDR "Stream Event:\n" C_RST);
    printf("  " C_CMD "on_update <stream> { cmd }"
           C_RST "\n");
    printf("  Waits for stream update then runs cmd\n");
    printf("\n");
}


/* ------------------------------------------------------------------ */
/* Topic: scripting — script files, I/O, builtins, traps              */
/* ------------------------------------------------------------------ */

/**
 * @brief Print help for scripting features and built-ins.
 */
void help_topic_scripting(void)
{
    printf("\n");
    printf(C_TITLE "========================================================\n"
           C_RST);
    printf(C_TITLE "              SCRIPTING FEATURES  (scripting)\n" C_RST);
    printf(C_TITLE "========================================================\n"
           C_RST);
    printf("\n");
    printf(C_HDR "Script Files:\n" C_RST);
    printf(C_CMD "  source <file>      " C_RST
           "Execute a script file\n");
    printf(C_CMD "  . <file>           " C_RST "Same (dot-source)\n");
    printf(C_CMD "  include_once <f>   " C_RST
           "Source only once per session\n");
    printf(C_CMD "  savescript <file>  " C_RST
           "Save variables & functions\n");
    printf(C_CMD "  savehistory <file> " C_RST "Save command history\n");
    printf("  Startup: "
           C_CMD "milk-cli -s <file>" C_RST
           "  run on launch\n");
    printf("  Auto-load: "
           C_CMD "~/.milkrc" C_RST " sourced at startup\n");
    printf("  Shebang:   "
           C_NOTE "#!/usr/bin/env milk-cli -s" C_RST "\n");
    printf("\n");
    printf(C_HDR "Built-in Commands:\n" C_RST);
    printf(C_CMD "  echo <str>         " C_RST "Print a line\n");
    printf(C_CMD "  printf \"fmt\" a..   " C_RST
           "Formatted output (%%s %%d %%f)\n");
    printf(C_CMD "  sleep <sec>        " C_RST
           "Pause (float-capable)\n");
    printf(C_CMD "  read [-p p] var    " C_RST
           "Read line from stdin\n");
    printf(C_CMD "  read -t N var      " C_RST
           "Timed read (seconds)\n");
    printf(C_CMD "  read -a ARR        " C_RST
           "Read words into array\n");
    printf(C_CMD "  exit [N]           " C_RST "Exit with status N\n");
    printf(C_CMD "  shift [N]          " C_RST "Shift $1..$9 by N\n");
    printf(C_CMD "  true / false       " C_RST "Set $? to 0 / 1\n");
    printf("\n");
    printf(C_HDR "Pipes & Redirection:\n" C_RST);
    printf("  " C_CMD "cmd1 | cmd2" C_RST
           "     pipe stdout → stdin\n");
    printf("  " C_CMD "cmd > file" C_RST "      write to file\n");
    printf("  " C_CMD "cmd >> file" C_RST "     append to file\n");
    printf("  " C_CMD "cmd < file" C_RST "      stdin from file\n");
    printf("  " C_CMD "cmd <<< \"str\"" C_RST "  here-string\n");
    printf("  " C_CMD "cmd 2>&1" C_RST "        stderr to stdout\n");
    printf("  " C_CMD "cmd 2>/dev/null" C_RST " discard stderr\n");
    printf("\n");
    printf(C_HDR "Brace & Glob Expansion:\n" C_RST);
    printf("  " C_CMD "{1..5}" C_RST " → 1 2 3 4 5\n");
    printf("  " C_CMD "{0..10..2}" C_RST " → 0 2 4 6 8 10\n");
    printf("  " C_CMD "*.fits" C_RST " expands to matching files\n");
    printf("  " C_CMD "data_??.bin" C_RST " single-char wildcard\n");
    printf("\n");
    printf(C_HDR "Heredocs:\n" C_RST);
    printf("  " C_CMD "VAR=<<EOF" C_RST "\n");
    printf("  " C_CMD "  line 1" C_RST "\n");
    printf("  " C_CMD "EOF" C_RST " → multi-line variable\n");
    printf("\n");
    printf(C_HDR "Signal Traps:\n" C_RST);
    printf("  " C_CMD "trap 'cmd' EXIT INT" C_RST " handler\n");
    printf("  " C_CMD "trap 'rm /tmp/f' EXIT" C_RST " cleanup\n");
    printf("\n");
    printf(C_HDR "Shell Options:\n" C_RST);
    printf("  " C_CMD "set -e" C_RST "  exit on error\n");
    printf("  " C_CMD "set -x" C_RST "  trace commands\n");
    printf("  " C_CMD "set +e" C_RST "  / " C_CMD "set +x" C_RST
           "  disable above\n");
    printf("\n");
    printf(C_HDR "Environment & Read-only:\n" C_RST);
    printf("  " C_CMD "export VAR=val" C_RST "  env var\n");
    printf("  " C_CMD "readonly VAR=val" C_RST " immutable\n");
    printf("\n");
    printf(C_HDR "Aliases & Indirect Expansion:\n" C_RST);
    printf("  " C_CMD "alias n='cmd'" C_RST " create alias\n");
    printf("  " C_CMD "unalias n" C_RST "     remove alias\n");
    printf("  " C_CMD "${!var}" C_RST "       indirect expansion\n");
    printf("\n");
    printf(C_HDR "Miscellaneous:\n" C_RST);
    printf("  " C_CMD "getopts \"ab:\" opt" C_RST
           "  option parsing\n");
    printf("  " C_CMD "mapfile -t arr < file" C_RST
           "  lines → array\n");
    printf("  " C_CMD "cmd &" C_RST " background; "
           C_CMD "wait" C_RST " for bg jobs\n");
    printf("  " C_CMD "(cmd1; cmd2)" C_RST " subshell\n");
    printf("  " C_CMD "~/path" C_RST " → $HOME/path\n");
    printf("  " C_CMD "basename / dirname" C_RST
           "  path utilities\n");
    printf("  " C_CMD "pushd / popd / dirs" C_RST
           "  directory stack\n");
    printf("\n");
}


/* ------------------------------------------------------------------ */
/* Topic: milk — milk-specific runtime features                       */
/* ------------------------------------------------------------------ */

/**
 * @brief Print help for milk-specific CLI features.
 */
void help_topic_milk(void)
{
    printf("\n");
    printf(C_TITLE "========================================================\n"
           C_RST);
    printf(C_TITLE "          MILK-SPECIFIC FEATURES  (milk)\n" C_RST);
    printf(C_TITLE "========================================================\n"
           C_RST);
    printf("\n");
    printf(C_HDR "Stream Metadata Dot-Expansion:\n" C_RST);
    printf("  " C_CMD "${s.xsize}" C_RST
           "         Stream width dimension\n");
    printf("  " C_CMD "${s.ysize}" C_RST
           "         Stream height dimension\n");
    printf("  " C_CMD "${s.type}" C_RST
           "          Stream datatype code\n");
    printf("  " C_CMD "${s.cnt0}" C_RST
           "          Frame counter (total)\n");
    printf("  " C_CMD "${s.cnt1}" C_RST
           "          Frame counter (recent)\n");
    printf("\n");
    printf(C_HDR "Waiting for Resources:\n" C_RST);
    printf("  " C_CMD "waitfor_stream s T" C_RST
           "  Block up to T sec for SHM stream\n");
    printf("  " C_CMD "waitfor_fps f T   " C_RST
           "  Block up to T sec for FPS\n");
    printf("  " C_CMD "on_update <name> { cmd }" C_RST
           "  Trigger on stream write\n");
    printf("\n");
    printf(C_HDR "FPS Parameters:\n" C_RST);
    printf("  " C_CMD "@fpsname.param   " C_RST
           "Read FPS parameter value\n");
    printf("  " C_CMD "fpsset fps p v   " C_RST
           "Write FPS parameter\n");
    printf("  " C_CMD "fparam <fpsname> " C_RST
           "Interactive FPS parameter editor\n");
    printf("\n");
    printf(C_HDR "Stream Management:\n" C_RST);
    printf("  " C_CMD "milk-FITS2shm f.fits s " C_RST
           "Load FITS into SHM stream\n");
    printf("  " C_CMD "milk-shm2FITS s f.fits " C_RST
           "Save SHM stream to FITS\n");
    printf("  " C_CMD "milk-stream-help        " C_RST
           "Stream usage guide\n");
    printf("\n");
    printf(C_HDR "FPS Executables:\n" C_RST);
    printf("  " C_CMD "milk-fpsexec-list       " C_RST
           "List all fpsexec programs\n");
    printf("  " C_CMD "milk-fpsexec-<name> -h1 " C_RST
           "One-line description\n");
    printf("  " C_CMD "milk-fpsCTRL           " C_RST
           "TUI parameter controller\n");
    printf("  " C_CMD "milk-fps-help           " C_RST
           "FPS usage guide\n");
    printf("\n");
    printf(C_HDR "Process Monitoring:\n" C_RST);
    printf("  " C_CMD "milk-streamCTRL        " C_RST
           "TUI stream monitor\n");
    printf("  " C_CMD "milk-procCTRL          " C_RST
           "TUI process monitor\n");
    printf("  " C_CMD "milk-procinfo-help      " C_RST
           "Processinfo guide\n");
    printf("\n");
}


/* ------------------------------------------------------------------ */
/* Topic dispatch by name                                              */
/* ------------------------------------------------------------------ */

/**
 * @brief Dispatch help output to the correct topic function.
 *
 * @param topic  Topic keyword string, or NULL / "" for index.
 * @return  0 on success, 1 if topic not found.
 */
int help_topic_dispatch(const char *topic)
{
    if (!topic || topic[0] == '\0')
    {
        return 1; /* caller should print the index */
    }

    if (strcmp(topic, "cmdopts") == 0)
    {
        help_topic_cmdopts();
    }
    else if (strcmp(topic, "syntax") == 0)
    {
        help_topic_syntax();
    }
    else if (strcmp(topic, "commands") == 0)
    {
        help_topic_commands();
    }
    else if (strcmp(topic, "variables") == 0)
    {
        help_topic_variables();
    }
    else if (strcmp(topic, "flowcontrol") == 0)
    {
        help_topic_flowcontrol();
    }
    else if (strcmp(topic, "scripting") == 0)
    {
        help_topic_scripting();
    }
    else if (strcmp(topic, "milk") == 0)
    {
        help_topic_milk();
    }
    else
    {
        return 1; /* unknown topic */
    }
    return 0;
}


/* ------------------------------------------------------------------ */
/* Main help index (replaces the old monolithic function)             */
/* ------------------------------------------------------------------ */

/**
 * @brief Print a compact help index listing all available topics.
 *
 * The full content for each topic is obtained via help-<topic> CLI
 * commands or  milk-cli-help <topic>  from the shell.
 */
/**
 * @brief Print only the available-topics list.
 *
 * Shown both in the full help index and as a concise
 * hint when an unknown topic is supplied.
 */
void print_help_topic_list(void)
{
    printf(C_HDR "Available help topics:\n" C_RST);
    printf("  " C_CMD "cmdopts     " C_RST
           "Command-line flags (-h, -s, -n\xe2\x80\xa6)\n");
    printf("  " C_CMD "syntax      " C_RST
           "Syntax, tab completion, piping\n");
    printf("  " C_CMD "commands    " C_RST
           "Built-in CLI commands (?, cmd?\xe2\x80\xa6)\n");
    printf("  " C_CMD "variables   " C_RST
           "Variables, arrays, arithmetic\n");
    printf("  " C_CMD "flowcontrol " C_RST
           "if/while/for/case/function\n");
    printf("  " C_CMD "scripting   " C_RST
           "Script files, I/O, builtins\n");
    printf("  " C_CMD "milk        " C_RST
           "Streams, FPS, milk-specific\n");
    printf("\n");
}

void print_milk_cli_help(void)
{
    printf("\n");
    printf(C_TITLE
           "========================================\n" C_RST);
    printf(C_TITLE
           "           milk-cli \xe2\x80\x94 HELP INDEX\n" C_RST);
    printf(C_TITLE
           "========================================\n" C_RST);
    printf("\n");
    print_help_topic_list();
    printf(C_NOTE "From the shell:\n" C_RST);
    printf("  $ " C_CMD "milk-cli-help <topic>\n" C_RST);
    printf("\n");
    printf(C_HDR "Quick reference:\n" C_RST);
    printf("  " C_CMD "cmd? [name]   " C_RST
           "Help for a specific command\n");
    printf("  " C_CMD "m? [module]   " C_RST
           "List commands in a module\n");
    printf("  " C_CMD "h? [string]   " C_RST
           "Search command descriptions\n");
    printf("  " C_CMD "fhelp         " C_RST
           "Interactive fuzzy command search\n");
    printf("  " C_CMD "quit / exit   " C_RST
           "Exit the milk shell\n");
    printf("\n");
}


errno_t help()
{
    if ((data.cmdargtoken[1].type == CMDARGTOKEN_TYPE_STRING) ||
            (data.cmdargtoken[1].type == CMDARGTOKEN_TYPE_RAWSTRING))
    {
        const char *topic = data.cmdargtoken[1].val.string;
        if (help_topic_dispatch(topic) != 0)
        {
            printf(C_ERR "Unknown help topic: \"%s\"" C_RST "\n\n",
                   topic);
            print_help_topic_list();
        }
    }
    else
    {
        print_milk_cli_help();
    }

    return RETURN_SUCCESS;
}

errno_t helpreadline()
{

    EXECUTE_SYSTEM_COMMAND(
        "more %s/src/CommandLineInterface/doc/helpreadline.md",
        dcsourcedir);

    return RETURN_SUCCESS;
}

errno_t help_cmd()
{
    if((data.cmdargtoken[1].type == CMDARGTOKEN_TYPE_STRING) ||
            (data.cmdargtoken[1].type == CMDARGTOKEN_TYPE_EXISTINGIMAGE) ||
            (data.cmdargtoken[1].type == CMDARGTOKEN_TYPE_COMMAND) ||
            (data.cmdargtoken[1].type == CMDARGTOKEN_TYPE_RAWSTRING))
    {
        help_command(data.cmdargtoken[1].val.string);
    }
    else
    {
        list_commands();
    }

    return RETURN_SUCCESS;
}

errno_t cmdinfosearch()
{
    if((data.cmdargtoken[1].type == CMDARGTOKEN_TYPE_STRING) ||
            (data.cmdargtoken[1].type == CMDARGTOKEN_TYPE_EXISTINGIMAGE) ||
            (data.cmdargtoken[1].type == CMDARGTOKEN_TYPE_COMMAND) ||
            (data.cmdargtoken[1].type == CMDARGTOKEN_TYPE_RAWSTRING))
    {
        command_info_search(data.cmdargtoken[1].val.string);
    }
    else
    {
        list_commands();
    }

    return RETURN_SUCCESS;
}

errno_t help_module()
{

    if(data.cmdargtoken[1].type == 3)
    {
        list_commands_module(data.cmdargtoken[1].val.string);
    }
    else
    {
        long i;
        printf("\n");
        printf("%2s  %10s %32s %10s %7s    %20s %s\n",
               "#",
               "shortname",
               "Name",
               "Package",
               "Version",
               "last compiled",
               "description");
        printf(
            "--------------------------------------------------------------"
            "-----------------------------------------"
            "-------\n");
        for(i = 0; i < data.NBmodule; i++)
        {
            printf(
                "%2ld %10s \033[1m%32s\033[0m %10s %2d.%02d.%02d    "
                "%11s %8s  %s\n",
                i,
                data.module[i].shortname,
                data.module[i].name,
                data.module[i].package,
                data.module[i].versionmajor,
                data.module[i].versionminor,
                data.module[i].versionpatch,
                data.module[i].datestring,
                data.module[i].timestring,
                data.module[i].info);
        }
        printf(
            "--------------------------------------------------------------"
            "-----------------------------------------"
            "\n");
        printf("\n");
    }

    return RETURN_SUCCESS;
}


// -----------------------------------------------------------------------------
// Interactive Fuzzy Help (fhelp)
// -----------------------------------------------------------------------------

#include <termios.h>
#include <sys/ioctl.h>
#include <ctype.h>
#ifdef USE_READLINE
#include <readline/readline.h>
#include <readline/history.h>
#endif

#include <sys/select.h>

/**
 * @brief Read one byte from stdin using read(), bypassing stdio buffering.
 *
 * All TUI input loops must use this instead of getchar() so that
 * select() and read() operate on the same file-descriptor buffer.
 * When getchar() is used, stdio pre-reads multiple bytes into its
 * internal buffer, leaving the kernel fd empty; select() then sees
 * no data and incorrectly treats an arrow-key ESC sequence as a
 * bare ESC press.
 *
 * Returns the byte read (0-255 as unsigned char cast to int),
 * or -1 on error / EOF.
 */
static int tui_readchar(void)
{
    unsigned char ch;
    ssize_t n = read(STDIN_FILENO, &ch, 1);
    return (n == 1) ? (int)ch : -1;
}

/**
 * @brief Wait up to ms milliseconds for stdin fd to have data.
 *
 * Returns 1 if data is ready, 0 on timeout.
 * Must be paired with tui_readchar() (not getchar()) so that
 * select() and read() both observe the same kernel buffer.
 */
static int tui_stdin_wait_ms(int ms)
{
    fd_set fds;
    struct timeval tv;

    FD_ZERO(&fds);
    FD_SET(STDIN_FILENO, &fds);
    tv.tv_sec  = 0;
    tv.tv_usec = ms * 1000;
    return select(STDIN_FILENO + 1,
                  &fds, NULL, NULL, &tv) > 0;
}

// Simple Levenshtein distance for fuzzy matching
static int fuzzy_match_score(const char *query, const char *target)
{
    if (!query || !query[0]) return 10000; // Empty query matches perfectly
    
    int score = 0;
    const char *q = query;
    const char *t = target;
    
    while (*q && *t) {
        if (tolower(*q) == tolower(*t)) {
            score += 10;
            q++;
        }
        t++;
    }
    
    // Penalty for target length (prefer shorter exact matches)
    score -= strlen(target);
    
    if (*q) return -1000; // Didn't match all characters in query
    return score;
}

// Structure for sorting matches
typedef struct {
    int index;
    int score;
} MatchScore;

static int compare_matches(const void *a, const void *b)
{
    return ((MatchScore*)b)->score - ((MatchScore*)a)->score;
}

int cli_fhelp(void)
{
    struct termios oldt, newt;
    char query[128] = {0};
    int query_len = 0;
    int selected = 0;
    int num_matches = 0;
    MatchScore matches[1024];

    // Setup raw terminal mode
    tcgetattr(STDIN_FILENO, &oldt);
    newt = oldt;
    newt.c_lflag &= ~(ICANON | ECHO | ISIG);
    tcsetattr(STDIN_FILENO, TCSANOW, &newt);

    while (1) {
        // Compute matches
        num_matches = 0;
        for (long i = 0; i < data.NBcmd; i++) {
            if (num_matches >= 1024) break;
            
            int s1 = fuzzy_match_score(query, data.cmd[i].key);
            int s2 = fuzzy_match_score(query, data.cmd[i].info);
            int best_score = s1 > s2 ? s1 : s2;
            
            if (best_score > -500) {
                matches[num_matches].index = i;
                matches[num_matches].score = best_score;
                num_matches++;
            }
        }
        
        qsort(matches, num_matches, sizeof(MatchScore), compare_matches);

        // Clamp selection
        if (selected >= num_matches) selected = num_matches - 1;
        if (selected < 0) selected = 0;

        // Render UI
        printf("\033[2J\033[H"); // Clear screen
        printf("\033[1;36m>> Interactive Fuzzy Help <<\033[0m\n");
        printf("Search: \033[1m%s\033[0m_\n\n", query);
        
        int display_count = num_matches > 20 ? 20 : num_matches;
        
        for (int i = 0; i < display_count; i++) {
            int cmd_idx = matches[i].index;
            if (i == selected) {
                printf("\033[1;33m> %-20s : %s\033[0m\n", data.cmd[cmd_idx].key, data.cmd[cmd_idx].info);
            } else {
                printf("  %-20s : %s\n", data.cmd[cmd_idx].key, data.cmd[cmd_idx].info);
            }
        }
        
        printf("\n\033[2m[Up/Down/PgUp/PgDn] Navigate  [Enter] Select  [Esc/Ctrl+C] Cancel\033[0m\n");

        // Input loop
        int c = tui_readchar();
        if (c == 27) { // Escape seq
            if (tui_stdin_wait_ms(50)) {
                int b1 = tui_readchar();
                int b2 = tui_readchar();
                if (b1 == '[') {
                    if (b2 == 'A') selected--; // Up
                    else if (b2 == 'B') selected++; // Down
                    else if (b2 == '5') { tui_readchar(); selected -= 10; } // PgUp
                    else if (b2 == '6') { tui_readchar(); selected += 10; } // PgDn
                }
            } else {
                selected = -1;
                break; // Bare ESC — cancel
            }
        } else if (c == 10 || c == 13) { // Enter
            break; // Select
        } else if (c == 3 || c == 4) { // Ctrl+C or Ctrl+D
            selected = -1;
            break;
        } else if (c == 127 || c == 8) { // Backspace
            if (query_len > 0) {
                query_len--;
                query[query_len] = '\0';
                selected = 0;
            }
        } else if (c >= 32 && c <= 126 && query_len < 127) { // Printable
            query[query_len++] = (char)c;
            query[query_len] = '\0';
            selected = 0;
        }
    }

    // Restore terminal
    tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
    printf("\033[2J\033[H"); // Clear screen on exit

    if (selected >= 0 && selected < num_matches) {
        int cmd_idx = matches[selected].index;
        printf("Selected command: \033[1m%s\033[0m\n", data.cmd[cmd_idx].key);
        
#ifdef USE_READLINE
        // Stuff the selected command into the readline input stream. 
        // This is safer than modifying the rl_line_buffer directly while inside the handler
        // because readline will process these stuffed characters on its very next read cycle
        // and echo them correctly as if the user is typing them at the prompt.
        for (size_t i = 0; i < strlen(data.cmd[cmd_idx].key); i++) {
            rl_stuff_char(data.cmd[cmd_idx].key[i]);
        }
        rl_stuff_char(' ');
#else
        // If not using readline, just print it and they can copy-paste.
#endif
    }

    return RETURN_SUCCESS;
}


// -----------------------------------------------------------------------------
// Interactive Fuzzy History (fhist)
// -----------------------------------------------------------------------------

int cli_fhist(void)
{
#ifdef USE_READLINE
    HIST_ENTRY **hlist = history_list();
    if(hlist == NULL || history_length == 0)
    {
        printf("No history\n");
        return RETURN_SUCCESS;
    }

    struct termios oldt, newt;
    char query[128] = {0};
    int query_len = 0;
    int selected = 0;
    int num_matches = 0;
    MatchScore matches[1024];

    // Setup raw terminal mode
    tcgetattr(STDIN_FILENO, &oldt);
    newt = oldt;
    newt.c_lflag &= ~(ICANON | ECHO | ISIG);
    tcsetattr(STDIN_FILENO, TCSANOW, &newt);

    while (1) {
        // Compute matches
        num_matches = 0;
        // Search backwards through history
        for (int i = history_length - 1; i >= 0; i--) {
            if (num_matches >= 1024) break;
            
            // Skip duplicates (simple lookback to recent matching entries)
            // It's common to have the same command consecutively in history
            int is_dup = 0;
            for (int j = 0; j < num_matches; j++) {
                if (strcmp(hlist[i]->line, hlist[matches[j].index]->line) == 0) {
                    is_dup = 1;
                    break;
                }
            }
            if (is_dup) continue;
            
            int score = fuzzy_match_score(query, hlist[i]->line);
            
            if (score > -500) {
                matches[num_matches].index = i;
                // Penalize older entries so recent ones stay at top if scores match
                matches[num_matches].score = score - (history_length - i);
                num_matches++;
            }
        }
        
        // Sort if we have a query. If not, keep chronological (reversed).
        if (query_len > 0) {
            qsort(matches, num_matches, sizeof(MatchScore), compare_matches);
        }

        // Clamp selection
        if (selected >= num_matches) selected = num_matches - 1;
        if (selected < 0) selected = 0;

        // Render UI
        printf("\033[2J\033[H"); // Clear screen
        printf("\033[1;36m>> Interactive Fuzzy History Search <<\033[0m\n");
        printf("Search: \033[1m%s\033[0m_\n\n", query);
        
        int display_count = num_matches > 20 ? 20 : num_matches;
        
        for (int i = 0; i < display_count; i++) {
            int hist_idx = matches[i].index;
            if (i == selected) {
                printf("\033[1;33m> %s\033[0m\n", hlist[hist_idx]->line);
            } else {
                printf("  %s\n", hlist[hist_idx]->line);
            }
        }
        
        printf("\n\033[2m[Up/Down/PgUp/PgDn] Navigate  [Enter] Select  [Esc/Ctrl+C] Cancel\033[0m\n");

        // Input loop
        int c = tui_readchar();
        if (c == 27) { // Escape seq
            if (tui_stdin_wait_ms(50)) {
                int b1 = tui_readchar();
                int b2 = tui_readchar();
                if (b1 == '[') {
                    if (b2 == 'A') selected--; // Up
                    else if (b2 == 'B') selected++; // Down
                    else if (b2 == '5') { tui_readchar(); selected -= 10; } // PgUp
                    else if (b2 == '6') { tui_readchar(); selected += 10; } // PgDn
                }
            } else {
                selected = -1;
                break; // Bare ESC — cancel
            }
        } else if (c == 10 || c == 13) { // Enter
            break; // Select
        } else if (c == 3 || c == 4) { // Ctrl+C or Ctrl+D
            selected = -1;
            break;
        } else if (c == 127 || c == 8) { // Backspace
            if (query_len > 0) {
                query_len--;
                query[query_len] = '\0';
                selected = 0;
            }
        } else if (c >= 32 && c <= 126 && query_len < 127) { // Printable
            query[query_len++] = (char)c;
            query[query_len] = '\0';
            selected = 0;
        }
    }

    // Restore terminal
    tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
    printf("\033[2J\033[H"); // Clear screen on exit

    if (selected >= 0 && selected < num_matches) {
        int hist_idx = matches[selected].index;
        printf("Selected history: \033[1m%s\033[0m\n", hlist[hist_idx]->line);
        
        for (size_t i = 0; i < strlen(hlist[hist_idx]->line); i++) {
            rl_stuff_char(hlist[hist_idx]->line[i]);
        }
    }
#else
    printf("Readline not available. History fuzzy search requires readline.\n");
#endif
    return RETURN_SUCCESS;
}


// -----------------------------------------------------------------------------
// Interactive FPS Parameter Edit (fparam)
// -----------------------------------------------------------------------------

int cli_fparam(void)
{
    if (data.cmdargtoken[1].type != CMDARGTOKEN_TYPE_STRING) {
        printf("Usage: fparam <fpsname>\n");
        return RETURN_SUCCESS;
    }
    
    char *fpsname = data.cmdargtoken[1].val.string;
    
    FUNCTION_PARAMETER_STRUCT fps;
    fps.SMfd = -1;

    if (function_parameter_struct_connect(fpsname, &fps, 0) == -1) {
        printf("Error: cannot connect to FPS '%s'.\n", fpsname);
        return RETURN_SUCCESS;
    }

    struct termios oldt, newt;
    int selected = 0;
    
    // collect active params
    int active_pindices[1024];
    int num_params = 0;
    
    for (int pindex = 0; pindex < fps.md->NBparamMAX; pindex++) {
        if (fps.parray[pindex].fpflag & FPFLAG_USED) {
            if (num_params < 1024) {
               active_pindices[num_params++] = pindex;
            }
        }
    }

    tcgetattr(STDIN_FILENO, &oldt);
    newt = oldt;
    newt.c_lflag &= ~(ICANON | ECHO | ISIG);
    tcsetattr(STDIN_FILENO, TCSANOW, &newt);

    char error_msg[200] = {0};

    while(1) {
        if (selected >= num_params) selected = num_params - 1;
        if (selected < 0) selected = 0;

        printf("\033[2J\033[H"); // Clear screen
        printf("\033[1;36m>> Interactive FPS Parameter Editor : %s <<\033[0m\n\n", fpsname);
        
        // determine the display window
        int display_count = 20;
        int start_idx = selected - (display_count/2);
        if (start_idx < 0) start_idx = 0;
        if (start_idx + display_count > num_params) start_idx = num_params - display_count;
        if (start_idx < 0) start_idx = 0;
        
        // Render rows
        for (int i = start_idx; i < start_idx + display_count && i < num_params; i++) {
            int pidx = active_pindices[i];
            char valstring[200];
            if (fps.parray[pidx].type == FPTYPE_STREAMNAME) {
                snprintf(valstring, 200, "%s", fps.parray[pidx].val.string[0]);
            } else {
                functionparameter_GetParamValueString(&fps.parray[pidx], valstring, 200);
            }
            
            const char *display_keyword = fps.parray[pidx].keywordfull;
            int prefix_len = strlen(fps.md->name);
            if (strncmp(display_keyword, fps.md->name, prefix_len) == 0 && display_keyword[prefix_len] == '.') {
                display_keyword += prefix_len + 1;
            }
            
            if (i == selected) {
                printf("\033[1;33m> %-30s : %-20s  (%s)\033[0m\n", display_keyword, valstring, fps.parray[pidx].description);
            } else {
                printf("  %-30s : %-20s  (%s)\n", display_keyword, valstring, fps.parray[pidx].description);
            }
        }
        
        printf("\n\033[2m[Up/Down/PgUp/PgDn] Navigate  [Enter] Edit  [Esc/q] Quit\033[0m\n");
        if (error_msg[0]) {
            printf("\033[1;31mError: %s\033[0m\n", error_msg);
            error_msg[0] = '\0';
        }
        
        // Input loop
        int c = tui_readchar();
        if (c == 27) { // Escape seq
            if (tui_stdin_wait_ms(50)) {
                int b1 = tui_readchar();
                int b2 = tui_readchar();
                if (b1 == '[') {
                    if (b2 == 'A') selected--; // Up
                    else if (b2 == 'B') selected++; // Down
                    else if (b2 == '5') { tui_readchar(); selected -= 10; } // PgUp
                    else if (b2 == '6') { tui_readchar(); selected += 10; } // PgDn
                }
            } else {
                break; // Bare ESC — quit
            }
        } else if (c == 'q' || c == 'Q') {
            break;
        } else if (c == 3 || c == 4) { // Ctrl+C or Ctrl+D
            break;
        } else if (c == 10 || c == 13) {
            // Edit the selected parameter
            int pidx = active_pindices[selected];
            
            // disable raw mode to get input
            tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
            
            printf("\033[2J\033[H");
            printf("Editing parameter: \033[1;36m%s\033[0m\n", fps.parray[pidx].keywordfull);
            printf("Description: %s\n", fps.parray[pidx].description);
            
            char valstring[200];
            if (fps.parray[pidx].type == FPTYPE_STREAMNAME) {
                snprintf(valstring, 200, "%s", fps.parray[pidx].val.string[0]);
            } else {
                functionparameter_GetParamValueString(&fps.parray[pidx], valstring, 200);
            }
            printf("Current value: %s\n", valstring);
            printf("New value (leave empty to cancel): ");
            
            char inputbuf[256];
            if (fgets(inputbuf, sizeof(inputbuf), stdin) != NULL) {
                // strip newline
                int len = strlen(inputbuf);
                if (len > 0 && inputbuf[len-1] == '\n') inputbuf[len-1] = '\0';
                
                if (strlen(inputbuf) > 0) {
                    // Update parameter logic
                    int pindex = pidx;
                    int type = fps.parray[pindex].type;
                    int vOK = 1;
                    char *endptr;
                    long lval;
                    double fval;

                    switch(type) {
                        case FPTYPE_INT32:
                            lval = strtol(inputbuf, &endptr, 10);
                            if (*endptr != '\0') vOK = 0;
                            else fps.parray[pindex].val.i32[0] = (int32_t)lval;
                            break;
                        case FPTYPE_UINT32:
                            lval = strtol(inputbuf, &endptr, 10);
                            if (*endptr != '\0' || lval < 0) vOK = 0;
                            else fps.parray[pindex].val.ui32[0] = (uint32_t)lval;
                            break;
                        case FPTYPE_INT64:
                            lval = strtol(inputbuf, &endptr, 10);
                            if (*endptr != '\0') vOK = 0;
                            else fps.parray[pindex].val.i64[0] = (int64_t)lval;
                            break;
                        case FPTYPE_UINT64:
                            lval = strtol(inputbuf, &endptr, 10);
                            if (*endptr != '\0' || lval < 0) vOK = 0;
                            else fps.parray[pindex].val.ui64[0] = (uint64_t)lval;
                            break;
                        case FPTYPE_FLOAT32:
                            fval = strtod(inputbuf, &endptr);
                            if (*endptr != '\0') vOK = 0;
                            else fps.parray[pindex].val.f32[0] = (float)fval;
                            break;
                        case FPTYPE_FLOAT64:
                            fval = strtod(inputbuf, &endptr);
                            if (*endptr != '\0') vOK = 0;
                            else fps.parray[pindex].val.f64[0] = fval;
                            break;
                        case FPTYPE_STRING:
                        case FPTYPE_FILENAME:
                        case FPTYPE_FITSFILENAME:
                        case FPTYPE_EXECFILENAME:
                        case FPTYPE_DIRNAME:
                        case FPTYPE_STREAMNAME:
                        case FPTYPE_FPSNAME:
                            strncpy(fps.parray[pindex].val.string[0],
                                inputbuf,
                                FUNCTION_PARAMETER_STRMAXLEN-1);
                            break;
                        case FPTYPE_ONOFF:
                            if(strcasecmp(inputbuf, "ON") == 0 || strcmp(inputbuf, "1") == 0) {
                                fps.parray[pindex].fpflag |= FPFLAG_ONOFF;
                                fps.parray[pindex].val.i64[0] = 1;
                            } else if(strcasecmp(inputbuf, "OFF") == 0 || strcmp(inputbuf, "0") == 0) {
                                fps.parray[pindex].fpflag &= ~FPFLAG_ONOFF;
                                fps.parray[pindex].val.i64[0] = 0;
                            } else {
                                vOK = 0;
                            }
                            break;
                        case FPTYPE_TIMESPEC:
                            fval = strtod(inputbuf, &endptr);
                            if (*endptr != '\0') vOK = 0;
                            else {
                                struct timespec ts;
                                ts.tv_sec = (time_t)fval;
                                ts.tv_nsec = (long)((fval - (double)ts.tv_sec) * 1000000000.0);
                                if (ts.tv_nsec < 0) ts.tv_nsec = 0;
                                if (ts.tv_nsec >= 1000000000) ts.tv_nsec = 999999999;
                                
                                fps.parray[pindex].val.ts[0] = ts;
                            }
                            break;
                        default:
                            snprintf(error_msg, sizeof(error_msg), "Unsupported parameter type for editing.");
                            vOK = 0;
                    }

                    if (!vOK && error_msg[0] == '\0') {
                        snprintf(error_msg, sizeof(error_msg), "Invalid value format for the type.");
                    } else if (vOK) {
                        fps.parray[pindex].cnt0++;
                        fps.parray[pindex].value_cnt++;
                        fps.md->signal |= FUNCTION_PARAMETER_STRUCT_SIGNAL_UPDATE;
                    }
                }
            }
            
            // re-enable raw mode
            tcsetattr(STDIN_FILENO, TCSANOW, &newt);
        }
    }
    
    tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
    printf("\033[2J\033[H");
    
    function_parameter_struct_disconnect(&fps);
    return RETURN_SUCCESS;
}
