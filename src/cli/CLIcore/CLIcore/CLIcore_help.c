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

void print_milk_cli_help(void)
{
    printf("\n");
    printf(C_TITLE "========================================================\n" C_RST);
    printf(C_TITLE "                COMMAND LINE OPTIONS                    \n" C_RST);
    printf(C_TITLE "========================================================\n" C_RST);
    printf("\n");
    printf(C_CMD "  -h, --help      " C_RST "print this message and exit\n");
    printf(C_CMD "  -v, --version   " C_RST "print version and exit\n");
    printf(C_CMD "  -i, --info      " C_RST "print version, settings, info and exit\n");
    printf(C_CMD "  --verbose       " C_RST "be verbose\n");
    printf(C_CMD "  -d <level>      " C_RST "set debug level at startup\n");
    printf(C_CMD "  -o, --overwrite " C_RST "overwrite existing FITS files " C_NOTE "(USE WITH CAUTION)\n" C_RST);
    printf(C_CMD "  -e, --errorexit " C_RST "exit on error\n");
    printf(C_CMD "  -Z, --idle      " C_RST "only run process when X is idle\n");
    printf(C_CMD "  -A, --autocomplete " C_RST "enable autocomplete preview " C_NOTE "(ON by default)\n" C_RST);
    printf(C_CMD "  --no-autocomplete  " C_RST "disable inline autocomplete preview\n");
    printf(C_CMD "  --no-history-suggest " C_RST "disable history-based suggestions\n");
    printf(C_CMD "  --no-arg-hints     " C_RST "disable argument hint line\n");
    printf(C_CMD "  --no-fuzzy         " C_RST "disable fuzzy/substring matching\n");
    printf(C_CMD "  -f, --fifoflag  " C_RST "enable default fifo input\n");
    printf(C_CMD "  -F <fifoname>   " C_RST "specify custom fifo name\n");
    printf(C_CMD "  -s <file>       " C_RST "execute startup script\n");
    printf(C_CMD "  -n <name>       " C_RST "specify process name\n");
    printf(C_CMD "  -p <priority>   " C_RST "set RT priority (0-99)\n");

    printf("\n");
    printf(C_TITLE "========================================================\n" C_RST);
    printf(C_TITLE "                SYNTAX & INTERACTION                    \n" C_RST);
    printf(C_TITLE "========================================================\n" C_RST);
    printf("\n");
    printf(C_HDR "Syntax Rules:\n" C_RST);
    printf("  Spaces separate arguments. Use " C_BOLD "#" C_RST " for comments.\n");
    printf("  Example: " C_CMD "command arg1 arg2 # comment\n" C_RST);
    printf("\n");
    printf(C_HDR "Tab Completion & UX Features:\n" C_RST);
    printf("  1st arg: Match commands, then images, then files.\n");
    printf("  Subsequent: Match images, then files.\n");
    printf("  " C_NOTE "History:" C_RST " Commands are saved persistently across sessions (Up/Down).\n");
    printf("  " C_NOTE "Autocorrection:" C_RST " Mistyped commands suggest the closest match.\n");
    printf("  " C_NOTE "Fuzzy finding:" C_RST " " C_CMD "milk-fpsexec-list <keyword>" C_RST " filters matches.\n");
    printf("  " C_NOTE "Bash completion:" C_RST " Source scripts/milk-completion.sh for standard tab completion.\n");
    printf("\n");
    printf(C_HDR "Piping Commands:\n" C_RST);
    printf("  Commands can be piped to milk-cli via stdin:\n");
    printf("  " C_CMD "echo -e \"a=1\\nb=2\\nc=a+b\" | milk-cli\n" C_RST);
    printf("  Use " C_BOLD "\\n" C_RST " to separate multiple commands.\n");

    printf("\n");
    printf(C_TITLE "========================================================\n" C_RST);
    printf(C_TITLE "                 IMPORTANT COMMANDS                     \n" C_RST);
    printf(C_TITLE "========================================================\n" C_RST);
    printf("\n");
    printf(C_CMD "  ? / help        " C_RST "Show this overview\n");
    printf(C_CMD "  cmd? [cmd]      " C_RST "Help for specific command\n");
    printf(C_CMD "  m? [module]     " C_RST "List commands in module\n");
    printf(C_CMD "  h? [string]     " C_RST "Search command descriptions\n");
    printf(C_CMD "  ci              " C_RST "System info and memory usage\n");
    printf(C_CMD "  mem.listim      " C_RST "List images in memory\n");
    printf(C_CMD "  iofits.loadfits " C_RST "Load FITS file " C_NOTE "(requires CFITS)\n" C_RST);
    printf(C_CMD "  iofits.savefits " C_RST "Save FITS file " C_NOTE "(requires CFITS)\n" C_RST);
    printf(C_CMD "  quit / exit     " C_RST "Exit the milk shell\n");
    printf(C_CMD "  !<syscommand>   " C_RST "Execute shell command\n");

    printf("\n");
    printf(C_TITLE "========================================================\n" C_RST);
    printf(C_TITLE "                 SCRIPTING                              \n" C_RST);
    printf(C_TITLE "========================================================\n" C_RST);
    printf("\n");
    printf(C_HDR "Variables:\n" C_RST);
    printf("  " C_CMD "x=42" C_RST
           "              Set variable\n");
    printf("  " C_CMD "echo $x" C_RST
           "           Print variable\n");
    printf("  " C_CMD "echo ${x}" C_RST
           "         Braced form\n");
    printf("  " C_CMD "unset x" C_RST
           "           Remove variable\n");
    printf("  " C_CMD "vars" C_RST
           "              List all variables\n");
    printf("\n");
    printf(C_HDR "String Operations:\n" C_RST);
    printf("  " C_CMD "${#var}" C_RST
           "          String length\n");
    printf("  " C_CMD "${var:2:3}" C_RST
           "       Substring (offset:len)\n");
    printf("  " C_CMD "${var%%%%pat}" C_RST
           "     Strip longest suffix\n");
    printf("  " C_CMD "${var##pat}" C_RST
           "      Strip longest prefix\n");
    printf("\n");
    printf(C_HDR "Arrays:\n" C_RST);
    printf("  " C_CMD "arr=(a b c)" C_RST
           "      Create array\n");
    printf("  " C_CMD "${arr[0]}" C_RST
           "        Access element\n");
    printf("  " C_CMD "${arr[@]}" C_RST
           "        All elements\n");
    printf("  " C_CMD "${#arr[@]}" C_RST
           "       Array length\n");
    printf("\n");
    printf(C_HDR "Arithmetic:\n" C_RST);
    printf("  " C_CMD "y=$(( x + 5 ))" C_RST
           "  Integer +, -, *, /, %%\n");
    printf("\n");
    printf(C_HDR "FPS Parameter Access:\n" C_RST);
    printf("  " C_CMD "@fpsname.param" C_RST
           "    Read FPS parameter\n");
    printf("  " C_CMD "fpsset fps p v" C_RST
           "    Write FPS parameter\n");

    printf("\n");
    printf(C_TITLE "========================================================\n" C_RST);
    printf(C_TITLE "                 FLOW CONTROL                           \n" C_RST);
    printf(C_TITLE "========================================================\n" C_RST);
    printf("\n");
    printf(C_HDR "Conditionals:\n" C_RST);
    printf("  " C_CMD
           "if [ $x -gt 5 ]; then"
           C_RST "\n");
    printf("  " C_CMD "    echo big"
           C_RST "\n");
    printf("  " C_CMD "elif [ $x -gt 2 ]; then"
           C_RST "  ← cascading branch\n");
    printf("  " C_CMD "    echo medium"
           C_RST "\n");
    printf("  " C_CMD "else" C_RST "\n");
    printf("  " C_CMD "    echo small"
           C_RST "\n");
    printf("  " C_CMD "fi" C_RST "\n");
    printf("  Tests: "
           C_NOTE "-eq -ne -gt -ge -lt -le"
           C_RST " (numeric), "
           C_NOTE "= !=" C_RST
           " (string)\n");
    printf("  File tests: "
           C_NOTE "-f" C_RST
           " (file), "
           C_NOTE "-d" C_RST
           " (dir), "
           C_NOTE "-e" C_RST
           " (exists), "
           C_NOTE "-s" C_RST
           " (non-empty)\n");
    printf("  Negate: "
           C_CMD "[ ! expr ]" C_RST
           " logical NOT\n");
    printf("\n");
    printf(C_HDR "Loops:\n" C_RST);
    printf("  " C_CMD
           "while [ $n -lt 10 ]; do"
           C_RST " ... "
           C_CMD "done" C_RST "\n");
    printf("  " C_CMD
           "for x in a b c; do"
           C_RST " ... "
           C_CMD "done" C_RST "\n");
    printf("  " C_CMD "break" C_RST
           " exits loop, "
           C_CMD "continue" C_RST
           " next iter\n");
    printf("\n");
    printf(C_HDR "Case statement:\n" C_RST);
    printf("  " C_CMD
           "case $var in" C_RST "\n");
    printf("  " C_CMD
           "  yes) echo ok ;;" C_RST "\n");
    printf("  " C_CMD
           "  a|b) echo ab ;;" C_RST
           "  ← alternation\n");
    printf("  " C_CMD
           "  *) echo default ;;"
           C_RST "\n");
    printf("  " C_CMD "esac" C_RST "\n");
    printf("\n");
    printf(C_HDR "Functions:\n" C_RST);
    printf("  " C_CMD
           "function myfunc {" C_RST
           " ... "
           C_CMD "}" C_RST "\n");
    printf("  " C_CMD "myfunc arg1 arg2"
           C_RST "  call with "
           C_NOTE "$1..$9" C_RST
           " inside body\n");
    printf("  " C_CMD "return [val]" C_RST
           "      exit function, optionally"
           " set $?\n");
    printf("  Variables created inside a"
           " function are " C_NOTE "local"
           C_RST "\n");

    printf("\n");
    printf(C_TITLE "========================================================\n" C_RST);
    printf(C_TITLE "                 SCRIPT FILES                           \n" C_RST);
    printf(C_TITLE "========================================================\n" C_RST);
    printf("\n");
    printf(C_CMD "  source <file>     " C_RST
           "Execute a script file\n");
    printf(C_CMD "  . <file>          " C_RST
           "Same (dot-source)\n");
    printf(C_CMD "  include_once <f>  " C_RST
           "Source only once\n");
    printf(C_CMD "  savescript <file>  " C_RST
           "Save variables & functions\n");
    printf(C_CMD "  savehistory <file> " C_RST
           "Save command history\n");
    printf("  Startup: "
           C_CMD "milk-cli -s <file>" C_RST
           "  run script on launch\n");
    printf("  Auto-load: "
           C_CMD "~/.milkrc" C_RST
           " sourced at startup\n");
    printf("  Shebang: "
           C_NOTE "#!/usr/bin/env milk-cli -s"
           C_RST "\n");
    printf("\n");
    printf(C_HDR "Built-in Commands:\n" C_RST);
    printf(C_CMD "  sleep <sec>       " C_RST
           "Pause (float-capable)\n");
    printf(C_CMD "  printf \"fmt\" a.. " C_RST
           "Formatted output\n");
    printf(C_CMD "  read [-p p] var   " C_RST
           "Read line from stdin\n");
    printf("\n");
    printf(C_HDR "Logical Operators:\n" C_RST);
    printf("  " C_CMD
           "cmd1 && cmd2" C_RST
           "  run cmd2 if cmd1 succeeds\n");
    printf("  " C_CMD
           "cmd1 || cmd2" C_RST
           "  run cmd2 if cmd1 fails\n");
    printf("\n");
    printf(C_HDR "Brace Expansion:\n" C_RST);
    printf("  " C_CMD
           "{1..5}" C_RST
           " → 1 2 3 4 5\n");
    printf("  " C_CMD
           "{0..10..2}" C_RST
           " → 0 2 4 6 8 10\n");
    printf("\n");
    printf(C_HDR "Heredocs:\n" C_RST);
    printf("  " C_CMD "VAR=<<EOF" C_RST
           "\n");
    printf("  " C_CMD "  line 1" C_RST
           "\n");
    printf("  " C_CMD "  line 2" C_RST
           "\n");
    printf("  " C_CMD "EOF" C_RST
           "  → multi-line var\n");
    printf("\n");
    printf(C_HDR "Stream Events:\n" C_RST);
    printf("  " C_CMD
           "on_update <stream> { cmd }"
           C_RST "\n");
    printf("  Waits for stream update,"
           " then runs cmd\n");
    printf("\n");

    return;
}

errno_t help()
{
    print_milk_cli_help();

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
    newt.c_lflag &= ~(ICANON | ECHO);
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
        char c = getchar();
        if (c == 27) { // Escape seq
            char seq[2];
            seq[0] = getchar();
            seq[1] = getchar();
            if (seq[0] == '[') {
                if (seq[1] == 'A') selected--; // Up
                else if (seq[1] == 'B') selected++; // Down
                else if (seq[1] == '5') { selected -= 10; getchar(); } // PgUp
                else if (seq[1] == '6') { selected += 10; getchar(); } // PgDn
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
        } else if (c >= 32 && c <= 126 && query_len < 127) { // Printable chars
            query[query_len++] = c;
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
    newt.c_lflag &= ~(ICANON | ECHO);
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
        char c = getchar();
        if (c == 27) { // Escape seq
            // check if there is an escape sequence waiting
            int byteswaiting;
            ioctl(STDIN_FILENO, FIONREAD, &byteswaiting);
            if (byteswaiting > 0) {
                char seq[2];
                seq[0] = getchar();
                seq[1] = getchar();
                if (seq[0] == '[') {
                    if (seq[1] == 'A') selected--; // Up
                    else if (seq[1] == 'B') selected++; // Down
                    else if (seq[1] == '5') { selected -= 10; getchar(); } // PgUp
                    else if (seq[1] == '6') { selected += 10; getchar(); } // PgDn
                }
            } else {
                selected = -1;
                break; // Escape key alone
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
        } else if (c >= 32 && c <= 126 && query_len < 127) { // Printable chars
            query[query_len++] = c;
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
    newt.c_lflag &= ~(ICANON | ECHO);
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
        char c = getchar();
        if (c == 27) { // Escape seq
            // check if there is an escape sequence waiting
            int byteswaiting;
            ioctl(STDIN_FILENO, FIONREAD, &byteswaiting);
            if (byteswaiting > 0) {
                char seq[2];
                seq[0] = getchar();
                seq[1] = getchar();
                if (seq[0] == '[') {
                    if (seq[1] == 'A') selected--; // Up
                    else if (seq[1] == 'B') selected++; // Down
                    else if (seq[1] == '5') { selected -= 10; getchar(); } // PgUp
                    else if (seq[1] == '6') { selected += 10; getchar(); } // PgDn
                }
            } else {
                break; // Escape key alone
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
