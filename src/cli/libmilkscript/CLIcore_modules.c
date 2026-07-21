// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file CLIcore_modules.c
 *
 * @brief Modules functions
 *
 */

#include <dirent.h>
#include <dlfcn.h>
#include <stdio.h>
#include <string.h>

#include "CLIcore.h"
#include "milkdata.h"

#define KNRM "\x1B[0m"
#define KRED "\x1B[31m"
#define KGRN "\x1B[32m"
#define KYEL "\x1B[33m"
#define KBLU "\x1B[34m"
#define KMAG "\x1B[35m"
#define KCYN "\x1B[36m"
#define KWHT "\x1B[37m"
#define KRES "\033[0m"

// local valiables to keep track of library last loaded
static int   DLib_index;
static void *DLib_handle[1000];
static char  libnameloaded[STRINGMAXLEN_MODULE_SOFILENAME];

/**
 * @brief Load a shared library via dlopen.
 *
 * Opens @libname with RTLD_LAZY|RTLD_GLOBAL.
 * The library's init function (if any) is called
 * automatically, which triggers RegisterModule()
 * to add the module to the command table.
 *
 * Skips loading if the .so is already loaded
 * (checked via sofilename match).
 *
 * @param libname  Full path to the .so file
 */
errno_t load_sharedobj(const char *__restrict libname)
{
    DEBUG_TRACE_FSTART();

    DEBUG_TRACEPOINT("[%5d] Loading shared object \"%s\"", DLib_index, libname);
    strncpy(libnameloaded, libname, STRINGMAXLEN_MODULE_SOFILENAME - 1);

    // check if already loaded
    DEBUG_TRACEPOINT("--- %ld modules loaded ---", data.NBmodule);
    int mmatch = -1;
    for (int m = 0; m < data.NBmodule; m++)
    {
        //printf("  [%03d] %s\n", m, data.module[m].sofilename);
        if (strcmp(libnameloaded, data.module[m].sofilename) == 0)
        {
            mmatch = m;
        }
    }
    if (mmatch > -1)
    {
        printf("    Shared object %s already loaded - no action taken\n", libnameloaded);
        DEBUG_TRACE_FEXIT();
        return RETURN_FAILURE;
    }

    void *handle = dlopen(libname, RTLD_LAZY | RTLD_GLOBAL);
    if (!handle)
    {
        printf(KRED "FAILED TO LOAD : %s\n" KRES, libname);
        const char *errstr = dlerror();
        if (errstr)
        {
            fprintf(stderr, KRED "%s\n" KRES, errstr);
        }
        DEBUG_TRACE_FEXIT();
        return RETURN_FAILURE;
    }

    dlerror();
    if (!getenv("MILK_QUIET") && dcquiet == 0)
    {
        printf(KGRN "   LOADED : %s\n" KRES, libname);
    }
    DLib_handle[DLib_index] = handle;
    // increment number of libs dynamically loaded
    DLib_index++;

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}

errno_t newstyle_milk_module_registration(void *handle, const char *libname)
{
    /* New-style modules export __milk_module_info.
    We perform the registration code that priorly belonged to INIT_MILK_MODULE macro.

    Except that now this code lives within the CLI/script so, not the module so.
    */

    MILK_MODULE_INFO *info = (MILK_MODULE_INFO *) dlsym(handle, "__milk_module_info");
    if (info == NULL || info->mod_registered == 1)
    {
        return RETURN_SUCCESS;
    }

    /* Load declared deps before registering */
    if (info->deps)
    {
        for (int _di = 0; info->deps[_di]; _di++)
        {
            load_module_shared(info->deps[_di]);
        }
    }

    strncpy(data.moduleshortname_default, info->shortname_default,
            STRINGMAXLEN_MODULE_SHORTNAME - 1);
    strncpy(data.moduledatestring, info->date_string, STRINGMAXLEN_MODULE_DATESTRING - 1);
    strncpy(data.moduletimestring, info->time_string, STRINGMAXLEN_MODULE_TIMESTRING - 1);
    strncpy(data.modulename, info->name, STRINGMAXLEN_MODULE_NAME - 1);
    data.module_nbdep = 0;

    RegisterModule(info->source_file, info->package, info->description, info->version_major,
                   info->version_minor, info->version_patch);

    /* Stamp sofilename so load_module_shared
                 * can find this slot by Case-1 match. */
    strncpy(data.module[data.moduleindex].sofilename, libname, STRINGMAXLEN_MODULE_SOFILENAME - 1);

    if (info->reg_call)
    {
        (*(info->reg_call))();
    }

    strncpy(data.modulename, "", STRINGMAXLEN_MODULE_NAME - 1);
    strncpy(data.moduleshortname_default, "", STRINGMAXLEN_MODULE_SHORTNAME - 1);
    strncpy(data.moduleshortname, "", STRINGMAXLEN_MODULE_SHORTNAME - 1);

    info->mod_registered = 1;

    return RETURN_SUCCESS;
}


/**
 * @brief Load a module by name from the install
 *        directory.
 *
 * Constructs the path $MILK_INSTALLDIR/lib/lib<name>.so
 * and calls load_sharedobj(). Sets the module type
 * to MODULE_TYPE_CUSTOMLOAD and records the .so
 * filename and load name in the module struct.
 *
 * @param modulename  Module base name or absolute path
 */
errno_t load_module_shared(const char *__restrict modulename)
{
    DEBUG_TRACE_FSTART();
    char libname[STRINGMAXLEN_MODULE_SOFILENAME];

    // make locacl copy of module name
    char modulenameLC[STRINGMAXLEN_MODULE_SOFILENAME];

    {
        int slen = snprintf(modulenameLC, STRINGMAXLEN_MODULE_SOFILENAME, "%s", modulename);
        if (slen < 1)
        {
            PRINT_ERROR("snprintf wrote <1 char");
            abort(); // can't handle this error any other way
        }
        if (slen >= STRINGMAXLEN_MODULE_SOFILENAME)
        {
            PRINT_ERROR("snprintf string truncation");
            abort(); // can't handle this error any other way
        }
    }

    if (modulename[0] == '/')
    {
        strncpy(libname, modulename, STRINGMAXLEN_MODULE_SOFILENAME - 1);
    }
    else
    {
        // Assemble absolute path module filename
        //printf("Searching for shared object in directory MILK_INSTALLDIR/lib : %s/lib\n", getenv("MILK_INSTALLDIR"));
        DEBUG_TRACEPOINT("Searching for shared object in directory [dcinstalldir]/lib : "
                         "%s/lib",
                         dcinstalldir);

        {
            int slen = snprintf(libname, STRINGMAXLEN_MODULE_SOFILENAME, "%.100s/lib/lib%.50s.so",
                                dcinstalldir, modulenameLC);
            if (slen < 1)
            {
                PRINT_ERROR("snprintf wrote <1 char");
                abort(); // can't handle this error any other way
            }
            if (slen >= STRINGMAXLEN_MODULE_SOFILENAME)
            {
                PRINT_ERROR("snprintf string truncation");
                abort(); // can't handle this error any other way
            }
        }
    }

    DEBUG_TRACEPOINT("libname = %s", libname);

    DEBUG_TRACEPOINT("[%5d] Loading shared object \"%s\"", DLib_index, libname);


    strncpy(data.moduleloadname, modulenameLC, STRINGMAXLEN_MODULE_LOADNAME - 1);
    strncpy(data.modulesofilename, libname, STRINGMAXLEN_MODULE_SOFILENAME - 1);


    if (load_sharedobj(libname) == RETURN_SUCCESS)
    {
        // We inherit DLib_index AFTER INCREMENT from load_sharedobj (which has no recursive calls)
        // Now, WARNING, this function may recurse into other calls to load_module_shared
        newstyle_milk_module_registration(DLib_handle[DLib_index - 1], libname);
        // TODO fix load_module_shared_local to do the same new-style init
    }

    // Find the correct module slot for metadata.
    //
    // We cannot blindly use data.moduleindex because dlopen() may have transitively loaded other
    // libraries whose constructors changed it.
    // If the .so was already loaded (transitive dep of a prior dlopen), the constructor does not
    // re-run and moduleindex is stale.
    //
    // Strategy:
    //  1. Match by sofilename (set by a prior call)
    //  2. Among newly registered modules, match by
    //     module name as suffix of the load name
    //  3. Fall back to data.moduleindex

    int target_idx = -1;

    // Case 1: sofilename already set by prior call
    for (int m = 0; m < data.NBmodule; m++)
    {
        if (strcmp(data.module[m].sofilename, libname) == 0)
        {
            target_idx = m;
            break;
        }
    }

    // Case 2: match module name as suffix of load
    // name. Handles modules loaded transitively by
    // an earlier dlopen (sofilename still empty).
    // E.g. loadname "milkpsf" matches name "psf".
    if (target_idx < 0)
    {
        int llen = (int) strlen(modulenameLC);
        for (long m = 0; m < data.NBmodule; m++)
        {
            const char *mname = data.module[m].name;
            int         mlen  = (int) strlen(mname);
            if (mlen > 0 && mlen <= llen && strcmp(modulenameLC + llen - mlen, mname) == 0)
            {
                target_idx = (int) m;
                break;
            }
        }
    }

    // Case 3: fallback
    if (target_idx < 0)
    {
        target_idx = (int) data.moduleindex;
    }

    data.module[target_idx].type = MODULE_TYPE_CUSTOMLOAD;
    strncpy(data.module[target_idx].sofilename, libname, STRINGMAXLEN_MODULE_SOFILENAME - 1);
    strncpy(data.module[target_idx].loadname, modulenameLC, STRINGMAXLEN_MODULE_LOADNAME - 1);

    DEBUG_TRACE_FEXIT();
    return target_idx >= 0 ? RETURN_SUCCESS : RETURN_FAILURE;
}


/**
 * @brief Load all .so files from the ./milklib/
 *        directory.
 *
 * Scans the local milklib/ directory for shared
 * objects and loads them with multiple linker passes
 * (up to 4) to resolve inter-library dependencies.
 * A pass succeeds when all .so files load without
 * dlopen errors.
 */
errno_t load_module_shared_local()
{
    DEBUG_TRACE_FSTART();

    char libname[STRINGMAXLEN_FULLFILENAME + STRINGMAXLEN_DIRNAME];
    char dirname[STRINGMAXLEN_DIRNAME];

    char cwd[STRINGMAXLEN_DIRNAME];
    if (getcwd(cwd, sizeof(cwd)) == NULL)
    {
        PRINT_ERROR("getcwd failed");
        return RETURN_FAILURE;
    }

    WRITE_DIRNAME(dirname, "./milklib");

    if (dcquiet == 0)
    {
        printf("load modules from directory %s\n", dirname);
    }
    DIR           *d = opendir(dirname);
    struct dirent *dir;
    if (d == NULL)
    {
        if (dcquiet == 0)
        {
            printf("--> directory not found.\n");
        }
        return RETURN_SUCCESS;
    }
    int itermax             = 4; // number of passes
    int any_so_link_failure = 0;
    for (int iter = 0; iter < itermax; ++iter)
    {
        any_so_link_failure = 0;

        while ((dir = readdir(d)) != NULL) // iterate .so files
        {
            char *dot = strrchr(dir->d_name, '.');
            if (dot == NULL || strcmp(dot, ".so") != 0)
            {
                continue;
            }

            // after
            snprintf(libname, sizeof(libname), "%s/lib/%s", cwd, dir->d_name);
            //printf("%02d   (re-?) LOADING shared object  %40s -> %s\n", DLib_index, dir->d_name, libname);
            //fflush(stdout);

            // libname should be an abspath and start with a /
            // load_module_shared handles that.
            if (load_module_shared(libname) != RETURN_SUCCESS)
            {
                any_so_link_failure = 1;
            }
        }

        if (any_so_link_failure == 0)
        {
            iter > 0 ? printf(KGRN "        Linker pass #%d successful\n" KRES, iter) : 0;
            break;
        }
    }

    closedir(d);

    if (any_so_link_failure == 1)
    {
        printf("Some libraries could not be loaded -> EXITING\n");
        return RETURN_FAILURE;
    }
    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}


/**
 * @brief Register a module in the global module
 *        table.
 *
 * Called from each module's __attribute__((constructor))
 * init function during dlopen. Records the module
 * name, package, version, short name, and build
 * timestamps in data.module[].
 *
 * If the module exports a NULL-sentinel function-pointer
 * array named @c __reg_calls, each entry is invoked to
 * register the module's CLI commands.
 *
 * @param FileName      Source filename of the module
 * @param PackageName   Package the module belongs to
 * @param InfoString    One-line description
 * @param versionmajor  Major version number
 * @param versionminor  Minor version number
 * @param versionpatch  Patch version number
 */
errno_t RegisterModule(const char *__restrict FileName,
                       const char *__restrict PackageName,
                       const char *__restrict InfoString,
                       int versionmajor,
                       int versionminor,
                       int versionpatch)
{
    DEBUG_TRACE_FSTART();

    /* On the very first call, scan /proc/self/cmdline for -q /
     * --quiet / -h1 / --help-oneline.
     * This fires before main() when called from a
     * __attribute__((constructor)) in a linked .so, so it is
     * the only reliable place to intercept a command-line flag
     * at constructor time. */
    /*
    NOTE: deprecated requirement since this will NOT be called from constructors anymore,
    since we fetch a struct from modules and the "constructor" is deferred to code that is
    here and executes synchronously.
    */
    static int quiet_scanned = 0;
    if (!quiet_scanned)
    {
        quiet_scanned = 1;
        if (!getenv("MILK_QUIET"))
        {
            FILE *fp = fopen("/proc/self/cmdline", "r");
            if (fp)
            {
                char   buf[4096];
                size_t n = fread(buf, 1, sizeof(buf) - 1, fp);
                fclose(fp);
                buf[n] = '\0';

                /* Skip argv[0] */
                size_t i = 0;
                while (i < n && buf[i] != '\0')
                {
                    i++;
                }
                i++;

                while (i < n)
                {
                    const char *tok = buf + i;
                    /* Stop at bare '-' (stdin) or non-option */
                    if (tok[0] != '-' || tok[1] == '\0')
                    {
                        break;
                    }

                    if (strcmp(tok, "--quiet") == 0 || strcmp(tok, "-h1") == 0 ||
                        strcmp(tok, "--help-oneline") == 0)
                    {
                        setenv("MILK_QUIET", "1", 0);
                        milk_data.quiet = 1;
                        break;
                    }

                    /* Short option bundle: -q or -qE or -xqE etc.
                     * Stop interpreting as options when we hit '--' */
                    if (tok[1] != '-')
                    {
                        for (int k = 1; tok[k] != '\0'; k++)
                        {
                            if (tok[k] == 'q')
                            {
                                setenv("MILK_QUIET", "1", 0);
                                milk_data.quiet = 1;
                                goto quiet_found; /* break both loops */
                            }
                        }
                    }

                    while (i < n && buf[i] != '\0')
                    {
                        i++;
                    }
                    i++;
                }
            quiet_found:;
            }
        }
    } // if (!quiet_scanned)


    int OKmsg = 0;

    int moduleindex  = data.NBmodule;
    data.moduleindex = moduleindex; // current module index

    data.NBmodule++;


    if (strlen(data.modulename) == 0)
    {
        strncpy(data.module[moduleindex].name, "???", STRINGMAXLEN_MODULE_NAME - 1);
    }
    else
    {
        strncpy(data.module[moduleindex].name, data.modulename, STRINGMAXLEN_MODULE_NAME - 1);
    }

    int stringlen = strlen(data.moduleshortname);
    if (stringlen == 0)
    {
        // if no shortname provided, try to use default
        if (strlen(data.moduleshortname_default) > 0)
        {
            // otherwise, construct call key as <shortname_default>.<CLIkey>
            strncpy(data.moduleshortname, data.moduleshortname_default,
                    STRINGMAXLEN_MODULE_SHORTNAME - 1);
        }
    }

    strncpy(data.module[moduleindex].package, PackageName, STRINGMAXLEN_MODULE_PACKAGENAME - 1);
    strncpy(data.module[moduleindex].info, InfoString, STRINGMAXLEN_MODULE_INFOSTRING - 1);

    strncpy(data.module[moduleindex].shortname, data.moduleshortname,
            STRINGMAXLEN_MODULE_SHORTNAME - 1);

    strncpy(data.module[moduleindex].datestring, data.moduledatestring,
            STRINGMAXLEN_MODULE_DATESTRING - 1);
    strncpy(data.module[moduleindex].timestring, data.moduletimestring,
            STRINGMAXLEN_MODULE_TIMESTRING - 1);

    data.module[moduleindex].versionmajor = versionmajor;
    data.module[moduleindex].versionminor = versionminor;
    data.module[moduleindex].versionpatch = versionpatch;

    data.module[moduleindex].type = data.moduletype;


    //printf("--- libnameloaded : %s\n", libnameloaded);


    if (dcprogstatus == 0)
    {
        OKmsg = 1;
        if (!getenv("MILK_QUIET") && dcquiet == 0)
        {
            printf(".");
        }
        //	printf("  %02ld  LOADING %10s  module %40s\n", data.NBmodule, PackageName, FileName);
        //	fflush(stdout);
    }

    if (dcprogstatus == 1)
    {
        OKmsg = 1;
        DEBUG_TRACEPOINT("  %02d  Found unloaded shared object in ./libs/ -> LOADING "
                         "%10s  module %40s",
                         moduleindex, PackageName, FileName);
        fflush(stdout);
    }

    if (OKmsg == 0)
    {
        printf("  %02d  ERROR: module load requested outside of normal step "
               "-> LOADING %10s  module %40s",
               moduleindex, PackageName, FileName);
        fflush(stdout);
    }


    // default
    // may be overridden by load_module_shared
    //
    //data.moduletype = MODULE_TYPE_STARTUP;

    data.module[data.moduleindex].type = MODULE_TYPE_STARTUP;

    //strncpy(data.modulesofilename, "", STRINGMAXLEN_MODULE_SOFILENAME - 1);
    if (data.module[data.moduleindex].sofilename[0] != '/')
    {
        strncpy(data.module[data.moduleindex].sofilename, "", STRINGMAXLEN_MODULE_SOFILENAME - 1);
    }

    //strncpy(data.modulesofilename, "", STRINGMAXLEN_MODULE_SOFILENAME - 1);
    strncpy(data.module[data.moduleindex].loadname, "", STRINGMAXLEN_MODULE_LOADNAME - 1);

    // Copy dependency info from transient DATA fields
    data.module[data.moduleindex].nbdep = data.module_nbdep;
    for (int di = 0; di < data.module_nbdep; di++)
    {
        strncpy(data.module[data.moduleindex].depname[di], data.module_depname[di],
                STRINGMAXLEN_MODULE_LOADNAME - 1);
    }
    // Reset transient dep storage
    data.module_nbdep = 0;

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}


/**
 * @brief Register a CLI command (legacy API).
 *
 * Deprecated in favor of RegisterCLIcmd().
 * Stores the command key, source file, function
 * pointer, info/syntax/example strings, and C-call
 * prototype in data.cmd[] at index data.NBcmd.
 *
 * The command key is prefixed with the module's
 * shortname (if set) to form namespace.command.
 *
 * @return New data.NBcmd count
 */
uint32_t RegisterCLIcommand(const char *__restrict CLIkey,
                            const char *__restrict CLImodulesrc,
                            errno_t (*CLIfptr)(),
                            const char *__restrict CLIinfo,
                            const char *__restrict CLIsyntax,
                            const char *__restrict CLIexample,
                            const char *__restrict CLICcall)
{
    DEBUG_TRACE_FSTART();

    DEBUG_TRACEPOINT("FARG CLIkey %s -> command index %u / %d", CLIkey, data.NBcmd,
                     DATA_NB_MAX_COMMAND);

    data.cmd[data.NBcmd].moduleindex = data.moduleindex;

    if (data.cmd[data.NBcmd].moduleindex == -1)
    {
        strncpy(data.cmd[data.NBcmd].module, "MAIN", STRINGMAXLEN_MODULE_NAME - 1);
        strncpy(data.cmd[data.NBcmd].key, CLIkey, STRINGMAXLEN_CMD_KEY - 1);
    }
    else
    {
        if (strlen(data.module[data.moduleindex].shortname) == 0)
        {
            strncpy(data.cmd[data.NBcmd].key, CLIkey, STRINGMAXLEN_CMD_KEY - 1);
        }
        else
        {
            // otherwise, construct call key as <shortname>.<CLIkey>
            snprintf(data.cmd[data.NBcmd].key, STRINGMAXLEN_CMD_KEY, "%.30s.%.30s",
                     data.module[data.moduleindex].shortname, CLIkey);
        }
    }

    DEBUG_TRACEPOINT("set module name");
    if (strlen(data.modulename) == 0)
    {
        strncpy(data.cmd[data.NBcmd].module, "unknown", STRINGMAXLEN_MODULE_NAME - 1);
    }
    else
    {
        strncpy(data.cmd[data.NBcmd].module, data.modulename, STRINGMAXLEN_MODULE_NAME - 1);
    }

    DEBUG_TRACEPOINT("load function data");

    strncpy(data.cmd[data.NBcmd].srcfile, CLImodulesrc, STRINGMAXLEN_CMD_SRCFILE - 1);

    data.cmd[data.NBcmd].fp = CLIfptr;

    strncpy(data.cmd[data.NBcmd].info, CLIinfo, STRINGMAXLEN_CMD_INFO - 1);

    strncpy(data.cmd[data.NBcmd].syntax, CLIsyntax, STRINGMAXLEN_CMD_SYNTAX - 1);

    strncpy(data.cmd[data.NBcmd].example, CLIexample, STRINGMAXLEN_CMD_EXAMPLE - 1);

    strncpy(data.cmd[data.NBcmd].Ccall, CLICcall, STRINGMAXLEN_CMD_CCALL - 1);

    data.cmd[data.NBcmd].nbarg = 0;

    // Set defaults
    data.cmd[data.NBcmd].cmdsettings.procinfo_loopcntMax    = 1;
    data.cmd[data.NBcmd].cmdsettings.procinfo_MeasureTiming = 1;

    data.NBcmd++;

    DEBUG_TRACEPOINT("Done1");

    DEBUG_TRACE_FEXIT();

    DEBUG_TRACEPOINT("NBcmd = %u", data.NBcmd);

    return (data.NBcmd);
}


/**
 * @brief Register a CLI command (current API).
 *
 * Replaces the legacy RegisterCLIcommand().
 * Uses a CLICMDDATA struct to define the command
 * key, source file, description, argument
 * definitions, and flags.
 *
 * Automatically builds the CLI syntax and example
 * strings from the argument definitions, and
 * initializes per-command processinfo settings.
 *
 * @param CLIcmddata  Command definition struct
 * @param CLIfptr     Function pointer to execute
 * @return Index of the newly registered command
 */
uint32_t RegisterCLIcmd(CLICMDDATA CLIcmddata, errno_t (*CLIfptr)())
{
    // Command registration logic
    DEBUG_TRACE_FSTART();

    /* Guard against constructor ordering race:
     * per-command __attribute__((constructor))
     * may not have run yet, leaving key empty.
     */
    if (CLIcmddata.key[0] == '\0')
    {
        DEBUG_TRACE_FEXIT();
        return data.NBcmd;
    }

    data.cmd[data.NBcmd].moduleindex = data.moduleindex;
    if (data.cmd[data.NBcmd].moduleindex == -1)
    {
        strncpy(data.cmd[data.NBcmd].module, "MAIN", STRINGMAXLEN_MODULE_NAME - 1);
        strncpy(data.cmd[data.NBcmd].key, CLIcmddata.key, STRINGMAXLEN_CMD_KEY - 1);
    }
    else
    {
        if (strlen(data.module[data.moduleindex].shortname) == 0)
        {
            strncpy(data.cmd[data.NBcmd].key, CLIcmddata.key, STRINGMAXLEN_CMD_KEY);
        }
        else
        {
            // otherwise, construct call key as <shortname>.<CLIkey>
            int slen = snprintf(data.cmd[data.NBcmd].key, STRINGMAXLEN_CMD_KEY, "%.30s.%.30s",
                                data.module[data.moduleindex].shortname, CLIcmddata.key);
            if (slen < 1)
            {
                PRINT_ERROR("failed to write call key");
                abort();
            }
            if (slen >= STRINGMAXLEN_CMD_KEY)
            {
                PRINT_ERROR("call key string too long");
                abort();
            }
        }
    }

    if (strlen(data.modulename) == 0)
    {
        strncpy(data.cmd[data.NBcmd].module, "unknown", STRINGMAXLEN_MODULE_NAME - 1);
    }
    else
    {
        strncpy(data.cmd[data.NBcmd].module, data.modulename, STRINGMAXLEN_MODULE_NAME - 1);
    }


    DEBUG_TRACEPOINT("settingsrcfile to %s", CLIcmddata.sourcefilename);
    strncpy(data.cmd[data.NBcmd].srcfile, CLIcmddata.sourcefilename, STRINGMAXLEN_CMD_SRCFILE - 1);
    data.cmd[data.NBcmd].fp = CLIfptr;
    strncpy(data.cmd[data.NBcmd].info, CLIcmddata.description, STRINGMAXLEN_CMD_INFO - 1);


    // assemble argument syntax string for help
    char          argstring[STRINGMAXLEN_CMD_SYNTAX];
    CLICMDARGDEF *farg_visible = NULL;
    if (CLIcmddata.nbarg > 0)
    {
        farg_visible = (CLICMDARGDEF *) calloc(CLIcmddata.nbarg, sizeof(CLICMDARGDEF));
        if (farg_visible == NULL)
        {
            PRINT_ERROR("calloc failed for farg_visible");
            abort();
        }
    }
    int nbarg_visible = 0;
    for (int argi = 0; argi < CLIcmddata.nbarg; argi++)
    {
        if (CLIcmddata.funcfpscliarg[argi].fpflag & FPFLAG_PRIMARY_CLI_INPUT)
        {
            farg_visible[nbarg_visible] = CLIcmddata.funcfpscliarg[argi];
            nbarg_visible++;
        }
    }

    CLIhelp_make_argstring(farg_visible, nbarg_visible, argstring);
    strncpy(data.cmd[data.NBcmd].syntax, argstring, STRINGMAXLEN_CMD_SYNTAX - 1);


    // assemble example string for help
    char cmdexamplestring[STRINGMAXLEN_CMD_EXAMPLE];
    CLIhelp_make_cmdexamplestring(farg_visible, nbarg_visible, CLIcmddata.key, cmdexamplestring);
    strncpy(data.cmd[data.NBcmd].example, cmdexamplestring, STRINGMAXLEN_CMD_EXAMPLE - 1);

    free(farg_visible);


    strncpy(data.cmd[data.NBcmd].Ccall, "--callstring--", STRINGMAXLEN_CMD_CCALL - 1);


    DEBUG_TRACEPOINT("define arguments to CLI function from content of "
                     "CLIcmddata.funcfpscliarg");
    data.cmd[data.NBcmd].nbarg = 0; // count only primary mandatory arguments
    for (int argi = 0; argi < CLIcmddata.nbarg; argi++)
    {
        if (CLIcmddata.funcfpscliarg[argi].fpflag & FPFLAG_PRIMARY_CLI_INPUT)
        {
            data.cmd[data.NBcmd].nbarg++;
        }
    }

    // Still allocate the full array for all parameters (including hidden ones)
    data.cmd[data.NBcmd].nbparam = CLIcmddata.nbarg;
    if (CLIcmddata.nbarg > 0)
    {
        data.cmd[data.NBcmd].argdata =
            (CLICMDARGDATA *) calloc(CLIcmddata.nbarg, sizeof(CLICMDARGDATA));
        if (data.cmd[data.NBcmd].argdata == NULL)
        {
            PRINT_ERROR("calloc failed for argdata");
            free(farg_visible);
            abort();
        }

        for (int argi = 0; argi < CLIcmddata.nbarg; argi++)
        {
            data.cmd[data.NBcmd].argdata[argi].type   = CLIcmddata.funcfpscliarg[argi].type;
            data.cmd[data.NBcmd].argdata[argi].fpflag = CLIcmddata.funcfpscliarg[argi].fpflag;

            strncpy(data.cmd[data.NBcmd].argdata[argi].descr, CLIcmddata.funcfpscliarg[argi].descr,
                    STRINGMAXLEN_FPSCLIARG_DESCR - 1);

            strncpy(data.cmd[data.NBcmd].argdata[argi].fpstag,
                    CLIcmddata.funcfpscliarg[argi].fpstag, STRINGMAXLEN_FPSCLIARG_TAG - 1);

            strncpy(data.cmd[data.NBcmd].argdata[argi].example,
                    CLIcmddata.funcfpscliarg[argi].example, STRINGMAXLEN_FPSCLIARG_EXAMPLE - 1);

            // Set default values
            switch (data.cmd[data.NBcmd].argdata[argi].type)
            {
            case FPTYPE_FLOAT32:
                data.cmd[data.NBcmd].argdata[argi].val.f32 =
                    atof(CLIcmddata.funcfpscliarg[argi].example);
                break;

            case FPTYPE_FLOAT64:
                data.cmd[data.NBcmd].argdata[argi].val.f64 =
                    atof(CLIcmddata.funcfpscliarg[argi].example);
                break;

            case FPTYPE_INT32:
                data.cmd[data.NBcmd].argdata[argi].val.i32 =
                    (int32_t) atol(CLIcmddata.funcfpscliarg[argi].example);
                break;

            case FPTYPE_UINT32:
                data.cmd[data.NBcmd].argdata[argi].val.ui32 =
                    (uint32_t) atol(CLIcmddata.funcfpscliarg[argi].example);
                break;

            case FPTYPE_INT64:
                data.cmd[data.NBcmd].argdata[argi].val.i64 =
                    (int64_t) atol(CLIcmddata.funcfpscliarg[argi].example);
                break;

            case FPTYPE_UINT64:
                data.cmd[data.NBcmd].argdata[argi].val.ui64 =
                    (uint64_t) atol(CLIcmddata.funcfpscliarg[argi].example);
                break;

            case FPTYPE_ONOFF:
                data.cmd[data.NBcmd].argdata[argi].val.ui64 =
                    (int64_t) atol(CLIcmddata.funcfpscliarg[argi].example);
                break;

            case FPTYPE_STRING:
            case FPTYPE_STRING_NOT_STREAM:
            case FPTYPE_STREAMNAME:
            case FPTYPE_FILENAME:
            case FPTYPE_FITSFILENAME:
            case FPTYPE_FPSNAME:
            case FPTYPE_DIRNAME:
            case FPTYPE_EXECFILENAME:
                strncpy(data.cmd[data.NBcmd].argdata[argi].val.s,
                        CLIcmddata.funcfpscliarg[argi].example, STRINGMAXLEN_CLICMDARG - 1);
                break;
            }
        }
    }

    DEBUG_TRACEPOINT("define CLI function flags from content of CLIcmddata.flags");


    data.cmd[data.NBcmd].cmdsettings.flags = CLIcmddata.flags;


    // set default values
    //
    data.cmd[data.NBcmd].cmdsettings.procinfo_loopcntMax    = 1;
    data.cmd[data.NBcmd].cmdsettings.procinfo_MeasureTiming = 1;

    data.cmd[data.NBcmd].cmdsettings.triggerdelay.tv_sec  = 0;
    data.cmd[data.NBcmd].cmdsettings.triggerdelay.tv_nsec = 0;

    data.cmd[data.NBcmd].cmdsettings.triggertimeout.tv_sec  = 1;
    data.cmd[data.NBcmd].cmdsettings.triggertimeout.tv_nsec = 0;

    data.NBcmd++;

    DEBUG_TRACE_FEXIT();

    return ((uint32_t) ((int) data.NBcmd - 1));
}
