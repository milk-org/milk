/**
 * @file CLIcore_modules.c
 *
 * @brief Modules functions
 *
 */

#include <dirent.h>
#include <dlfcn.h>

#include "CommandLineInterface/CLIcore.h"

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




errno_t load_sharedobj(
    const char *__restrict libname
)
{
    DEBUG_TRACE_FSTART();

    DEBUG_TRACEPOINT("[%5d] Loading shared object \"%s\"", DLib_index, libname);
    strncpy(libnameloaded, libname, STRINGMAXLEN_MODULE_SOFILENAME - 1);

    // check if already loaded
    DEBUG_TRACEPOINT("--- %ld modules loaded ---", data.NBmodule);
    int mmatch = -1;
    for(int m = 0; m < data.NBmodule; m++)
    {
        //printf("  [%03d] %s\n", m, data.module[m].sofilename);
        if(strcmp(libnameloaded, data.module[m].sofilename) == 0)
        {
            mmatch = m;
        }
    }
    if(mmatch > -1)
    {
        printf("    Shared object %s already loaded - no action taken\n",
               libnameloaded);
        DEBUG_TRACE_FEXIT();
        return RETURN_FAILURE;
    }

    DLib_handle[DLib_index] = dlopen(libname, RTLD_LAZY | RTLD_GLOBAL);
    if(!DLib_handle[DLib_index])
    {
        fprintf(stderr, KRED "%s\n" KRES, dlerror());
        //exit(EXIT_FAILURE);
    }
    else
    {
        dlerror();
        printf(KGRN "   LOADED : %s\n" KRES, libnameloaded);
        // increment number of libs dynamically loaded
        DLib_index++;
    }

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}



errno_t load_module_shared(
    const char *__restrict modulename
)
{
    DEBUG_TRACE_FSTART();
    char libname[STRINGMAXLEN_MODULE_SOFILENAME];

    // make locacl copy of module name
    char modulenameLC[STRINGMAXLEN_MODULE_SOFILENAME];

    {
        int slen = snprintf(modulenameLC,
                            STRINGMAXLEN_MODULE_SOFILENAME,
                            "%s",
                            modulename);
        if(slen < 1)
        {
            PRINT_ERROR("snprintf wrote <1 char");
            abort(); // can't handle this error any other way
        }
        if(slen >= STRINGMAXLEN_MODULE_SOFILENAME)
        {
            PRINT_ERROR("snprintf string truncation");
            abort(); // can't handle this error any other way
        }
    }

    // Assemble absolute path module filename
    //printf("Searching for shared object in directory MILK_INSTALLDIR/lib : %s/lib\n", getenv("MILK_INSTALLDIR"));
    DEBUG_TRACEPOINT(
        "Searching for shared object in directory [data.installdir]/lib : "
        "%s/lib",
        data.installdir);

    {
        int slen = snprintf(libname,
                            STRINGMAXLEN_MODULE_SOFILENAME,
                            "%s/lib/lib%s.so",
                            data.installdir,
                            modulenameLC);
        if(slen < 1)
        {
            PRINT_ERROR("snprintf wrote <1 char");
            abort(); // can't handle this error any other way
        }
        if(slen >= STRINGMAXLEN_MODULE_SOFILENAME)
        {
            PRINT_ERROR("snprintf string truncation");
            abort(); // can't handle this error any other way
        }
    }

    DEBUG_TRACEPOINT("libname = %s", libname);

    DEBUG_TRACEPOINT("[%5d] Loading shared object \"%s\"", DLib_index, libname);





    // a custom module is about to be loaded, so we set the type accordingly
    // this variable will be written by module register function into module struct
    strncpy(data.moduleloadname,
            modulenameLC,
            STRINGMAXLEN_MODULE_LOADNAME - 1);
    strncpy(data.modulesofilename, libname, STRINGMAXLEN_MODULE_SOFILENAME - 1);



    if(load_sharedobj(libname) == RETURN_SUCCESS)
    {
        // RegisterModule called here
    }


    data.module[data.moduleindex].type = MODULE_TYPE_CUSTOMLOAD;

    strncpy(data.module[data.moduleindex].sofilename,
            data.modulesofilename,
            STRINGMAXLEN_MODULE_SOFILENAME - 1);

    strncpy(data.module[data.moduleindex].loadname,
            data.moduleloadname,
            STRINGMAXLEN_MODULE_LOADNAME - 1);


    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}




errno_t load_module_shared_ALL()
{
    DEBUG_TRACE_FSTART();

    char           libname[STRINGMAXLEN_FULLFILENAME];
    char           dirname[STRINGMAXLEN_DIRNAME];
    DIR           *d;
    struct dirent *dir;
    int            iter;
    int            loopOK;
    int            itermax;

    WRITE_DIRNAME(dirname, "%s/lib", data.installdir);

    if(data.quiet == 0)
    {
        printf("LOAD MODULES SHARED ALL: %s\n", dirname);
    }

    loopOK  = 0;
    iter    = 0;
    itermax = 4; // number of passes
    while((loopOK == 0) && (iter < itermax))
    {
        loopOK = 1;
        d      = opendir(dirname);
        if(d)
        {
            while((dir = readdir(d)) != NULL)
            {
                char *dot = strrchr(dir->d_name, '.');
                if(dot && !strcmp(dot, ".so"))
                {
                    WRITE_FULLFILENAME(libname,
                                       "%s/lib/%s",
                                       data.installdir,
                                       dir->d_name);
                    //printf("%02d   (re-?) LOADING shared object  %40s -> %s\n", DLib_index, dir->d_name, libname);
                    //fflush(stdout);

                    printf(
                        "    [%5d] Loading shared object "
                        "\"%s\"\n",
                        DLib_index,
                        libname);
                    DLib_handle[DLib_index] =
                        dlopen(libname, RTLD_LAZY | RTLD_GLOBAL);
                    if(!DLib_handle[DLib_index])
                    {
                        fprintf(stderr,
                                KMAG
                                "        WARNING: linker "
                                "pass # %d, module # %d\n  "
                                "        %s\n" KRES,
                                iter,
                                DLib_index,
                                dlerror());
                        fflush(stderr);
                        //exit(EXIT_FAILURE);
                        loopOK = 0;
                    }
                    else
                    {
                        dlerror();
                        // increment number of libs dynamically loaded
                        DLib_index++;
                    }
                }
            }

            closedir(d);
        }
        if(iter > 0)
            if(loopOK == 1)
            {
                printf(KGRN "        Linker pass #%d successful\n" KRES, iter);
            }
        iter++;
    }

    if(loopOK != 1)
    {
        printf("Some libraries could not be loaded -> EXITING\n");
        exit(2);
    }

    //printf("All libraries successfully loaded\n");

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}






errno_t RegisterModule(const char *__restrict FileName,
                       const char *__restrict PackageName,
                       const char *__restrict InfoString,
                       int versionmajor,
                       int versionminor,
                       int versionpatch)
{
    DEBUG_TRACE_FSTART();

    int OKmsg = 0;

    int moduleindex =  data.NBmodule;
    data.moduleindex = moduleindex; // current module index

    data.NBmodule++;



    if(strlen(data.modulename) == 0)
    {
        strncpy(data.module[moduleindex].name, "???", STRINGMAXLEN_MODULE_NAME - 1);
    }
    else
    {
        strncpy(data.module[moduleindex].name, data.modulename,
                STRINGMAXLEN_MODULE_NAME - 1);
    }

    int stringlen = strlen(data.moduleshortname);
    if(stringlen == 0)
    {
        // if no shortname provided, try to use default
        if(strlen(data.moduleshortname_default) > 0)
        {
            // otherwise, construct call key as <shortname_default>.<CLIkey>
            strncpy(data.moduleshortname, data.moduleshortname_default,
                    STRINGMAXLEN_MODULE_SHORTNAME - 1);
        }
    }

    strncpy(data.module[moduleindex].package, PackageName,
            STRINGMAXLEN_MODULE_PACKAGENAME - 1);
    strncpy(data.module[moduleindex].info, InfoString,
            STRINGMAXLEN_MODULE_INFOSTRING - 1);

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


    if(data.progStatus == 0)
    {
        OKmsg = 1;
        if(!getenv("MILK_QUIET"))
        {
            printf(".");
        }
        //	printf("  %02ld  LOADING %10s  module %40s\n", data.NBmodule, PackageName, FileName);
        //	fflush(stdout);
    }

    if(data.progStatus == 1)
    {
        OKmsg = 1;
        DEBUG_TRACEPOINT(
            "  %02d  Found unloaded shared object in ./libs/ -> LOADING "
            "%10s  module %40s",
            moduleindex,
            PackageName,
            FileName);
        fflush(stdout);
    }

    if(OKmsg == 0)
    {
        printf(
            "  %02d  ERROR: module load requested outside of normal step "
            "-> LOADING %10s  module %40s",
            moduleindex,
            PackageName,
            FileName);
        fflush(stdout);
    }


    // default
    // may be overridden by load_module_shared
    //
    //data.moduletype = MODULE_TYPE_STARTUP;

    data.module[data.moduleindex].type = MODULE_TYPE_STARTUP;

    //strncpy(data.modulesofilename, "", STRINGMAXLEN_MODULE_SOFILENAME - 1);
    strncpy(data.module[data.moduleindex].sofilename,
            "",
            STRINGMAXLEN_MODULE_SOFILENAME - 1);

    //strncpy(data.modulesofilename, "", STRINGMAXLEN_MODULE_SOFILENAME - 1);
    strncpy(data.module[data.moduleindex].loadname,
            "",
            STRINGMAXLEN_MODULE_LOADNAME - 1);

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}




// Legacy function
//
uint32_t RegisterCLIcommand(const char *__restrict CLIkey,
                            const char *__restrict CLImodulesrc,
                            errno_t (*CLIfptr)(),
                            const char *__restrict CLIinfo,
                            const char *__restrict CLIsyntax,
                            const char *__restrict CLIexample,
                            const char *__restrict CLICcall)
{
    DEBUG_TRACE_FSTART();

    DEBUG_TRACEPOINT("FARG CLIkey %s -> command index %u / %d",
                     CLIkey,
                     data.NBcmd,
                     DATA_NB_MAX_COMMAND);

    data.cmd[data.NBcmd].moduleindex = data.moduleindex;

    if(data.cmd[data.NBcmd].moduleindex == -1)
    {
        strncpy(data.cmd[data.NBcmd].module, "MAIN", STRINGMAXLEN_MODULE_NAME - 1);
        strncpy(data.cmd[data.NBcmd].key, CLIkey, STRINGMAXLEN_CMD_KEY - 1);
    }
    else
    {

        if(strlen(data.module[data.moduleindex].shortname) == 0)
        {
            strncpy(data.cmd[data.NBcmd].key, CLIkey, STRINGMAXLEN_CMD_KEY - 1);
        }
        else
        {
            // otherwise, construct call key as <shortname>.<CLIkey>
            snprintf(data.cmd[data.NBcmd].key,
                     STRINGMAXLEN_CMD_KEY,
                     "%s.%s",
                     data.module[data.moduleindex].shortname,
                     CLIkey);
        }
    }

    DEBUG_TRACEPOINT("set module name");
    if(strlen(data.modulename) == 0)
    {
        strncpy(data.cmd[data.NBcmd].module, "unknown", STRINGMAXLEN_MODULE_NAME - 1);
    }
    else
    {
        strncpy(data.cmd[data.NBcmd].module, data.modulename,
                STRINGMAXLEN_MODULE_NAME - 1);
    }

    DEBUG_TRACEPOINT("load function data");

    strncpy(data.cmd[data.NBcmd].srcfile,
            CLImodulesrc,
            STRINGMAXLEN_CMD_SRCFILE - 1);

    data.cmd[data.NBcmd].fp = CLIfptr;

    strncpy(data.cmd[data.NBcmd].info, CLIinfo, STRINGMAXLEN_CMD_INFO - 1);

    strncpy(data.cmd[data.NBcmd].syntax,
            CLIsyntax,
            STRINGMAXLEN_CMD_SYNTAX - 1);

    strncpy(data.cmd[data.NBcmd].example,
            CLIexample,
            STRINGMAXLEN_CMD_EXAMPLE - 1);

    strncpy(data.cmd[data.NBcmd].Ccall, CLICcall, STRINGMAXLEN_CMD_CCALL - 1);

    data.cmd[data.NBcmd].nbarg = 0;
    data.NBcmd++;

    DEBUG_TRACEPOINT("Done1");

    DEBUG_TRACE_FEXIT();

    DEBUG_TRACEPOINT("NBcmd = %u", data.NBcmd);

    return (data.NBcmd);
}




/**
 Register command
Replaces legacy function RegisterCLIcommand
*/
uint32_t RegisterCLIcmd(
    CLICMDDATA CLIcmddata,
    errno_t (*CLIfptr)()
)
{
    DEBUG_TRACE_FSTART();

    data.cmd[data.NBcmd].moduleindex = data.moduleindex;
    if(data.cmd[data.NBcmd].moduleindex == -1)
    {
        strncpy(data.cmd[data.NBcmd].module, "MAIN", STRINGMAXLEN_MODULE_NAME - 1);
        strncpy(data.cmd[data.NBcmd].key, CLIcmddata.key, STRINGMAXLEN_CMD_KEY - 1);
    }
    else
    {

        if(strlen(data.module[data.moduleindex].shortname) == 0)
        {
            strncpy(data.cmd[data.NBcmd].key, CLIcmddata.key, STRINGMAXLEN_CMD_KEY);
        }
        else
        {
            // otherwise, construct call key as <shortname>.<CLIkey>
            int slen = snprintf(data.cmd[data.NBcmd].key,
                                STRINGMAXLEN_CMD_KEY,
                                "%s.%s",
                                data.module[data.moduleindex].shortname,
                                CLIcmddata.key);
            if(slen < 1)
            {
                PRINT_ERROR("failed to write call key");
                abort();
            }
            if(slen >= STRINGMAXLEN_CMD_KEY)
            {
                PRINT_ERROR("call key string too long");
                abort();
            }
        }
    }

    if(strlen(data.modulename) == 0)
    {
        strncpy(data.cmd[data.NBcmd].module, "unknown", STRINGMAXLEN_MODULE_NAME - 1);
    }
    else
    {
        strncpy(data.cmd[data.NBcmd].module, data.modulename,
                STRINGMAXLEN_MODULE_NAME - 1);
    }

    DEBUG_TRACEPOINT("settingsrcfile to %s", CLIcmddata.sourcefilename);
    strncpy(data.cmd[data.NBcmd].srcfile, CLIcmddata.sourcefilename,
            STRINGMAXLEN_CMD_SRCFILE - 1);
    data.cmd[data.NBcmd].fp = CLIfptr;
    strncpy(data.cmd[data.NBcmd].info, CLIcmddata.description,
            STRINGMAXLEN_CMD_INFO - 1);

    // assemble argument syntax string for help
    char argstring[STRINGMAXLEN_CMD_SYNTAX];
    CLIhelp_make_argstring(CLIcmddata.funcfpscliarg,
                           CLIcmddata.nbarg,
                           argstring);
    strncpy(data.cmd[data.NBcmd].syntax, argstring, STRINGMAXLEN_CMD_SYNTAX - 1);

    // assemble example string for help
    char cmdexamplestring[STRINGMAXLEN_CMD_EXAMPLE];
    CLIhelp_make_cmdexamplestring(CLIcmddata.funcfpscliarg,
                                  CLIcmddata.nbarg,
                                  CLIcmddata.key,
                                  cmdexamplestring);
    strncpy(data.cmd[data.NBcmd].example, cmdexamplestring,
            STRINGMAXLEN_CMD_EXAMPLE - 1);

    strncpy(data.cmd[data.NBcmd].Ccall, "--callstring--",
            STRINGMAXLEN_CMD_CCALL - 1);

    DEBUG_TRACEPOINT(
        "define arguments to CLI function from content of "
        "CLIcmddata.funcfpscliarg");
    data.cmd[data.NBcmd].nbarg = CLIcmddata.nbarg;
    if(CLIcmddata.nbarg > 0)
    {
        data.cmd[data.NBcmd].argdata =
            (CLICMDARGDATA *) malloc(sizeof(CLICMDARGDATA) * CLIcmddata.nbarg);

        for(int argi = 0; argi < CLIcmddata.nbarg; argi++)
        {
            data.cmd[data.NBcmd].argdata[argi].type =
                CLIcmddata.funcfpscliarg[argi].type;
            data.cmd[data.NBcmd].argdata[argi].flag =
                CLIcmddata.funcfpscliarg[argi].flag;

            strncpy(data.cmd[data.NBcmd].argdata[argi].descr,
                    CLIcmddata.funcfpscliarg[argi].descr,
                    STRINGMAXLEN_FPSCLIARG_DESCR - 1);

            strncpy(data.cmd[data.NBcmd].argdata[argi].fpstag,
                    CLIcmddata.funcfpscliarg[argi].fpstag,
                    STRINGMAXLEN_FPSCLIARG_TAG - 1);

            strncpy(data.cmd[data.NBcmd].argdata[argi].example,
                    CLIcmddata.funcfpscliarg[argi].example,
                    STRINGMAXLEN_FPSCLIARG_EXAMPLE - 1);

            // Set default values
            switch(data.cmd[data.NBcmd].argdata[argi].type)
            {

                /*case CLIARG_FLOAT:
                    data.cmd[data.NBcmd].argdata[argi].val.f = atof(
                                CLIcmddata.funcfpscliarg[argi].example);
                    break;*/

                case CLIARG_FLOAT32:
                    data.cmd[data.NBcmd].argdata[argi].val.f32 =
                        atof(CLIcmddata.funcfpscliarg[argi].example);
                    break;

                case CLIARG_FLOAT64:
                    data.cmd[data.NBcmd].argdata[argi].val.f64 =
                        atof(CLIcmddata.funcfpscliarg[argi].example);
                    break;

                /*case CLIARG_LONG:
                    data.cmd[data.NBcmd].argdata[argi].val.l = atol(
                                CLIcmddata.funcfpscliarg[argi].example);
                    break;*/

                case CLIARG_INT32:
                    data.cmd[data.NBcmd].argdata[argi].val.i32 =
                        (int32_t) atol(CLIcmddata.funcfpscliarg[argi].example);
                    break;

                case CLIARG_UINT32:
                    data.cmd[data.NBcmd].argdata[argi].val.ui32 =
                        (uint32_t) atol(CLIcmddata.funcfpscliarg[argi].example);
                    break;

                case CLIARG_INT64:
                    data.cmd[data.NBcmd].argdata[argi].val.i64 =
                        (int64_t) atol(CLIcmddata.funcfpscliarg[argi].example);
                    break;

                case CLIARG_UINT64:
                    data.cmd[data.NBcmd].argdata[argi].val.ui64 =
                        (uint64_t) atol(CLIcmddata.funcfpscliarg[argi].example);
                    break;

                case CLIARG_ONOFF:
                    data.cmd[data.NBcmd].argdata[argi].val.ui64 =
                        (int64_t) atol(CLIcmddata.funcfpscliarg[argi].example);
                    break;

                case CLIARG_STR_NOT_IMG:
                    strncpy(data.cmd[data.NBcmd].argdata[argi].val.s,
                            CLIcmddata.funcfpscliarg[argi].example,
                            STRINGMAXLEN_CLICMDARG - 1);
                    break;

                case CLIARG_IMG:
                    strncpy(data.cmd[data.NBcmd].argdata[argi].val.s,
                            CLIcmddata.funcfpscliarg[argi].example,
                            STRINGMAXLEN_CLICMDARG - 1);
                    break;

                case CLIARG_STR:
                    strncpy(data.cmd[data.NBcmd].argdata[argi].val.s,
                            CLIcmddata.funcfpscliarg[argi].example,
                            STRINGMAXLEN_CLICMDARG - 1);
                    break;
            }
        }
    }

    DEBUG_TRACEPOINT(
        "define CLI function flags from content of CLIcmddata.flags");


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

    return ((uint32_t)((int) data.NBcmd - 1));
}
