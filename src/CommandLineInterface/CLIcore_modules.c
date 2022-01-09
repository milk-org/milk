/**
 * @file CLIcore_modules.c
 *
 * @brief Modules functions
 *
 */


#include <dlfcn.h>
#include <dirent.h>

#include "CommandLineInterface/CLIcore.h"



#define KNRM  "\x1B[0m"
#define KRED  "\x1B[31m"
#define KGRN  "\x1B[32m"
#define KYEL  "\x1B[33m"
#define KBLU  "\x1B[34m"
#define KMAG  "\x1B[35m"
#define KCYN  "\x1B[36m"
#define KWHT  "\x1B[37m"
#define KRES  "\033[0m"




// local valiables to keep track of library last loaded
static int DLib_index;
static void *DLib_handle[1000];
static char libnameloaded[STRINGMAXLEN_MODULE_SOFILENAME];



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
        fprintf(stderr, KRED"%s\n"KRES, dlerror());
        //exit(EXIT_FAILURE);
    }
    else
    {
        dlerror();
        printf(KGRN"   LOADED : %s\n"KRES, libnameloaded);
        // increment number of libs dynamically loaded
        DLib_index ++;
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
        int slen = snprintf(modulenameLC, STRINGMAXLEN_MODULE_SOFILENAME, "%s",
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
    DEBUG_TRACEPOINT("Searching for shared object in directory [data.installdir]/lib : %s/lib",
                     data.installdir);

    {
        int slen = snprintf(libname, STRINGMAXLEN_MODULE_SOFILENAME,
                            "%s/lib/lib%s.so", data.installdir, modulenameLC);
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
    data.moduletype = MODULE_TYPE_CUSTOMLOAD;
    strncpy(data.moduleloadname, modulenameLC, STRINGMAXLEN_MODULE_LOADNAME - 1);
    strncpy(data.modulesofilename, libname, STRINGMAXLEN_MODULE_SOFILENAME - 1);
    if(load_sharedobj(libname) == RETURN_SUCCESS)
    {
        //
    }
    // reset to default for next load
    data.moduletype = MODULE_TYPE_STARTUP;
    strncpy(data.moduleloadname, "", STRINGMAXLEN_MODULE_LOADNAME - 1);
    strncpy(data.modulesofilename, "", STRINGMAXLEN_MODULE_SOFILENAME - 1);

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}





errno_t load_module_shared_ALL()
{
    DEBUG_TRACE_FSTART();

    char libname[STRINGMAXLEN_FULLFILENAME];
    char dirname[STRINGMAXLEN_DIRNAME];
    DIR           *d;
    struct dirent *dir;
    int iter;
    int loopOK;
    int itermax;

    WRITE_DIRNAME(dirname, "%s/lib", data.installdir);

    if(data.quiet == 0)
    {
        printf("LOAD MODULES SHARED ALL: %s\n", dirname);
    }

    loopOK = 0;
    iter = 0;
    itermax = 4; // number of passes
    while((loopOK == 0) && (iter < itermax))
    {
        loopOK = 1;
        d = opendir(dirname);
        if(d)
        {
            while((dir = readdir(d)) != NULL)
            {
                char *dot = strrchr(dir->d_name, '.');
                if(dot && !strcmp(dot, ".so"))
                {
                    WRITE_FULLFILENAME(libname, "%s/lib/%s", data.installdir, dir->d_name);
                    //printf("%02d   (re-?) LOADING shared object  %40s -> %s\n", DLib_index, dir->d_name, libname);
                    //fflush(stdout);

                    printf("    [%5d] Loading shared object \"%s\"\n", DLib_index, libname);
                    DLib_handle[DLib_index] = dlopen(libname, RTLD_LAZY | RTLD_GLOBAL);
                    if(!DLib_handle[DLib_index])
                    {
                        fprintf(stderr, KMAG
                                "        WARNING: linker pass # %d, module # %d\n          %s\n" KRES, iter,
                                DLib_index, dlerror());
                        fflush(stderr);
                        //exit(EXIT_FAILURE);
                        loopOK = 0;
                    }
                    else
                    {
                        dlerror();
                        // increment number of libs dynamically loaded
                        DLib_index ++;
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





errno_t RegisterModule(
    const char *__restrict FileName,
    const char *__restrict PackageName,
    const char *__restrict InfoString,
    int versionmajor,
    int versionminor,
    int versionpatch
)
{
    DEBUG_TRACE_FSTART();

    int OKmsg = 0;

    //printf("REGISTERING MODULE %s\n", FileName);

    if(strlen(data.modulename) == 0)
    {
        strcpy(data.module[data.NBmodule].name, "???");
    }
    else
    {
        strcpy(data.module[data.NBmodule].name,         data.modulename);
    }


    int stringlen = strlen(data.moduleshortname);
    if(stringlen == 0)
    {
        // if no shortname provided, try to use default
        if(strlen(data.moduleshortname_default) > 0)
        {
            // otherwise, construct call key as <shortname_default>.<CLIkey>
            strcpy(data.moduleshortname, data.moduleshortname_default);
        }
    }

    data.moduleindex = data.NBmodule; // current module index

    strcpy(data.module[data.NBmodule].package,      PackageName);
    strcpy(data.module[data.NBmodule].info,         InfoString);

    strcpy(data.module[data.NBmodule].shortname,    data.moduleshortname);

    strcpy(data.module[data.NBmodule].datestring,   data.moduledatestring);
    strcpy(data.module[data.NBmodule].timestring,   data.moduletimestring);

    data.module[data.NBmodule].versionmajor = versionmajor;
    data.module[data.NBmodule].versionminor = versionminor;
    data.module[data.NBmodule].versionpatch = versionpatch;

    data.module[data.NBmodule].type = data.moduletype;

    strncpy(data.module[data.NBmodule].loadname, data.moduleloadname,
            STRINGMAXLEN_MODULE_LOADNAME - 1);
    strncpy(data.module[data.NBmodule].sofilename, data.modulesofilename,
            STRINGMAXLEN_MODULE_SOFILENAME - 1);

    //printf("--- libnameloaded : %s\n", libnameloaded);
    strncpy(data.module[data.NBmodule].sofilename, libnameloaded,
            STRINGMAXLEN_MODULE_SOFILENAME - 1);

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
        DEBUG_TRACEPOINT("  %02ld  Found unloaded shared object in ./libs/ -> LOADING %10s  module %40s",
                         data.NBmodule,
                         PackageName,
                         FileName);
        fflush(stdout);
    }

    if(OKmsg == 0)
    {
        printf("  %02ld  ERROR: module load requested outside of normal step -> LOADING %10s  module %40s",
               data.NBmodule,
               PackageName,
               FileName);
        fflush(stdout);
    }

    data.NBmodule++;


    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}




// Legacy function
//
uint32_t RegisterCLIcommand(
    const char *__restrict CLIkey,
    const char *__restrict CLImodulesrc,
    errno_t (*CLIfptr)(),
    const char *__restrict CLIinfo,
    const char *__restrict CLIsyntax,
    const char *__restrict CLIexample,
    const char *__restrict CLICcall
)
{
    DEBUG_TRACE_FSTART();

    DEBUG_TRACEPOINT("FARG CLIkey %s -> command index %u / %d", CLIkey, data.NBcmd,
                     DATA_NB_MAX_COMMAND);




    data.cmd[data.NBcmd].moduleindex = data.moduleindex;

    if(data.cmd[data.NBcmd].moduleindex == -1)
    {
        strcpy(data.cmd[data.NBcmd].module, "MAIN");
        strcpy(data.cmd[data.NBcmd].key, CLIkey);
    }
    else
    {

        if(strlen(data.module[data.moduleindex].shortname) == 0)
        {
            strcpy(data.cmd[data.NBcmd].key, CLIkey);
        }
        else
        {
            // otherwise, construct call key as <shortname>.<CLIkey>
            sprintf(data.cmd[data.NBcmd].key, "%s.%s",
                    data.module[data.moduleindex].shortname, CLIkey);
        }
    }

    DEBUG_TRACEPOINT("set module name");
    if(strlen(data.modulename) == 0)
    {
        strcpy(data.cmd[data.NBcmd].module, "unknown");
    }
    else
    {
        strcpy(data.cmd[data.NBcmd].module, data.modulename);
    }

    DEBUG_TRACEPOINT("load function data");

    strncpy(data.cmd[data.NBcmd].srcfile, CLImodulesrc,
            STRINGMAXLEN_CMD_SRCFILE - 1);

    data.cmd[data.NBcmd].fp = CLIfptr;

    strncpy(data.cmd[data.NBcmd].info,    CLIinfo, STRINGMAXLEN_CMD_INFO - 1);

    strncpy(data.cmd[data.NBcmd].syntax,  CLIsyntax, STRINGMAXLEN_CMD_SYNTAX - 1);

    strncpy(data.cmd[data.NBcmd].example, CLIexample, STRINGMAXLEN_CMD_EXAMPLE - 1);

    strncpy(data.cmd[data.NBcmd].Ccall,   CLICcall, STRINGMAXLEN_CMD_CCALL - 1);

    data.cmd[data.NBcmd].nbarg = 0;
    data.NBcmd++;

    DEBUG_TRACEPOINT("Done1");

    DEBUG_TRACE_FEXIT();

    DEBUG_TRACEPOINT("NBcmd = %u", data.NBcmd);

    return(data.NBcmd);
}




// Register command
// Replaces legacy function RegisterCLIcommand
//
uint32_t RegisterCLIcmd(
    CLICMDDATA CLIcmddata,
    errno_t (*CLIfptr)()
)
{
    DEBUG_TRACE_FSTART();


    data.cmd[data.NBcmd].moduleindex = data.moduleindex;
    if(data.cmd[data.NBcmd].moduleindex == -1)
    {
        strcpy(data.cmd[data.NBcmd].module, "MAIN");
        strcpy(data.cmd[data.NBcmd].key, CLIcmddata.key);
    }
    else
    {

        if(strlen(data.module[data.moduleindex].shortname) == 0)
        {
            strcpy(data.cmd[data.NBcmd].key, CLIcmddata.key);
        }
        else
        {
            // otherwise, construct call key as <shortname>.<CLIkey>
            int slen = snprintf(data.cmd[data.NBcmd].key, STRINGMAXLEN_CMD_KEY, "%s.%s",
                                data.module[data.moduleindex].shortname, CLIcmddata.key);
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
        strcpy(data.cmd[data.NBcmd].module, "unknown");
    }
    else
    {
        strcpy(data.cmd[data.NBcmd].module, data.modulename);
    }

    DEBUG_TRACEPOINT("settingsrcfile to %s", CLIcmddata.sourcefilename);
    strcpy(data.cmd[data.NBcmd].srcfile, CLIcmddata.sourcefilename);
    data.cmd[data.NBcmd].fp = CLIfptr;
    strcpy(data.cmd[data.NBcmd].info,    CLIcmddata.description);

    // assemble argument syntax string for help
    char argstring[STRINGMAXLEN_CMD_SYNTAX];
    CLIhelp_make_argstring(CLIcmddata.funcfpscliarg, CLIcmddata.nbarg, argstring);
    strcpy(data.cmd[data.NBcmd].syntax,  argstring);

    // assemble example string for help
    char cmdexamplestring[STRINGMAXLEN_CMD_EXAMPLE];
    CLIhelp_make_cmdexamplestring(CLIcmddata.funcfpscliarg, CLIcmddata.nbarg,
                                  CLIcmddata.key,
                                  cmdexamplestring);
    strcpy(data.cmd[data.NBcmd].example, cmdexamplestring);

    strcpy(data.cmd[data.NBcmd].Ccall,   "--callstring--");


    DEBUG_TRACEPOINT("define arguments to CLI function from content of CLIcmddata.funcfpscliarg");
    data.cmd[data.NBcmd].nbarg = CLIcmddata.nbarg;
    if(CLIcmddata.nbarg > 0)
    {
        data.cmd[data.NBcmd].argdata =
            (CLICMDARGDATA *) malloc(sizeof(CLICMDARGDATA) * CLIcmddata.nbarg);

        for(int argi = 0; argi < CLIcmddata.nbarg; argi++)
        {
            data.cmd[data.NBcmd].argdata[argi].type = CLIcmddata.funcfpscliarg[argi].type;
            data.cmd[data.NBcmd].argdata[argi].flag = CLIcmddata.funcfpscliarg[argi].flag;
            strcpy(data.cmd[data.NBcmd].argdata[argi].descr,
                   CLIcmddata.funcfpscliarg[argi].descr);
            strcpy(data.cmd[data.NBcmd].argdata[argi].fpstag,
                   CLIcmddata.funcfpscliarg[argi].fpstag);
            strcpy(data.cmd[data.NBcmd].argdata[argi].example,
                   CLIcmddata.funcfpscliarg[argi].example);

            // Set default values
            switch(data.cmd[data.NBcmd].argdata[argi].type)
            {

            /*case CLIARG_FLOAT:
                data.cmd[data.NBcmd].argdata[argi].val.f = atof(
                            CLIcmddata.funcfpscliarg[argi].example);
                break;*/

            case CLIARG_FLOAT32:
                data.cmd[data.NBcmd].argdata[argi].val.f32 = atof(
                            CLIcmddata.funcfpscliarg[argi].example);
                break;

            case CLIARG_FLOAT64:
                data.cmd[data.NBcmd].argdata[argi].val.f64 = atof(
                            CLIcmddata.funcfpscliarg[argi].example);
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


            case CLIARG_STR_NOT_IMG:
                strcpy(data.cmd[data.NBcmd].argdata[argi].val.s,
                       CLIcmddata.funcfpscliarg[argi].example);
                break;

            case CLIARG_IMG:
                strcpy(data.cmd[data.NBcmd].argdata[argi].val.s,
                       CLIcmddata.funcfpscliarg[argi].example);
                break;

            case CLIARG_STR:
                strcpy(data.cmd[data.NBcmd].argdata[argi].val.s,
                       CLIcmddata.funcfpscliarg[argi].example);
                break;
            }
        }
    }

    DEBUG_TRACEPOINT("define CLI function flags from content of CLIcmddata.flags");
    data.cmd[data.NBcmd].cmdsettings.flags = CLIcmddata.flags;

    data.cmd[data.NBcmd].cmdsettings.procinfo_loopcntMax = 1;
    data.cmd[data.NBcmd].cmdsettings.procinfo_MeasureTiming = 1;

    data.NBcmd++;

    DEBUG_TRACE_FEXIT();

    return((uint32_t)((int) data.NBcmd - 1));
}

