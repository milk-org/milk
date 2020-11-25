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
    const char *restrict libname
)
{
    //printf("[%5d] Loading shared object \"%s\"\n", DLib_index, libname);
    strncpy(libnameloaded, libname, STRINGMAXLEN_MODULE_SOFILENAME);


    // check if already loaded
    //printf("--- %ld modules loaded ---\n", data.NBmodule);
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
        printf("Shared object %s already loaded - no action taken\n", libnameloaded);
        return RETURN_FAILURE;
    }



    DLib_handle[DLib_index] = dlopen(libname, RTLD_LAZY | RTLD_GLOBAL);
    if(!DLib_handle[DLib_index])
    {
        fprintf(stderr, "%s\n", dlerror());
        //exit(EXIT_FAILURE);
    }
    else
    {
        dlerror();
        printf("  ----- LOADED : %s <- %s\n", libnameloaded, libname);
        // increment number of libs dynamically loaded
        DLib_index ++;
    }


    return RETURN_SUCCESS;
}




errno_t load_module_shared(
    const char *restrict modulename
)
{
    char libname[STRINGMAXLEN_MODULE_SOFILENAME];

	
	// module name local copy
    char modulenameLC[STRINGMAXLEN_MODULE_SOFILENAME];

    {
        int slen = snprintf(modulenameLC, STRINGMAXLEN_MODULE_SOFILENAME, "%s", modulename);
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


    {
        int slen = snprintf(libname, STRINGMAXLEN_MODULE_SOFILENAME,
                            "%s/lib/lib%s.so", getenv("MILK_INSTALLDIR"), modulenameLC);
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

    printf("libname = %s\n", libname);

    printf("[%5d] Loading shared object \"%s\"\n", DLib_index, libname);

	// a custom module is about to be loaded, so we set the type accordingly
	// this variable will be written by module register function into module struct
	data.moduletype = MODULE_TYPE_CUSTOMLOAD; 
	strncpy(data.moduleloadname, modulenameLC, STRINGMAXLEN_MODULE_LOADNAME);
	strncpy(data.modulesofilename, libname, STRINGMAXLEN_MODULE_SOFILENAME);
    if ( load_sharedobj(libname) == RETURN_SUCCESS )
    {
		// 
	}
	// reset to default for next load
	data.moduletype = MODULE_TYPE_STARTUP; 
	strncpy(data.moduleloadname, "", STRINGMAXLEN_MODULE_LOADNAME);
	strncpy(data.modulesofilename, "", STRINGMAXLEN_MODULE_SOFILENAME);

    return RETURN_SUCCESS;
}





errno_t load_module_shared_ALL()
{
    char libname[500];
    char dirname[200];
    DIR           *d;
    struct dirent *dir;
    int iter;
    int loopOK;
    int itermax;

    sprintf(dirname, "%s/lib", data.sourcedir);

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
                    sprintf(libname, "%s/lib/%s", data.sourcedir, dir->d_name);
                    //printf("%02d   (re-?) LOADING shared object  %40s -> %s\n", DLib_index, dir->d_name, libname);
                    //fflush(stdout);

                    //printf("[%5d] Loading shared object \"%s\"\n", DLib_index, libname);
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


    return RETURN_SUCCESS;
}





errno_t RegisterModule(
    const char *restrict FileName,
    const char *restrict PackageName,
    const char *restrict InfoString,
    int versionmajor,
    int versionminor,
    int versionpatch
)
{
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
	
	strncpy(data.module[data.NBmodule].loadname, data.moduleloadname, STRINGMAXLEN_MODULE_LOADNAME);
	strncpy(data.module[data.NBmodule].sofilename, data.modulesofilename, STRINGMAXLEN_MODULE_SOFILENAME);	

	//printf("--- libnameloaded : %s\n", libnameloaded);
	strncpy(data.module[data.NBmodule].sofilename, libnameloaded, STRINGMAXLEN_MODULE_SOFILENAME);

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
        printf("  %02ld  Found unloaded shared object in ./libs/ -> LOADING %10s  module %40s\n",
               data.NBmodule,
               PackageName,
               FileName);
        fflush(stdout);
    }

    if(OKmsg == 0)
    {
        printf("  %02ld  ERROR: module load requested outside of normal step -> LOADING %10s  module %40s\n",
               data.NBmodule,
               PackageName,
               FileName);
        fflush(stdout);
    }

    data.NBmodule++;


    return RETURN_SUCCESS;
}





uint32_t RegisterCLIcommand(
    const char *restrict CLIkey,
    const char *restrict CLImodulesrc,
    errno_t (*CLIfptr)(),
    const char *restrict CLIinfo,
    const char *restrict CLIsyntax,
    const char *restrict CLIexample,
    const char *restrict CLICcall
)
{
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
            sprintf(data.cmd[data.NBcmd].key, "%s.%s", data.module[data.moduleindex].shortname, CLIkey);
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

    strcpy(data.cmd[data.NBcmd].modulesrc, CLImodulesrc);
    data.cmd[data.NBcmd].fp = CLIfptr;
    strcpy(data.cmd[data.NBcmd].info,    CLIinfo);
    strcpy(data.cmd[data.NBcmd].syntax,  CLIsyntax);
    strcpy(data.cmd[data.NBcmd].example, CLIexample);
    strcpy(data.cmd[data.NBcmd].Ccall,   CLICcall);
    data.NBcmd++;

    return(data.NBcmd);
}



