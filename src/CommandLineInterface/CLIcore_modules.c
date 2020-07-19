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



int DLib_index;
void *DLib_handle[1000];





errno_t load_sharedobj(
    const char *restrict libname
)
{
    printf("[%5d] Loading shared object \"%s\"\n", DLib_index, libname);

    DLib_handle[DLib_index] = dlopen(libname, RTLD_LAZY | RTLD_GLOBAL);
    if(!DLib_handle[DLib_index])
    {
        fprintf(stderr, "%s\n", dlerror());
        //exit(EXIT_FAILURE);
    }
    else
    {
        dlerror();
        // increment number of libs dynamically loaded
        DLib_index ++;
    }

    return RETURN_SUCCESS;
}




errno_t load_module_shared(
    const char *restrict modulename
)
{
    int STRINGMAXLEN_LIBRARYNAME = 200;
    char libname[STRINGMAXLEN_LIBRARYNAME];
    char modulenameLC[STRINGMAXLEN_LIBRARYNAME];
    //    char c;
    //    int n;
    //    int (*libinitfunc) ();
    //    char *error;
    //    char initfuncname[200];

    {
        int slen = snprintf(modulenameLC, STRINGMAXLEN_LIBRARYNAME, "%s", modulename);
        if(slen < 1)
        {
            PRINT_ERROR("snprintf wrote <1 char");
            abort(); // can't handle this error any other way
        }
        if(slen >= STRINGMAXLEN_LIBRARYNAME)
        {
            PRINT_ERROR("snprintf string truncation");
            abort(); // can't handle this error any other way
        }
    }

    /*    for(n=0; n<strlen(modulenameLC); n++)
        {
            c = modulenameLC[n];
            modulenameLC[n] = tolower(c);
        }
    */

    //    sprintf(libname, "%s/lib/lib%s.so", data.sourcedir, modulenameLC);
    {
        int slen = snprintf(libname, STRINGMAXLEN_LIBRARYNAME,
                            "%s/lib/lib%s.so", getenv("MILK_INSTALLDIR"), modulenameLC);
        if(slen < 1)
        {
            PRINT_ERROR("snprintf wrote <1 char");
            abort(); // can't handle this error any other way
        }
        if(slen >= STRINGMAXLEN_LIBRARYNAME)
        {
            PRINT_ERROR("snprintf string truncation");
            abort(); // can't handle this error any other way
        }
    }

    printf("libname = %s\n", libname);


    printf("[%5d] Loading shared object \"%s\"\n", DLib_index, libname);

    load_sharedobj(libname);

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
