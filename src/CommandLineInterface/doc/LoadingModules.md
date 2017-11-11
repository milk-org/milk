# Loading Modules {#page_LoadingModules}


## Compiling and linking modules with Autotools

Source code is organized in modules. Modules are compiled and linked using the libtool / autotools process, as shared libtool convenience libraries. 

See : https://www.gnu.org/software/automake/manual/html_node/Libtool-Convenience-Libraries.html

The list of modules currently appears in multiple files associated with the build process. The build process uses autotools, which has the advantage of supporting many platforms. 

Users adding modules are required to add the modules to 3 files, as described in sections 1.1, 1.2, and 1.3.

### Adding modules to `configure.ac`
 
Modules are listed in file `<srcdir>/configure.ac` under the "AC_CONFIG_FILES"

see: https://www.gnu.org/savannah-checkouts/gnu/autoconf/manual/autoconf-2.69/html_node/Configuration-Files.html


### Adding module functions to the command line interface

Modules are listed in file `<srcdir>/src/initmodules.c`

The init_modules() function in this file is executed to register commands provided in each modules in the command line interface

### Including modules in the recursive build and link process

Modules are listed in file `<srcdir>/src/Makefile.am`

This tells autotools' recursive build process to add the module directory in the list of subdirectories

This adds the library to the list of object files linked to the executable


## Linking at runtile with dlopen()

Alternatively, users may create their own libraries and link them at runtime using the `mload` function of the comamnd line interface. At any time, a module can be loaded from the CLI using:

	> mload <mymodulename>

See: http://www.linux-mag.com/id/1028/ for instructions to create shared libraries that can be loaded as modules. 

The largely self-explanatory source code, extracted from `src/CLIcore.c` is:


~~~
static int_fast8_t load_module_shared(char *modulename)
{
    char libname[200];
    char modulenameLC[200];
    char c;
    int n;
    int (*libinitfunc) ();
    char *error;
    char initfuncname[200];



    sprintf(modulenameLC, "%s", modulename);
    for(n=0; n<strlen(modulenameLC); n++)
    {
        c = modulenameLC[n];
        modulenameLC[n] = tolower(c);
    }

    sprintf(libname, "src/%s/.libs/lib%s.so", modulename, modulenameLC);
    printf("libname = %s\n", libname);


    printf("[%5d] Loading object \"%s\"\n", DLib_index, libname);


    DLib_handle[DLib_index] = dlopen(libname, RTLD_LAZY);
    if (!DLib_handle[DLib_index]) {
        fprintf(stderr, "%s\n", dlerror());
        exit(EXIT_FAILURE);
    }

    dlerror();

	sprintf(initfuncname, "initlib_%s", modulenameLC);
    libinitfunc = dlsym(DLib_handle[DLib_index], initfuncname);
    if ((error = dlerror()) != NULL) {
        fputs(error, stderr);
    exit(1);
	}

	(*libinitfunc)();

	// increment number of libs dynamically loaded
	DLib_index ++;

    return 0;
}
~~~

The library should include a function `initlib_<modulename>` to be executed when the module is loaded. This function should register functions to the command line interface, as done for all other modules that are part of the distribution.


