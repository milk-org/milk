# Loading, Creating Additional Modules {#page_LoadingModules}

@note This file: ./src/CommandLineInterface/doc/LoadingModules.md

[TOC]




---

Users can create additional modules, and bu following a few milk-specific conventions, link their functions to the milk CLI. Additional modules can be added to the existing build process or compiled separately and then loaded at runtime.


---



# Loading modules {#page_LoadingModules_Loading}


## Automatic loading from `./lib/` directory {#page_LoadingModules_loading_libdir}

Any shared object in the `./lib/` subdirectory of source code will be loaded upon startup.




## Loading module from within CLI with soload {#page_LoadingModules_loading_soload}

Pre-compiled modules can be loaded with the `soload` command within the CLI:

	milk> soload <fullsoname>

The `fullsoname` is the shared object file name, path relative to current directory. For example "../mylib/myodule.so".

Provided that the module follows milk conventions, loading the module will add the corresponding functions to the CLI. This can be checked by probing the module functions:

	milk> m? <modulename>
	

	
## Loading module from within CLI with mload {#page_LoadingModules_loading_mload}

By default, modules shared objects are installed in `/usr/local/lib`, and are named `lib<ModuleName>.so`. With these assumptions satisfied, modules can be loaded with the `mload` command:

	milk> mload <modulename>



## Using environment variable `CLI_ADD_LIBS` to load shared objects {#page_LoadingModules_loading_envvar}

Upon startup, milk will read the CLI_ADD_LIBS environment variable to load shared objects. For example:

	CLI_ADD_LIBS="/usr/local/libs/libMyFirstModule.so /usr/local/libs/libMySecondModule.so" milk
	
will load modules `MyFirstModule` and `MySecondModule`.

@note Shared object names can be separated by space, semicolumn, or comma.





---


# Writing and compiling milk modules {#page_LoadingModules_compiling}



## Principles {#page_LoadingModules_compiling_principles}

Modules that are always loaded upon startup are explicitely listed in CLImain.c. Additional modules may be loaded using the C dlopen() command. The library should include a function `initlib_<modulename>` to be executed when the module is loaded. This function should register functions to the command line interface, as done for all other modules that are part of the distribution.

\see http://www.linux-mag.com/id/1028/ for instructions to create shared libraries that can be loaded as modules. 
\see https://www.gnu.org/software/automake/manual/html_node/Libtool-Convenience-Libraries.html





## Adding modules to the main package compilation {#page_LoadingModules_compiling_compiling_cmake}

The preferred way to add modules is to have them within the main source code directory alongside default modules, following the same conventions and locations as the default modules. A new module should then have the following files in the `./src/<ModuleName>/` directory:

- `CMakeLists.txt` file
- source code files (.c and .h files)


The `EXTRAMODULES` option is then used to add entry(ies) to the list of compiled modules. For example:

	cmake .. -DEXTRAMODULES="WFpropagate;OpticsMaterials"

will compile modules `WFpropagate` and `OpticsMaterials` in addition to default modules. The extra modules shared objects will be `/usr/local/lib/libWFpropagate.so` and `/usr/local/lib/libOpticsMaterials.so`, and can be loaded with any of the methods described in @ref page_LoadingModules_Loading.


@attention Adding entries with the EXTRAMODULES option will compile the corresponding shared objects, but will not have them loaded upon execution of the main executable. See next section for automatic loading.





## Automatic loading

Several options are available to have the additional module(s) automatically loaded every time:

- Copy or link the shared object to the `./src/lib/` directory (see @ref page_LoadingModules_loading_libdir). 

For example:

	ln -s /usr/local/lib/libWFpropagate.so ~./lib/libWFpropagate.so


- Create a system-wide environment variable CLI_ADD_LIBS in `~/.bashrc` (see @ref page_LoadingModules_loading_envvar)

@note Several versions of the executable can also be defined, each with its own set of automatically loaded modules. For example, the following line can be saved as an executable script:

	CLI_ADD_LIBS="/usr/local/libs/libWFpropagate.so" milk




## Custom compilation: Example code {#page_LoadingModules_compiling_examplecode}

Additional modules may also be compiled independently from the main compile process. This is not the preferred option, and there is a performance hit, as the benefits of link-time optimization will be lost.

An example module is included in milk, in directory exampleModule.

To compile it:

	cd exampleModule
	gcc -c -I.. -fPIC exampleModule.c
	gcc -c -fPIC compute_pi.c -Ofast
	gcc -shared -I.. -fPIC exampleModule.o compute_pi.o -o libexamplemodule.so -lc
	
To load the new modules automatically on milk startup, create a sym link in the milk's lib directory:

	cd ~/src/milk/lib
	ln -s ../src/exampleModule/libexamplemodule.so libexamplemodule.so

Alternatively, you can load it from the CLI at anytime:

	milk > soload "<pathtomilk>/src/exampleModule/libexamplemodule.so"
	


---


