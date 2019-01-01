# Loading and Creating Modules {#page_LoadingModules}

@note This file: ./src/CommandLineInterface/doc/LoadingModules.md


Users can create additional modules, and bu following a few milk-specific conventions, link their functions to the milk CLI. Additional modules can be added to the existing build process (see "Compiling and linking modules with Autotools") or compiled separately and then loaded at runtime (see "Linking at runtime").

The second option is more flexible, and allows user to choose compiliation options, and when/if to load the module. It will however require to pay attention to dependencies.



---

## Compiling and linking modules with Autotools

Source code is organized in modules. Modules are compiled and linked using the libtool / autotools process, as shared libtool convenience libraries. 

See : https://www.gnu.org/software/automake/manual/html_node/Libtool-Convenience-Libraries.html

The list of modules currently appears in multiple files associated with the build process. The build process uses autotools, which has the advantage of supporting many platforms. 

Users adding modules are required to add the modules to 3 files, as described below.

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





---

## Linking at runtime with dlopen()

### Loading modules into milk

Alternatively, users may create their own libraries and link them at runtime using the `soload` or function of the command line interface. At any time, a module can be loaded from the CLI using:

	milk > soload <mymodulename.so>

See: http://www.linux-mag.com/id/1028/ for instructions to create shared libraries that can be loaded as modules. 

The library should include a function `initlib_<modulename>` to be executed when the module is loaded. This function should register functions to the command line interface, as done for all other modules that are part of the distribution.

### Example code

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
	
