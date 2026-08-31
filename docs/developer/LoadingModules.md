# Loading, Creating Additional Modules

Users can create additional modules, and following a few milk-specific conventions, link their
functions to the milk CLI. Additional modules can be added to the existing build process or compiled
separately and then loaded at runtime.

---

## 1. Overview

milk is oganized in modules, each compiled as a shared object and loaded at runtime.

## 1.1. Compiling

While milk comes with its own set of modules, others can be added at runtime. Modules that conform
to milk specifications can be downloaded and added by following these steps:

    # execute these commands from the source root directory (for example ~./milk/)
    # move to source subdir, where module directories are located
    $ cd ./src

    # Download a new module
    $ git clone https://github.com/milk-org/NewModuleName

    # move to build directory
    $ cd ../_build

    # Add new module to compile script
    $ cmake .. -DEXTRAMODULES="NewModuleName"

    # compile and install
    $ sudo make install

## 1.2. Runtime linking

Note that the commands above will compile the module into a shared object, but it will not be loaded
by default at runtime. There are 4 options to load the module into the milk executable, detailed in
the following section. For example, the new module may be loaded within the milk command line
interface as follows:

    # start milk
    $ milk-cli

    # load module from command line interface
    milk-cli > mload NewModuleName

    # check list of modules. NewModuleName should appear
    milk-cli > m?

## 1.3. A simple example

milk includes an example module, which is not compiled or linked by default, to demonstrate how to
write and add modules.

The example module is in directory `src/milk_module_example` :

    $ cd src/milk_module_example
    $ ls
    CMakeLists.txt
    examplefunc1.c
    examplefunc1.h
    examplefunc2_FPS.c
    examplefunc2_FPS.h
    examplefunc4_streamprocess.c
    examplefunc4_streamprocess.h
    examplefunc_fps_cli_poc.c    # V2 template
    milk_module_example.c
    milk_module_example.h

Check the source code (well documented) to see how modules are written. To compile :

    $ cd _build
    $ cmake .. -DEXTRAMODULES="milk_module_example"
    $ sudo make install

Lets run it :

    # start milk
    $ milk-cli

    # load module as "mex" (short for module example)
    milk-cli > mloadas milk_module_example mex

    # check we have it
    milk-cli > m?

    # run the test function, which creates an image
    # note you can tab complete after typying "mex." to list functions
    milk-cli > mex.func1 im1 1.2

    # check the image is in memory
    milk-cli > mem.listim

---

## 2. Linking Modules to CLI

## 2.1. Automatic linking from `./lib/` directory

Any shared object in the `./lib/` subdirectory of source code will be loaded upon startup.

## 2.2. Linking module from within CLI with soload

Pre-compiled modules can be linked with the `soload` command within the CLI:

    milk-cli > soload <fullsoname>

The `fullsoname` is the shared object file name, path relative to current directory. For example
"../mylib/myodule.so".

Provided that the module follows milk conventions, linking the module will add the corresponding
functions to the CLI. This can be checked by probing the module functions:

    milk-cli > m?                # lists linked modules
    milk-cli > m? <modulename>   # lists CLI commands in the module

## 2.3. Linking module from within CLI with mload

By default, modules shared objects are installed in `/usr/local/lib`, and are named
`lib<ModuleName>.so`. With these assumptions satisfied, modules can be linked from within the CLI
with the `mload` command:

    milk-cli > mload <modulename>

Alternatively, a short name can be specified

    milk-cli > mloadas <modulename> <shortname>

Module functions are called from the command line interface prompt:

    milk-cli > <shortname>.<functionname> <arguments...>

## 2.4. Using environment variable `CLI_ADD_LIBS` to link shared objects

Upon startup, milk will read the CLI_ADD_LIBS environment variable to link shared objects. For
example:

    CLI_ADD_LIBS="/usr/local/libs/libMyFirstModule.so /usr/local/libs/libMySecondModule.so" milk-cli

will link modules `MyFirstModule` and `MySecondModule`.

<!-- prettier-ignore -->
!!! note
    Shared object names can be separated by space, semicolon, or comma.

---

## 3. Writing and Compiling Modules

## 3.1. Principles

Modules that are always loaded upon startup are registered in the module's `initlib_<modulename>`
function, which is called automatically when the shared object is loaded. Additional modules may be
loaded using the C `dlopen()` command. The library should include a function `initlib_<modulename>`
to be executed when the module is loaded. This function should register functions to the command
line interface, as done for all other modules that are part of the distribution.

\see <http://www.linux-mag.com/id/1028/> for instructions to create shared libraries that can be
loaded as modules. \see
<https://www.gnu.org/software/automake/manual/html_node/Libtool-Convenience-Libraries.html>

## 3.2. Adding modules to the main package compilation

The preferred way to add modules is to have them within the main source code directory alongside
default modules, following the same conventions and locations as the default modules. A new module
should then have the following files in the `./src/<ModuleName>/` directory:

- `CMakeLists.txt` file
- source code files (.c and .h files)

The `EXTRAMODULES` option is then used to add entry(ies) to the list of compiled modules. For
example:

    cmake .. -DEXTRAMODULES="WFpropagate;OpticsMaterials"

will compile modules `WFpropagate` and `OpticsMaterials` in addition to default modules. The extra
modules shared objects will be `/usr/local/lib/libWFpropagate.so` and
`/usr/local/lib/libOpticsMaterials.so`, and can be loaded with any of the methods described in the
linking section.

<!-- prettier-ignore -->
!!! warning
    Adding entries with the EXTRAMODULES option will compile the corresponding shared objects, but will
    not have them loaded upon execution of the main executable by default. See section below on automatic loading.

## 3.3. Automatic loading

Several options are available to have the additional module(s) automatically loaded every time:

- Copy or link the shared object to the `./lib/` directory.

For example:

    ln -s /usr/local/lib/libWFpropagate.so ~/src/milk/lib/libWFpropagate.so

- Create a system-wide environment variable CLI_ADD_LIBS in `~/.bashrc`.

<!-- prettier-ignore -->
!!! note
    Several versions of the executable can also be defined, each with its own set of automatically
    loaded modules. For example, the following line can be saved as an executable script:

    CLI_ADD_LIBS="/usr/local/libs/libWFpropagate.so" milk-cli

## 3.4. Adding new module to github

We assume here that you have created a module and you would like to push it to the main github
package org (we assume here milk-org).

    # First, create the repo in github, then run the following commands:
    cd ./MyModuleName/
    git init
    git add .
    git commit -m "First commit"
    git remote add origin https://github.com/milk-org/MyModuleName
    git config credential.helper store       # For convenience
    git push --set-upstream origin main
    # Now we create framework-dev branch
    git checkout -b framework-dev
    git push --set-upstream origin framework-dev

## 4. Custom modules

Additional custom modules may also be compiled independently from the main compile process. This is
not the preferred option, and there is a performance hit, as the benefits of link-time optimization
will be lost.

To compile a custom module :

    cd exampleModule
    gcc -c -I.. -fPIC exampleModule.c
    gcc -c -fPIC compute_pi.c -Ofast
    gcc -shared -I.. -fPIC exampleModule.o compute_pi.o -o libexamplemodule.so -lc

To load the new modules automatically on milk startup, create a sym link in the milk's lib
directory:

    cd ~/src/milk/lib
    ln -s ../src/exampleModule/libexamplemodule.so libexamplemodule.so

Alternatively, you can load it from the CLI at anytime:

    milk-cli > soload "<pathtomilk>/src/exampleModule/libexamplemodule.so"

---

---

← [Documentation Index](../index.md)
