
# Module Files {#page_ModuleFiles}

The code is arranged in modules. Source code, documentation and additional files for each modules are located in :

	./src/<modulename>/



|  File / Directory                       |  Content                           |
|-----------------------------------------|------------------------------------|
| ./src/<modulename>/<modulename>.c       | C source code                      |
| ./src/<modulename>/<modulename>.h       | Function prototypes                |
| ./src/<modulename>/Makefile.am          | Makefile input                     |
| ./src/<modulename>/doc/                 | documentation                      |
| ./src/<modulename>/docdir/              | extended documentation (optional)  |
| ./src/<modulename>/data/                | module data file       (optional)  |
| ./src/<modulename>/scripts/             | high level scripts     (optional)  |
| ./src/<modulename>/examples/            | examples               (optional)  |


Modules are compiled to object files, stored in ./libs/ (static and shared objects).


