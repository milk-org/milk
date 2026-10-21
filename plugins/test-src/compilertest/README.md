# For compilation testing.

To be tested:

- standalones
- cli
- no cli
- cuda off
- cuda on

These files test the new compilation system. The module is made of `MILK_RUNNABLE` units (one
per file, built with `milk_add_runnable_module()`); the executables come from `EXPORT_FPSEXEC`,
and `compilertest.c` is the module registry.

The main CMakeLists.txt provides symbols for us

- the user passes the -DUSE_<X>=ON symbols
- the cmake/have_<X>.cmake files:
  - Success: set -DHAVE_<X>(=TRUE, but it's really a set/unset variable)
  - Failure: set -DUSE_<X>=OFF

request_cuda.c:

- Contains a // MILK_CMAKE_REQUEST_CUDA
- Must use #ifdef HAVE_CUDA guards

mandate_cuda.c:

- Contains a // MILK_CMAKE_MANDATE_CUDA guard
- Does not need to use #ifdef HAVE_CUDA guards
- Will be excluded entirely if -DUSE_CUDA=OFF (because unset by user, forced OFF, or missing CUDA)
- When excluded, its runnable is NULL in the registry (weak declaration in `compilertest.c`): the
  CLI command is absent and the executable is not built (see
  [Managing Dependencies](../../../docs/developer/dependency_system.md))

mandate_lapacke.c: same, with `// MILK_CMAKE_MANDATE_LAPACKE`.

## Tests

### CLI OFF

cmake .. -DUSE_CUDA=ON -DUSE_CLI=OFF
make, make install
milk-fpsexec-compilertest-mandatecuda fpsinit
milk-fpsexec-compilertest-mandatecuda runstart
---> CublasSaxpy
milk-fpsexec-compilertest-requestcuda fpsinit
milk-fpsexec-compilertest-requestcuda runstart
---> CublasSaxpy

cmake .. -DUSE_CUDA=OFF -DUSE_CLI=OFF
make, make install
milk-fpsexec-compilertest-mandatecuda fpsinit
---> Does not exist
milk-fpsexec-compilertest-requestcuda fpsinit
milk-fpsexec-compilertest-requestcuda runstart
---> Doing nothing

### CLI ON

Same as above.

cmake .. -DUSE_CUDA=ON

milk-cli
mload milkcompilertest
compilertest.mandcuda

cmake .. -DUSE_CUDA=OFF

milk-cli
mload milkcompilertest
---> loads without error
compilertest.reqcuda
---> runs
compilertest.mandcuda
---> unknown command
---> CublasSaxpy
compilertest.reqcuda
---> CublasSaxpy

cmake .. -DUSE_CUDA=OFF

milk-cli
mload milkcompilertest
compilertest.mandcuda
---> not found
compilertest.reqcuda
---> Doing nothing
