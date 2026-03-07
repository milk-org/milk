---
description: Create a new fpsexec standalone executable based on V2 compute unit template
---

This workflow automates the creation of a new fpsexec compute unit standalone executable. It handles copying the standard V2 template, renaming key fields, and configuring the module's CMakeLists.txt to build the executable.

1. Ask the user for the following required information if they haven't provided it yet:
   - Target directory (e.g., `~/src/milk/src/COREMOD_arith`)
   - C filename for the new unit (e.g., `my_new_op.c`)
   - Executable suffix / CLI cmdkey (e.g., `myop`, which creates `milk-fpsexec-myop`)
   - FPS shared memory name (e.g., `myopfps`, no spaces)
   - Module library name (e.g., `milkCOREMODarith`)
   - A one-line description for the `-h1` / `--help-oneline` output

2. Copy the template `~/src/milk/src/milk_module_example/examplefunc_fps_cli_poc.c` to the target directory with the given C filename.

3. Update the newly created C file to replace the default placeholders with the user's specifics:
   - Locate the `FPS_APP_INFO` struct in section 1 and update:
     - `.fps_name` to the provided FPS shared memory name.
     - `.cmdkey` to the provided CLI cmdkey.
     - `.description` to the provided one-line description.
   - Locate the `CLIADDCMD_milk_module_example__fpscli` function in section 7 and rename it to a fitting name using the module and cmdkey (e.g., `CLIADDCMD_COREMOD_arith__myop`).

4. Append the CMake target definitions to the `CMakeLists.txt` file in the target directory to build and install the new executable. Example configuration to append:
   ```cmake
   add_executable(milk-fpsexec-[cmdkey] [C filename])
   target_link_libraries(milk-fpsexec-[cmdkey] CLIcore [Module library name] milkfpsCLI)
   target_include_directories(milk-fpsexec-[cmdkey] PRIVATE ${PROJECT_SOURCE_DIR}/.. ${CMAKE_CURRENT_SOURCE_DIR})
   target_compile_definitions(milk-fpsexec-[cmdkey] PRIVATE FPS_STANDALONE)
   install(TARGETS milk-fpsexec-[cmdkey] DESTINATION bin)
   ```

5. Notify the user that the boilerplate for the new V2 compute unit has been set up successfully. Instruct them to modify the `FPS_PARAMS` and `fpsexec()` computation logic in the generated C file as needed. Remind them they can use the `/compile-test` workflow afterwards to verify everything compiles.
