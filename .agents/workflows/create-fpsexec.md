---
description: Create a new fpsexec standalone executable based on V2 compute unit template
---

This workflow automates the creation of a new fpsexec compute unit standalone executable. It handles copying the standard V2 template, renaming key fields, and configuring the module's CMakeLists.txt to build the executable.

See also: [Developer Tutorial](docs/developer/tutorial.md) ·
[Adding Plugins](docs/developer/plugins.md) ·
[Template Source Code](docs/developer/TemplateSourceCode.md)

1. Ask the user for the following required information if they haven't provided it yet:
   - Target directory (e.g., `~/src/milk/src/COREMOD_arith`)
   - C filename for the new unit (e.g., `my_new_op.c`)
   - Executable suffix / CLI cmdkey (e.g., `myop`, which creates `milk-fpsexec-myop`)
   - FPS shared memory name (e.g., `myopfps`, no spaces)
   - Module library name (e.g., `milkCOREMODarith`)
   - A one-line description for the `-h1` / `--help-oneline` output
   - Whether this is a milk or cacao standalone
   - If cacao: whether it needs plugin libraries (fft/imagegen/imagefilter/imagebasic)

2. **Parameter Design**: Consult the `pseudocode-to-compute-unit` skill. Analyze the intended computation and explicitly map variables to `FPTYPE_*` parameters. Determine which inputs will trigger the computation and which are tunable scalars.

3. Copy the template `~/src/milk/src/milk_module_example/examplefunc_fps_cli_poc.c` to the target directory with the given C filename.

3. Update the newly created C file to replace the default placeholders with the user's specifics:
   - Locate the `FPS_APP_INFO` struct in section 1 and update:
     - `.fps_name` to the provided FPS shared memory name.
     - `.cmdkey` to the provided CLI cmdkey.
     - `.description` to the provided one-line description.
   - Locate the `CLIADDCMD_milk_module_example__fpscli` function in section 7 and rename it to a fitting name using the module and cmdkey (e.g., `CLIADDCMD_COREMOD_arith__myop`).
   - The template already includes the correct `#ifdef MILK_NO_CLI` conditional include pattern — no changes needed there.

4. Append a **single-line** CMake target to the `CMakeLists.txt` file in the target directory. Use the standard function that matches the project:
   ```cmake
   # For milk modules:
   add_milk_standalone(cmdkey source_file.c)

   # For cacao modules (no plugin deps):
   add_cacao_standalone(cmdkey source_file.c)

   # For cacao modules that use fft/imagegen/imagefilter/imagebasic:
   add_cacao_standalone_plugins(cmdkey source_file.c)
   ```
   **DO NOT** use the old 4-line pattern (`add_executable` / `target_link_libraries` / `target_include_directories` / `target_compile_definitions`). These functions handle all of that automatically.

   If the standalone needs additional libraries beyond the standard set, add a `target_link_libraries` call on the next line:
   ```cmake
   add_cacao_standalone(my-func myfunction.c)
   target_link_libraries(cacao-fpsexec-my-func PUBLIC milkstatistic)
   ```

5. Notify the user that the boilerplate for the new V2 compute unit has been set up successfully. Instruct them to modify the `FPS_PARAMS` and `fpsexec()` computation logic in the generated C file as needed. Remind them they can use the [`/compile-test`](compile-test.md) workflow afterwards to verify everything compiles.
