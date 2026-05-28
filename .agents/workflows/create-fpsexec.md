---
description: Create a new fpsexec standalone executable based on V2 compute unit template
---

This workflow automates the creation of a new fpsexec compute unit standalone executable. It handles copying the standard V2 template, renaming key fields, and configuring the module's CMakeLists.txt to build the executable.

See also: [Developer Tutorial](docs/developer/tutorial.md) ·
[Adding Plugins](docs/developer/plugins.md) ·
[Template Source Code](docs/developer/TemplateSourceCode.md)

**Skills to consult** (in order):

1. `pseudocode-to-compute-unit` — algorithm
   analysis and parameter mapping
2. `fps-parameter-guide` — FPTYPE/FPFLAG
   reference and X-macro syntax
3. `api-quick-reference` — IMGID API, stream
   write protocol, required headers
4. `cmake-patterns` — standalone target setup

**Rules to review**: `fpsexec-conventions`,
`common-agent-mistakes`

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

4. Update the newly created C file:
   - **Section 1** (`FPS_APP_INFO`): set
     `.fps_name`, `.cmdkey`, `.description`.
   - **Section 2**: declare local C variables
     for all parameters. Use
     `char var[FUNCTION_PARAMETER_STRMAXLEN]`
     for string types.
   - **Section 3**: update the `FPS_PARAMS`
     X-macro. Use `var` (no `&`) for strings,
     `&var` for scalars.
   - **Section 4**: implement the computation
     in `fpsexec()`. Parameters are synced.
   - **Section 5**: bindings and `CLIcmddata`
     are auto-generated from `FPS_PARAMS`.
     Ensure `FPS_CMDSETTINGS_INIT(dft,
CLIcmddata, FPS_app_info)` is present.
   - **Section 6**: update `compute_function()`
     wrapper with IMGID setup and the
     processinfo loop.
   - **Section 7**: rename
     `CLIADDCMD_milk_module_example__fpscli`
     to match your module/function.

5. Update CMakeLists.txt:

   a. **Add to SOURCEFILES**: append the new
   `.c` file to the module's `SOURCEFILES`
   list so it compiles into the shared lib.

   b. **Add standalone target**: use the
   standard helper:

   ```cmake
   # For milk modules:
   add_milk_standalone(cmdkey source_file.c)

   # For cacao modules:
   add_cacao_standalone(cmdkey source_file.c)

   # For cacao with plugin deps:
   add_cacao_standalone_plugins(
       cmdkey source_file.c)
   ```

   If extra link deps are needed:

   ```cmake
   add_cacao_standalone(my-func myfunction.c)
   target_link_libraries(
       cacao-fpsexec-my-func
       PUBLIC milkstatistic)
   ```

   **DO NOT** use the old 4-line pattern.

6. **Register**: add the `CLIADDCMD_*()` call
   in the module's `initModule()` function and
   `#include` the header.

7. **Compile and Smoke Test**:

   Run [`/compile-test`](compile-test.md), then:

   ```bash
   milk-fpsexec-<name> -h
   milk-fpsexec-<name> -h1
   milk-fpsexec-list | grep <name>
   ```

   Verify help renders and the executable
   appears in the list.
