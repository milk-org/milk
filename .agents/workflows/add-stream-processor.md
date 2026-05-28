---
description: Scaffold a stream processing loop compute unit
---

# Add a Stream Processor

Use this workflow when creating a compute unit that
reads from an input stream, processes frames, and
writes to an output stream in a continuous loop.
The key template is
`src/milk_module_example/examplefunc4_streamprocess.c`.

**Skills to consult** (in order):

1. `fps-parameter-guide` — parameter types,
   stream flag combos
2. `imagestream-internals` — stream write
   protocol, semaphore model
3. `api-quick-reference` — IMGID lifecycle,
   required headers
4. `cmake-patterns` — standalone target setup

**Rules to review**: `fpsexec-conventions`,
`common-agent-mistakes`

## 1. Gather Information

Ask the user for:

- **Target directory** (e.g.,
  `~/src/milk/plugins/milk-extra-src/mymodule`)
- **C filename** (e.g., `mystream_proc.c`)
- **Executable suffix / CLI cmdkey** (e.g.,
  `mystreamproc`, creates `milk-fpsexec-mystreamproc`)
- **FPS shared memory name** (e.g., `mystreamproc`)
- **Module library name** (e.g., `milkmymodule`)
- **Input stream parameter name** (e.g., `.in_name`)
- **Output stream parameter name** (e.g., `.out_name`)
- **One-line description**
- **milk or cacao** standalone

## 2. Copy and Rename the Template

Copy `src/milk_module_example/examplefunc4_streamprocess.c`
to the target directory with the given filename.

## 3. Update Identity (Section 1)

In the `FPS_APP_INFO` struct, update:

- `.fps_name` to the FPS shared memory name
- `.cmdkey` to the CLI keyword
- `.description` to the one-line description

## 4. Update Parameters (Sections 2–3)

- Replace `.in_name` / `.out_name` with the user's
  parameter names if different.
- Add or remove parameters in `FPS_PARAMS` as needed.
- For each parameter, declare a matching local C
  variable in section 2.

## 5. Update the Stream Processing Function

Modify `streamprocess()`:

- Update `resolveIMGID()` and `imcreateIMGID()` calls
  to match parameter names.
- Replace the per-pixel sqrt() example with the
  user's actual processing logic.
- Keep `processinfo_update_output_stream()` in the
  compute loop — it handles semaphore posting and
  timing metadata.

## 6. Key Patterns to Preserve

The stream processor template uses patterns that
differ from a basic FPS compute unit:

1. **IMGID lifecycle**: Create `IMGID` from the name
   parameter, then `resolveIMGID()` to connect to
   shared memory. Use `imgid_copy()` if output should
   match input format.

2. **Loop structure**:

   ```c
   INSERT_STD_PROCINFO_COMPUTEFUNC_INIT
   INSERT_STD_PROCINFO_COMPUTEFUNC_LOOPSTART
   {
       streamprocess(&inimg, &outimg);
       processinfo_update_output_stream(
           processinfo, outimg.im, inimg.im);
   }
   INSERT_STD_PROCINFO_COMPUTEFUNC_END
   ```

   Set `outimg->md->write = 1` before modifying
   pixels in your processing function.
   `processinfo_update_output_stream()` handles
   `write = 0` and semaphore posting.

3. **Cleanup**: Call `imgid_free()` for all IMGIDs
   after the loop ends.

## 7. Update CMake

Append the standalone target to `CMakeLists.txt`:

```cmake
# For milk:
add_milk_standalone(cmdkey source_file.c)

# For cacao:
add_cacao_standalone(cmdkey source_file.c)
```

Also add the `.c` file to `SOURCEFILES` in
the module's library build.

## 8. Update CLI Registration

Rename the `CLIADDCMD_milk_module_example__streamprocess`
function to match the module and function name.
Add the call to the module's `initModule()`.

## 9. Compile and Verify

Run the [`/compile-test`](compile-test.md)
workflow, then verify:

```bash
milk-fpsexec-<name> -h
milk-fpsexec-<name> -h1
milk-fpsexec-list | grep <name>
```

## 10. Notify

Tell the user the stream processor boilerplate is
ready. Remind them to customize the processing logic
in `streamprocess()`.
