# CMake Build Helpers (`cmake/`)

Shared CMake modules for `MILK_RUNNABLE` compile-once modules, `fpsexec` standalone stub generation,
build-layer guards, optional external dependency detection, static LTO archives, and Profile-Guided
Optimization (PGO).

See also: [Programmer's Guide §6](../docs/arch/programmers_guide.md#6-cmakelists-txt-conventions) ·
[Dependency System](../docs/developer/dependency_system.md) · [PGO & LTO](../docs/operations/pgo.md)

---

## Files

| File                        | Description                                                                                                       |
| :-------------------------- | :---------------------------------------------------------------------------------------------------------------- |
| `milk_runnable.cmake`       | `milk_add_runnable_module()`, `milk_runnable_module_usage()`, and `milk_add_fpsexec_stubs()` for `MILK_RUNNABLE`  |
| `milk_standalone.cmake`     | Common engine link set (`_MILK_STANDALONE_LIBS`), embedded build tags, and PGO/LTO helpers for `fpsexec` binaries |
| `milk_static_lto.cmake`     | `milk_link_module()`: links `_static` module archives when `USE_STATIC_LTO=ON`                                    |
| `milk_layer_guards.cmake`   | Build-tier `#include` and engine-layer isolation checks (also run as a CTest script)                              |
| `milk_extensions.cmake`     | `milk_apply_extensions()`: scans C/C++ sources for `MILK_CMAKE_REQUEST_*` / `MILK_CMAKE_MANDATE_*` directives     |
| `pgo.cmake`                 | Profile-Guided Optimization (`USE_PGO=GENERATE` / `USE_PGO=USE`) compiler and linker configuration                |
| `have_blas_lapacke.cmake`   | Detects OpenBLAS / Intel MKL / LAPACKE and sets `HAVE_BLAS`, `HAVE_OPENBLAS`, `HAVE_MKL`, `HAVE_LAPACKE`          |
| `have_cfitsio.cmake`        | Detects `cfitsio` via `pkg-config` and sets `HAVE_CFITSIO`                                                        |
| `have_cuda.cmake`           | Detects CUDA compiler and toolkit (`CUDAToolkit`) and sets `HAVE_CUDA`                                            |
