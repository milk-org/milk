cmake_minimum_required(VERSION 3.17)

# nanobind extensions of pyMilk: ImageStreamIOWrap, FpsWrap, ProcessInfoWrap.
#
# PYMILK_ENGINE selects where the C engine comes from:
#   SYSTEM  - an installed milk, located through pkg-config (milk.pc). milk's own build is untouched.
#   BUNDLED - the same sources (src/engine) as static PIC archives, private to each module.
#
# ImageStreamIOWrap.cpp belongs to ImageStreamIO (M-4) but is always compiled here, into pyMilk.

project(pyMilkBindings LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Root of the milk source tree. F-1 (de-nesting pyMilk) only changes this default.
set(PYMILK_MILK_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../.." CACHE PATH "milk source tree root")
set(PYMILK_ISIO_SOURCE_DIR "${PYMILK_MILK_SOURCE_DIR}/src/engine/ImageStreamIO")

set(PYMILK_ENGINE "SYSTEM" CACHE STRING "Engine provisioning: SYSTEM or BUNDLED")
set_property(CACHE PYMILK_ENGINE PROPERTY STRINGS SYSTEM BUNDLED)

# =======================================
# ENGINE: provides milk_engine::<lib> targets and milk_engine_headers
# =======================================
if(PYMILK_ENGINE STREQUAL "SYSTEM")
  find_package(PkgConfig REQUIRED)
  if(DEFINED ENV{MILK_INSTALLDIR})
    set(ENV{PKG_CONFIG_PATH} "$ENV{MILK_INSTALLDIR}/lib/pkgconfig:$ENV{PKG_CONFIG_PATH}")
  endif()

  # Only the version and paths are taken from the .pc files: milk.pc's Libs line lists
  # the CLI dependencies (readline, ncurses, cfitsio...), which the bindings don't need.
  pkg_check_modules(MILK REQUIRED milk)
  pkg_check_modules(ISIO REQUIRED ImageStreamIO)
  pkg_get_variable(MILK_PC_INCLUDEDIR milk includedir)
  pkg_get_variable(MILK_PC_LIBDIR milk libdir)
  message(STATUS "pyMilk engine: SYSTEM, milk ${MILK_VERSION}, ImageStreamIO ${ISIO_VERSION}, libs in ${MILK_PC_LIBDIR}")

  # milk installs some headers in per-library subdirectories, but its code includes them by bare name.
  add_library(milk_engine_headers INTERFACE)
  target_include_directories(milk_engine_headers INTERFACE
    "${MILK_PC_INCLUDEDIR}"
    "${MILK_PC_INCLUDEDIR}/ImageStreamIO"
    "${MILK_PC_INCLUDEDIR}/libmilkcommon"
    "${MILK_PC_INCLUDEDIR}/libmilkdata"
    "${MILK_PC_INCLUDEDIR}/libfps"
    "${MILK_PC_INCLUDEDIR}/libcomputeunit")

  foreach(lib ImageStreamIO milkcommon milkprocessinfo milkfps milkcomputeunit)
    find_library(PYMILK_LIB_${lib} NAMES ${lib} PATHS "${MILK_PC_LIBDIR}" NO_DEFAULT_PATH)
    if(NOT PYMILK_LIB_${lib})
      message(FATAL_ERROR "lib${lib} not found in ${MILK_PC_LIBDIR}")
    endif()
    add_library(milk_engine::${lib} SHARED IMPORTED)
    set_target_properties(milk_engine::${lib} PROPERTIES IMPORTED_LOCATION "${PYMILK_LIB_${lib}}")
    target_link_libraries(milk_engine::${lib} INTERFACE milk_engine_headers)
  endforeach()

  # The ImageStreamIOWrap source comes from this checkout, the library from the install:
  # refuse to mix ImageStreamIO versions.
  file(STRINGS "${PYMILK_ISIO_SOURCE_DIR}/CMakeLists.txt" _isio_ver REGEX "set *\\( *VERSION_(MAJOR|MINOR|PATCH) ")
  string(REGEX REPLACE ".*VERSION_MAJOR +([0-9]+).*VERSION_MINOR +([0-9]+).*VERSION_PATCH +([0-9]+).*"
         "\\1.\\2.\\3" _isio_src_version "${_isio_ver}")
  if(NOT _isio_src_version STREQUAL ISIO_VERSION)
    message(FATAL_ERROR "ImageStreamIO version mismatch: source tree ${_isio_src_version}, "
                        "installed ${ISIO_VERSION}. Rebuild and install milk from this checkout.")
  endif()

  set(PYMILK_ENGINE_RPATH "${MILK_PC_LIBDIR}")

  # CUDA: follow the installed libImageStreamIO.
  execute_process(COMMAND readelf -d "${PYMILK_LIB_ImageStreamIO}" OUTPUT_VARIABLE _isio_dyn ERROR_QUIET)
  if(_isio_dyn MATCHES "libcudart")
    set(_isio_cuda ON)
  else()
    set(_isio_cuda OFF)
  endif()
  set(USE_CUDA ${_isio_cuda} CACHE BOOL "Build with CUDA (default: as the installed libImageStreamIO)")
elseif(PYMILK_ENGINE STREQUAL "BUNDLED")
  message(STATUS "pyMilk engine: BUNDLED, static engine from ${PYMILK_MILK_SOURCE_DIR}/src/engine")

  # The engine's <lib>_static archives are built under USE_STATIC_LTO (no LTO flags reach here).
  # PIC so they link into the extension modules; no CLI and no cfitsio dependency.
  set(USE_STATIC_LTO ON)
  set(USE_CFITSIO OFF)
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
  set(CMAKE_C_VISIBILITY_PRESET hidden)
  add_compile_definitions(MILK_NO_CLI)
  set(USE_CUDA OFF CACHE BOOL "Build with CUDA")
  if(USE_CUDA)
    find_package(CUDAToolkit REQUIRED)
  endif()
  add_subdirectory("${PYMILK_MILK_SOURCE_DIR}/src/engine" engine EXCLUDE_FROM_ALL)

  # Same header layout as an installed milk (see the SYSTEM branch).
  set(_eng "${PYMILK_MILK_SOURCE_DIR}/src/engine")
  add_library(milk_engine_headers INTERFACE)
  target_include_directories(milk_engine_headers INTERFACE
    "${_eng}" "${_eng}/ImageStreamIO" "${_eng}/libmilkcommon" "${_eng}/libmilkdata"
    "${_eng}/libfps" "${_eng}/libcomputeunit" "${_eng}/libprocessinfo"
    "${CMAKE_BINARY_DIR}/config")

  foreach(lib ImageStreamIO milkcommon milkprocessinfo milkfps milkcomputeunit)
    add_library(milk_engine::${lib} INTERFACE IMPORTED)
    target_link_libraries(milk_engine::${lib} INTERFACE ${lib}_static milk_engine_headers)
  endforeach()

  # Keep the engine symbols private to each module: no clash with a libImageStreamIO.so
  # or libmilk*.so loaded in the same process.
  add_link_options(-Wl,--exclude-libs,ALL)
  set(PYMILK_ENGINE_RPATH "")
else()
  message(FATAL_ERROR "Unknown PYMILK_ENGINE '${PYMILK_ENGINE}' (SYSTEM or BUNDLED).")
endif()

# =======================================
# NANOBIND MODULES
# =======================================
find_package(Python 3.9 COMPONENTS Interpreter Development.Module REQUIRED)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(ImageStreamIOWrap "${PYMILK_ISIO_SOURCE_DIR}/python_module/ImageStreamIOWrap.cpp")
target_link_libraries(ImageStreamIOWrap PRIVATE milk_engine::ImageStreamIO)
if(USE_CUDA)
  find_package(CUDAToolkit REQUIRED)
  target_compile_definitions(ImageStreamIOWrap PRIVATE HAVE_CUDA)
  target_link_libraries(ImageStreamIOWrap PRIVATE CUDA::cudart)
endif()

nanobind_add_module(FpsWrap FpsWrap.cpp)
target_link_libraries(FpsWrap PRIVATE milk_engine::milkfps milk_engine::milkcomputeunit
                                      milk_engine::milkprocessinfo milk_engine::milkcommon
                                      milk_engine::ImageStreamIO)

nanobind_add_module(ProcessInfoWrap ProcessInfoWrap.cpp)
target_compile_definitions(ProcessInfoWrap PRIVATE PYMILK_ENGINE="${PYMILK_ENGINE}")
target_link_libraries(ProcessInfoWrap PRIVATE milk_engine::milkprocessinfo milk_engine::milkcommon
                                              milk_engine::ImageStreamIO)

set(PYMILK_MODULES ImageStreamIOWrap FpsWrap ProcessInfoWrap)

# The modules are used from the build tree (setup.py drops them into pyMilk/), so the
# build RPATH is what ships: point it at the engine libraries.
set_target_properties(${PYMILK_MODULES} PROPERTIES BUILD_RPATH "${PYMILK_ENGINE_RPATH}"
                                                    INSTALL_RPATH "${PYMILK_ENGINE_RPATH}")

if(CMAKE_BUILD_TYPE STREQUAL "Coverage")
  foreach(mod ${PYMILK_MODULES})
    target_compile_definitions(${mod} PRIVATE COVERAGE_BUILD=1)
    target_compile_options(${mod} PRIVATE --coverage)
    target_link_options(${mod} PRIVATE --coverage)
  endforeach()
endif()
