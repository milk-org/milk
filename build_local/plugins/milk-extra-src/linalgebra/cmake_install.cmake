# Install script for directory: /home/oguyon/src/milk/plugins/milk-extra-src/linalgebra

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/home/oguyon/local_milk/milk-1.03.00")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set path to fallback-tool for dependency-resolution.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libmilklinalgebra.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libmilklinalgebra.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libmilklinalgebra.so"
         RPATH "/home/oguyon/local_milk/milk-1.03.00/lib:/usr/lib/x86_64-linux-gnu/openblas-pthread:/usr/local/lib:/usr/local/include")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES "/home/oguyon/src/milk/build_local/plugins/milk-extra-src/linalgebra/libmilklinalgebra.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libmilklinalgebra.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libmilklinalgebra.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libmilklinalgebra.so"
         OLD_RPATH "/usr/lib/x86_64-linux-gnu/openblas-pthread:/usr/local/lib:/usr/local/include:/home/oguyon/src/milk/build_local/src/cli/CLIcore:/home/oguyon/src/milk/build_local/src/cli/libmilkscript:/home/oguyon/src/milk/build_local/src/coremods/COREMOD_arith:/home/oguyon/src/milk/build_local/src/coremods/COREMOD_iofits:/home/oguyon/src/milk/build_local/src/coremods/COREMOD_memory:/home/oguyon/src/milk/build_local/src/coremods/COREMOD_tools:/home/oguyon/src/milk/build_local/src/engine/libfpsseq:/home/oguyon/src/milk/build_local/src/engine/libfps:/home/oguyon/src/milk/build_local/src/engine/libprocessinfo:/home/oguyon/src/milk/build_local/src/engine/libmilkdata:/home/oguyon/src/milk/build_local/src/engine/ImageStreamIO:"
         NEW_RPATH "/home/oguyon/local_milk/milk-1.03.00/lib:/usr/lib/x86_64-linux-gnu/openblas-pthread:/usr/local/lib:/usr/local/include")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libmilklinalgebra.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  include("/home/oguyon/src/milk/build_local/plugins/milk-extra-src/linalgebra/CMakeFiles/milklinalgebra.dir/install-cxx-module-bmi-Release.cmake" OPTIONAL)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/linalgebra" TYPE FILE FILES
    "/home/oguyon/src/milk/plugins/milk-extra-src/linalgebra/linalgebra.h"
    "/home/oguyon/src/milk/plugins/milk-extra-src/linalgebra/basis_rotate_match.h"
    "/home/oguyon/src/milk/plugins/milk-extra-src/linalgebra/cublas_Coeff2Map_Loop.h"
    "/home/oguyon/src/milk/plugins/milk-extra-src/linalgebra/linalgebra_types.h"
    "/home/oguyon/src/milk/plugins/milk-extra-src/linalgebra/linalgebrainit.h"
    "/home/oguyon/src/milk/plugins/milk-extra-src/linalgebra/cublas_linalgebratest.h"
    "/home/oguyon/src/milk/plugins/milk-extra-src/linalgebra/GPU_loop_MultMat_free.h"
    "/home/oguyon/src/milk/plugins/milk-extra-src/linalgebra/GPU_loop_MultMat_execute.h"
    "/home/oguyon/src/milk/plugins/milk-extra-src/linalgebra/GPU_loop_MultMat_setup.h"
    "/home/oguyon/src/milk/plugins/milk-extra-src/linalgebra/GPU_SVD_computeControlMatrix.h"
    "/home/oguyon/src/milk/plugins/milk-extra-src/linalgebra/GPUloadCmat.h"
    "/home/oguyon/src/milk/plugins/milk-extra-src/linalgebra/GramSchmidt.h"
    "/home/oguyon/src/milk/plugins/milk-extra-src/linalgebra/magma_compute_SVDpseudoInverse_SVD.h"
    "/home/oguyon/src/milk/plugins/milk-extra-src/linalgebra/magma_compute_SVDpseudoInverse.h"
    "/home/oguyon/src/milk/plugins/milk-extra-src/linalgebra/magma_MatMatMult_testPseudoInverse.h"
    "/home/oguyon/src/milk/plugins/milk-extra-src/linalgebra/modalremap.h"
    "/home/oguyon/src/milk/plugins/milk-extra-src/linalgebra/MVM_CPU.h"
    "/home/oguyon/src/milk/plugins/milk-extra-src/linalgebra/PCAmatch.h"
    "/home/oguyon/src/milk/plugins/milk-extra-src/linalgebra/cublas_PCA.h"
    "/home/oguyon/src/milk/plugins/milk-extra-src/linalgebra/printGPUMATMULTCONF.h"
    "/home/oguyon/src/milk/plugins/milk-extra-src/linalgebra/MVMextractModes.h"
    "/home/oguyon/src/milk/plugins/milk-extra-src/linalgebra/Qexpand.h"
    "/home/oguyon/src/milk/plugins/milk-extra-src/linalgebra/SGEMM.h"
    "/home/oguyon/src/milk/plugins/milk-extra-src/linalgebra/SingularValueDecomp.h"
    "/home/oguyon/src/milk/plugins/milk-extra-src/linalgebra/SingularValueDecomp_mkM.h"
    "/home/oguyon/src/milk/plugins/milk-extra-src/linalgebra/SingularValueDecomp_mkU.h"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE PROGRAM FILES
    "/home/oguyon/src/milk/plugins/milk-extra-src/linalgebra/scripts/milk-linmap-calib"
    "/home/oguyon/src/milk/plugins/milk-extra-src/linalgebra/scripts/milk-modes-project"
    "/home/oguyon/src/milk/plugins/milk-extra-src/linalgebra/scripts/milk-modes-project-U"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libmilklinalgebra_compute.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libmilklinalgebra_compute.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libmilklinalgebra_compute.so"
         RPATH "/home/oguyon/local_milk/milk-1.03.00/lib:/usr/local/lib:/usr/lib/x86_64-linux-gnu/openblas-pthread:/usr/local/include")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES "/home/oguyon/src/milk/build_local/plugins/milk-extra-src/linalgebra/libmilklinalgebra_compute.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libmilklinalgebra_compute.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libmilklinalgebra_compute.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libmilklinalgebra_compute.so"
         OLD_RPATH "/usr/local/lib:/usr/lib/x86_64-linux-gnu/openblas-pthread:/usr/local/include:/home/oguyon/src/milk/build_local/src/engine/libfps:/home/oguyon/src/milk/build_local/src/engine/libprocessinfo:/home/oguyon/src/milk/build_local/src/engine/ImageStreamIO:"
         NEW_RPATH "/home/oguyon/local_milk/milk-1.03.00/lib:/usr/local/lib:/usr/lib/x86_64-linux-gnu/openblas-pthread:/usr/local/include")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libmilklinalgebra_compute.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  include("/home/oguyon/src/milk/build_local/plugins/milk-extra-src/linalgebra/CMakeFiles/milklinalgebra_compute.dir/install-cxx-module-bmi-Release.cmake" OPTIONAL)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-linalg-gramschmidt" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-linalg-gramschmidt")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-linalg-gramschmidt"
         RPATH "/home/oguyon/local_milk/milk-1.03.00/lib:/usr/local/include:/usr/local/lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/home/oguyon/src/milk/build_local/plugins/milk-extra-src/linalgebra/milk-fpsexec-linalg-gramschmidt")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-linalg-gramschmidt" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-linalg-gramschmidt")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-linalg-gramschmidt"
         OLD_RPATH "/usr/local/include:/usr/local/lib:/home/oguyon/src/milk/build_local/src/engine/libfps:/home/oguyon/src/milk/build_local/src/engine/libfpsseq:/home/oguyon/src/milk/build_local/src/engine/libmilkdata:/home/oguyon/src/milk/build_local/src/coremods/COREMOD_memory:/home/oguyon/src/milk/build_local/src/coremods/COREMOD_tools:/home/oguyon/src/milk/build_local/src/coremods/COREMOD_arith:/home/oguyon/src/milk/build_local/src/coremods/COREMOD_iofits:/home/oguyon/src/milk/build_local/src/engine/libprocessinfo:/home/oguyon/src/milk/build_local/src/engine/ImageStreamIO:"
         NEW_RPATH "/home/oguyon/local_milk/milk-1.03.00/lib:/usr/local/include:/usr/local/lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-linalg-gramschmidt")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  include("/home/oguyon/src/milk/build_local/plugins/milk-extra-src/linalgebra/CMakeFiles/milk-fpsexec-linalg-gramschmidt.dir/install-cxx-module-bmi-Release.cmake" OPTIONAL)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-linalg-MVMextract" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-linalg-MVMextract")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-linalg-MVMextract"
         RPATH "/home/oguyon/local_milk/milk-1.03.00/lib:/usr/local/include:/usr/local/lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/home/oguyon/src/milk/build_local/plugins/milk-extra-src/linalgebra/milk-fpsexec-linalg-MVMextract")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-linalg-MVMextract" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-linalg-MVMextract")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-linalg-MVMextract"
         OLD_RPATH "/usr/local/include:/usr/local/lib:/home/oguyon/src/milk/build_local/src/engine/libfps:/home/oguyon/src/milk/build_local/src/engine/libfpsseq:/home/oguyon/src/milk/build_local/src/engine/libmilkdata:/home/oguyon/src/milk/build_local/src/coremods/COREMOD_memory:/home/oguyon/src/milk/build_local/src/coremods/COREMOD_tools:/home/oguyon/src/milk/build_local/src/coremods/COREMOD_arith:/home/oguyon/src/milk/build_local/src/coremods/COREMOD_iofits:/home/oguyon/src/milk/build_local/src/engine/libprocessinfo:/home/oguyon/src/milk/build_local/src/engine/ImageStreamIO:"
         NEW_RPATH "/home/oguyon/local_milk/milk-1.03.00/lib:/usr/local/include:/usr/local/lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-linalg-MVMextract")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  include("/home/oguyon/src/milk/build_local/plugins/milk-extra-src/linalgebra/CMakeFiles/milk-fpsexec-linalg-MVMextract.dir/install-cxx-module-bmi-Release.cmake" OPTIONAL)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-linalg-PCAmatch" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-linalg-PCAmatch")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-linalg-PCAmatch"
         RPATH "/home/oguyon/local_milk/milk-1.03.00/lib:/usr/local/include:/usr/local/lib:/usr/lib/x86_64-linux-gnu/openblas-pthread")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/home/oguyon/src/milk/build_local/plugins/milk-extra-src/linalgebra/milk-fpsexec-linalg-PCAmatch")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-linalg-PCAmatch" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-linalg-PCAmatch")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-linalg-PCAmatch"
         OLD_RPATH "/usr/local/include:/usr/local/lib:/usr/lib/x86_64-linux-gnu/openblas-pthread:/home/oguyon/src/milk/build_local/src/engine/libfps:/home/oguyon/src/milk/build_local/src/engine/libfpsseq:/home/oguyon/src/milk/build_local/src/engine/libmilkdata:/home/oguyon/src/milk/build_local/src/coremods/COREMOD_memory:/home/oguyon/src/milk/build_local/src/coremods/COREMOD_tools:/home/oguyon/src/milk/build_local/src/coremods/COREMOD_arith:/home/oguyon/src/milk/build_local/src/coremods/COREMOD_iofits:/home/oguyon/src/milk/build_local/plugins/milk-extra-src/linalgebra:/home/oguyon/src/milk/build_local/src/engine/libprocessinfo:/home/oguyon/src/milk/build_local/src/engine/ImageStreamIO:"
         NEW_RPATH "/home/oguyon/local_milk/milk-1.03.00/lib:/usr/local/include:/usr/local/lib:/usr/lib/x86_64-linux-gnu/openblas-pthread")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-linalg-PCAmatch")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  include("/home/oguyon/src/milk/build_local/plugins/milk-extra-src/linalgebra/CMakeFiles/milk-fpsexec-linalg-PCAmatch.dir/install-cxx-module-bmi-Release.cmake" OPTIONAL)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-linalg-Qexpand" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-linalg-Qexpand")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-linalg-Qexpand"
         RPATH "/home/oguyon/local_milk/milk-1.03.00/lib:/usr/local/include:/usr/local/lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/home/oguyon/src/milk/build_local/plugins/milk-extra-src/linalgebra/milk-fpsexec-linalg-Qexpand")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-linalg-Qexpand" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-linalg-Qexpand")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-linalg-Qexpand"
         OLD_RPATH "/usr/local/include:/usr/local/lib:/home/oguyon/src/milk/build_local/src/engine/libfps:/home/oguyon/src/milk/build_local/src/engine/libfpsseq:/home/oguyon/src/milk/build_local/src/engine/libmilkdata:/home/oguyon/src/milk/build_local/src/coremods/COREMOD_memory:/home/oguyon/src/milk/build_local/src/coremods/COREMOD_tools:/home/oguyon/src/milk/build_local/src/coremods/COREMOD_arith:/home/oguyon/src/milk/build_local/src/coremods/COREMOD_iofits:/home/oguyon/src/milk/build_local/src/engine/libprocessinfo:/home/oguyon/src/milk/build_local/src/engine/ImageStreamIO:"
         NEW_RPATH "/home/oguyon/local_milk/milk-1.03.00/lib:/usr/local/include:/usr/local/lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-linalg-Qexpand")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  include("/home/oguyon/src/milk/build_local/plugins/milk-extra-src/linalgebra/CMakeFiles/milk-fpsexec-linalg-Qexpand.dir/install-cxx-module-bmi-Release.cmake" OPTIONAL)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-linalg-SGEMM" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-linalg-SGEMM")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-linalg-SGEMM"
         RPATH "/home/oguyon/local_milk/milk-1.03.00/lib:/usr/local/include:/usr/local/lib:/usr/lib/x86_64-linux-gnu/openblas-pthread")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/home/oguyon/src/milk/build_local/plugins/milk-extra-src/linalgebra/milk-fpsexec-linalg-SGEMM")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-linalg-SGEMM" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-linalg-SGEMM")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-linalg-SGEMM"
         OLD_RPATH "/usr/local/include:/usr/local/lib:/usr/lib/x86_64-linux-gnu/openblas-pthread:/home/oguyon/src/milk/build_local/src/engine/libfps:/home/oguyon/src/milk/build_local/src/engine/libfpsseq:/home/oguyon/src/milk/build_local/src/engine/libmilkdata:/home/oguyon/src/milk/build_local/src/coremods/COREMOD_memory:/home/oguyon/src/milk/build_local/src/coremods/COREMOD_tools:/home/oguyon/src/milk/build_local/src/coremods/COREMOD_arith:/home/oguyon/src/milk/build_local/src/coremods/COREMOD_iofits:/home/oguyon/src/milk/build_local/plugins/milk-extra-src/linalgebra:/home/oguyon/src/milk/build_local/src/engine/libprocessinfo:/home/oguyon/src/milk/build_local/src/engine/ImageStreamIO:"
         NEW_RPATH "/home/oguyon/local_milk/milk-1.03.00/lib:/usr/local/include:/usr/local/lib:/usr/lib/x86_64-linux-gnu/openblas-pthread")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-linalg-SGEMM")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  include("/home/oguyon/src/milk/build_local/plugins/milk-extra-src/linalgebra/CMakeFiles/milk-fpsexec-linalg-SGEMM.dir/install-cxx-module-bmi-Release.cmake" OPTIONAL)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-linalg-SVD" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-linalg-SVD")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-linalg-SVD"
         RPATH "/home/oguyon/local_milk/milk-1.03.00/lib:/usr/lib/x86_64-linux-gnu/openblas-pthread:/usr/local/include:/usr/local/lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/home/oguyon/src/milk/build_local/plugins/milk-extra-src/linalgebra/milk-fpsexec-linalg-SVD")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-linalg-SVD" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-linalg-SVD")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-linalg-SVD"
         OLD_RPATH "/usr/lib/x86_64-linux-gnu/openblas-pthread:/usr/local/include:/usr/local/lib:/home/oguyon/src/milk/build_local/src/engine/libfps:/home/oguyon/src/milk/build_local/src/engine/libfpsseq:/home/oguyon/src/milk/build_local/src/engine/libmilkdata:/home/oguyon/src/milk/build_local/src/coremods/COREMOD_memory:/home/oguyon/src/milk/build_local/src/coremods/COREMOD_tools:/home/oguyon/src/milk/build_local/src/coremods/COREMOD_arith:/home/oguyon/src/milk/build_local/src/coremods/COREMOD_iofits:/home/oguyon/src/milk/build_local/plugins/milk-extra-src/linalgebra:/home/oguyon/src/milk/build_local/src/engine/libprocessinfo:/home/oguyon/src/milk/build_local/src/engine/ImageStreamIO:"
         NEW_RPATH "/home/oguyon/local_milk/milk-1.03.00/lib:/usr/lib/x86_64-linux-gnu/openblas-pthread:/usr/local/include:/usr/local/lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-linalg-SVD")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  include("/home/oguyon/src/milk/build_local/plugins/milk-extra-src/linalgebra/CMakeFiles/milk-fpsexec-linalg-SVD.dir/install-cxx-module-bmi-Release.cmake" OPTIONAL)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-linalg-SVDmkM" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-linalg-SVDmkM")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-linalg-SVDmkM"
         RPATH "/home/oguyon/local_milk/milk-1.03.00/lib:/usr/local/include:/usr/local/lib:/usr/lib/x86_64-linux-gnu/openblas-pthread")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/home/oguyon/src/milk/build_local/plugins/milk-extra-src/linalgebra/milk-fpsexec-linalg-SVDmkM")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-linalg-SVDmkM" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-linalg-SVDmkM")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-linalg-SVDmkM"
         OLD_RPATH "/usr/local/include:/usr/local/lib:/usr/lib/x86_64-linux-gnu/openblas-pthread:/home/oguyon/src/milk/build_local/src/engine/libfps:/home/oguyon/src/milk/build_local/src/engine/libfpsseq:/home/oguyon/src/milk/build_local/src/engine/libmilkdata:/home/oguyon/src/milk/build_local/src/coremods/COREMOD_memory:/home/oguyon/src/milk/build_local/src/coremods/COREMOD_tools:/home/oguyon/src/milk/build_local/src/coremods/COREMOD_arith:/home/oguyon/src/milk/build_local/src/coremods/COREMOD_iofits:/home/oguyon/src/milk/build_local/plugins/milk-extra-src/linalgebra:/home/oguyon/src/milk/build_local/src/engine/libprocessinfo:/home/oguyon/src/milk/build_local/src/engine/ImageStreamIO:"
         NEW_RPATH "/home/oguyon/local_milk/milk-1.03.00/lib:/usr/local/include:/usr/local/lib:/usr/lib/x86_64-linux-gnu/openblas-pthread")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-linalg-SVDmkM")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  include("/home/oguyon/src/milk/build_local/plugins/milk-extra-src/linalgebra/CMakeFiles/milk-fpsexec-linalg-SVDmkM.dir/install-cxx-module-bmi-Release.cmake" OPTIONAL)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-linalg-SVDmkU" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-linalg-SVDmkU")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-linalg-SVDmkU"
         RPATH "/home/oguyon/local_milk/milk-1.03.00/lib:/usr/local/include:/usr/local/lib:/usr/lib/x86_64-linux-gnu/openblas-pthread")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/home/oguyon/src/milk/build_local/plugins/milk-extra-src/linalgebra/milk-fpsexec-linalg-SVDmkU")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-linalg-SVDmkU" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-linalg-SVDmkU")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-linalg-SVDmkU"
         OLD_RPATH "/usr/local/include:/usr/local/lib:/usr/lib/x86_64-linux-gnu/openblas-pthread:/home/oguyon/src/milk/build_local/src/engine/libfps:/home/oguyon/src/milk/build_local/src/engine/libfpsseq:/home/oguyon/src/milk/build_local/src/engine/libmilkdata:/home/oguyon/src/milk/build_local/src/coremods/COREMOD_memory:/home/oguyon/src/milk/build_local/src/coremods/COREMOD_tools:/home/oguyon/src/milk/build_local/src/coremods/COREMOD_arith:/home/oguyon/src/milk/build_local/src/coremods/COREMOD_iofits:/home/oguyon/src/milk/build_local/plugins/milk-extra-src/linalgebra:/home/oguyon/src/milk/build_local/src/engine/libprocessinfo:/home/oguyon/src/milk/build_local/src/engine/ImageStreamIO:"
         NEW_RPATH "/home/oguyon/local_milk/milk-1.03.00/lib:/usr/local/include:/usr/local/lib:/usr/lib/x86_64-linux-gnu/openblas-pthread")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-linalg-SVDmkU")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  include("/home/oguyon/src/milk/build_local/plugins/milk-extra-src/linalgebra/CMakeFiles/milk-fpsexec-linalg-SVDmkU.dir/install-cxx-module-bmi-Release.cmake" OPTIONAL)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-linalg-basisrotmatch" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-linalg-basisrotmatch")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-linalg-basisrotmatch"
         RPATH "/home/oguyon/local_milk/milk-1.03.00/lib:/usr/local/include:/usr/local/lib:/usr/lib/x86_64-linux-gnu/openblas-pthread")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/home/oguyon/src/milk/build_local/plugins/milk-extra-src/linalgebra/milk-fpsexec-linalg-basisrotmatch")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-linalg-basisrotmatch" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-linalg-basisrotmatch")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-linalg-basisrotmatch"
         OLD_RPATH "/usr/local/include:/usr/local/lib:/usr/lib/x86_64-linux-gnu/openblas-pthread:/home/oguyon/src/milk/build_local/src/engine/libfps:/home/oguyon/src/milk/build_local/src/engine/libfpsseq:/home/oguyon/src/milk/build_local/src/engine/libmilkdata:/home/oguyon/src/milk/build_local/src/coremods/COREMOD_memory:/home/oguyon/src/milk/build_local/src/coremods/COREMOD_tools:/home/oguyon/src/milk/build_local/src/coremods/COREMOD_arith:/home/oguyon/src/milk/build_local/src/coremods/COREMOD_iofits:/home/oguyon/src/milk/build_local/plugins/milk-extra-src/linalgebra:/home/oguyon/src/milk/build_local/src/engine/libprocessinfo:/home/oguyon/src/milk/build_local/src/engine/ImageStreamIO:"
         NEW_RPATH "/home/oguyon/local_milk/milk-1.03.00/lib:/usr/local/include:/usr/local/lib:/usr/lib/x86_64-linux-gnu/openblas-pthread")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-linalg-basisrotmatch")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  include("/home/oguyon/src/milk/build_local/plugins/milk-extra-src/linalgebra/CMakeFiles/milk-fpsexec-linalg-basisrotmatch.dir/install-cxx-module-bmi-Release.cmake" OPTIONAL)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-linalg-cublasPCA" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-linalg-cublasPCA")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-linalg-cublasPCA"
         RPATH "/home/oguyon/local_milk/milk-1.03.00/lib:/usr/local/include:/usr/local/lib:/usr/lib/x86_64-linux-gnu/openblas-pthread")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/home/oguyon/src/milk/build_local/plugins/milk-extra-src/linalgebra/milk-fpsexec-linalg-cublasPCA")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-linalg-cublasPCA" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-linalg-cublasPCA")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-linalg-cublasPCA"
         OLD_RPATH "/usr/local/include:/usr/local/lib:/usr/lib/x86_64-linux-gnu/openblas-pthread:/home/oguyon/src/milk/build_local/src/engine/libfps:/home/oguyon/src/milk/build_local/src/engine/libfpsseq:/home/oguyon/src/milk/build_local/src/engine/libmilkdata:/home/oguyon/src/milk/build_local/src/coremods/COREMOD_memory:/home/oguyon/src/milk/build_local/src/coremods/COREMOD_tools:/home/oguyon/src/milk/build_local/src/coremods/COREMOD_arith:/home/oguyon/src/milk/build_local/src/coremods/COREMOD_iofits:/home/oguyon/src/milk/build_local/plugins/milk-extra-src/linalgebra:/home/oguyon/src/milk/build_local/src/engine/libprocessinfo:/home/oguyon/src/milk/build_local/src/engine/ImageStreamIO:"
         NEW_RPATH "/home/oguyon/local_milk/milk-1.03.00/lib:/usr/local/include:/usr/local/lib:/usr/lib/x86_64-linux-gnu/openblas-pthread")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-linalg-cublasPCA")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  include("/home/oguyon/src/milk/build_local/plugins/milk-extra-src/linalgebra/CMakeFiles/milk-fpsexec-linalg-cublasPCA.dir/install-cxx-module-bmi-Release.cmake" OPTIONAL)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-linalg-modalremap" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-linalg-modalremap")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-linalg-modalremap"
         RPATH "/home/oguyon/local_milk/milk-1.03.00/lib:/usr/local/include:/usr/local/lib:/usr/lib/x86_64-linux-gnu/openblas-pthread")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/home/oguyon/src/milk/build_local/plugins/milk-extra-src/linalgebra/milk-fpsexec-linalg-modalremap")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-linalg-modalremap" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-linalg-modalremap")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-linalg-modalremap"
         OLD_RPATH "/usr/local/include:/usr/local/lib:/usr/lib/x86_64-linux-gnu/openblas-pthread:/home/oguyon/src/milk/build_local/src/engine/libfps:/home/oguyon/src/milk/build_local/src/engine/libfpsseq:/home/oguyon/src/milk/build_local/src/engine/libmilkdata:/home/oguyon/src/milk/build_local/src/coremods/COREMOD_memory:/home/oguyon/src/milk/build_local/src/coremods/COREMOD_tools:/home/oguyon/src/milk/build_local/src/coremods/COREMOD_arith:/home/oguyon/src/milk/build_local/src/coremods/COREMOD_iofits:/home/oguyon/src/milk/build_local/plugins/milk-extra-src/linalgebra:/home/oguyon/src/milk/build_local/src/engine/libprocessinfo:/home/oguyon/src/milk/build_local/src/engine/ImageStreamIO:"
         NEW_RPATH "/home/oguyon/local_milk/milk-1.03.00/lib:/usr/local/include:/usr/local/lib:/usr/lib/x86_64-linux-gnu/openblas-pthread")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-linalg-modalremap")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  include("/home/oguyon/src/milk/build_local/plugins/milk-extra-src/linalgebra/CMakeFiles/milk-fpsexec-linalg-modalremap.dir/install-cxx-module-bmi-Release.cmake" OPTIONAL)
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
if(CMAKE_INSTALL_LOCAL_ONLY)
  file(WRITE "/home/oguyon/src/milk/build_local/plugins/milk-extra-src/linalgebra/install_local_manifest.txt"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
endif()
