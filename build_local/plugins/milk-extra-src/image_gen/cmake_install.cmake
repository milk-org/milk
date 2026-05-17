# Install script for directory: /home/oguyon/src/milk/plugins/milk-extra-src/image_gen

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
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libmilkimagegen.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libmilkimagegen.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libmilkimagegen.so"
         RPATH "/home/oguyon/local_milk/milk-1.03.00/lib:/usr/local/lib:/usr/local/include")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES "/home/oguyon/src/milk/build_local/plugins/milk-extra-src/image_gen/libmilkimagegen.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libmilkimagegen.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libmilkimagegen.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libmilkimagegen.so"
         OLD_RPATH "/usr/local/lib:/usr/local/include:/home/oguyon/src/milk/build_local/plugins/milk-extra-src/statistic:/home/oguyon/src/milk/build_local/src/cli/CLIcore:/home/oguyon/src/milk/build_local/src/cli/libmilkscript:/home/oguyon/src/milk/build_local/src/coremods/COREMOD_arith:/home/oguyon/src/milk/build_local/src/coremods/COREMOD_iofits:/home/oguyon/src/milk/build_local/src/coremods/COREMOD_memory:/home/oguyon/src/milk/build_local/src/coremods/COREMOD_tools:/home/oguyon/src/milk/build_local/src/engine/libfpsseq:/home/oguyon/src/milk/build_local/src/engine/libfps:/home/oguyon/src/milk/build_local/src/engine/libprocessinfo:/home/oguyon/src/milk/build_local/src/engine/libmilkdata:/home/oguyon/src/milk/build_local/src/engine/ImageStreamIO:"
         NEW_RPATH "/home/oguyon/local_milk/milk-1.03.00/lib:/usr/local/lib:/usr/local/include")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libmilkimagegen.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  include("/home/oguyon/src/milk/build_local/plugins/milk-extra-src/image_gen/CMakeFiles/milkimagegen.dir/install-cxx-module-bmi-Release.cmake" OPTIONAL)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/image_gen" TYPE FILE FILES
    "/home/oguyon/src/milk/plugins/milk-extra-src/image_gen/image_gen.h"
    "/home/oguyon/src/milk/plugins/milk-extra-src/image_gen/mkdisk.h"
    "/home/oguyon/src/milk/plugins/milk-extra-src/image_gen/mkpolygon.h"
    "/home/oguyon/src/milk/plugins/milk-extra-src/image_gen/mkspdisk.h"
    "/home/oguyon/src/milk/plugins/milk-extra-src/image_gen/mkrandomim.h"
    "/home/oguyon/src/milk/plugins/milk-extra-src/image_gen/voronoi.h"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/image_gen" TYPE FILE FILES
    "/home/oguyon/src/milk/plugins/milk-extra-src/image_gen/mkdisk.h"
    "/home/oguyon/src/milk/plugins/milk-extra-src/image_gen/mkpolygon.h"
    "/home/oguyon/src/milk/plugins/milk-extra-src/image_gen/mkspdisk.h"
    "/home/oguyon/src/milk/plugins/milk-extra-src/image_gen/mkrandomim.h"
    "/home/oguyon/src/milk/plugins/milk-extra-src/image_gen/voronoi.h"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libmilkimagegen_compute.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libmilkimagegen_compute.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libmilkimagegen_compute.so"
         RPATH "/home/oguyon/local_milk/milk-1.03.00/lib:/usr/local/lib:/usr/local/include")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES "/home/oguyon/src/milk/build_local/plugins/milk-extra-src/image_gen/libmilkimagegen_compute.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libmilkimagegen_compute.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libmilkimagegen_compute.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libmilkimagegen_compute.so"
         OLD_RPATH "/usr/local/lib:/usr/local/include:/home/oguyon/src/milk/build_local/plugins/milk-extra-src/statistic:/home/oguyon/src/milk/build_local/src/coremods/COREMOD_memory:/home/oguyon/src/milk/build_local/src/coremods/COREMOD_iofits:/home/oguyon/src/milk/build_local/src/engine/libfps:/home/oguyon/src/milk/build_local/src/engine/libprocessinfo:/home/oguyon/src/milk/build_local/src/engine/ImageStreamIO:"
         NEW_RPATH "/home/oguyon/local_milk/milk-1.03.00/lib:/usr/local/lib:/usr/local/include")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libmilkimagegen_compute.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  include("/home/oguyon/src/milk/build_local/plugins/milk-extra-src/image_gen/CMakeFiles/milkimagegen_compute.dir/install-cxx-module-bmi-Release.cmake" OPTIONAL)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-imggen-mkrandom" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-imggen-mkrandom")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-imggen-mkrandom"
         RPATH "/home/oguyon/local_milk/milk-1.03.00/lib:/usr/local/include:/usr/local/lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/home/oguyon/src/milk/build_local/plugins/milk-extra-src/image_gen/milk-fpsexec-imggen-mkrandom")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-imggen-mkrandom" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-imggen-mkrandom")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-imggen-mkrandom"
         OLD_RPATH "/usr/local/include:/usr/local/lib:/home/oguyon/src/milk/build_local/src/engine/libfps:/home/oguyon/src/milk/build_local/src/coremods/COREMOD_tools:/home/oguyon/src/milk/build_local/src/coremods/COREMOD_arith:/home/oguyon/src/milk/build_local/plugins/milk-extra-src/image_gen:/home/oguyon/src/milk/build_local/plugins/milk-extra-src/statistic:/home/oguyon/src/milk/build_local/src/cli/CLIcore:/home/oguyon/src/milk/build_local/src/cli/libmilkscript:/home/oguyon/src/milk/build_local/src/engine/libfpsseq:/home/oguyon/src/milk/build_local/src/engine/libmilkdata:/home/oguyon/src/milk/build_local/src/coremods/COREMOD_iofits:/home/oguyon/src/milk/build_local/src/coremods/COREMOD_memory:/home/oguyon/src/milk/build_local/src/engine/libprocessinfo:/home/oguyon/src/milk/build_local/src/engine/ImageStreamIO:"
         NEW_RPATH "/home/oguyon/local_milk/milk-1.03.00/lib:/usr/local/include:/usr/local/lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-imggen-mkrandom")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  include("/home/oguyon/src/milk/build_local/plugins/milk-extra-src/image_gen/CMakeFiles/milk-fpsexec-imggen-mkrandom.dir/install-cxx-module-bmi-Release.cmake" OPTIONAL)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-imggen-voronoi" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-imggen-voronoi")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-imggen-voronoi"
         RPATH "/home/oguyon/local_milk/milk-1.03.00/lib:/usr/local/include:/usr/local/lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/home/oguyon/src/milk/build_local/plugins/milk-extra-src/image_gen/milk-fpsexec-imggen-voronoi")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-imggen-voronoi" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-imggen-voronoi")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-imggen-voronoi"
         OLD_RPATH "/usr/local/include:/usr/local/lib:/home/oguyon/src/milk/build_local/src/engine/libfps:/home/oguyon/src/milk/build_local/src/engine/libfpsseq:/home/oguyon/src/milk/build_local/src/engine/libmilkdata:/home/oguyon/src/milk/build_local/src/coremods/COREMOD_memory:/home/oguyon/src/milk/build_local/src/coremods/COREMOD_tools:/home/oguyon/src/milk/build_local/src/coremods/COREMOD_arith:/home/oguyon/src/milk/build_local/src/coremods/COREMOD_iofits:/home/oguyon/src/milk/build_local/src/engine/libprocessinfo:/home/oguyon/src/milk/build_local/src/engine/ImageStreamIO:"
         NEW_RPATH "/home/oguyon/local_milk/milk-1.03.00/lib:/usr/local/include:/usr/local/lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-imggen-voronoi")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  include("/home/oguyon/src/milk/build_local/plugins/milk-extra-src/image_gen/CMakeFiles/milk-fpsexec-imggen-voronoi.dir/install-cxx-module-bmi-Release.cmake" OPTIONAL)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-imggen-mkdisk" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-imggen-mkdisk")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-imggen-mkdisk"
         RPATH "/home/oguyon/local_milk/milk-1.03.00/lib:/usr/local/include:/usr/local/lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/home/oguyon/src/milk/build_local/plugins/milk-extra-src/image_gen/milk-fpsexec-imggen-mkdisk")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-imggen-mkdisk" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-imggen-mkdisk")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-imggen-mkdisk"
         OLD_RPATH "/usr/local/include:/usr/local/lib:/home/oguyon/src/milk/build_local/src/engine/libfps:/home/oguyon/src/milk/build_local/src/engine/libfpsseq:/home/oguyon/src/milk/build_local/src/engine/libmilkdata:/home/oguyon/src/milk/build_local/src/coremods/COREMOD_tools:/home/oguyon/src/milk/build_local/src/coremods/COREMOD_arith:/home/oguyon/src/milk/build_local/plugins/milk-extra-src/image_gen:/home/oguyon/src/milk/build_local/plugins/milk-extra-src/statistic:/home/oguyon/src/milk/build_local/src/coremods/COREMOD_memory:/home/oguyon/src/milk/build_local/src/coremods/COREMOD_iofits:/home/oguyon/src/milk/build_local/src/engine/libprocessinfo:/home/oguyon/src/milk/build_local/src/engine/ImageStreamIO:"
         NEW_RPATH "/home/oguyon/local_milk/milk-1.03.00/lib:/usr/local/include:/usr/local/lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-imggen-mkdisk")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  include("/home/oguyon/src/milk/build_local/plugins/milk-extra-src/image_gen/CMakeFiles/milk-fpsexec-imggen-mkdisk.dir/install-cxx-module-bmi-Release.cmake" OPTIONAL)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-imggen-mkspdisk" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-imggen-mkspdisk")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-imggen-mkspdisk"
         RPATH "/home/oguyon/local_milk/milk-1.03.00/lib:/usr/local/include:/usr/local/lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/home/oguyon/src/milk/build_local/plugins/milk-extra-src/image_gen/milk-fpsexec-imggen-mkspdisk")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-imggen-mkspdisk" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-imggen-mkspdisk")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-imggen-mkspdisk"
         OLD_RPATH "/usr/local/include:/usr/local/lib:/home/oguyon/src/milk/build_local/src/engine/libfps:/home/oguyon/src/milk/build_local/src/engine/libfpsseq:/home/oguyon/src/milk/build_local/src/engine/libmilkdata:/home/oguyon/src/milk/build_local/src/coremods/COREMOD_tools:/home/oguyon/src/milk/build_local/src/coremods/COREMOD_arith:/home/oguyon/src/milk/build_local/plugins/milk-extra-src/image_gen:/home/oguyon/src/milk/build_local/plugins/milk-extra-src/statistic:/home/oguyon/src/milk/build_local/src/coremods/COREMOD_memory:/home/oguyon/src/milk/build_local/src/coremods/COREMOD_iofits:/home/oguyon/src/milk/build_local/src/engine/libprocessinfo:/home/oguyon/src/milk/build_local/src/engine/ImageStreamIO:"
         NEW_RPATH "/home/oguyon/local_milk/milk-1.03.00/lib:/usr/local/include:/usr/local/lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-imggen-mkspdisk")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  include("/home/oguyon/src/milk/build_local/plugins/milk-extra-src/image_gen/CMakeFiles/milk-fpsexec-imggen-mkspdisk.dir/install-cxx-module-bmi-Release.cmake" OPTIONAL)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-imggen-mkpolygon" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-imggen-mkpolygon")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-imggen-mkpolygon"
         RPATH "/home/oguyon/local_milk/milk-1.03.00/lib:/usr/local/include:/usr/local/lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/home/oguyon/src/milk/build_local/plugins/milk-extra-src/image_gen/milk-fpsexec-imggen-mkpolygon")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-imggen-mkpolygon" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-imggen-mkpolygon")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-imggen-mkpolygon"
         OLD_RPATH "/usr/local/include:/usr/local/lib:/home/oguyon/src/milk/build_local/src/engine/libfps:/home/oguyon/src/milk/build_local/src/engine/libfpsseq:/home/oguyon/src/milk/build_local/src/engine/libmilkdata:/home/oguyon/src/milk/build_local/src/coremods/COREMOD_tools:/home/oguyon/src/milk/build_local/src/coremods/COREMOD_arith:/home/oguyon/src/milk/build_local/plugins/milk-extra-src/image_gen:/home/oguyon/src/milk/build_local/plugins/milk-extra-src/statistic:/home/oguyon/src/milk/build_local/src/coremods/COREMOD_memory:/home/oguyon/src/milk/build_local/src/coremods/COREMOD_iofits:/home/oguyon/src/milk/build_local/src/engine/libprocessinfo:/home/oguyon/src/milk/build_local/src/engine/ImageStreamIO:"
         NEW_RPATH "/home/oguyon/local_milk/milk-1.03.00/lib:/usr/local/include:/usr/local/lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-imggen-mkpolygon")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  include("/home/oguyon/src/milk/build_local/plugins/milk-extra-src/image_gen/CMakeFiles/milk-fpsexec-imggen-mkpolygon.dir/install-cxx-module-bmi-Release.cmake" OPTIONAL)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-imggen-mkgauss" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-imggen-mkgauss")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-imggen-mkgauss"
         RPATH "/home/oguyon/local_milk/milk-1.03.00/lib:/usr/local/include:/usr/local/lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/home/oguyon/src/milk/build_local/plugins/milk-extra-src/image_gen/milk-fpsexec-imggen-mkgauss")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-imggen-mkgauss" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-imggen-mkgauss")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-imggen-mkgauss"
         OLD_RPATH "/usr/local/include:/usr/local/lib:/home/oguyon/src/milk/build_local/src/engine/libfps:/home/oguyon/src/milk/build_local/src/engine/libfpsseq:/home/oguyon/src/milk/build_local/src/engine/libmilkdata:/home/oguyon/src/milk/build_local/src/coremods/COREMOD_tools:/home/oguyon/src/milk/build_local/src/coremods/COREMOD_arith:/home/oguyon/src/milk/build_local/plugins/milk-extra-src/image_gen:/home/oguyon/src/milk/build_local/plugins/milk-extra-src/statistic:/home/oguyon/src/milk/build_local/src/coremods/COREMOD_memory:/home/oguyon/src/milk/build_local/src/coremods/COREMOD_iofits:/home/oguyon/src/milk/build_local/src/engine/libprocessinfo:/home/oguyon/src/milk/build_local/src/engine/ImageStreamIO:"
         NEW_RPATH "/home/oguyon/local_milk/milk-1.03.00/lib:/usr/local/include:/usr/local/lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-imggen-mkgauss")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  include("/home/oguyon/src/milk/build_local/plugins/milk-extra-src/image_gen/CMakeFiles/milk-fpsexec-imggen-mkgauss.dir/install-cxx-module-bmi-Release.cmake" OPTIONAL)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-imggen-mkfiberclpoverlap" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-imggen-mkfiberclpoverlap")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-imggen-mkfiberclpoverlap"
         RPATH "/home/oguyon/local_milk/milk-1.03.00/lib:/usr/local/include:/usr/local/lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/home/oguyon/src/milk/build_local/plugins/milk-extra-src/image_gen/milk-fpsexec-imggen-mkfiberclpoverlap")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-imggen-mkfiberclpoverlap" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-imggen-mkfiberclpoverlap")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-imggen-mkfiberclpoverlap"
         OLD_RPATH "/usr/local/include:/usr/local/lib:/home/oguyon/src/milk/build_local/src/engine/libfps:/home/oguyon/src/milk/build_local/src/engine/libfpsseq:/home/oguyon/src/milk/build_local/src/engine/libmilkdata:/home/oguyon/src/milk/build_local/src/coremods/COREMOD_tools:/home/oguyon/src/milk/build_local/src/coremods/COREMOD_arith:/home/oguyon/src/milk/build_local/plugins/milk-extra-src/image_gen:/home/oguyon/src/milk/build_local/plugins/milk-extra-src/statistic:/home/oguyon/src/milk/build_local/src/coremods/COREMOD_memory:/home/oguyon/src/milk/build_local/src/coremods/COREMOD_iofits:/home/oguyon/src/milk/build_local/src/engine/libprocessinfo:/home/oguyon/src/milk/build_local/src/engine/ImageStreamIO:"
         NEW_RPATH "/home/oguyon/local_milk/milk-1.03.00/lib:/usr/local/include:/usr/local/lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-imggen-mkfiberclpoverlap")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  include("/home/oguyon/src/milk/build_local/plugins/milk-extra-src/image_gen/CMakeFiles/milk-fpsexec-imggen-mkfiberclpoverlap.dir/install-cxx-module-bmi-Release.cmake" OPTIONAL)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-imggen-mkslopexy" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-imggen-mkslopexy")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-imggen-mkslopexy"
         RPATH "/home/oguyon/local_milk/milk-1.03.00/lib:/usr/local/include:/usr/local/lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/home/oguyon/src/milk/build_local/plugins/milk-extra-src/image_gen/milk-fpsexec-imggen-mkslopexy")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-imggen-mkslopexy" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-imggen-mkslopexy")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-imggen-mkslopexy"
         OLD_RPATH "/usr/local/include:/usr/local/lib:/home/oguyon/src/milk/build_local/src/engine/libfps:/home/oguyon/src/milk/build_local/src/engine/libfpsseq:/home/oguyon/src/milk/build_local/src/engine/libmilkdata:/home/oguyon/src/milk/build_local/src/coremods/COREMOD_tools:/home/oguyon/src/milk/build_local/src/coremods/COREMOD_arith:/home/oguyon/src/milk/build_local/plugins/milk-extra-src/image_gen:/home/oguyon/src/milk/build_local/plugins/milk-extra-src/statistic:/home/oguyon/src/milk/build_local/src/coremods/COREMOD_memory:/home/oguyon/src/milk/build_local/src/coremods/COREMOD_iofits:/home/oguyon/src/milk/build_local/src/engine/libprocessinfo:/home/oguyon/src/milk/build_local/src/engine/ImageStreamIO:"
         NEW_RPATH "/home/oguyon/local_milk/milk-1.03.00/lib:/usr/local/include:/usr/local/lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-imggen-mkslopexy")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  include("/home/oguyon/src/milk/build_local/plugins/milk-extra-src/image_gen/CMakeFiles/milk-fpsexec-imggen-mkslopexy.dir/install-cxx-module-bmi-Release.cmake" OPTIONAL)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-imggen-mkdist" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-imggen-mkdist")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-imggen-mkdist"
         RPATH "/home/oguyon/local_milk/milk-1.03.00/lib:/usr/local/include:/usr/local/lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/home/oguyon/src/milk/build_local/plugins/milk-extra-src/image_gen/milk-fpsexec-imggen-mkdist")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-imggen-mkdist" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-imggen-mkdist")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-imggen-mkdist"
         OLD_RPATH "/usr/local/include:/usr/local/lib:/home/oguyon/src/milk/build_local/src/engine/libfps:/home/oguyon/src/milk/build_local/src/engine/libfpsseq:/home/oguyon/src/milk/build_local/src/engine/libmilkdata:/home/oguyon/src/milk/build_local/src/coremods/COREMOD_tools:/home/oguyon/src/milk/build_local/src/coremods/COREMOD_arith:/home/oguyon/src/milk/build_local/plugins/milk-extra-src/image_gen:/home/oguyon/src/milk/build_local/plugins/milk-extra-src/statistic:/home/oguyon/src/milk/build_local/src/coremods/COREMOD_memory:/home/oguyon/src/milk/build_local/src/coremods/COREMOD_iofits:/home/oguyon/src/milk/build_local/src/engine/libprocessinfo:/home/oguyon/src/milk/build_local/src/engine/ImageStreamIO:"
         NEW_RPATH "/home/oguyon/local_milk/milk-1.03.00/lib:/usr/local/include:/usr/local/lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-imggen-mkdist")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  include("/home/oguyon/src/milk/build_local/plugins/milk-extra-src/image_gen/CMakeFiles/milk-fpsexec-imggen-mkdist.dir/install-cxx-module-bmi-Release.cmake" OPTIONAL)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-imggen-mkhexsegpup" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-imggen-mkhexsegpup")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-imggen-mkhexsegpup"
         RPATH "/home/oguyon/local_milk/milk-1.03.00/lib:/usr/local/include:/usr/local/lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/home/oguyon/src/milk/build_local/plugins/milk-extra-src/image_gen/milk-fpsexec-imggen-mkhexsegpup")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-imggen-mkhexsegpup" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-imggen-mkhexsegpup")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-imggen-mkhexsegpup"
         OLD_RPATH "/usr/local/include:/usr/local/lib:/home/oguyon/src/milk/build_local/src/engine/libfps:/home/oguyon/src/milk/build_local/src/engine/libfpsseq:/home/oguyon/src/milk/build_local/src/engine/libmilkdata:/home/oguyon/src/milk/build_local/src/coremods/COREMOD_tools:/home/oguyon/src/milk/build_local/src/coremods/COREMOD_arith:/home/oguyon/src/milk/build_local/plugins/milk-extra-src/image_gen:/home/oguyon/src/milk/build_local/plugins/milk-extra-src/statistic:/home/oguyon/src/milk/build_local/src/coremods/COREMOD_memory:/home/oguyon/src/milk/build_local/src/coremods/COREMOD_iofits:/home/oguyon/src/milk/build_local/src/engine/libprocessinfo:/home/oguyon/src/milk/build_local/src/engine/ImageStreamIO:"
         NEW_RPATH "/home/oguyon/local_milk/milk-1.03.00/lib:/usr/local/include:/usr/local/lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-imggen-mkhexsegpup")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  include("/home/oguyon/src/milk/build_local/plugins/milk-extra-src/image_gen/CMakeFiles/milk-fpsexec-imggen-mkhexsegpup.dir/install-cxx-module-bmi-Release.cmake" OPTIONAL)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-imggen-segs2wfmodes" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-imggen-segs2wfmodes")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-imggen-segs2wfmodes"
         RPATH "/home/oguyon/local_milk/milk-1.03.00/lib:/usr/local/include:/usr/local/lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/home/oguyon/src/milk/build_local/plugins/milk-extra-src/image_gen/milk-fpsexec-imggen-segs2wfmodes")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-imggen-segs2wfmodes" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-imggen-segs2wfmodes")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-imggen-segs2wfmodes"
         OLD_RPATH "/usr/local/include:/usr/local/lib:/home/oguyon/src/milk/build_local/src/engine/libfps:/home/oguyon/src/milk/build_local/src/engine/libfpsseq:/home/oguyon/src/milk/build_local/src/engine/libmilkdata:/home/oguyon/src/milk/build_local/src/coremods/COREMOD_tools:/home/oguyon/src/milk/build_local/src/coremods/COREMOD_arith:/home/oguyon/src/milk/build_local/plugins/milk-extra-src/image_gen:/home/oguyon/src/milk/build_local/plugins/milk-extra-src/statistic:/home/oguyon/src/milk/build_local/src/coremods/COREMOD_memory:/home/oguyon/src/milk/build_local/src/coremods/COREMOD_iofits:/home/oguyon/src/milk/build_local/src/engine/libprocessinfo:/home/oguyon/src/milk/build_local/src/engine/ImageStreamIO:"
         NEW_RPATH "/home/oguyon/local_milk/milk-1.03.00/lib:/usr/local/include:/usr/local/lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-imggen-segs2wfmodes")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  include("/home/oguyon/src/milk/build_local/plugins/milk-extra-src/image_gen/CMakeFiles/milk-fpsexec-imggen-segs2wfmodes.dir/install-cxx-module-bmi-Release.cmake" OPTIONAL)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-imggen-mkrect" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-imggen-mkrect")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-imggen-mkrect"
         RPATH "/home/oguyon/local_milk/milk-1.03.00/lib:/usr/local/include:/usr/local/lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/home/oguyon/src/milk/build_local/plugins/milk-extra-src/image_gen/milk-fpsexec-imggen-mkrect")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-imggen-mkrect" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-imggen-mkrect")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-imggen-mkrect"
         OLD_RPATH "/usr/local/include:/usr/local/lib:/home/oguyon/src/milk/build_local/src/engine/libfps:/home/oguyon/src/milk/build_local/src/engine/libfpsseq:/home/oguyon/src/milk/build_local/src/engine/libmilkdata:/home/oguyon/src/milk/build_local/src/coremods/COREMOD_tools:/home/oguyon/src/milk/build_local/src/coremods/COREMOD_arith:/home/oguyon/src/milk/build_local/plugins/milk-extra-src/image_gen:/home/oguyon/src/milk/build_local/plugins/milk-extra-src/statistic:/home/oguyon/src/milk/build_local/src/coremods/COREMOD_memory:/home/oguyon/src/milk/build_local/src/coremods/COREMOD_iofits:/home/oguyon/src/milk/build_local/src/engine/libprocessinfo:/home/oguyon/src/milk/build_local/src/engine/ImageStreamIO:"
         NEW_RPATH "/home/oguyon/local_milk/milk-1.03.00/lib:/usr/local/include:/usr/local/lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-imggen-mkrect")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  include("/home/oguyon/src/milk/build_local/plugins/milk-extra-src/image_gen/CMakeFiles/milk-fpsexec-imggen-mkrect.dir/install-cxx-module-bmi-Release.cmake" OPTIONAL)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-imggen-mkline" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-imggen-mkline")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-imggen-mkline"
         RPATH "/home/oguyon/local_milk/milk-1.03.00/lib:/usr/local/include:/usr/local/lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/home/oguyon/src/milk/build_local/plugins/milk-extra-src/image_gen/milk-fpsexec-imggen-mkline")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-imggen-mkline" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-imggen-mkline")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-imggen-mkline"
         OLD_RPATH "/usr/local/include:/usr/local/lib:/home/oguyon/src/milk/build_local/src/engine/libfps:/home/oguyon/src/milk/build_local/src/engine/libfpsseq:/home/oguyon/src/milk/build_local/src/engine/libmilkdata:/home/oguyon/src/milk/build_local/src/coremods/COREMOD_tools:/home/oguyon/src/milk/build_local/src/coremods/COREMOD_arith:/home/oguyon/src/milk/build_local/plugins/milk-extra-src/image_gen:/home/oguyon/src/milk/build_local/plugins/milk-extra-src/statistic:/home/oguyon/src/milk/build_local/src/coremods/COREMOD_memory:/home/oguyon/src/milk/build_local/src/coremods/COREMOD_iofits:/home/oguyon/src/milk/build_local/src/engine/libprocessinfo:/home/oguyon/src/milk/build_local/src/engine/ImageStreamIO:"
         NEW_RPATH "/home/oguyon/local_milk/milk-1.03.00/lib:/usr/local/include:/usr/local/lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-imggen-mkline")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  include("/home/oguyon/src/milk/build_local/plugins/milk-extra-src/image_gen/CMakeFiles/milk-fpsexec-imggen-mkline.dir/install-cxx-module-bmi-Release.cmake" OPTIONAL)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-imggen-mklincoord" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-imggen-mklincoord")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-imggen-mklincoord"
         RPATH "/home/oguyon/local_milk/milk-1.03.00/lib:/usr/local/include:/usr/local/lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/home/oguyon/src/milk/build_local/plugins/milk-extra-src/image_gen/milk-fpsexec-imggen-mklincoord")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-imggen-mklincoord" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-imggen-mklincoord")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-imggen-mklincoord"
         OLD_RPATH "/usr/local/include:/usr/local/lib:/home/oguyon/src/milk/build_local/src/engine/libfps:/home/oguyon/src/milk/build_local/src/engine/libfpsseq:/home/oguyon/src/milk/build_local/src/engine/libmilkdata:/home/oguyon/src/milk/build_local/src/coremods/COREMOD_tools:/home/oguyon/src/milk/build_local/src/coremods/COREMOD_arith:/home/oguyon/src/milk/build_local/plugins/milk-extra-src/image_gen:/home/oguyon/src/milk/build_local/plugins/milk-extra-src/statistic:/home/oguyon/src/milk/build_local/src/coremods/COREMOD_memory:/home/oguyon/src/milk/build_local/src/coremods/COREMOD_iofits:/home/oguyon/src/milk/build_local/src/engine/libprocessinfo:/home/oguyon/src/milk/build_local/src/engine/ImageStreamIO:"
         NEW_RPATH "/home/oguyon/local_milk/milk-1.03.00/lib:/usr/local/include:/usr/local/lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-imggen-mklincoord")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  include("/home/oguyon/src/milk/build_local/plugins/milk-extra-src/image_gen/CMakeFiles/milk-fpsexec-imggen-mklincoord.dir/install-cxx-module-bmi-Release.cmake" OPTIONAL)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-imggen-mkgridpix" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-imggen-mkgridpix")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-imggen-mkgridpix"
         RPATH "/home/oguyon/local_milk/milk-1.03.00/lib:/usr/local/include:/usr/local/lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/home/oguyon/src/milk/build_local/plugins/milk-extra-src/image_gen/milk-fpsexec-imggen-mkgridpix")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-imggen-mkgridpix" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-imggen-mkgridpix")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-imggen-mkgridpix"
         OLD_RPATH "/usr/local/include:/usr/local/lib:/home/oguyon/src/milk/build_local/src/engine/libfps:/home/oguyon/src/milk/build_local/src/engine/libfpsseq:/home/oguyon/src/milk/build_local/src/engine/libmilkdata:/home/oguyon/src/milk/build_local/src/coremods/COREMOD_tools:/home/oguyon/src/milk/build_local/src/coremods/COREMOD_arith:/home/oguyon/src/milk/build_local/plugins/milk-extra-src/image_gen:/home/oguyon/src/milk/build_local/plugins/milk-extra-src/statistic:/home/oguyon/src/milk/build_local/src/coremods/COREMOD_memory:/home/oguyon/src/milk/build_local/src/coremods/COREMOD_iofits:/home/oguyon/src/milk/build_local/src/engine/libprocessinfo:/home/oguyon/src/milk/build_local/src/engine/ImageStreamIO:"
         NEW_RPATH "/home/oguyon/local_milk/milk-1.03.00/lib:/usr/local/include:/usr/local/lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-imggen-mkgridpix")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  include("/home/oguyon/src/milk/build_local/plugins/milk-extra-src/image_gen/CMakeFiles/milk-fpsexec-imggen-mkgridpix.dir/install-cxx-module-bmi-Release.cmake" OPTIONAL)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-imggen-mkrndim" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-imggen-mkrndim")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-imggen-mkrndim"
         RPATH "/home/oguyon/local_milk/milk-1.03.00/lib:/usr/local/include:/usr/local/lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/home/oguyon/src/milk/build_local/plugins/milk-extra-src/image_gen/milk-fpsexec-imggen-mkrndim")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-imggen-mkrndim" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-imggen-mkrndim")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-imggen-mkrndim"
         OLD_RPATH "/usr/local/include:/usr/local/lib:/home/oguyon/src/milk/build_local/src/engine/libfps:/home/oguyon/src/milk/build_local/src/engine/libfpsseq:/home/oguyon/src/milk/build_local/src/engine/libmilkdata:/home/oguyon/src/milk/build_local/src/coremods/COREMOD_tools:/home/oguyon/src/milk/build_local/src/coremods/COREMOD_arith:/home/oguyon/src/milk/build_local/plugins/milk-extra-src/image_gen:/home/oguyon/src/milk/build_local/plugins/milk-extra-src/statistic:/home/oguyon/src/milk/build_local/src/coremods/COREMOD_memory:/home/oguyon/src/milk/build_local/src/coremods/COREMOD_iofits:/home/oguyon/src/milk/build_local/src/engine/libprocessinfo:/home/oguyon/src/milk/build_local/src/engine/ImageStreamIO:"
         NEW_RPATH "/home/oguyon/local_milk/milk-1.03.00/lib:/usr/local/include:/usr/local/lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-imggen-mkrndim")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  include("/home/oguyon/src/milk/build_local/plugins/milk-extra-src/image_gen/CMakeFiles/milk-fpsexec-imggen-mkrndim.dir/install-cxx-module-bmi-Release.cmake" OPTIONAL)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-imggen-mkrndgim" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-imggen-mkrndgim")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-imggen-mkrndgim"
         RPATH "/home/oguyon/local_milk/milk-1.03.00/lib:/usr/local/include:/usr/local/lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/home/oguyon/src/milk/build_local/plugins/milk-extra-src/image_gen/milk-fpsexec-imggen-mkrndgim")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-imggen-mkrndgim" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-imggen-mkrndgim")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-imggen-mkrndgim"
         OLD_RPATH "/usr/local/include:/usr/local/lib:/home/oguyon/src/milk/build_local/src/engine/libfps:/home/oguyon/src/milk/build_local/src/engine/libfpsseq:/home/oguyon/src/milk/build_local/src/engine/libmilkdata:/home/oguyon/src/milk/build_local/src/coremods/COREMOD_tools:/home/oguyon/src/milk/build_local/src/coremods/COREMOD_arith:/home/oguyon/src/milk/build_local/plugins/milk-extra-src/image_gen:/home/oguyon/src/milk/build_local/plugins/milk-extra-src/statistic:/home/oguyon/src/milk/build_local/src/coremods/COREMOD_memory:/home/oguyon/src/milk/build_local/src/coremods/COREMOD_iofits:/home/oguyon/src/milk/build_local/src/engine/libprocessinfo:/home/oguyon/src/milk/build_local/src/engine/ImageStreamIO:"
         NEW_RPATH "/home/oguyon/local_milk/milk-1.03.00/lib:/usr/local/include:/usr/local/lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-imggen-mkrndgim")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  include("/home/oguyon/src/milk/build_local/plugins/milk-extra-src/image_gen/CMakeFiles/milk-fpsexec-imggen-mkrndgim.dir/install-cxx-module-bmi-Release.cmake" OPTIONAL)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-imggen-im2coord" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-imggen-im2coord")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-imggen-im2coord"
         RPATH "/home/oguyon/local_milk/milk-1.03.00/lib:/usr/local/include:/usr/local/lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/home/oguyon/src/milk/build_local/plugins/milk-extra-src/image_gen/milk-fpsexec-imggen-im2coord")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-imggen-im2coord" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-imggen-im2coord")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-imggen-im2coord"
         OLD_RPATH "/usr/local/include:/usr/local/lib:/home/oguyon/src/milk/build_local/src/engine/libfps:/home/oguyon/src/milk/build_local/src/engine/libfpsseq:/home/oguyon/src/milk/build_local/src/engine/libmilkdata:/home/oguyon/src/milk/build_local/src/coremods/COREMOD_tools:/home/oguyon/src/milk/build_local/src/coremods/COREMOD_arith:/home/oguyon/src/milk/build_local/plugins/milk-extra-src/image_gen:/home/oguyon/src/milk/build_local/plugins/milk-extra-src/statistic:/home/oguyon/src/milk/build_local/src/coremods/COREMOD_memory:/home/oguyon/src/milk/build_local/src/coremods/COREMOD_iofits:/home/oguyon/src/milk/build_local/src/engine/libprocessinfo:/home/oguyon/src/milk/build_local/src/engine/ImageStreamIO:"
         NEW_RPATH "/home/oguyon/local_milk/milk-1.03.00/lib:/usr/local/include:/usr/local/lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-imggen-im2coord")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  include("/home/oguyon/src/milk/build_local/plugins/milk-extra-src/image_gen/CMakeFiles/milk-fpsexec-imggen-im2coord.dir/install-cxx-module-bmi-Release.cmake" OPTIONAL)
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
if(CMAKE_INSTALL_LOCAL_ONLY)
  file(WRITE "/home/oguyon/src/milk/build_local/plugins/milk-extra-src/image_gen/install_local_manifest.txt"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
endif()
