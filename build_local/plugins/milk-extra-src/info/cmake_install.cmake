# Install script for directory: /home/oguyon/src/milk/plugins/milk-extra-src/info

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
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libmilkinfo.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libmilkinfo.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libmilkinfo.so"
         RPATH "/home/oguyon/local_milk/milk-1.03.00/lib:/usr/local/lib:/usr/local/include")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES "/home/oguyon/src/milk/build_local/plugins/milk-extra-src/info/libmilkinfo.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libmilkinfo.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libmilkinfo.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libmilkinfo.so"
         OLD_RPATH "/usr/local/lib:/usr/local/include:/home/oguyon/src/milk/build_local/src/cli/CLIcore:/home/oguyon/src/milk/build_local/src/cli/libmilkscript:/home/oguyon/src/milk/build_local/src/coremods/COREMOD_arith:/home/oguyon/src/milk/build_local/src/coremods/COREMOD_iofits:/home/oguyon/src/milk/build_local/src/coremods/COREMOD_memory:/home/oguyon/src/milk/build_local/src/coremods/COREMOD_tools:/home/oguyon/src/milk/build_local/src/engine/libfpsseq:/home/oguyon/src/milk/build_local/src/engine/libfps:/home/oguyon/src/milk/build_local/src/engine/libprocessinfo:/home/oguyon/src/milk/build_local/src/engine/libmilkdata:/home/oguyon/src/milk/build_local/src/engine/ImageStreamIO:"
         NEW_RPATH "/home/oguyon/local_milk/milk-1.03.00/lib:/usr/local/lib:/usr/local/include")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libmilkinfo.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  include("/home/oguyon/src/milk/build_local/plugins/milk-extra-src/info/CMakeFiles/milkinfo.dir/install-cxx-module-bmi-Release.cmake" OPTIONAL)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/info" TYPE FILE FILES
    "/home/oguyon/src/milk/plugins/milk-extra-src/info/info.h"
    "/home/oguyon/src/milk/plugins/milk-extra-src/info/cubeMatchMatrix.h"
    "/home/oguyon/src/milk/plugins/milk-extra-src/info/cubestats.h"
    "/home/oguyon/src/milk/plugins/milk-extra-src/info/image_stats.h"
    "/home/oguyon/src/milk/plugins/milk-extra-src/info/improfile.h"
    "/home/oguyon/src/milk/plugins/milk-extra-src/info/kbdhit.h"
    "/home/oguyon/src/milk/plugins/milk-extra-src/info/percentile.h"
    "/home/oguyon/src/milk/plugins/milk-extra-src/info/print_header.h"
    "/home/oguyon/src/milk/plugins/milk-extra-src/info/stream_monproc.h"
    "/home/oguyon/src/milk/plugins/milk-extra-src/info/streamtiming_stats.h"
    "/home/oguyon/src/milk/plugins/milk-extra-src/info/timediff.h"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libmilkinfo_compute.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libmilkinfo_compute.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libmilkinfo_compute.so"
         RPATH "/home/oguyon/local_milk/milk-1.03.00/lib:/usr/local/lib:/usr/local/include")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES "/home/oguyon/src/milk/build_local/plugins/milk-extra-src/info/libmilkinfo_compute.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libmilkinfo_compute.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libmilkinfo_compute.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libmilkinfo_compute.so"
         OLD_RPATH "/usr/local/lib:/usr/local/include:/home/oguyon/src/milk/build_local/src/coremods/COREMOD_memory:/home/oguyon/src/milk/build_local/src/coremods/COREMOD_iofits:/home/oguyon/src/milk/build_local/src/engine/libfps:/home/oguyon/src/milk/build_local/src/engine/libprocessinfo:/home/oguyon/src/milk/build_local/src/engine/ImageStreamIO:"
         NEW_RPATH "/home/oguyon/local_milk/milk-1.03.00/lib:/usr/local/lib:/usr/local/include")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libmilkinfo_compute.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  include("/home/oguyon/src/milk/build_local/plugins/milk-extra-src/info/CMakeFiles/milkinfo_compute.dir/install-cxx-module-bmi-Release.cmake" OPTIONAL)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/stream-monproc-disp" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/stream-monproc-disp")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/stream-monproc-disp"
         RPATH "/home/oguyon/local_milk/milk-1.03.00/lib:/usr/local/include")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/home/oguyon/src/milk/build_local/plugins/milk-extra-src/info/stream-monproc-disp")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/stream-monproc-disp" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/stream-monproc-disp")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/stream-monproc-disp"
         OLD_RPATH "/usr/local/include:/home/oguyon/src/milk/build_local/src/engine/libmilkdata:/home/oguyon/src/milk/build_local/src/engine/libfps:/home/oguyon/src/milk/build_local/src/engine/libprocessinfo:/home/oguyon/src/milk/build_local/src/engine/ImageStreamIO:"
         NEW_RPATH "/home/oguyon/local_milk/milk-1.03.00/lib:/usr/local/include")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/stream-monproc-disp")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  include("/home/oguyon/src/milk/build_local/plugins/milk-extra-src/info/CMakeFiles/stream-monproc-disp.dir/install-cxx-module-bmi-Release.cmake" OPTIONAL)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-info-strmonproc" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-info-strmonproc")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-info-strmonproc"
         RPATH "/home/oguyon/local_milk/milk-1.03.00/lib:/usr/local/include:/usr/local/lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/home/oguyon/src/milk/build_local/plugins/milk-extra-src/info/milk-fpsexec-info-strmonproc")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-info-strmonproc" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-info-strmonproc")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-info-strmonproc"
         OLD_RPATH "/usr/local/include:/usr/local/lib:/home/oguyon/src/milk/build_local/src/engine/libfps:/home/oguyon/src/milk/build_local/src/engine/libfpsseq:/home/oguyon/src/milk/build_local/src/engine/libmilkdata:/home/oguyon/src/milk/build_local/src/coremods/COREMOD_memory:/home/oguyon/src/milk/build_local/src/coremods/COREMOD_tools:/home/oguyon/src/milk/build_local/src/coremods/COREMOD_arith:/home/oguyon/src/milk/build_local/src/coremods/COREMOD_iofits:/home/oguyon/src/milk/build_local/src/engine/libprocessinfo:/home/oguyon/src/milk/build_local/src/engine/ImageStreamIO:"
         NEW_RPATH "/home/oguyon/local_milk/milk-1.03.00/lib:/usr/local/include:/usr/local/lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-fpsexec-info-strmonproc")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  include("/home/oguyon/src/milk/build_local/plugins/milk-extra-src/info/CMakeFiles/milk-fpsexec-info-strmonproc.dir/install-cxx-module-bmi-Release.cmake" OPTIONAL)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-shmimmon" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-shmimmon")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-shmimmon"
         RPATH "/home/oguyon/local_milk/milk-1.03.00/lib:/usr/local/include")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/home/oguyon/src/milk/build_local/plugins/milk-extra-src/info/milk-shmimmon")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-shmimmon" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-shmimmon")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-shmimmon"
         OLD_RPATH "/usr/local/include:/home/oguyon/src/milk/build_local/src/engine/ImageStreamIO:"
         NEW_RPATH "/home/oguyon/local_milk/milk-1.03.00/lib:/usr/local/include")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/milk-shmimmon")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  include("/home/oguyon/src/milk/build_local/plugins/milk-extra-src/info/CMakeFiles/milk-shmimmon.dir/install-cxx-module-bmi-Release.cmake" OPTIONAL)
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
if(CMAKE_INSTALL_LOCAL_ONLY)
  file(WRITE "/home/oguyon/src/milk/build_local/plugins/milk-extra-src/info/install_local_manifest.txt"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
endif()
