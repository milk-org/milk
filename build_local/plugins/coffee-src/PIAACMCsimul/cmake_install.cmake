# Install script for directory: /home/oguyon/src/milk/plugins/coffee-src/PIAACMCsimul

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
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libcoffeePIAACMCsimul.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libcoffeePIAACMCsimul.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libcoffeePIAACMCsimul.so"
         RPATH "/home/oguyon/local_milk/milk-1.03.00/lib:/usr/local/lib:/usr/local/include")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES "/home/oguyon/src/milk/build_local/plugins/coffee-src/PIAACMCsimul/libcoffeePIAACMCsimul.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libcoffeePIAACMCsimul.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libcoffeePIAACMCsimul.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libcoffeePIAACMCsimul.so"
         OLD_RPATH "/usr/local/lib:/usr/local/include:/home/oguyon/src/milk/build_local/plugins/milk-extra-src/info:/home/oguyon/src/milk/build_local/plugins/OpticsMaterials-src/OpticsMaterials:/home/oguyon/src/milk/build_local/plugins/milk-extra-src/linopt_imtools:/home/oguyon/src/milk/build_local/plugins/milk-extra-src/image_basic:/home/oguyon/src/milk/build_local/plugins/milk-extra-src/image_filter:/home/oguyon/src/milk/build_local/plugins/WFpropagate-src/WFpropagate:/home/oguyon/src/milk/build_local/plugins/milk-extra-src/image_gen:/home/oguyon/src/milk/build_local/plugins/milk-extra-src/statistic:/home/oguyon/src/milk/build_local/plugins/milk-extra-src/fft:/home/oguyon/src/milk/build_local/src/cli/CLIcore:/home/oguyon/src/milk/build_local/src/cli/libmilkscript:/home/oguyon/src/milk/build_local/src/coremods/COREMOD_memory:/home/oguyon/src/milk/build_local/src/coremods/COREMOD_arith:/home/oguyon/src/milk/build_local/src/coremods/COREMOD_iofits:/home/oguyon/src/milk/build_local/src/coremods/COREMOD_tools:/home/oguyon/src/milk/build_local/src/engine/libfpsseq:/home/oguyon/src/milk/build_local/src/engine/libfps:/home/oguyon/src/milk/build_local/src/engine/libprocessinfo:/home/oguyon/src/milk/build_local/src/engine/libmilkdata:/home/oguyon/src/milk/build_local/src/engine/ImageStreamIO:"
         NEW_RPATH "/home/oguyon/local_milk/milk-1.03.00/lib:/usr/local/lib:/usr/local/include")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libcoffeePIAACMCsimul.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  include("/home/oguyon/src/milk/build_local/plugins/coffee-src/PIAACMCsimul/CMakeFiles/coffeePIAACMCsimul.dir/install-cxx-module-bmi-Release.cmake" OPTIONAL)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/PIAACMCsimul" TYPE FILE FILES
    "/home/oguyon/src/milk/plugins/coffee-src/PIAACMCsimul/PIAACMCsimul.h"
    "/home/oguyon/src/milk/plugins/coffee-src/PIAACMCsimul/exec_compute_image.h"
    "/home/oguyon/src/milk/plugins/coffee-src/PIAACMCsimul/exec_computePSF_no_fpm.h"
    "/home/oguyon/src/milk/plugins/coffee-src/PIAACMCsimul/exec_optimize_PIAA_shapes_fpmtransm.h"
    "/home/oguyon/src/milk/plugins/coffee-src/PIAACMCsimul/init_piaacmcopticaldesign.h"
    "/home/oguyon/src/milk/plugins/coffee-src/PIAACMCsimul/init_piaacmcopticalsystem.h"
    "/home/oguyon/src/milk/plugins/coffee-src/PIAACMCsimul/PIAACMC_f_evalmask.h"
    "/home/oguyon/src/milk/plugins/coffee-src/PIAACMCsimul/PIAACMCsimul_achromFPMsol_eval.h"
    "/home/oguyon/src/milk/plugins/coffee-src/PIAACMCsimul/PIAACMCsimul_achromFPMsol_eval_zonezderivative.c"
    "/home/oguyon/src/milk/plugins/coffee-src/PIAACMCsimul/PIAACMCsimul_CA2propCubeInt.h"
    "/home/oguyon/src/milk/plugins/coffee-src/PIAACMCsimul/PIAACMCsimul_computePSF.h"
    "/home/oguyon/src/milk/plugins/coffee-src/PIAACMCsimul/PIAACMCsimul_eval_poly_design.h"
    "/home/oguyon/src/milk/plugins/coffee-src/PIAACMCsimul/PIAACMCsimul_exec.h"
    "/home/oguyon/src/milk/plugins/coffee-src/PIAACMCsimul/PIAACMCsimul_loadsavepiaacmcconf.h"
    "/home/oguyon/src/milk/plugins/coffee-src/PIAACMCsimul/PIAACMCsimul_measure_transm_curve.h"
    "/home/oguyon/src/milk/plugins/coffee-src/PIAACMCsimul/PIAACMCsimul_run.h"
    "/home/oguyon/src/milk/plugins/coffee-src/PIAACMCsimul/FocalPlaneMask/exec_multizone_fpm_calib.h"
    "/home/oguyon/src/milk/plugins/coffee-src/PIAACMCsimul/FocalPlaneMask/exec_optimize_fpm_zones.h"
    "/home/oguyon/src/milk/plugins/coffee-src/PIAACMCsimul/FocalPlaneMask/exec_optimize_fpmtransmission.h"
    "/home/oguyon/src/milk/plugins/coffee-src/PIAACMCsimul/FocalPlaneMask/FPM_process.h"
    "/home/oguyon/src/milk/plugins/coffee-src/PIAACMCsimul/FocalPlaneMask/FPMresp_resample.h"
    "/home/oguyon/src/milk/plugins/coffee-src/PIAACMCsimul/FocalPlaneMask/FPMresp_rmzones.h"
    "/home/oguyon/src/milk/plugins/coffee-src/PIAACMCsimul/FocalPlaneMask/mkFocalPlaneMask.h"
    "/home/oguyon/src/milk/plugins/coffee-src/PIAACMCsimul/FocalPlaneMask/mkFPM_zonemap.h"
    "/home/oguyon/src/milk/plugins/coffee-src/PIAACMCsimul/FocalPlaneMask/rings2sectors.h"
    "/home/oguyon/src/milk/plugins/coffee-src/PIAACMCsimul/LyotStop/exec_optimize_lyot_stop_position.h"
    "/home/oguyon/src/milk/plugins/coffee-src/PIAACMCsimul/LyotStop/exec_optimize_lyot_stops_shapes_positions.h"
    "/home/oguyon/src/milk/plugins/coffee-src/PIAACMCsimul/LyotStop/geomProp.h"
    "/home/oguyon/src/milk/plugins/coffee-src/PIAACMCsimul/LyotStop/mkLyotMask.h"
    "/home/oguyon/src/milk/plugins/coffee-src/PIAACMCsimul/LyotStop/mkSimpleLyotStop.h"
    "/home/oguyon/src/milk/plugins/coffee-src/PIAACMCsimul/LyotStop/optimizeLyotStop.h"
    "/home/oguyon/src/milk/plugins/coffee-src/PIAACMCsimul/PIAAshape/exec_optimize_PIAA_shapes.h"
    "/home/oguyon/src/milk/plugins/coffee-src/PIAACMCsimul/PIAAshape/init_geomPIAA_rad.h"
    "/home/oguyon/src/milk/plugins/coffee-src/PIAACMCsimul/PIAAshape/load2DRadialApodization.h"
    "/home/oguyon/src/milk/plugins/coffee-src/PIAACMCsimul/PIAAshape/makePIAAshapes.h"
    "/home/oguyon/src/milk/plugins/coffee-src/PIAACMCsimul/PIAAshape/mkPIAAMshapes_from_RadSag.h"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE PROGRAM FILES
    "/home/oguyon/src/milk/plugins/coffee-src/PIAACMCsimul/scripts/coffee-optloop"
    "/home/oguyon/src/milk/plugins/coffee-src/PIAACMCsimul/scripts/coffee-runclean"
    "/home/oguyon/src/milk/plugins/coffee-src/PIAACMCsimul/scripts/coffee-runPIAACMC"
    "/home/oguyon/src/milk/plugins/coffee-src/PIAACMCsimul/scripts/coffee-setconf"
    "/home/oguyon/src/milk/plugins/coffee-src/PIAACMCsimul/scripts/coffee-syncscripts"
    "/home/oguyon/src/milk/plugins/coffee-src/PIAACMCsimul/scripts/coffee-run"
    "/home/oguyon/src/milk/plugins/coffee-src/PIAACMCsimul/scripts/coffee-runopt"
    "/home/oguyon/src/milk/plugins/coffee-src/PIAACMCsimul/scripts/coffee-runPIAACMCdesign"
    "/home/oguyon/src/milk/plugins/coffee-src/PIAACMCsimul/scripts/coffee-sim"
    "/home/oguyon/src/milk/plugins/coffee-src/PIAACMCsimul/scripts/coffee-waitforfile1"
    )
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
if(CMAKE_INSTALL_LOCAL_ONLY)
  file(WRITE "/home/oguyon/src/milk/build_local/plugins/coffee-src/PIAACMCsimul/install_local_manifest.txt"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
endif()
