# CMake generated Testfile for 
# Source directory: /home/oguyon/src/milk/src/coremods/COREMOD_arith
# Build directory: /home/oguyon/src/milk/build_local/src/coremods/COREMOD_arith
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(CLIfunc-COREMOD_arith-module "milk-exec" "-T" "m?")
set_tests_properties(CLIfunc-COREMOD_arith-module PROPERTIES  LABELS "CLImodule" PASS_REGULAR_EXPRESSION "COREMOD_arith" TIMEOUT "1" _BACKTRACE_TRIPLES "/home/oguyon/src/milk/src/coremods/COREMOD_arith/CMakeLists.txt;89;add_test;/home/oguyon/src/milk/src/coremods/COREMOD_arith/CMakeLists.txt;0;")
add_test(CLIfunc-COREMOD_arith-extractim "milk-exec" "-T" "cmd? arith.extractim")
set_tests_properties(CLIfunc-COREMOD_arith-extractim PROPERTIES  LABELS "CLIfunc" PASS_REGULAR_EXPRESSION "src:" TIMEOUT "1" _BACKTRACE_TRIPLES "/home/oguyon/src/milk/src/coremods/COREMOD_arith/CMakeLists.txt;102;add_test;/home/oguyon/src/milk/src/coremods/COREMOD_arith/CMakeLists.txt;0;")
add_test(CLIfunc-COREMOD_arith-extract3Dim "milk-exec" "-T" "cmd? arith.extract3Dim")
set_tests_properties(CLIfunc-COREMOD_arith-extract3Dim PROPERTIES  LABELS "CLIfunc" PASS_REGULAR_EXPRESSION "src:" TIMEOUT "1" _BACKTRACE_TRIPLES "/home/oguyon/src/milk/src/coremods/COREMOD_arith/CMakeLists.txt;102;add_test;/home/oguyon/src/milk/src/coremods/COREMOD_arith/CMakeLists.txt;0;")
add_test(CLIfunc-COREMOD_arith-setpix "milk-exec" "-T" "cmd? arith.setpix")
set_tests_properties(CLIfunc-COREMOD_arith-setpix PROPERTIES  LABELS "CLIfunc" PASS_REGULAR_EXPRESSION "src:" TIMEOUT "1" _BACKTRACE_TRIPLES "/home/oguyon/src/milk/src/coremods/COREMOD_arith/CMakeLists.txt;102;add_test;/home/oguyon/src/milk/src/coremods/COREMOD_arith/CMakeLists.txt;0;")
add_test(CLIfunc-COREMOD_arith-setpix1Drange "milk-exec" "-T" "cmd? arith.setpix1Drange")
set_tests_properties(CLIfunc-COREMOD_arith-setpix1Drange PROPERTIES  LABELS "CLIfunc" PASS_REGULAR_EXPRESSION "src:" TIMEOUT "1" _BACKTRACE_TRIPLES "/home/oguyon/src/milk/src/coremods/COREMOD_arith/CMakeLists.txt;102;add_test;/home/oguyon/src/milk/src/coremods/COREMOD_arith/CMakeLists.txt;0;")
add_test(CLIfunc-COREMOD_arith-setrow "milk-exec" "-T" "cmd? arith.setrow")
set_tests_properties(CLIfunc-COREMOD_arith-setrow PROPERTIES  LABELS "CLIfunc" PASS_REGULAR_EXPRESSION "src:" TIMEOUT "1" _BACKTRACE_TRIPLES "/home/oguyon/src/milk/src/coremods/COREMOD_arith/CMakeLists.txt;102;add_test;/home/oguyon/src/milk/src/coremods/COREMOD_arith/CMakeLists.txt;0;")
add_test(CLIfunc-COREMOD_arith-setcol "milk-exec" "-T" "cmd? arith.setcol")
set_tests_properties(CLIfunc-COREMOD_arith-setcol PROPERTIES  LABELS "CLIfunc" PASS_REGULAR_EXPRESSION "src:" TIMEOUT "1" _BACKTRACE_TRIPLES "/home/oguyon/src/milk/src/coremods/COREMOD_arith/CMakeLists.txt;102;add_test;/home/oguyon/src/milk/src/coremods/COREMOD_arith/CMakeLists.txt;0;")
add_test(CLIfunc-COREMOD_arith-imzero "milk-exec" "-T" "cmd? arith.imzero")
set_tests_properties(CLIfunc-COREMOD_arith-imzero PROPERTIES  LABELS "CLIfunc" PASS_REGULAR_EXPRESSION "src:" TIMEOUT "1" _BACKTRACE_TRIPLES "/home/oguyon/src/milk/src/coremods/COREMOD_arith/CMakeLists.txt;102;add_test;/home/oguyon/src/milk/src/coremods/COREMOD_arith/CMakeLists.txt;0;")
add_test(CLIfunc-COREMOD_arith-imtrunc "milk-exec" "-T" "cmd? arith.imtrunc")
set_tests_properties(CLIfunc-COREMOD_arith-imtrunc PROPERTIES  LABELS "CLIfunc" PASS_REGULAR_EXPRESSION "src:" TIMEOUT "1" _BACKTRACE_TRIPLES "/home/oguyon/src/milk/src/coremods/COREMOD_arith/CMakeLists.txt;102;add_test;/home/oguyon/src/milk/src/coremods/COREMOD_arith/CMakeLists.txt;0;")
add_test(CLIfunc-COREMOD_arith-cropmask "milk-exec" "-T" "cmd? arith.cropmask")
set_tests_properties(CLIfunc-COREMOD_arith-cropmask PROPERTIES  LABELS "CLIfunc" PASS_REGULAR_EXPRESSION "src:" TIMEOUT "1" _BACKTRACE_TRIPLES "/home/oguyon/src/milk/src/coremods/COREMOD_arith/CMakeLists.txt;102;add_test;/home/oguyon/src/milk/src/coremods/COREMOD_arith/CMakeLists.txt;0;")
