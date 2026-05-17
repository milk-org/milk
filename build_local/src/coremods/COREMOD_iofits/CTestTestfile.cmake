# CMake generated Testfile for 
# Source directory: /home/oguyon/src/milk/src/coremods/COREMOD_iofits
# Build directory: /home/oguyon/src/milk/build_local/src/coremods/COREMOD_iofits
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(CLIfunc-COREMOD_iofits-module "milk-exec" "-T" "m?")
set_tests_properties(CLIfunc-COREMOD_iofits-module PROPERTIES  LABELS "CLImodule" PASS_REGULAR_EXPRESSION "COREMOD_iofits" TIMEOUT "1" _BACKTRACE_TRIPLES "/home/oguyon/src/milk/src/coremods/COREMOD_iofits/CMakeLists.txt;43;add_test;/home/oguyon/src/milk/src/coremods/COREMOD_iofits/CMakeLists.txt;0;")
add_test(CLIfunc-COREMOD_iofits-loadfits "milk-exec" "-T" "cmd? iofits.loadfits")
set_tests_properties(CLIfunc-COREMOD_iofits-loadfits PROPERTIES  LABELS "CLIfunc" PASS_REGULAR_EXPRESSION "src:" TIMEOUT "1" _BACKTRACE_TRIPLES "/home/oguyon/src/milk/src/coremods/COREMOD_iofits/CMakeLists.txt;56;add_test;/home/oguyon/src/milk/src/coremods/COREMOD_iofits/CMakeLists.txt;0;")
add_test(CLIfunc-COREMOD_iofits-saveFITS "milk-exec" "-T" "cmd? iofits.saveFITS")
set_tests_properties(CLIfunc-COREMOD_iofits-saveFITS PROPERTIES  LABELS "CLIfunc" PASS_REGULAR_EXPRESSION "src:" TIMEOUT "1" _BACKTRACE_TRIPLES "/home/oguyon/src/milk/src/coremods/COREMOD_iofits/CMakeLists.txt;56;add_test;/home/oguyon/src/milk/src/coremods/COREMOD_iofits/CMakeLists.txt;0;")
add_test(CLIfunc-COREMOD_iofits-breakcube "milk-exec" "-T" "cmd? iofits.breakcube")
set_tests_properties(CLIfunc-COREMOD_iofits-breakcube PROPERTIES  LABELS "CLIfunc" PASS_REGULAR_EXPRESSION "src:" TIMEOUT "1" _BACKTRACE_TRIPLES "/home/oguyon/src/milk/src/coremods/COREMOD_iofits/CMakeLists.txt;56;add_test;/home/oguyon/src/milk/src/coremods/COREMOD_iofits/CMakeLists.txt;0;")
add_test(CLIfunc-COREMOD_iofits-imgs2cube "milk-exec" "-T" "cmd? iofits.imgs2cube")
set_tests_properties(CLIfunc-COREMOD_iofits-imgs2cube PROPERTIES  LABELS "CLIfunc" PASS_REGULAR_EXPRESSION "src:" TIMEOUT "1" _BACKTRACE_TRIPLES "/home/oguyon/src/milk/src/coremods/COREMOD_iofits/CMakeLists.txt;56;add_test;/home/oguyon/src/milk/src/coremods/COREMOD_iofits/CMakeLists.txt;0;")
