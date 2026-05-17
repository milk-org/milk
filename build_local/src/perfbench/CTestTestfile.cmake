# CMake generated Testfile for 
# Source directory: /home/oguyon/src/milk/src/perfbench
# Build directory: /home/oguyon/src/milk/build_local/src/perfbench
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(perf-clitest "/home/oguyon/src/milk/src/perfbench/tests/test-perf-clitest.sh" "100" "/home/oguyon/src/milk/build_local/perfresults")
set_tests_properties(perf-clitest PROPERTIES  ENVIRONMENT "PATH=/home/oguyon/src/milk/build_local/src/perfbench:/home/oguyon/src/milk/build_local/src/milk_module_example:/home/oguyon/src/milk/build_local/src/engine/libfps:/home/oguyon/src/milk/scripts:/home/oguyon/.opencode/bin:/home/oguyon/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin:/snap/bin:/usr/local/milk/bin:/home/linuxbrew/.linuxbrew/bin:/snap/ghostty/718/bin:/usr/local/milk/bin:/home/linuxbrew/.linuxbrew/bin" LABELS "perf" TIMEOUT "30" _BACKTRACE_TRIPLES "/home/oguyon/src/milk/src/perfbench/CMakeLists.txt;98;add_test;/home/oguyon/src/milk/src/perfbench/CMakeLists.txt;0;")
