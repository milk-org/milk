# milk-CTRL CTest check (key-table documentation parity).
# Included from src/cli/CLIcore/CMakeLists.txt: relative paths resolve against that directory.
if(BUILD_TESTING)
  add_test(NAME milkctrl_keys_doc
           COMMAND bash ${CMAKE_SOURCE_DIR}/src/cli/overview/tests/check_keys_doc.sh $<TARGET_FILE:milk-CTRL>
                   ${CMAKE_SOURCE_DIR}/docs/guide/milk-CTRL.md)
endif()
