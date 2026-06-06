# =======================================
# PROFILE-GUIDED OPTIMIZATION (PGO)
# =======================================
# Usage: cmake -DUSE_PGO=GENERATE ..   # instrument cmake -DUSE_PGO=USE .. #
# optimized See docs/pgo.md for the full workflow.
#
# Global flags apply to shared libraries. Standalone executables get per-target
# profile directories via milk_pgo_target() in MilkStandalone.cmake.
set(USE_PGO
    ""
    CACHE STRING "PGO mode: GENERATE or USE (empty = off)")
set(PGO_DIR
    "${CMAKE_BINARY_DIR}/pgo"
    CACHE PATH "Directory for PGO profile data")
if(USE_PGO STREQUAL "GENERATE")
  message(STATUS "PGO: instrumentation build (profiles → ${PGO_DIR})")
  add_compile_options(-fprofile-generate=${PGO_DIR}/shared)
  add_link_options(-fprofile-generate=${PGO_DIR}/shared)
elseif(USE_PGO STREQUAL "USE")
  message(STATUS "PGO: optimized build (profiles ← ${PGO_DIR})")
  add_compile_options(-fprofile-use=${PGO_DIR}/shared -fprofile-correction)
  add_link_options(-fprofile-use=${PGO_DIR}/shared)
endif()
