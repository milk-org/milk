# ~~~
# have_magma.cmake
#
# Scans for MAGMA library
#
# Provides    HAVE_MAGMA symbol (unset / TRUE)
# May disable USE_MAGMA  symbol (OFF   / ON)
# ~~~

function(apply_magma_to_target target)
  if(NOT HAVE_MAGMA)
    message(
      WARNING "apply_magma_to_target: MAGMA not found, skipping ${target}")
    return()
  endif()
  target_include_directories(${target} PRIVATE ${MAGMA_INCLUDE_DIRS})
  target_link_directories(${target} PRIVATE ${MAGMA_LIBRARY_DIRS})
  target_link_libraries(${target} PRIVATE ${MAGMA_LIBRARIES})
  target_compile_options(${target} PRIVATE ${MAGMA_CFLAGS_OTHER})
endfunction()

pkg_check_modules(MAGMA magma)

if(MAGMA_FOUND)
  message(STATUS "Found MAGMA: ${MAGMA_LIBRARIES}")
  set(HAVE_MAGMA TRUE)
  add_compile_definitions(HAVE_MAGMA)
  include_directories(${MAGMA_INCLUDE_DIRS})
else()
  message(WARNING "No MAGMA support found -- USE_MAGMA is deactivated")
  set(USE_MAGMA OFF)
endif()
