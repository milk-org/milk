# ~~~
# have_cfitsio.cmake
#
# Scans for CFITSIO library
#
# Provides    HAVE_CFITSIO symbol (unset / TRUE)
# May disable USE_CFITSIO  symbol (OFF   / ON)
# ~~~

pkg_check_modules(CFITSIO cfitsio)

if(CFITSIO_FOUND)
  message("Found cfitsio version ${CFITSIO_VERSION}")
  set(HAVE_CFITSIO TRUE)
  add_compile_definitions(HAVE_CFITSIO)
  include_directories(${CFITSIO_INCLUDE_DIRS})
else()
  message(WARNING "No CFITSIO support found -- USE_CFITSIO is deactivated")
  set(USE_CFITSIO OFF)
endif()

function(apply_cfitsio_to_target target)
  if(NOT CFITSIO_FOUND)
    message(
      WARNING "apply_cfitsio_to_target: CFITSIO not found, skipping ${target}")
    return()
  endif()
  target_compile_definitions(${target} PRIVATE USE_CFITSIO=1)
  target_link_directories(${target} PRIVATE ${CFITSIO_LIBRARY_DIRS})
  target_link_libraries(${target} PRIVATE ${CFITSIO_LIBRARIES})
endfunction()
