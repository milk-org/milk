#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "ImageStreamIO::ImageStreamIO" for configuration "Release"
set_property(TARGET ImageStreamIO::ImageStreamIO APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(ImageStreamIO::ImageStreamIO PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libImageStreamIO.so"
  IMPORTED_SONAME_RELEASE "libImageStreamIO.so"
  )

list(APPEND _cmake_import_check_targets ImageStreamIO::ImageStreamIO )
list(APPEND _cmake_import_check_files_for_ImageStreamIO::ImageStreamIO "${_IMPORT_PREFIX}/lib/libImageStreamIO.so" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
