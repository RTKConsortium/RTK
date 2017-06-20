# This file sets up include directories, link directories, and
# compiler settings for a project to use RTK.  It should not be
# included directly, but rather through the RTK_USE_FILE setting
# obtained from RTKConfig.cmake.

# Find ITK (required)
find_package(ITK REQUIRED HINTS "${ITK_DIR}")
include(${ITK_USE_FILE})
set(ITK_LIBRARIES_SANS_VTK)
foreach(lib ${ITK_LIBRARIES})
  if(NOT lib MATCHES "ITKVtkGlue")
    list(APPEND ITK_LIBRARIES_SANS_VTK ${lib})
  endif()
endforeach()
set(ITK_LIBRARIES ${ITK_LIBRARIES_SANS_VTK})

# Add include directories needed to use RTK.
include_directories(BEFORE ${RTK_INCLUDE_DIRS})

# Add link directories needed to use RTK.
link_directories(${RTK_LIBRARY_DIRS})
