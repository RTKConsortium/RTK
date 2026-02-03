cmake_minimum_required(VERSION 3.9.5 FATAL_ERROR)

# This project is designed to be built outside the RTK source tree.
project(InlineReconstruction)

# Find ITK with RTK
find_package(ITK REQUIRED COMPONENTS RTK)
include(itkVersion)
if("${ITK_VERSION_MAJOR}" VERSION_LESS "6")
  include(${ITK_USE_FILE})
endif()

# Executable(s)
add_executable(InlineReconstruction InlineReconstruction.cxx)
target_link_libraries(InlineReconstruction ${ITK_LIBRARIES})
