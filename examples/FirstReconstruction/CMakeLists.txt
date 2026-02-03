cmake_minimum_required(VERSION 3.9.5 FATAL_ERROR)

# This project is designed to be built outside the RTK source tree.
project(FirstReconstruction)

# Find ITK with RTK
find_package(ITK REQUIRED COMPONENTS RTK)
if("${ITK_VERSION_MAJOR}" VERSION_LESS "6")
  include(${ITK_USE_FILE})
endif()

# Executable(s)
add_executable(FirstReconstruction FirstReconstruction.cxx)
target_link_libraries(FirstReconstruction ${ITK_LIBRARIES})
if(${RTK_USE_CUDA})
  add_executable(FirstCudaReconstruction FirstCudaReconstruction.cxx)
  target_link_libraries(FirstCudaReconstruction ${ITK_LIBRARIES})
endif()
