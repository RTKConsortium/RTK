cmake_minimum_required(VERSION 3.9.5 FATAL_ERROR)

# This project is designed to be built outside the RTK source tree.
project(FirstReconstruction)

# Find ITK with RTK
find_package(ITK REQUIRED COMPONENTS RTK)
include(${ITK_USE_FILE})
        
# Executable(s)
add_executable(FirstReconstruction FirstReconstruction.cxx )
target_link_libraries(FirstReconstruction ${RTK_LIBRARIES})
if(${RTK_USE_CUDA})
  add_executable(FirstCudaReconstruction FirstCudaReconstruction.cxx )
  target_link_libraries(FirstCudaReconstruction ${RTK_LIBRARIES})
endif()
