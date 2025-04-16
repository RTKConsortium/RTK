cmake_minimum_required(VERSION 3.9.5 FATAL_ERROR)

# This project is designed to be built outside the RTK source tree.
project(AddNoise)

# Find ITK with RTK
find_package(ITK REQUIRED COMPONENTS RTK)
include(${ITK_USE_FILE})

# Executable(s)
add_executable(AddNoise AddNoise.cxx)
target_link_libraries(AddNoise ${ITK_LIBRARIES})
