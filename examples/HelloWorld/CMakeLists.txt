cmake_minimum_required (VERSION 2.8)

# This project is designed to be built outside the RTK source tree.
project(HelloWorld)

# Find the RTK libraries and includes
find_package(RTK REQUIRED)
include(${RTK_USE_FILE})

# Executable
add_executable(HelloWorld HelloWorld.cxx )
target_link_libraries(HelloWorld ${RTK_LIBRARIES})
target_link_libraries(HelloWorld ${ITK_LIBRARIES})
