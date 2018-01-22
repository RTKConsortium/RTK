cmake_minimum_required (VERSION 2.8)

# This project is designed to be built outside the RTK source tree.
project(HelloWorld)

# if this is built from within the project, finding rtk is not necessary
if (NOT TARGET RTK)
  # Find the RTK libraries and includes
  find_package(RTK REQUIRED)
  include(${RTK_USE_FILE})
endif()

# Executable
add_executable(HelloWorld HelloWorld.cxx )
target_link_libraries(HelloWorld ${RTK_LIBRARIES})
target_link_libraries(HelloWorld ${ITK_LIBRARIES})


MESSAGE("DEBUG RTK_LIBRARIES ${RTK_LIBRARIES}")