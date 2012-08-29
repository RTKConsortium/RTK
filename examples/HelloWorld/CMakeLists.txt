CMAKE_MINIMUM_REQUIRED (VERSION 2.8)

# This project is designed to be built outside the RTK source tree.
PROJECT(HelloWorld)

# Find the RTK libraries and includes
FIND_PACKAGE(RTK REQUIRED)
INCLUDE(${RTK_USE_FILE})

# Executable
ADD_EXECUTABLE(HelloWorld HelloWorld.cxx )
TARGET_LINK_LIBRARIES(HelloWorld ${RTK_LIBRARIES})
TARGET_LINK_LIBRARIES(HelloWorld ${ITK_LIBRARIES})
