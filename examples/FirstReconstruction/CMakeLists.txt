CMAKE_MINIMUM_REQUIRED (VERSION 2.8)

# This project is designed to be built outside the RTK source tree.
PROJECT(FirstReconstruction)

# Find the RTK libraries and includes
FIND_PACKAGE(RTK REQUIRED)
INCLUDE(${RTK_USE_FILE})

# Executable
ADD_EXECUTABLE(FirstReconstruction FirstReconstruction.cxx )
TARGET_LINK_LIBRARIES(FirstReconstruction ${RTK_LIBRARIES})
TARGET_LINK_LIBRARIES(FirstReconstruction ${ITK_LIBRARIES})
