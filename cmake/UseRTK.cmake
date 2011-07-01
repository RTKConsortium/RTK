# This file sets up include directories, link directories, and
# compiler settings for a project to use RTK.  It should not be
# included directly, but rather through the RTK_USE_FILE setting
# obtained from RTKConfig.cmake.

# Find ITK (required)
FIND_PACKAGE(ITK REQUIRED)
INCLUDE(${ITK_USE_FILE})

# Add include directories needed to use RTK.
INCLUDE_DIRECTORIES(BEFORE ${RTK_INCLUDE_DIRS})

# Add link directories needed to use RTK.
LINK_DIRECTORIES(${RTK_LIBRARY_DIRS})
