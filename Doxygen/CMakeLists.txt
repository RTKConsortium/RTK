#
# Build the documentation
#
INCLUDE (${CMAKE_ROOT}/Modules/Documentation.cmake)

IF (BUILD_DOXYGEN)

  #
  # Configure the script and the doxyfile, then add target
  #
  CONFIGURE_FILE(${PROJECT_SOURCE_DIR}/Doxygen/doxygen.config.in
                 ${PROJECT_BINARY_DIR}/Doxygen/doxygen.config)

  CONFIGURE_FILE(${PROJECT_SOURCE_DIR}/Doxygen/itkdoxygen.pl.in
                 ${PROJECT_BINARY_DIR}/Doxygen/itkdoxygen.pl)

  ADD_CUSTOM_TARGET(Documentation 
    COMMAND ${DOXYGEN_EXECUTABLE} ${PROJECT_BINARY_DIR}/Doxygen/doxygen.config
    MAIN_DEPENDENCY ${PROJECT_BINARY_DIR}/Doxygen/doxygen.config
    DEPENDS         ${PROJECT_BINARY_DIR}/Doxygen/itkdoxygen.pl
    WORKING_DIRECTORY ${PROJECT_BINARY_DIR}/Doxygen
    )

ENDIF (BUILD_DOXYGEN)
