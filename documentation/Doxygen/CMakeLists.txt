#
# Build the documentation
#
INCLUDE (${CMAKE_ROOT}/Modules/Documentation.cmake)


FIND_PACKAGE(LATEX)
IF(PDFLATEX_COMPILER)
  FILE(MAKE_DIRECTORY ${PROJECT_BINARY_DIR}/Doxygen/html)
  ADD_CUSTOM_TARGET( GeometryDocumentation
                     COMMAND ${PDFLATEX_COMPILER} ${PROJECT_SOURCE_DIR}/documentation/geometry.tex
                     DEPENDS ${PROJECT_SOURCE_DIR}/documentation/geometry.tex
                     WORKING_DIRECTORY ${PROJECT_BINARY_DIR}/Doxygen/html
                    )
  SET(LATEXTARGETS GeometryDocumentation)
ELSE()
  SET(LATEXTARGETS "")
ENDIF()

FIND_PACKAGE(Doxygen)
IF (BUILD_DOXYGEN)

  #
  # Configure the script and the doxyfile, then add target
  #
  CONFIGURE_FILE(${PROJECT_SOURCE_DIR}/documentation/Doxygen/doxygen.config.in
                 ${PROJECT_BINARY_DIR}/Doxygen/doxygen.config)

  CONFIGURE_FILE(${PROJECT_SOURCE_DIR}/documentation/Doxygen/itkdoxygen.pl.in
                 ${PROJECT_BINARY_DIR}/Doxygen/itkdoxygen.pl)

  ADD_CUSTOM_TARGET(Documentation 
    COMMAND ${DOXYGEN_EXECUTABLE} ${PROJECT_BINARY_DIR}/Doxygen/doxygen.config
    MAIN_DEPENDENCY ${PROJECT_BINARY_DIR}/Doxygen/doxygen.config
    DEPENDS         ${PROJECT_BINARY_DIR}/Doxygen/itkdoxygen.pl ${LATEXTARGETS}
    WORKING_DIRECTORY ${PROJECT_BINARY_DIR}/Doxygen
    )

ENDIF (BUILD_DOXYGEN)
