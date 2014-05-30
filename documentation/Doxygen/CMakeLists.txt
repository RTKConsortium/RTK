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

  # Information on how to retrieve the ITK documentation tag file
  SET(ITK_DOXYGEN_TAG_LOCATION "http://public.kitware.com/pub/itk/NightlyDoxygen/InsightDoxygenDocTag")
  SET(ITK_DOXYGEN_COMPRESSED_TAG_FILE ${PROJECT_BINARY_DIR}/Doxygen/InsightDoxygen.tag.gz)
  SET(ITK_DOXYGEN_TAG_FILE ${PROJECT_BINARY_DIR}/Doxygen/InsightDoxygen.tag)

  # Get the ITK documentation tag file
  IF(NOT EXISTS ${ITK_DOXYGEN_COMPRESSED_TAG_FILE})
    FILE( DOWNLOAD
          ${ITK_DOXYGEN_TAG_LOCATION}
	  ${ITK_DOXYGEN_COMPRESSED_TAG_FILE}
          TIMEOUT 60
          STATUS statusITKDoxygenTagFile
          SHOW_PROGRESS )
    LIST(GET statusITKDoxygenTagFile 0 statusITKDoxygenTagFile)
    IF(statusITKDoxygenTagFile)
	    FILE(REMOVE ${ITK_DOXYGEN_COMPRESSED_TAG_FILE})
    ENDIF()
  ENDIF(NOT EXISTS ${ITK_DOXYGEN_COMPRESSED_TAG_FILE})
  IF(EXISTS ${ITK_DOXYGEN_COMPRESSED_TAG_FILE})
    FIND_PROGRAM(GZIP_TOOL NAMES gzip)
    IF(GZIP_TOOL)
      EXECUTE_PROCESS(COMMAND ${GZIP_TOOL} -dkf ${ITK_DOXYGEN_COMPRESSED_TAG_FILE})    
      SET(DOXYGEN_TAGFILES_PARAMETER "${ITK_DOXYGEN_TAG_FILE}=http://www.itk.org/Doxygen/html/")
    ENDIF()
  ELSE()
    SET(DOXYGEN_TAGFILES_PARAMETER "")
  ENDIF()

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
