if(RTK_BUILD_DOXYGEN)
  find_package(UnixCommands)
  find_package(Doxygen)
  find_package(Gnuplot)
  find_package(HTMLHelp)
  find_package(Perl)
  find_package(Wget)

  #
  # Configure the script and the doxyfile, then add target
  #

  # Information on how to retrieve the ITK documentation tag file
  set(ITK_DOXYGEN_TAG_LOCATION "https://itk.org/files/NightlyDoxygen/InsightDoxygenDocTag.gz")
  set(ITK_DOXYGEN_COMPRESSED_TAG_FILE ${PROJECT_BINARY_DIR}/Doxygen/InsightDoxygen.tag.gz)
  set(ITK_DOXYGEN_TAG_FILE ${PROJECT_BINARY_DIR}/Doxygen/InsightDoxygen.tag)

  # Get the ITK documentation tag file
  if(NOT EXISTS ${ITK_DOXYGEN_COMPRESSED_TAG_FILE})
    file( DOWNLOAD
          ${ITK_DOXYGEN_TAG_LOCATION}
	  ${ITK_DOXYGEN_COMPRESSED_TAG_FILE}
          TIMEOUT 60
          STATUS statusITKDoxygenTagFile
          SHOW_PROGRESS )
    list(GET statusITKDoxygenTagFile 0 statusITKDoxygenTagFile)
    if(statusITKDoxygenTagFile)
	    file(REMOVE ${ITK_DOXYGEN_COMPRESSED_TAG_FILE})
    endif()
  endif()
  if(EXISTS ${ITK_DOXYGEN_COMPRESSED_TAG_FILE})
    find_program(GZIP_TOOL NAMES gzip)
    if(GZIP_TOOL)
      execute_process(COMMAND ${GZIP_TOOL} -dkf ${ITK_DOXYGEN_COMPRESSED_TAG_FILE})
      set(DOXYGEN_TAGFILES_PARAMETER "${ITK_DOXYGEN_TAG_FILE}=https://www.itk.org/Doxygen/html/")
    endif()
  else()
    set(DOXYGEN_TAGFILES_PARAMETER "")
  endif()

  configure_file(${PROJECT_SOURCE_DIR}/documentation/Doxygen/doxygen.config.in
                 ${PROJECT_BINARY_DIR}/Doxygen/doxygen.config)

  configure_file(${PROJECT_SOURCE_DIR}/documentation/Doxygen/itkdoxygen.pl.in
                 ${PROJECT_BINARY_DIR}/Doxygen/itkdoxygen.pl)

  add_custom_target(Documentation
    COMMAND ${DOXYGEN_EXECUTABLE} ${PROJECT_BINARY_DIR}/Doxygen/doxygen.config
    MAIN_DEPENDENCY ${PROJECT_BINARY_DIR}/Doxygen/doxygen.config
    DEPENDS         ${PROJECT_BINARY_DIR}/Doxygen/itkdoxygen.pl ${LATEXTARGETS}
    WORKING_DIRECTORY ${PROJECT_BINARY_DIR}/Doxygen
    )

endif ()
