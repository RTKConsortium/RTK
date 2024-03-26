if(RTK_BUILD_SPHINX)

  set(RTK_DOC_OUTPUT_DIR ${CMAKE_BINARY_DIR}/documentation)

  # We must also duplicate the source tree since the images need to be present
  # with the source.
  add_custom_target(copy_sources ALL
    COMMAND ${CMAKE_COMMAND} -E copy_directory "${RTK_SOURCE_DIR}/examples" "${RTK_DOC_OUTPUT_DIR}/examples"
    COMMAND ${CMAKE_COMMAND} -E copy_directory "${RTK_SOURCE_DIR}/documentation" "${RTK_DOC_OUTPUT_DIR}/documentation"
    COMMAND ${CMAKE_COMMAND} -E copy "${RTK_SOURCE_DIR}/index.rst" "${RTK_DOC_OUTPUT_DIR}/index.rst"
    COMMAND ${CMAKE_COMMAND} -E copy "${RTK_SOURCE_DIR}/conf.py" "${RTK_DOC_OUTPUT_DIR}/conf.py"
    COMMAND ${CMAKE_COMMAND} -E copy "${RTK_SOURCE_DIR}/GettingStarted.md" "${RTK_DOC_OUTPUT_DIR}/GettingStarted.md"
    COMMAND ${CMAKE_COMMAND} -E copy "${RTK_SOURCE_DIR}/INSTALLATION.md" "${RTK_DOC_OUTPUT_DIR}/INSTALLATION.md"
    COMMENT "Copying documentation sources"
    )

  find_program(SPHINX_EXECUTABLE
    NAMES sphinx-build sphinx-build.exe
    DOC "Sphinx documentation generator"
  )

  add_custom_target( sphinx_doc ALL
    COMMAND ${SPHINX_EXECUTABLE}
    -c ${RTK_DOC_OUTPUT_DIR}
    ${RTK_DOC_OUTPUT_DIR}
    "${RTK_DOC_OUTPUT_DIR}/sphinx_output"
    COMMENT "Generating sphinx documentation"
    WORKING_DIRECTORY ${RTK_DOC_OUTPUT_DIR}
    )

  set_property(
    DIRECTORY APPEND PROPERTY
    ADDITIONAL_MAKE_CLEAN_FILES
    "${RTK_DOC_OUTPUT_DIR}/sphinx_output"
    )

endif()
